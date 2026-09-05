import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ConvBlock(nn.Module):
    """Simple lightweight Conv-BatchNorm-ReLU block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ChangeMaskDecoder(nn.Module):
    """
    Multi-scale decoder with skip connections for bi-temporal change segmentation.
    Outputs 7-class logits for T1 and T2 semantic change masks (0 = No Change, 1-6 = Land Cover).
    """
    def __init__(self, num_classes=7):
        super().__init__()
        # Bottleneck projection: inputs are [f1_4, f2_4, |f1_4 - f2_4|] -> 512*3 = 1536 ch
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 * 3, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Stage 3: 256 (from bottle) + 256 (diff_3) = 512 -> 128
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = ConvBlock(256 + 256, 128)

        # Stage 2: 128 (from stage3) + 128 (diff_2) = 256 -> 64
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = ConvBlock(128 + 128, 64)

        # Stage 1: 64 (from stage2) + 64 (diff_1) = 128 -> 64
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBlock(64 + 64, 64)

        # Final upsampling to original resolution (4x: 64x64 -> 256x256)
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Semantic change mask heads for T1 and T2
        self.head_mask1 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.head_mask2 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, f1_feats, f2_feats):
        f1_1, f1_2, f1_3, f1_4 = f1_feats
        f2_1, f2_2, f2_3, f2_4 = f2_feats

        # Multi-scale absolute difference features
        diff_1 = torch.abs(f1_1 - f2_1)
        diff_2 = torch.abs(f1_2 - f2_2)
        diff_3 = torch.abs(f1_3 - f2_3)
        diff_4 = torch.abs(f1_4 - f2_4)

        # Bottleneck: combine T1, T2, and their absolute difference
        f_cat = torch.cat([f1_4, f2_4, diff_4], dim=1)
        x = self.bottleneck(f_cat)  # (B, 256, 8, 8)

        # Decoder stages with skip connections
        x = self.up3(x)  # (B, 256, 16, 16)
        x = self.conv3(torch.cat([x, diff_3], dim=1))  # (B, 128, 16, 16)

        x = self.up2(x)  # (B, 128, 32, 32)
        x = self.conv2(torch.cat([x, diff_2], dim=1))  # (B, 64, 32, 32)

        x = self.up1(x)  # (B, 64, 64, 64)
        x = self.conv1(torch.cat([x, diff_1], dim=1))  # (B, 64, 64, 64)

        x = self.final_up(x)  # (B, 64, 256, 256)
        x = self.final_conv(x)  # (B, 32, 256, 256)

        logits_mask1 = self.head_mask1(x)
        logits_mask2 = self.head_mask2(x)
        return logits_mask1, logits_mask2, diff_4


class QuestionEncoder(nn.Module):
    """
    BiGRU question encoder mapping question tokens into a fixed-size embedding.
    """
    def __init__(self, vocab_size=60, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim * 2, 256)

    def forward(self, q_tokens):
        # q_tokens: (B, seq_len)
        embedded = self.embedding(q_tokens)  # (B, seq_len, embed_dim)
        _, h_n = self.gru(embedded)  # h_n: (2, B, hidden_dim)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, hidden_dim * 2)
        return self.proj(h_cat)  # (B, 256)


class SpatialCrossAttentionVQAHead(nn.Module):
    """
    Spatial Cross-Attention VQA Head:
    Allows question embedding to directly attend across 2D spatial patches (8x8=64)
    of the visual difference feature map to isolate the precise region asked about.
    """
    def __init__(self, visual_dim=512, q_dim=256, num_answers=19, num_heads=4, dropout=0.2):
        super().__init__()
        self.v_proj = nn.Linear(visual_dim, 256)
        self.q_proj = nn.Linear(q_dim, 256)
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_answers)
        )

    def forward(self, diff_map, q_feat):
        # diff_map: (B, 512, H, W), q_feat: (B, 256)
        B, C, H, W = diff_map.shape
        # Spatial tokens: (B, H*W, 512)
        v_tokens = diff_map.flatten(2).transpose(1, 2)
        v_proj = self.v_proj(v_tokens)                    # (B, 64, 256)
        q_proj = self.q_proj(q_feat).unsqueeze(1)         # (B, 1, 256)

        # Cross-Attention: Question attends to spatial patches
        attn_out, _ = self.cross_attn(query=q_proj, key=v_proj, value=v_proj)  # (B, 1, 256)
        v_attended = self.norm1(attn_out.squeeze(1) + q_proj.squeeze(1))       # (B, 256)

        # Global context pool
        v_gap = self.norm2(v_proj.mean(dim=1))                                 # (B, 256)

        # Combine attended representation and global context
        fused = torch.cat([v_attended, v_gap], dim=-1)                         # (B, 512)
        logits = self.classifier(fused)
        return logits


class BiTemporalChangeModel(nn.Module):
    """
    Joint Bi-Temporal Change Detection (Task 3) + Change-VQA (Task 4) model.
    Supports both pretrained ImageNet weights and training from scratch.
    """
    def __init__(self, vocab_size=60, num_classes=7, num_answers=19, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Task 3: Change Mask Decoder
        self.mask_decoder = ChangeMaskDecoder(num_classes=num_classes)

        # Task 4: Question Encoder & Spatial Cross-Attention VQA Head
        self.question_encoder = QuestionEncoder(vocab_size=vocab_size)
        self.vqa_head = SpatialCrossAttentionVQAHead(visual_dim=512, q_dim=256, num_answers=num_answers)

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)  # 64 ch, H/4
        f2 = self.layer2(f1) # 128 ch, H/8
        f3 = self.layer3(f2) # 256 ch, H/16
        f4 = self.layer4(f3) # 512 ch, H/32
        return (f1, f2, f3, f4)

    def forward(self, t1, t2, q_tokens=None):
        f1_feats = self.extract_features(t1)
        f2_feats = self.extract_features(t2)

        # Task 3: Multi-Scale Change Mask Prediction
        logits_mask1, logits_mask2, diff_4 = self.mask_decoder(f1_feats, f2_feats)

        # Task 4: Spatial Cross-Attention VQA Prediction
        logits_vqa = None
        if q_tokens is not None:
            q_emb = self.question_encoder(q_tokens)  # (B, 256)
            logits_vqa = self.vqa_head(diff_4, q_emb)

        return {
            'logits_mask1': logits_mask1,
            'logits_mask2': logits_mask2,
            'logits_vqa': logits_vqa
        }


# --- Rule-Based Description Generator for Task 3 ---
def generate_change_description(pred_mask1: torch.Tensor, pred_mask2: torch.Tensor) -> str:
    """
    Computes per-class percentage changes between T1 and T2 predicted masks
    and produces a concise, readable textual change summary (Task 3).
    """
    class_names_clean = {
        1: "Non-vegetated ground surface",
        2: "Tree cover",
        3: "Low vegetation",
        4: "Water body",
        5: "Built-up area",
        6: "Playground area"
    }

    total_pixels = pred_mask1.numel()
    statements = []

    changed_pixels = ((pred_mask1 > 0) | (pred_mask2 > 0)).sum().item()
    total_change_pct = (changed_pixels / total_pixels) * 100.0

    for cls_idx, name in class_names_clean.items():
        cnt1 = (pred_mask1 == cls_idx).sum().item()
        cnt2 = (pred_mask2 == cls_idx).sum().item()
        pct1 = (cnt1 / total_pixels) * 100.0
        pct2 = (cnt2 / total_pixels) * 100.0
        delta = pct2 - pct1

        if abs(delta) >= 0.5:
            if delta > 0:
                statements.append(f"{name} increased by {abs(delta):.1f}%")
            else:
                statements.append(f"{name} decreased by {abs(delta):.1f}%")

    if not statements:
        if total_change_pct < 1.0:
            return "No significant land-cover change detected between the two images."
        else:
            return f"Minor land-cover changes detected affecting {total_change_pct:.1f}% of the area."

    return f"Between the two images, " + "; ".join(statements) + f" (total changed area: {total_change_pct:.1f}%)."
