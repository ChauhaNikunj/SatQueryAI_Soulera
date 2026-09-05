import os
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import CDVQADataset, QuestionTokenizer, ANS_TO_IDX, IDX_TO_ANS
from model import BiTemporalChangeModel, generate_change_description


def parse_args():
    parser = argparse.ArgumentParser(description="State-of-the-Art Training on RTX 4050 6GB")
    parser.add_argument("--im1_dir", type=str, default="C:/satquery/im1")
    parser.add_argument("--im2_dir", type=str, default="C:/satquery/im2")
    parser.add_argument("--label1_dir", type=str, default="C:/satquery/label1")
    parser.add_argument("--label2_dir", type=str, default="C:/satquery/label2")
    parser.add_argument("--cdvqa_dir", type=str, default="C:/satquery/CDVQA-main")
    parser.add_argument("--subsample_count", type=int, default=0, help="0 = Full Dataset (65K QA pairs), or e.g. 800")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights")
    parser.add_argument("--image_size", type=int, default=256, help="256 or 512 native resolution")
    parser.add_argument("--batch_size", type=int, default=16, help="16 for 256px (optimal for RTX 4050)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dice_weight", type=float, default=0.5, help="Weight for multi-class Dice loss")
    parser.add_argument("--lambda_vqa", type=float, default=1.0, help="Weight for VQA loss")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing for non-ordinal VQA")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="C:/satquery/checkpoints")
    return parser.parse_args()


class RobustChangeLoss(nn.Module):
    """
    1. Class-weighted Cross-Entropy: down-weights background (0.5) and heavily penalizes
       missing changed classes (2.5 - 3.5).
    2. Binary Change Dice Loss: penalizes predicting 'no change' when real changes exist.
       Mathematically impossible to collapse to all-background!
    3. Ordinal-smoothed CrossEntropy for 10% VQA numerical ratio bands.
    """
    def __init__(self, num_classes=7, dice_weight=0.5, label_smoothing=0.05, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        self.device = device

        # Background gets 0.5; Changed classes get 2.5 to 3.5
        weights = torch.tensor([0.5, 3.0, 2.5, 2.5, 3.0, 2.5, 3.5], device=device)
        self.ce_mask = nn.CrossEntropyLoss(weight=weights)

    def binary_change_dice(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        p_change = 1.0 - probs[:, 0, :, :]
        gt_change = (targets > 0).float()

        inter = (p_change * gt_change).sum(dim=(1, 2))
        card = p_change.sum(dim=(1, 2)) + gt_change.sum(dim=(1, 2))
        dice = (2.0 * inter + 1e-4) / (card + 1e-4)
        return 1.0 - dice.mean()

    def ordinal_vqa_loss(self, logits, targets):
        B, num_classes = logits.shape
        smooth_targets = torch.zeros_like(logits)

        for b in range(B):
            t = targets[b].item()
            if 8 <= t <= 18:
                smooth_targets[b, t] = 0.70
                if t > 8:
                    smooth_targets[b, t - 1] += 0.15
                else:
                    smooth_targets[b, t] += 0.15
                if t < 18:
                    smooth_targets[b, t + 1] += 0.15
                else:
                    smooth_targets[b, t] += 0.15
            else:
                smooth_targets[b].fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets[b, t] = 1.0 - self.label_smoothing

        log_probs = F.log_softmax(logits, dim=-1)
        return -(smooth_targets * log_probs).sum(dim=-1).mean()

    def forward(self, logits_m1, logits_m2, logits_vqa, m1, m2, ans_target):
        l_ce1 = self.ce_mask(logits_m1, m1)
        l_ce2 = self.ce_mask(logits_m2, m2)
        l_dice1 = self.binary_change_dice(logits_m1, m1)
        l_dice2 = self.binary_change_dice(logits_m2, m2)

        loss_mask = (l_ce1 + l_ce2) + self.dice_weight * (l_dice1 + l_dice2)
        loss_vqa = self.ordinal_vqa_loss(logits_vqa, ans_target)
        return loss_mask, loss_vqa


def compute_vqa_accuracies(preds, targets):
    strict_correct = (preds == targets).sum().item()
    adj_correct = 0

    for p, t in zip(preds, targets):
        p_val = p.item()
        t_val = t.item()
        if p_val == t_val:
            adj_correct += 1
        elif 8 <= t_val <= 18 and 8 <= p_val <= 18 and abs(p_val - t_val) == 1:
            adj_correct += 1

    return strict_correct, adj_correct


def train_one_epoch(model, loader, optimizer, criterion, lambda_vqa, grad_accum, device):
    model.train()
    running_loss = 0.0
    running_mask = 0.0
    running_vqa = 0.0
    correct_vqa = 0
    adj_vqa = 0
    total_vqa = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        t1 = batch['t1'].to(device, non_blocking=True)
        t2 = batch['t2'].to(device, non_blocking=True)
        m1 = batch['mask1'].to(device, non_blocking=True)
        m2 = batch['mask2'].to(device, non_blocking=True)
        q_tokens = batch['question_tokens'].to(device, non_blocking=True)
        ans_target = batch['ans_target'].to(device, non_blocking=True)

        out = model(t1, t2, q_tokens)
        loss_mask, loss_vqa = criterion(
            out['logits_mask1'], out['logits_mask2'], out['logits_vqa'],
            m1, m2, ans_target
        )
        total_loss = (loss_mask + lambda_vqa * loss_vqa) / grad_accum

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            optimizer.zero_grad()
            continue

        total_loss.backward()

        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += total_loss.item() * grad_accum
        running_mask += loss_mask.item()
        running_vqa += loss_vqa.item()

        preds = out['logits_vqa'].argmax(dim=1)
        sc, ac = compute_vqa_accuracies(preds, ans_target)
        correct_vqa += sc
        adj_vqa += ac
        total_vqa += ans_target.size(0)

    num_b = len(loader)
    top1 = (correct_vqa / total_vqa) * 100.0
    adj1 = (adj_vqa / total_vqa) * 100.0
    return running_loss / num_b, running_mask / num_b, running_vqa / num_b, top1, adj1


@torch.no_grad()
def evaluate(model, loader, criterion, lambda_vqa, device):
    model.eval()
    running_loss = 0.0
    correct_vqa = 0
    adj_vqa = 0
    total_vqa = 0

    intersection_strict = 0
    union_strict = 0
    intersection_trimap = 0
    union_trimap = 0
    total_pixels_evaluated = 0
    total_pixels_correct = 0

    kernel = np.ones((5, 5), np.uint8)  # 2-pixel radius margin for boundary relaxation

    for batch in loader:
        t1 = batch['t1'].to(device, non_blocking=True)
        t2 = batch['t2'].to(device, non_blocking=True)
        m1 = batch['mask1'].to(device, non_blocking=True)
        m2 = batch['mask2'].to(device, non_blocking=True)
        q_tokens = batch['question_tokens'].to(device, non_blocking=True)
        ans_target = batch['ans_target'].to(device, non_blocking=True)

        out = model(t1, t2, q_tokens)
        loss_mask, loss_vqa = criterion(
            out['logits_mask1'], out['logits_mask2'], out['logits_vqa'],
            m1, m2, ans_target
        )
        total_loss = loss_mask + lambda_vqa * loss_vqa

        running_loss += total_loss.item()
        preds = out['logits_vqa'].argmax(dim=1)
        sc, ac = compute_vqa_accuracies(preds, ans_target)
        correct_vqa += sc
        adj_vqa += ac
        total_vqa += ans_target.size(0)

        # Mask predictions
        pred_m1 = out['logits_mask1'].argmax(dim=1)
        pred_m2 = out['logits_mask2'].argmax(dim=1)

        # Overall Pixel Accuracy
        pix_corr = ((pred_m1 == m1) & (pred_m2 == m2)).sum().item()
        total_pixels_correct += pix_corr
        total_pixels_evaluated += m1.numel()

        # Binary change maps
        gt_change = ((m1 > 0) | (m2 > 0)).cpu().numpy()
        pred_change = ((pred_m1 > 0) | (pred_m2 > 0)).cpu().numpy()

        for b in range(gt_change.shape[0]):
            g_b = gt_change[b]
            p_b = pred_change[b]

            # Strict IoU
            inter = np.logical_and(p_b, g_b).sum()
            union = np.logical_or(p_b, g_b).sum()
            intersection_strict += inter
            union_strict += union

            # Trimap (2px relaxed boundary)
            dilated = cv2.dilate(g_b.astype(np.uint8), kernel)
            eroded = cv2.erode(g_b.astype(np.uint8), kernel)
            boundary = (dilated - eroded) > 0
            valid = ~boundary

            intersection_trimap += np.logical_and(p_b[valid], g_b[valid]).sum()
            union_trimap += np.logical_or(p_b[valid], g_b[valid]).sum()

    avg_loss = running_loss / len(loader)
    top1 = (correct_vqa / total_vqa) * 100.0 if total_vqa > 0 else 0.0
    adj1 = (adj_vqa / total_vqa) * 100.0 if total_vqa > 0 else 0.0
    strict_iou = (intersection_strict / (union_strict + 1e-7)) * 100.0
    trimap_iou = (intersection_trimap / (union_trimap + 1e-7)) * 100.0
    overall_acc = (total_pixels_correct / (total_pixels_evaluated + 1e-7)) * 100.0

    return avg_loss, top1, adj1, strict_iou, trimap_iou, overall_acc


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    subsample_display = "FULL DATASET (1600 pairs, 65K QA)" if args.subsample_count == 0 else f"{args.subsample_count} pairs"
    print("=" * 80)
    print("MAXIMUM ACCURACY PIPELINE: BI-TEMPORAL CHANGE + CHANGE-VQA")
    print(f"GPU: {torch.cuda.get_device_name(0)} (6GB VRAM)")
    print(f"Pretrained Backbone: {args.pretrained} | Cross-Attention VQA: YES")
    print(f"Image Resolution: {args.image_size}x{args.image_size} | Dataset: {subsample_display}")
    print(f"Batch Size: {args.batch_size} (Grad Accum: {args.grad_accum}) | Epochs: {args.epochs}")
    print("=" * 80)

    tokenizer = QuestionTokenizer()
    sub_count = 1600 if args.subsample_count == 0 else args.subsample_count

    # If image_size is 512 and full dataset, disable RAM caching if system RAM is limited,
    # or keep cache_in_ram=True if RAM is sufficient.
    cache_ram = (args.image_size <= 256 or sub_count <= 800)

    train_dataset = CDVQADataset(
        im1_dir=args.im1_dir,
        im2_dir=args.im2_dir,
        label1_dir=args.label1_dir,
        label2_dir=args.label2_dir,
        cdvqa_dir=args.cdvqa_dir,
        split='Train',
        image_size=args.image_size,
        subsample_count=sub_count,
        tokenizer=tokenizer,
        augment=True,
        cache_in_ram=cache_ram
    )

    val_dataset = CDVQADataset(
        im1_dir=args.im1_dir,
        im2_dir=args.im2_dir,
        label1_dir=args.label1_dir,
        label2_dir=args.label2_dir,
        cdvqa_dir=args.cdvqa_dir,
        split='Val',
        image_size=args.image_size,
        subsample_count=150,
        tokenizer=tokenizer,
        augment=False,
        cache_in_ram=cache_ram
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = BiTemporalChangeModel(
        vocab_size=tokenizer.vocab_size + 10,
        num_classes=7,
        num_answers=len(ANS_TO_IDX),
        pretrained=args.pretrained
    ).to(device)

    criterion = RobustChangeLoss(
        num_classes=7,
        dice_weight=args.dice_weight,
        label_smoothing=args.label_smoothing,
        device=device
    )

    backbone_params = [p for n, p in model.named_parameters() if 'conv1' in n or 'layer' in n or 'bn1' in n]
    head_params = [p for n, p in model.named_parameters() if not ('conv1' in n or 'layer' in n or 'bn1' in n)]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.5},
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_score = 0.0
    print("\n---> Training Loop Started...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        train_loss, train_mask, train_vqa, tr_top1, tr_adj = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            lambda_vqa=args.lambda_vqa,
            grad_accum=args.grad_accum,
            device=device
        )
        scheduler.step()

        val_loss, val_top1, val_adj, strict_iou, trimap_iou, overall_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            lambda_vqa=args.lambda_vqa,
            device=device
        )

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"VQA Top-1: {val_top1:.1f}% (Adj: {val_adj:.1f}%) | "
              f"Change IoU: {strict_iou:.1f}% (Trimap 2px: {trimap_iou:.1f}%) | "
              f"Pixel OA: {overall_acc:.1f}% | VRAM: {peak_vram_mb:.0f}MB")

        ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_top1': val_top1,
            'val_adj': val_adj,
            'strict_iou': strict_iou,
            'trimap_iou': trimap_iou,
            'overall_acc': overall_acc
        }, ckpt_path)

        score = val_adj + trimap_iou
        if score > best_score:
            best_score = score
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  --> Saved new BEST model: VQA Adj={val_adj:.1f}%, Trimap IoU={trimap_iou:.1f}%, OA={overall_acc:.1f}%")

    total_hours = (time.time() - start_time) / 3600.0
    print(f"\nTraining completed in {total_hours:.2f} hours.")
    print(f"Best model saved at: {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()
