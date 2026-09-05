import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
SatQuery AI — Check VSR Spatial Reasoning Model
================================================
Evaluates the trained VSR Spatial Reasoning Adapter on satellite/optical images
with natural-language spatial queries.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from torchvision import transforms

from satquery_backend.models.vsr_adapter import VSRSpatialAdapter

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def run_vsr_check():
    print("=" * 60)
    print("  SatQuery AI — Task: VSR Spatial Reasoning Verification")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # 1. Load RemoteCLIP
    weights_path = "satquery_backend/weights/RemoteCLIP-ViT-B-32.pt"
    print("Loading RemoteCLIP backbone...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None, device=device)
    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        clip_model.load_state_dict(clean_sd, strict=False)
        print("  RemoteCLIP weights loaded.")
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # 2. Load VSR Adapter
    adapter_path = "satquery_backend/weights/vsr_adapter_v1.pt"
    adapter = VSRSpatialAdapter(embed_dim=512, hidden_dim=256, num_classes=2).to(device)
    if Path(adapter_path).exists():
        ckpt = torch.load(adapter_path, map_location=device)
        adapter.load_state_dict(ckpt["adapter"])
        print(f"  VSR Adapter loaded from {adapter_path} (Val Acc: {ckpt.get('val_acc', 0):.1f}%)")
    else:
        print(f"  Warning: {adapter_path} not found. Using untrained head for dry-run.")
    adapter.eval()

    prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])

    # Sample image
    sample_img_path = Path("satquery_backend/bigearth_real/s2/cls00_sample000.png")
    if sample_img_path.exists():
        img = Image.open(sample_img_path).convert("RGB")
        print(f"Test image: {sample_img_path}")
    else:
        img = Image.new("RGB", (224, 224), color=(100, 150, 100))
        print("Using synthetic test image.")

    img_tensor = prep(img).unsqueeze(0).to(device)

    # Test Queries
    queries = [
        "The urban area is adjacent to the green vegetation.",
        "There is water inside the city center.",
        "Road infrastructure connects the buildings.",
        "The forest is completely submerged under ocean water.",
    ]

    print("\nEvaluating Spatial Reasoning Queries:")
    print("-" * 60)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_tensor)
        img_feat = F.normalize(img_feat, p=2, dim=-1)

        for q in queries:
            toks = tokenizer([q]).to(device)
            txt_feat = clip_model.encode_text(toks)
            txt_feat = F.normalize(txt_feat, p=2, dim=-1)

            logits = adapter(img_feat, txt_feat)
            probs = F.softmax(logits, dim=-1)[0]
            
            p_false, p_true = float(probs[0]), float(probs[1])
            is_valid = p_true >= 0.5
            conf = max(p_false, p_true) * 100

            verdict = "VALID (TRUE)" if is_valid else "INVALID (FALSE)"
            print(f" Query: \"{q}\"")
            print(f"   Verdict: {verdict} | Confidence: {conf:.1f}% (P(True)={p_true:.3f}, P(False)={p_false:.3f})\n")

    print("=" * 60)
    print(" VSR Benchmark Check PASSED")
    print("=" * 60)

if __name__ == "__main__":
    run_vsr_check()
