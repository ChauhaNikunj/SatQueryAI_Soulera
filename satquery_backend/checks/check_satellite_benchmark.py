from __future__ import annotations

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import time
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from satquery_backend.train_satellite_benchmark import SatelliteVisionAdapter, EUROSAT_PROMPTS

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def run_check():
    print("=" * 68)
    print("  🛰️  SatQuery AI — Satellite Benchmark (EuroSAT Sentinel-2) Check")
    print("=" * 68)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    weights_path = "satquery_backend/weights/RemoteCLIP-ViT-B-32.pt"
    adapter_path = "satquery_backend/weights/satellite_benchmark_adapter.pt"

    # 1. Load RemoteCLIP
    print("Loading RemoteCLIP Vision-Language Backbone...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None, device=device)
    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        clip_model.load_state_dict(clean_sd, strict=False)
        print("  RemoteCLIP weights loaded.")
    clip_model.eval()

    # 2. Load Adapter
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    adapter = SatelliteVisionAdapter(in_dim=512, hidden_dim=256, num_classes=len(classes)).to(device)
    if Path(adapter_path).exists():
        ckpt = torch.load(adapter_path, map_location=device)
        adapter.load_state_dict(ckpt["adapter"])
        val_acc = ckpt.get("val_acc", 0.0)
        classes = ckpt.get("classes", classes)
        print(f"  Satellite Vision Adapter loaded (Validation Accuracy: {val_acc:.2f}%)")
    else:
        print(f"  Warning: {adapter_path} not found. Run train_satellite_benchmark.py first.")
    adapter.eval()

    prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])

    # Test several real satellite images across different categories
    test_samples = [
        ("Forest Satellite Image", Path("satquery_backend/data/eurosat/2750/Forest/Forest_1.jpg")),
        ("Sea / Lake / Dam Satellite Image", Path("satquery_backend/data/eurosat/2750/SeaLake/SeaLake_1.jpg")),
        ("River Satellite Image", Path("satquery_backend/data/eurosat/2750/River/River_1.jpg")),
        ("Highway Satellite Image", Path("satquery_backend/data/eurosat/2750/Highway/Highway_1.jpg")),
        ("Residential Urban Satellite Image", Path("satquery_backend/data/eurosat/2750/Residential/Residential_1.jpg")),
    ]

    print("\n" + "-" * 68)
    print(f"{'Target Category':<30} | {'Predicted Class':<18} | {'Confidence':<10}")
    print("-" * 68)

    for label, img_path in test_samples:
        if not img_path.exists():
            print(f"File {img_path} not found.")
            continue

        img = Image.open(img_path).convert("RGB")
        tensor = prep(img).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            feat = clip_model.encode_image(tensor)
            feat = F.normalize(feat, p=2, dim=-1)
            logits = adapter(feat)
            probs = F.softmax(logits, dim=-1)[0]

        top_val, top_idx = torch.topk(probs, k=1)
        pred_class = classes[top_idx.item()]
        conf = float(top_val.item()) * 100.0
        latency = (time.perf_counter() - t0) * 1000.0

        print(f"{label:<30} | {pred_class:<18} | {conf:>6.1f}% ({latency:.1f}ms)")

    print("-" * 68)
    print(" Satellite Benchmark Verification COMPLETE")
    print("=" * 68)

if __name__ == "__main__":
    run_check()
