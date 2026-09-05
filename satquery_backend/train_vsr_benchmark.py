from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import open_clip
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from satquery_backend.models.vsr_adapter import VSRSpatialAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vsr_train")

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class VSRBenchmarkDataset(Dataset):
    """
    Dataset loader for VSR benchmark with local image caching.
    """
    def __init__(self, samples: List[dict], cache_dir: Path, transform=None):
        self.samples = samples
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _fetch_image(self, item: dict) -> Image.Image:
        img_filename = item["image"]
        cached_path = self.cache_dir / img_filename
        
        if cached_path.exists():
            try:
                return Image.open(cached_path).convert("RGB")
            except Exception:
                pass
                
        # Download from URL if not cached
        url = item["image_link"]
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img.save(cached_path)
                return img
        except Exception as e:
            log.warning("Failed to fetch %s: %s", url, e)
            
        # Fallback dummy RGB image if network fails
        return Image.new("RGB", (224, 224), color=(128, 128, 128))

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        image = self._fetch_image(item)
        img_t = self.transform(image)
        caption = item["caption"]
        label = int(item["label"])  # 0 or 1
        relation = item.get("relation", "")
        
        return {
            "image": img_t,
            "caption": caption,
            "label": torch.tensor(label, dtype=torch.long),
            "relation": relation,
        }


def load_remoteclip(weights_path: str, device: torch.device):
    """Load RemoteCLIP ViT-B/32 backbone."""
    log.info("Loading RemoteCLIP from %s ...", weights_path)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=None,
        device=device,
    )
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip prefix if needed
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_sd, strict=False)
        log.info("Loaded pretrained RemoteCLIP weights.")
    else:
        log.warning("Weights not found at %s. Using default init.", weights_path)
        
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, tokenizer, preprocess


def prefetch_images(samples: List[dict], cache_dir: Path, max_workers: int = 16):
    """Prefetch images in parallel to speed up training."""
    log.info("Prefetching %d benchmark images into %s ...", len(samples), cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _download_single(item):
        filename = item["image"]
        dest = cache_dir / filename
        if not dest.exists():
            try:
                r = requests.get(item["image_link"], timeout=8)
                if r.status_code == 200:
                    with open(dest, "wb") as f:
                        f.write(r.content)
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_download_single, samples))
    log.info("Prefetching completed.")


def train_vsr(
    num_samples: int = 300,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    weights_path: str = "satquery_backend/weights/RemoteCLIP-ViT-B-32.pt",
    output_path: str = "satquery_backend/weights/vsr_adapter_v1.pt",
):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Training Device: %s", device)

    # 1. Load VSR Dataset from Hugging Face
    log.info("Fetching VSR Benchmark dataset from Hugging Face (cambridgeltl/vsr_random)...")
    raw_train = load_dataset("cambridgeltl/vsr_random", split=f"train[:{num_samples}]")
    raw_val = load_dataset("cambridgeltl/vsr_random", split="validation[:80]")
    
    train_samples = [s for s in raw_train]
    val_samples = [s for s in raw_val]
    log.info("Loaded %d train samples and %d val samples.", len(train_samples), len(val_samples))

    # Prefetch images
    cache_dir = Path("satquery_backend/data/vsr_cache")
    prefetch_images(train_samples + val_samples, cache_dir)

    # 2. Setup Backbone & Tokenizer
    clip_model, tokenizer, _ = load_remoteclip(weights_path, device)

    # 3. Setup Dataset & DataLoader
    train_dataset = VSRBenchmarkDataset(train_samples, cache_dir)
    val_dataset = VSRBenchmarkDataset(val_samples, cache_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Setup Spatial Adapter Model
    adapter = VSRSpatialAdapter(embed_dim=512, hidden_dim=256, num_classes=2).to(device)
    optimizer = AdamW(adapter.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("Starting training loop for %d epochs...", epochs)
    print("=" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^10} | {'Val Acc':^10} | {'LR':^8}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        adapter.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            captions = batch["caption"]
            labels = batch["label"].to(device)

            # Extract frozen CLIP features
            with torch.no_grad():
                img_feats = clip_model.encode_image(imgs)
                img_feats = F.normalize(img_feats, p=2, dim=-1)

                tokens = tokenizer(captions).to(device)
                txt_feats = clip_model.encode_text(tokens)
                txt_feats = F.normalize(txt_feats, p=2, dim=-1)

            optimizer.zero_grad()
            logits = adapter(img_feats, txt_feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total) * 100.0

        # Validation loop
        adapter.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                captions = batch["caption"]
                labels = batch["label"].to(device)

                img_feats = clip_model.encode_image(imgs)
                img_feats = F.normalize(img_feats, p=2, dim=-1)

                tokens = tokenizer(captions).to(device)
                txt_feats = clip_model.encode_text(tokens)
                txt_feats = F.normalize(txt_feats, p=2, dim=-1)

                logits = adapter(img_feats, txt_feats)
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss = val_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total) * 100.0
        cur_lr = scheduler.get_last_lr()[0]

        print(f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc:^10.1f}% | {val_loss:^10.4f} | {val_acc:^9.1f}% | {cur_lr:^8.2e}")

        if val_acc > best_val_acc or epoch == epochs:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "adapter": adapter.state_dict(),
                "val_acc": val_acc,
                "embed_dim": 512,
                "hidden_dim": 256,
            }, output_path)

    print("=" * 70)
    log.info("Training complete! Best Val Accuracy: %.2f%%", best_val_acc)
    log.info("Saved VSR model checkpoint to: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VSR Spatial Reasoning Adapter")
    parser.add_argument("--samples", type=int, default=300, help="Number of benchmark training samples")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_vsr(
        num_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
