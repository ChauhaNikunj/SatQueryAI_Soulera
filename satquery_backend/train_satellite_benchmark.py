from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import os
import time
from typing import List, Dict

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("satellite_train")

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

EUROSAT_PROMPTS = [
    "A satellite optical image of annual crop agricultural fields.",
    "A satellite optical image of a dense green forest and canopy.",
    "A satellite optical image of herbaceous grassland and open vegetation.",
    "A satellite optical image of a highway road transportation corridor.",
    "A satellite optical image of an industrial zone and commercial buildings.",
    "A satellite optical image of open green pastures and meadows.",
    "A satellite optical image of permanent orchards and crop plantations.",
    "A satellite optical image of a residential urban neighborhood and houses.",
    "A satellite optical image of a winding river and flowing water channel.",
    "A satellite optical image of a sea, lake, reservoir, or water dam body.",
]


class SatelliteVisionAdapter(nn.Module):
    """
    Satellite Domain Adaptation Head for RemoteCLIP.
    Maps satellite visual features into refined semantic space.
    """
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, num_classes: int = 10, dropout: float = 0.15):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


def load_remoteclip(weights_path: str, device: torch.device):
    """Load RemoteCLIP ViT-B/32 backbone."""
    log.info("Loading RemoteCLIP from %s ...", weights_path)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=None,
        device=device,
    )
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_sd, strict=False)
        log.info("Loaded pretrained RemoteCLIP weights.")
    else:
        log.warning("Weights not found at %s. Using default init.", weights_path)
        
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, tokenizer


def train_satellite_model(
    data_dir: str = "satquery_backend/data/eurosat/2750",
    weights_path: str = "satquery_backend/weights/RemoteCLIP-ViT-B-32.pt",
    output_path: str = "satquery_backend/weights/satellite_benchmark_adapter.pt",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    subset_size: int = 3000,
):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Training Device: %s", device)

    prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])

    val_prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])

    log.info("Loading EuroSAT dataset from %s ...", data_dir)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=prep)
    classes = full_dataset.classes
    log.info("Loaded %d images across %d classes: %s", len(full_dataset), len(classes), classes)

    # Use a stratified subset for fast high-accuracy training
    if subset_size and subset_size < len(full_dataset):
        indices = torch.randperm(len(full_dataset))[:subset_size].tolist()
        dataset = torch.utils.data.Subset(full_dataset, indices)
        log.info("Using representative subset of %d satellite images for training.", len(dataset))
    else:
        dataset = full_dataset

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load RemoteCLIP Backbone
    clip_model, tokenizer = load_remoteclip(weights_path, device)

    # Pre-compute text features for prompt alignment
    text_tokens = tokenizer(EUROSAT_PROMPTS).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, p=2, dim=-1)

    # Adapter Model
    adapter = SatelliteVisionAdapter(in_dim=512, hidden_dim=256, num_classes=len(classes)).to(device)
    optimizer = AdamW(adapter.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("Starting Satellite Model Training for %d Epochs...", epochs)
    print("=" * 72)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^10} | {'Val Acc':^10} | {'LR':^8}")
    print("=" * 72)

    for epoch in range(1, epochs + 1):
        adapter.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                feats = clip_model.encode_image(imgs)
                feats = F.normalize(feats, p=2, dim=-1)

            optimizer.zero_grad()
            logits = adapter(feats)
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

        # Validation
        adapter.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = clip_model.encode_image(imgs)
                feats = F.normalize(feats, p=2, dim=-1)

                logits = adapter(feats)
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
                "classes": classes,
                "prompts": EUROSAT_PROMPTS,
            }, output_path)

    print("=" * 72)
    log.info("Satellite Benchmark Training Completed! Best Val Accuracy: %.2f%%", best_val_acc)
    log.info("Saved weights to: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Satellite Vision Adapter on EuroSAT")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--subset", type=int, default=3000, help="Number of satellite images to train on")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_satellite_model(
        epochs=args.epochs,
        subset_size=args.subset,
        batch_size=args.batch_size,
        lr=args.lr,
    )
