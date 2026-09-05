"""
satquery_backend/train_adapter.py
===================================
Train the FusionAdapter on BigEarthNet v2.0 optical-SAR patch pairs.

Usage
-----
    python satquery_backend/train_adapter.py \
        --data_dir  /path/to/bigearth_pairs/ \
        --clip_weights satquery_backend/weights/RemoteCLIP-ViT-B-32.pt \
        --output    satquery_backend/weights/adapter_v1.pt \
        --epochs    50 \
        --batch_size 32

Expected data_dir layout
-------------------------
    bigearth_pairs/
    ├── s2/          <- optical Sentinel-2 patches (PNG or TIF)
    ├── s1/          <- SAR Sentinel-1 patches (same filenames as s2/)
    └── labels.csv   <- columns: filename, label_indices (multi-label, comma-sep ints)
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from satquery_backend.models.optical_sar_fusion import (
    NUM_CLASSES,
    OpticalSARFusionModel,
    OpticalPreprocessor,
    SARPreprocessor,
)
from satquery_backend.utils.raster_io import load_geotiff, normalize_optical, normalize_sar


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class BigEarthNetPairDataset(Dataset):
    """
    Minimal BigEarthNet v2.0 optical-SAR pair dataset.

    Parameters
    ----------
    data_dir : str
        Root directory with s2/, s1/, and labels.csv.
    split : float
        Fraction of data to use for training (rest = val). Default 0.8.
    train : bool
        If True load training split; else validation split.
    """

    def __init__(self, data_dir: str, split: float = 0.8, train: bool = True):
        self.s2_dir = Path(data_dir) / "s2"
        self.s1_dir = Path(data_dir) / "s1"
        self.opt_prep = OpticalPreprocessor()
        self.sar_prep = SARPreprocessor()

        # Load label CSV
        labels_path = Path(data_dir) / "labels.csv"
        self.samples: list[tuple[str, list[int]]] = []
        with open(labels_path) as f:
            for row in csv.DictReader(f):
                idxs = [int(x) for x in row["label_indices"].split(",") if x.strip()]
                self.samples.append((row["filename"], idxs))

        # Train / val split
        cut = int(len(self.samples) * split)
        self.samples = self.samples[:cut] if train else self.samples[cut:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, label_idxs = self.samples[idx]

        # Load optical
        opt_path = self.s2_dir / fname
        if opt_path.suffix.lower() in {".tif", ".tiff"}:
            arr, _ = load_geotiff(str(opt_path))
            opt_pil = normalize_optical(arr[[3, 2, 1]] if arr.shape[0] >= 4 else arr[:3])
        else:
            opt_pil = Image.open(opt_path).convert("RGB")
        opt_t = self.opt_prep(opt_pil)

        # Load SAR
        sar_path = self.s1_dir / fname
        if sar_path.suffix.lower() in {".tif", ".tiff"}:
            arr, _ = load_geotiff(str(sar_path))
            sar_pil = normalize_sar(arr)
        else:
            sar_pil = Image.open(sar_path).convert("RGB")
        sar_t = self.sar_prep(sar_pil)

        # Multi-hot label vector
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for i in label_idxs:
            if 0 <= i < NUM_CLASSES:
                label[i] = 1.0

        return opt_t, sar_t, label


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = OpticalSARFusionModel(
        clip_weights_path=args.clip_weights,
        device=str(device),
        use_cross_attention=args.use_cross_attn,
    )

    # Only adapter params are updated
    optimizer = AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()   # multi-label

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = BigEarthNetPairDataset(args.data_dir, train=True)
    val_ds   = BigEarthNetPairDataset(args.data_dir, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        t0 = time.perf_counter()
        train_loss = 0.0
        for opt_t, sar_t, labels in train_dl:
            opt_t, sar_t, labels = opt_t.to(device), sar_t.to(device), labels.to(device)
            logits = model(opt_t, sar_t)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        scheduler.step()

        # ── Val ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for opt_t, sar_t, labels in val_dl:
                opt_t, sar_t, labels = opt_t.to(device), sar_t.to(device), labels.to(device)
                val_loss += criterion(model(opt_t, sar_t), labels).item()
        val_loss /= len(val_dl)

        elapsed = time.perf_counter() - t0
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"time={elapsed:.1f}s")

        # ── Save best ─────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"adapter": model.adapter.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss},
                       args.output)
            print(f"  ✅ Best adapter saved → {args.output}")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Adapter checkpoint: {args.output}")
    print(f"\nTo use it, set:  ADAPTER_WEIGHTS={args.output}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SatQuery FusionAdapter")
    parser.add_argument("--data_dir",    required=True,  help="BigEarthNet pairs root dir")
    parser.add_argument("--clip_weights",default="satquery_backend/weights/RemoteCLIP-ViT-B-32.pt")
    parser.add_argument("--output",      default="satquery_backend/weights/adapter_v1.pt")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--use_cross_attn", action="store_true",
                        help="Enable CrossAttentionGate (adds ~2.4M params)")
    train(parser.parse_args())
