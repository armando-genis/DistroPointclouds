#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train & evaluate a pedestrian detector on LiDAR Scan-Context grids.
Inputs:  sc_XXXXXX.npy  -> (160, 720) float32 grid
Labels:  pedestrians_XXXXXX.json -> [{"center_ring_idx": int, "center_sector_idx": int, "covered_cells": [[r,s], ...]}, ...]

Outputs:
- checkpoints/
- runs/val_images/ (overlays)
- metrics printed to console

Run:
  python3 train_sc_ped.py --sc_dir /path/to/scan_context --epochs 30 --batch_size 16

You can also export predictions on a folder:
  python3 train_sc_ped.py --sc_dir /path/to/scan_context --eval_only --ckpt checkpoints/best.pt
"""

import os, json, glob, math, argparse, random
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.ops import nms
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    H: int = 160
    W: int = 720
    heat_sigma: float = 1.8      # Gaussian radius in cells
    roll_aug: bool = True        # sector-roll augmentation
    ring_jitter: int = 0         # optional +/- rings jitter (set 0 to disable)
    clip_abs_z: float = 3.0      # example clipping for normalization
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 16
    workers: int = 4
    val_split: float = 0.15
    test_split: float = 0.10
    seed: int = 1337
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    hm_loss_w: float = 1.0
    mask_loss_w: float = 0.0     # set >0 to enable segmentation auxiliary loss
    ckpt_dir: str = "checkpoints"
    vis_dir: str = "runs/val_images"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)


# ----------------------------
# Utilities
# ----------------------------
def gaussian2d(shape: Tuple[int, int], center: Tuple[int, int], sigma: float) -> np.ndarray:
    R, S = shape
    rr = np.arange(R)[:, None]
    ss = np.arange(S)[None, :]
    d2 = (rr - center[0])**2 + (ss - center[1])**2
    return np.exp(-0.5 * d2 / (sigma**2))

def wrap_sector_distance(a: int, b: int, S: int) -> int:
    """Minimal circular distance on sector axis."""
    diff = abs(a - b)
    return min(diff, S - diff)

def bce_focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0, reduction="mean"):
    """
    Standard BCEWithLogits with a focal modulation.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal = (alpha * (1 - pt).pow(gamma)) * bce
    if reduction == "mean":
        return focal.mean()
    elif reduction == "sum":
        return focal.sum()
    else:
        return focal

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1,2,3))
    den = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps
    return 1.0 - (num / den).mean()

def save_overlay(sc: np.ndarray, heat_pred: np.ndarray, centers: List[Tuple[int,int]], out_path: str):
    """
    Save a visualization:
      - background: normalized SC
      - predicted heatmap: semi-transparent
      - GT centers: white points
    """
    scn = np.clip(sc, -CFG.clip_abs_z, CFG.clip_abs_z) / CFG.clip_abs_z
    scn = (scn + 1.0) / 2.0  # [-1,1] -> [0,1]

    fig = plt.figure(figsize=(7.2, 1.6), dpi=100)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(scn, cmap="viridis", aspect="auto", origin="lower")
    ax.imshow(heat_pred, cmap="hot", alpha=0.45, aspect="auto", origin="lower")
    if len(centers) > 0:
        ys = [c[0] for c in centers]
        xs = [c[1] for c in centers]
        ax.scatter(xs, ys, s=8, c="w")
    plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ----------------------------
# Dataset
# ----------------------------
class ScanContextPedDataset(Dataset):
    def __init__(self, sc_dir: str, augment: bool = True, use_mask: bool = False):
        self.sc_dir = sc_dir
        self.augment = augment
        self.use_mask = use_mask

        self.sc_files = sorted(glob.glob(os.path.join(sc_dir, "sc_*.npy")))
        assert len(self.sc_files) > 0, f"No sc_*.npy found in {sc_dir}"

    def __len__(self):
        return len(self.sc_files)

    def _load_json(self, sc_path: str) -> List[Dict]:
        base = os.path.splitext(os.path.basename(sc_path))[0]   # sc_XXXXXX
        json_path = os.path.join(os.path.dirname(sc_path), f"pedestrians_{base[3:]}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                return json.load(f)
        return []

    def __getitem__(self, idx):
        sc_path = self.sc_files[idx]
        sc = np.load(sc_path).astype(np.float32) # (160,720)

        # fixed normalization (example)
        sc = np.clip(sc, -CFG.clip_abs_z, CFG.clip_abs_z) / CFG.clip_abs_z    # [-1,1]

        # labels
        peds = self._load_json(sc_path)

        # build heatmap
        heat = np.zeros_like(sc, dtype=np.float32)
        centers = []
        for ped in peds:
            r = int(ped.get("center_ring_idx", -1))
            s = int(ped.get("center_sector_idx", -1))
            if 0 <= r < CFG.H and 0 <= s < CFG.W:
                centers.append((r,s))
                heat = np.maximum(heat, gaussian2d((CFG.H, CFG.W), (r, s), CFG.heat_sigma))

        # optional segmentation mask
        mask = None
        if self.use_mask:
            mask = np.zeros_like(sc, dtype=np.float32)
            for ped in peds:
                cells = np.array(ped.get("covered_cells", []), dtype=np.int32)
                if cells.size:
                    rr = np.clip(cells[:,0], 0, CFG.H-1)
                    ss = np.mod(cells[:,1], CFG.W)  # wrap safely
                    mask[rr, ss] = 1.0

        # augment (sector roll + optional small ring jitter)
        shift = 0
        ring_shift = 0
        if self.augment:
            shift = np.random.randint(0, CFG.W)
            sc = np.roll(sc, shift=shift, axis=1)
            heat = np.roll(heat, shift=shift, axis=1)
            if mask is not None:
                mask = np.roll(mask, shift=shift, axis=1)

            if CFG.ring_jitter != 0:
                ring_shift = np.random.randint(-CFG.ring_jitter, CFG.ring_jitter+1)
                sc = np.roll(sc, shift=ring_shift, axis=0)
                heat = np.roll(heat, shift=ring_shift, axis=0)
                if mask is not None:
                    mask = np.roll(mask, shift=ring_shift, axis=0)

        x = torch.from_numpy(sc)[None, ...]           # (1,H,W)
        y_heat = torch.from_numpy(heat)[None, ...]    # (1,H,W)
        sample = {"x": x, "y_heat": y_heat, "path": sc_path}

        if self.use_mask:
            y_mask = torch.from_numpy(mask)[None, ...]
            sample["y_mask"] = y_mask

        return sample


# ----------------------------
# Model (Tiny UNet-ish)
# ----------------------------
class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(c_in, c_out),
            ConvBNAct(c_out, c_out)
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(c_in, c_out)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(c_in, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class TinyUNet(nn.Module):
    def __init__(self, in_ch=1, base=32, out_heat=1, out_mask=0):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.d1  = Down(base, base*2)
        self.d2  = Down(base*2, base*4)
        self.d3  = Down(base*4, base*8)

        self.u2  = Up(base*8, base*4)
        self.u1  = Up(base*4, base*2)
        self.u0  = Up(base*2, base)

        self.head_heat = nn.Conv2d(base, out_heat, 1)
        self.use_mask = out_mask > 0
        if self.use_mask:
            self.head_mask = nn.Conv2d(base, out_mask, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x  = self.u2(x4, x3)
        x  = self.u1(x,  x2)
        x  = self.u0(x,  x1)

        heat = self.head_heat(x)
        if self.use_mask:
            mask = self.head_mask(x)
            return heat, mask
        return heat


# ----------------------------
# Peak extraction for eval
# ----------------------------
def extract_peaks(heatmap: np.ndarray, thr: float=0.4, ksize: int=7, topk: int=200) -> List[Tuple[int,int,float]]:
    """
    Extract local maxima from a heatmap with non-maximum suppression.
    Returns list of (r, s, score).
    """
    H, W = heatmap.shape
    # simple max pooling NMS
    t = torch.tensor(heatmap)[None, None, ...]
    pooled = F.max_pool2d(t, kernel_size=ksize, stride=1, padding=ksize//2)
    keep = (t == pooled) & (t >= thr)
    coords = torch.nonzero(keep[0,0], as_tuple=False)
    scores = t[0,0][keep[0,0]]
    if scores.numel() == 0:
        return []
    # limit topk
    scores, idxs = torch.topk(scores, k=min(topk, scores.numel()))
    coords = coords[idxs]
    out = [(int(coords[i,0]), int(coords[i,1]), float(scores[i])) for i in range(scores.numel())]
    return out


def match_centers(preds: List[Tuple[int,int,float]],
                  gts: List[Tuple[int,int]],
                  max_r_dist: int = 3,
                  max_s_dist: int = 6,
                  W: int = CFG.W) -> Tuple[int,int,int]:
    """
    Greedy matching with wrap-around in sector dimension.
    Returns: TP, FP, FN
    """
    used_gt = set()
    tp = 0
    for r,s,sc in preds:
        best = -1
        best_d = (1e9, 1e9)
        for i, (gr, gs) in enumerate(gts):
            if i in used_gt: continue
            dr = abs(r - gr)
            ds = wrap_sector_distance(s, gs, W)
            if dr <= max_r_dist and ds <= max_s_dist:
                # prioritize smallest (dr, ds)
                if (dr, ds) < best_d:
                    best_d = (dr, ds)
                    best = i
        if best >= 0:
            tp += 1
            used_gt.add(best)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


# ----------------------------
# Train / Validate
# ----------------------------
def run_one_epoch(model, loader, optimizer=None, scaler=None, use_mask=False):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_hm = 0.0
    total_mask = 0.0
    n = 0

    pbar = tqdm(loader, desc="train" if is_train else "valid", ncols=95)
    for batch in pbar:
        x = batch["x"].to(CFG.device)            # (B,1,H,W)
        y_heat = batch["y_heat"].to(CFG.device)  # (B,1,H,W)
        y_mask = batch.get("y_mask", None)
        if y_mask is not None:
            y_mask = y_mask.to(CFG.device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                out = model(x)
                if use_mask:
                    heat_logits, mask_logits = out
                else:
                    heat_logits = out

                hm_loss = bce_focal_loss_with_logits(
                    heat_logits, y_heat,
                    alpha=CFG.focal_alpha, gamma=CFG.focal_gamma, reduction="mean"
                )

                loss = CFG.hm_loss_w * hm_loss
                mask_loss_val = torch.tensor(0.0, device=CFG.device)
                if use_mask and y_mask is not None and CFG.mask_loss_w > 0.0:
                    mask_loss_val = dice_loss(mask_logits, y_mask)
                    loss = loss + CFG.mask_loss_w * mask_loss_val

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                out = model(x)
                if use_mask:
                    heat_logits, mask_logits = out
                else:
                    heat_logits = out
                hm_loss = bce_focal_loss_with_logits(
                    heat_logits, y_heat,
                    alpha=CFG.focal_alpha, gamma=CFG.focal_gamma, reduction="mean"
                )
                loss = CFG.hm_loss_w * hm_loss
                mask_loss_val = torch.tensor(0.0, device=CFG.device)
                if use_mask and "y_mask" in batch and CFG.mask_loss_w > 0.0:
                    mask_loss_val = dice_loss(mask_logits, y_mask)
                    loss = loss + CFG.mask_loss_w * mask_loss_val

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_hm   += float(hm_loss.item()) * bs
        total_mask += float(mask_loss_val.item()) * bs
        n += bs

        pbar.set_postfix(loss=f"{total_loss/n:.4f}", hm=f"{total_hm/n:.4f}",
                         msk=(f"{total_mask/n:.4f}" if use_mask and CFG.mask_loss_w>0 else "-"))

    return total_loss / max(1,n), total_hm / max(1,n), (total_mask / max(1,n))


def evaluate(model, loader, save_vis=False, max_batches=50):
    model.eval()
    TP=FP=FN=0
    os.makedirs(CFG.vis_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="eval", ncols=95)):
            x = batch["x"].to(CFG.device)
            paths = batch["path"]

            out = model(x)
            heat_logits = out[0] if isinstance(out, (tuple, list)) else out
            heat = torch.sigmoid(heat_logits).cpu().numpy()  # (B,1,H,W)

            for b in range(x.size(0)):
                heatmap = heat[b,0]
                # predicted peaks
                preds = extract_peaks(heatmap, thr=0.35, ksize=7, topk=200)
                # GT centers
                # we rebuild from stored heat target: safer to read JSON for exact centers
                sc_path = paths[b]
                gt_centers = []
                base = os.path.splitext(os.path.basename(sc_path))[0]
                json_path = os.path.join(os.path.dirname(sc_path), f"pedestrians_{base[3:]}.json")
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        peds = json.load(f)
                    for ped in peds:
                        r = int(ped.get("center_ring_idx", -1))
                        s = int(ped.get("center_sector_idx", -1))
                        if 0 <= r < CFG.H and 0 <= s < CFG.W:
                            gt_centers.append((r,s))

                tp, fp, fn = match_centers(preds, gt_centers, max_r_dist=3, max_s_dist=6, W=CFG.W)
                TP += tp; FP += fp; FN += fn

                if save_vis and i < max_batches:
                    sc = np.load(sc_path).astype(np.float32)
                    sc = np.clip(sc, -CFG.clip_abs_z, CFG.clip_abs_z) / CFG.clip_abs_z
                    out_png = os.path.join(CFG.vis_dir, f"vis_{i:04d}_{b}.png")
                    save_overlay(sc, heatmap, gt_centers, out_png)

    prec = TP / (TP + FP + 1e-9)
    rec  = TP / (TP + FN + 1e-9)
    f1   = 2*prec*rec / (prec+rec+1e-9)
    return {"TP":TP, "FP":FP, "FN":FN, "precision":prec, "recall":rec, "F1":f1}


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc_dir", required=True, help="Directory with sc_*.npy and pedestrians_*.json")
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    ap.add_argument("--use_mask", action="store_true", help="Enable covered-cells aux head & loss (set CFG.mask_loss_w>0)")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", type=str, default="")
    args = ap.parse_args()

    os.makedirs(CFG.ckpt_dir, exist_ok=True)
    os.makedirs(CFG.vis_dir, exist_ok=True)

    # dataset & split
    full = ScanContextPedDataset(args.sc_dir, augment=not args.eval_only, use_mask=args.use_mask)
    n_total = len(full)
    n_test  = int(CFG.test_split * n_total)
    n_val   = int(CFG.val_split * n_total)
    n_train = n_total - n_val - n_test
    train_set, val_set, test_set = random_split(full, [n_train, n_val, n_test],
                                                generator=torch.Generator().manual_seed(CFG.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=CFG.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=CFG.workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=CFG.workers, pin_memory=True)

    # model
    model = TinyUNet(in_ch=1, base=32, out_heat=1, out_mask=(1 if args.use_mask else 0)).to(CFG.device)

    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=CFG.device)
        model.load_state_dict(ckpt["model"])

    if args.eval_only:
        metrics = evaluate(model, test_loader, save_vis=True)
        print("Test metrics:", metrics)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG.device=="cuda"))

    best_f1 = -1.0
    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr = run_one_epoch(model, train_loader, optimizer=optimizer, scaler=scaler, use_mask=args.use_mask)
        vl = run_one_epoch(model, val_loader, optimizer=None, scaler=None, use_mask=args.use_mask)
        print(f"Train: loss={tr[0]:.4f} hm={tr[1]:.4f} mask={tr[2]:.4f}")
        print(f"Valid: loss={vl[0]:.4f} hm={vl[1]:.4f} mask={vl[2]:.4f}")

        # quick val metrics on peaks
        metrics = evaluate(model, val_loader, save_vis=True, max_batches=30)
        print(f"Val metrics: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['F1']:.3f}")

        # save last
        torch.save({"model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics}, os.path.join(CFG.ckpt_dir, "last.pt"))

        # save best by F1
        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            best_path = os.path.join(CFG.ckpt_dir, "best.pt")
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "metrics": metrics}, best_path)
            print(f"Saved best to {best_path} (F1={best_f1:.3f})")

    # final test
    print("\n== Final test on held-out split ==")
    ckpt_best = torch.load(os.path.join(CFG.ckpt_dir, "best.pt"), map_location=CFG.device)
    model.load_state_dict(ckpt_best["model"])
    test_metrics = evaluate(model, test_loader, save_vis=True)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
