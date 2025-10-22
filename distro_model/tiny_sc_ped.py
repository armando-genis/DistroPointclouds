# tiny_sc_ped.py
# Train & eval a TinyCNN on Scan Context (.npy) + pedestrian JSON labels
# - Input:  [B,1,H,W]   (H=ring_res, W=sector_res)
# - Target: [B,1,H,W]   binary pedestrian mask built from covered_cells
# - Metric: BCE+Dice loss, pixel IoU; inference -> connected components -> detections

import os, json, math, random
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# config
# ----------------------------
class CFG:
    scan_context_dir = "/workspace/DistroPointclouds/kitti/training/scan_context"   # <-- change me
    # H,W inferred from first npy; defaults used if discovery fails
    default_H, default_W = 160, 720
    train_split = 0.8
    batch_size = 8
    num_epochs = 30
    lr = 1e-3
    num_workers = 4
    seed = 1337
    pos_weight = 2.0         # class imbalance handling (tune)
    use_dice = True
    # augment
    aug_roll_sectors_max = 32  # cyclic roll along sectors (horizontal)
    aug_dropout_prob = 0.05    # random dropout of a few positive cells

torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)

# ----------------------------
# utils
# ----------------------------
def discover_ids(scan_context_dir: str) -> Tuple[List[int], int, int]:
    """Find sc_*.npy files and infer H,W from the first one."""
    npys = sorted(glob(os.path.join(scan_context_dir, "sc_*.npy")))
    if not npys:
        raise FileNotFoundError(f"No sc_*.npy in {scan_context_dir}")
    ids = [int(os.path.basename(p).split("_")[1].split(".")[0]) for p in npys]
    # infer shape
    arr = np.load(npys[0])
    H, W = (arr.shape[0], arr.shape[1]) if arr.ndim == 2 else (CFG.default_H, CFG.default_W)
    return ids, H, W

def to_tensor01(x: np.ndarray) -> torch.Tensor:
    # normalize [min,max] -> [0,1] and add [C=1] channel
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        x = (x - mn) / (mx - mn)
    return torch.from_numpy(x).unsqueeze(0)

def build_mask_from_json(H: int, W: int, json_path: str) -> np.ndarray:
    """Binary mask from 'covered_cells' lists."""
    m = np.zeros((H, W), dtype=np.uint8)
    if not os.path.exists(json_path):
        return m
    with open(json_path, "r") as f:
        data = json.load(f)
    for ped in data:
        for r, s in ped.get("covered_cells", []):
            if 0 <= r < H and 0 <= s < W:
                m[r, s] = 1
    return m

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # logits/targets: [B,1,H,W]
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=[1,2,3]) + eps
    den = probs.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3]) + eps
    return 1.0 - (num / den).mean()

def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    preds = (torch.sigmoid(logits) > thr).float()
    inter = (preds * targets).sum(dim=[1,2,3])
    union = ((preds + targets) >= 1).float().sum(dim=[1,2,3])
    return ((inter + eps) / (union + eps)).mean()

def random_roll_sectors(x: torch.Tensor, y: torch.Tensor, max_shift: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cyclic roll along width (sectors). x,y shape [1,H,W]."""
    if max_shift <= 0:
        return x, y
    shift = random.randint(-max_shift, max_shift)
    if shift == 0: return x, y
    return torch.roll(x, shifts=shift, dims=-1), torch.roll(y, shifts=shift, dims=-1)

def random_mask_dropout(y: torch.Tensor, p: float) -> torch.Tensor:
    """Drop a few positive pixels to improve robustness."""
    if p <= 0.0 or y.max() == 0:
        return y
    drop = (torch.rand_like(y) < p).float()
    return torch.clamp(y - (y * drop), 0.0, 1.0)

# ----------------------------
# dataset
# ----------------------------
class ScanContextPedDataset(Dataset):
    def __init__(self, scan_context_dir: str, ids: List[int], H: int, W: int, train: bool):
        self.dir = scan_context_dir
        self.ids = ids
        self.H, self.W = H, W
        self.train = train

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        sc_path = os.path.join(self.dir, f"sc_{sid:06d}.npy")
        pj_path = os.path.join(self.dir, f"pedestrians_{sid:06d}.json")

        sc = np.load(sc_path)                   # (H,W)
        x = to_tensor01(sc)                     # [1,H,W]
        y = torch.from_numpy(build_mask_from_json(self.H, self.W, pj_path)).float().unsqueeze(0)  # [1,H,W]

        # light augments for training
        if self.train:
            x, y = random_roll_sectors(x, y, CFG.aug_roll_sectors_max)
            y = random_mask_dropout(y, CFG.aug_dropout_prob)

        return x, y

# ----------------------------
# tiny model (encoder-decoder)
# ----------------------------
class TinyCNN(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4, W/4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/8, W/8
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)  # logits
        )

    def forward(self, x):  # [B,1,H,W] -> [B,1,H,W]
        return self.dec(self.enc(x))

# ----------------------------
# train / eval
# ----------------------------
def train_one_epoch(model, loader, opt, device, epoch):
    model.train()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CFG.pos_weight], device=device))
    running = 0.0
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = bce(logits, y)
        if CFG.use_dice:
            loss = loss + dice_loss_from_logits(logits, y)
        loss.backward()
        opt.step()
        running += loss.item()
        if bi % 50 == 0:
            print(f"[epoch {epoch:02d}][{bi:04d}/{len(loader)}] loss={loss.item():.4f}")
    return running / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CFG.pos_weight], device=device))
    total_loss, total_iou = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = bce(logits, y)
        if CFG.use_dice:
            loss = loss + dice_loss_from_logits(logits, y)
        total_loss += loss.item()
        total_iou += iou_from_logits(logits, y).item()
    n = max(1, len(loader))
    return total_loss / n, total_iou / n

# ----------------------------
# inference: mask -> detections (connected components)
# ----------------------------
def mask_to_detections(mask: np.ndarray, min_pixels: int = 20) -> List[Dict]:
    """
    mask: [H,W] binary
    returns a list of detections with (ring_idx, sector_idx, area_px)
    (center = mean of component indices)
    """
    import scipy.ndimage as ndi
    lab, n = ndi.label(mask > 0)
    detections = []
    for comp_id in range(1, n+1):
        coords = np.argwhere(lab == comp_id)  # [K,2] -> rows=rings, cols=sectors
        if coords.shape[0] < min_pixels:
            continue
        r_mean, s_mean = coords[:,0].mean(), coords[:,1].mean()
        detections.append({
            "center_ring_idx": float(r_mean),
            "center_sector_idx": float(s_mean),
            "area_pixels": int(coords.shape[0])
        })
    return detections

@torch.no_grad()
def run_inference(model_path: str, sc_npy_path: str, out_json: str = None, thr: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sc = np.load(sc_npy_path).astype(np.float32)
    x = to_tensor01(sc).unsqueeze(0).to(device)  # [1,1,H,W]
    logits = model(x)
    probs = torch.sigmoid(logits)[0,0].cpu().numpy()
    mask = (probs > thr).astype(np.uint8)

    dets = mask_to_detections(mask)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(dets, f, indent=2)
    return dets, mask, probs

# ----------------------------
# main
# ----------------------------
def main():
    ids, H, W = discover_ids(CFG.scan_context_dir)
    n_train = int(len(ids) * CFG.train_split)
    train_ids, val_ids = ids[:n_train], ids[n_train:]
    print(f"Found {len(ids)} samples; train={len(train_ids)}, val={len(val_ids)}, shape=({H},{W})")

    train_ds = ScanContextPedDataset(CFG.scan_context_dir, train_ids, H, W, train=True)
    val_ds   = ScanContextPedDataset(CFG.scan_context_dir, val_ids,   H, W, train=False)

    train_ld = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    val_ld   = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                          num_workers=max(1, CFG.num_workers//2), pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    best_iou, best_path = -1.0, "tinycnn_best.pth"
    for epoch in range(1, CFG.num_epochs+1):
        tr_loss = train_one_epoch(model, train_ld, opt, device, epoch)
        va_loss, va_iou = evaluate(model, val_ld, device)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_IoU={va_iou:.4f}")
        if va_iou > best_iou:
            best_iou = va_iou
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved new best to {best_path} (IoU={best_iou:.4f})")

if __name__ == "__main__":
    main()
