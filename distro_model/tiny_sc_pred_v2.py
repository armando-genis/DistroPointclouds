import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ===========================
# 1. DATASET CLASS WITH ADAPTIVE WEIGHTING
# ===========================
class ScanContextPedestrianDataset(Dataset):
    def __init__(self, scan_context_dir, sample_ids):
        """
        scan_context_dir: path to folder with sc_XXXXXX.npy and pedestrians_XXXXXX.json files
        sample_ids: list of sample IDs (integers)
        """
        self.scan_context_dir = scan_context_dir
        self.sample_ids = sample_ids
        
        # Filter out samples without pedestrians JSON (if any)
        self.valid_samples = []
        for sid in sample_ids:
            json_path = os.path.join(scan_context_dir, f"pedestrians_{sid:06d}.json")
            if os.path.exists(json_path):
                self.valid_samples.append(sid)
        
        print(f"Dataset initialized with {len(self.valid_samples)} valid samples out of {len(sample_ids)}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sid = self.valid_samples[idx]
        
        # Load scan context
        npy_path = os.path.join(self.scan_context_dir, f"sc_{sid:06d}.npy")
        sc = np.load(npy_path).astype(np.float32)  # Shape: (160, 720)
        
        # Normalize scan context to [0, 1]
        sc_min, sc_max = sc.min(), sc.max()
        if sc_max > sc_min:
            sc_norm = (sc - sc_min) / (sc_max - sc_min)
        else:
            sc_norm = sc
        
        # Load pedestrian JSON
        json_path = os.path.join(self.scan_context_dir, f"pedestrians_{sid:06d}.json")
        with open(json_path, 'r') as f:
            ped_list = json.load(f)
        
        # Create binary mask for all covered cells
        mask = np.zeros_like(sc_norm, dtype=np.float32)
        
        # Create adaptive weight map based on pedestrian size and distance
        weight_map = np.ones_like(sc_norm, dtype=np.float32)
        
        for ped in ped_list:
            covered = ped.get('covered_cells', [])
            distance = ped.get('distance', 10.0)
            num_cells = len(covered)
            
            # Adaptive weighting: smaller footprints get higher weight
            # This prevents the model from ignoring distant pedestrians
            if num_cells > 0:
                # Size-based weight (inverse to number of cells)
                size_weight = min(100.0 / num_cells, 5.0)
                
                # Distance-based weight (quadratic scaling for far objects)
                distance_weight = 1.0 + (distance / 20.0) ** 2
                
                # Combined weight (capped at 10x to prevent instability)
                ped_weight = min(size_weight * distance_weight, 10.0)
            else:
                ped_weight = 1.0
            
            for r, s in covered:
                if 0 <= r < mask.shape[0] and 0 <= s < mask.shape[1]:
                    mask[r, s] = 1.0
                    weight_map[r, s] = max(weight_map[r, s], ped_weight)
        
        # Convert to tensors
        x = torch.from_numpy(sc_norm).unsqueeze(0)  # Shape: (1, 160, 720)
        y = torch.from_numpy(mask).unsqueeze(0)      # Shape: (1, 160, 720)
        w = torch.from_numpy(weight_map).unsqueeze(0)  # Shape: (1, 160, 720)
        
        return {
            'input': x,
            'mask': y,
            'weight': w,
            'sample_id': sid,
            'num_pedestrians': len(ped_list),
            'ped_list': ped_list  # Keep for range-based evaluation
        }

# ===========================
# 2. MODEL ARCHITECTURE WITH MULTI-SCALE HEADS
# ===========================
class PolarConv2d(nn.Module):
    """Convolution with circular padding for polar coordinates"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding=0, dilation=dilation)
    
    def forward(self, x):
        # Circular padding on width (sectors), zero padding on height (rings)
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, self.pad, self.pad), mode='constant', value=0)
        return self.conv(x)

class MultiScaleOutput(nn.Module):
    """Multi-scale detection heads for different pedestrian sizes"""
    def __init__(self, in_channels=32):
        super().__init__()
        # Different kernel sizes for different pedestrian sizes
        self.small_obj_head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.medium_obj_head = nn.Conv2d(in_channels, 1, kernel_size=5, padding=2)
        self.large_obj_head = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        small = self.small_obj_head(x)
        medium = self.medium_obj_head(x)
        large = self.large_obj_head(x)
        
        # Distance-based blending (near objects use large kernel, far use small)
        h = x.shape[-2]
        device = x.device
        
        # Create distance-based weights
        near_weight = torch.zeros((1, 1, h, 1), device=device)
        near_weight[:, :, :60, :] = 1.0  # Near range
        near_weight[:, :, 60:90, :] = 0.5  # Transition
        
        far_weight = torch.zeros((1, 1, h, 1), device=device)
        far_weight[:, :, 90:, :] = 1.0  # Far range
        far_weight[:, :, 60:90, :] = 0.5  # Transition
        
        # Combine outputs
        output = large * near_weight + small * far_weight + 0.3 * medium
        
        return output

class ScanContextUNet(nn.Module):
    """U-Net architecture adapted for Scan Context geometry with multi-scale output"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self._conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._conv_block(512, 256)  # 256 + 256 from skip
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)  # 128 + 128 from skip
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)   # 64 + 64 from skip
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)    # 32 + 32 from skip
        
        # Multi-scale final output
        self.final = MultiScaleOutput(32)
    
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            PolarConv2d(in_c, out_c, 3),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            PolarConv2d(out_c, out_c, 3),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        bn = self.bottleneck(p4)
        
        # Decoder with skip connections
        d4 = self.upconv4(bn)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Multi-scale output
        out = self.final(d1)
        return out

# ===========================
# 3. LOSS FUNCTIONS
# ===========================
class AdaptiveFocalLoss(nn.Module):
    """Focal Loss with adaptive alpha based on object size"""
    def __init__(self, gamma=2.0, base_alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.base_alpha = base_alpha
    
    def forward(self, pred, target, pixel_weights=None):
        p = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Adaptive alpha based on pixel weights (higher for smaller objects)
        if pixel_weights is not None:
            alpha = self.base_alpha * pixel_weights
        else:
            alpha = self.base_alpha
        
        alpha_t = alpha * target + (1 - self.base_alpha) * (1 - target)
        loss = ce_loss * focal_weight * alpha_t
        
        return loss.mean()

# ===========================
# 4. TRAINING AND EVALUATION WITH RANGE METRICS
# ===========================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        inputs = batch['input'].to(device)
        masks = batch['mask'].to(device)
        weights = batch['weight'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss with adaptive weights
        loss = criterion(outputs, masks, weights)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / total_samples
    return avg_loss

def evaluate_with_ranges(model, dataloader, criterion, device):
    """Evaluate model with metrics for different distance ranges"""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    
    # Overall metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Range-specific metrics
    ranges = {
        'near': (0, 60),      # 0-7.5m
        'medium': (60, 120),  # 7.5-15m
        'far': (120, 160)     # 15-20m
    }
    range_metrics = {name: {'tp': 0, 'fp': 0, 'fn': 0} for name in ranges}
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating')
        for batch in progress_bar:
            inputs = batch['input'].to(device)
            masks = batch['mask'].to(device)
            weights = batch['weight'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, masks, weights)
            
            # Get predictions with distance-adaptive thresholds
            pred_probs = torch.sigmoid(outputs)
            
            # Use different thresholds for different ranges
            preds = torch.zeros_like(pred_probs)
            for range_name, (r_min, r_max) in ranges.items():
                if range_name == 'far':
                    threshold = 0.3  # Lower threshold for far objects
                elif range_name == 'medium':
                    threshold = 0.4
                else:
                    threshold = 0.5
                
                preds[:, :, r_min:r_max, :] = (pred_probs[:, :, r_min:r_max, :] > threshold).float()
            
            # Overall metrics
            tp = (preds * masks).sum()
            fp = (preds * (1 - masks)).sum()
            fn = ((1 - preds) * masks).sum()
            
            total_tp += tp.item()
            total_fp += fp.item()
            total_fn += fn.item()
            
            # Range-specific metrics
            for range_name, (r_min, r_max) in ranges.items():
                range_mask = torch.zeros_like(masks)
                range_mask[:, :, r_min:r_max, :] = 1.0
                
                tp = (preds * masks * range_mask).sum()
                fp = (preds * (1 - masks) * range_mask).sum()
                fn = ((1 - preds) * masks * range_mask).sum()
                
                range_metrics[range_name]['tp'] += tp.item()
                range_metrics[range_name]['fp'] += fp.item()
                range_metrics[range_name]['fn'] += fn.item()
            
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    # Calculate overall metrics
    avg_loss = total_loss / total_samples
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    
    overall_metrics = {
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }
    
    # Calculate range-specific metrics
    range_results = {}
    for name, metrics in range_metrics.items():
        if metrics['tp'] + metrics['fn'] > 0:  # Only if there are actual positives
            precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-8)
            recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            range_results[name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        else:
            range_results[name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    
    return overall_metrics, range_results

# ===========================
# 5. MAIN TRAINING SCRIPT
# ===========================
def main():
    # Configuration
    config = {
        'dataset_dir': '/workspace/DistroPointclouds/kitti/training/scan_context',
        'batch_size': 8,
        'num_epochs': 60,
        'learning_rate': 1e-3,
        'min_learning_rate': 1e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Get sample IDs
    all_npys = glob(os.path.join(config['dataset_dir'], "sc_*.npy"))
    if not all_npys:
        raise ValueError(f"No scan context files found in {config['dataset_dir']}")
    
    all_ids = sorted([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in all_npys])
    
    # Split into train/val/test (70/15/15)
    n_samples = len(all_ids)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:n_train + n_val]
    test_ids = all_ids[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_ids)} samples")
    print(f"  Val: {len(val_ids)} samples")
    print(f"  Test: {len(test_ids)} samples")
    print(f"  Total: {n_samples} samples\n")
    
    # Create datasets
    train_dataset = ScanContextPedestrianDataset(config['dataset_dir'], train_ids)
    val_dataset = ScanContextPedestrianDataset(config['dataset_dir'], val_ids)
    test_dataset = ScanContextPedestrianDataset(config['dataset_dir'], test_ids)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Initialize model
    device = torch.device(config['device'])
    model = ScanContextUNet(in_channels=1, out_channels=1).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Initialize loss function (Adaptive Focal Loss)
    criterion = AdaptiveFocalLoss(gamma=2.0, base_alpha=0.25)
    
    # Initialize optimizer with different learning rates for encoder/decoder
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'enc' in name or 'pool' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': config['learning_rate'] * 0.5},
        {'params': decoder_params, 'lr': config['learning_rate']}
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=config['min_learning_rate']
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_iou': [],
        'range_metrics': {'near': [], 'medium': [], 'far': []}
    }
    
    best_f1 = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    # Training loop
    print("Starting training...\n")
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics, range_metrics = evaluate_with_ranges(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_iou'].append(val_metrics['iou'])
        
        for range_name in ['near', 'medium', 'far']:
            history['range_metrics'][range_name].append(range_metrics[range_name]['f1'])
        
        # Print metrics
        print(f"\nOverall Metrics:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}")
        print(f"  Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"  Val IoU: {val_metrics['iou']:.4f}")
        
        print(f"\nRange-Specific F1 Scores:")
        print(f"  Near (0-7.5m): {range_metrics['near']['f1']:.4f}")
        print(f"  Medium (7.5-15m): {range_metrics['medium']['f1']:.4f}")
        print(f"  Far (15-20m): {range_metrics['far']['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'range_metrics': range_metrics,
                'config': config
            }, 'best_model.pth')
            print(f"\n✓ Saved new best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"{'='*60}")
    
    # Test evaluation
    print("\nEvaluating on test set...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics, test_range_metrics = evaluate_with_ranges(model, test_loader, criterion, device)
    
    print(f"\nTest Set Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  IoU: {test_metrics['iou']:.4f}")
    
    print(f"\nTest Set Range-Specific F1 Scores:")
    print(f"  Near (0-7.5m): {test_range_metrics['near']['f1']:.4f}")
    print(f"  Medium (7.5-15m): {test_range_metrics['medium']['f1']:.4f}")
    print(f"  Far (15-20m): {test_range_metrics['far']['f1']:.4f}")
    
    # Plot training history
    plot_training_history(history)
    print("\n✓ Training history saved to 'training_history.png'")

def plot_training_history(history):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 1].plot(history['val_f1'], linewidth=2, color='green')
    axes[0, 1].set_title('Validation F1 Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision vs Recall
    axes[0, 2].plot(history['val_precision'], label='Precision', linewidth=2)
    axes[0, 2].plot(history['val_recall'], label='Recall', linewidth=2)
    axes[0, 2].set_title('Precision vs Recall', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # IoU
    axes[1, 0].plot(history['val_iou'], linewidth=2, color='purple')
    axes[1, 0].set_title('Validation IoU', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Range-specific F1 scores
    axes[1, 1].plot(history['range_metrics']['near'], label='Near (0-7.5m)', linewidth=2)
    axes[1, 1].plot(history['range_metrics']['medium'], label='Medium (7.5-15m)', linewidth=2)
    axes[1, 1].plot(history['range_metrics']['far'], label='Far (15-20m)', linewidth=2)
    axes[1, 1].set_title('F1 Score by Distance', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.close()

# ===========================
# 6. INFERENCE FUNCTION
# ===========================
def inference(model_path, scan_context_path, device='cuda', visualize=False):
    """
    Run inference on a single scan context
    Returns predicted mask and optionally visualizes results
    """
    
    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = ScanContextUNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess scan context
    sc = np.load(scan_context_path).astype(np.float32)
    sc_min, sc_max = sc.min(), sc.max()
    if sc_max > sc_min:
        sc_norm = (sc - sc_min) / (sc_max - sc_min)
    else:
        sc_norm = sc
    
    # Convert to tensor
    x = torch.from_numpy(sc_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    # Run inference with distance-adaptive thresholds
    with torch.no_grad():
        output = model(x)
        pred_probs = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Apply distance-adaptive thresholds
        pred_mask = np.zeros_like(pred_probs)
        pred_mask[:60, :] = (pred_probs[:60, :] > 0.5).astype(float)  # Near
        pred_mask[60:120, :] = (pred_probs[60:120, :] > 0.4).astype(float)  # Medium
        pred_mask[120:, :] = (pred_probs[120:, :] > 0.3).astype(float)  # Far
    
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original scan context
        axes[0].imshow(sc_norm, cmap='viridis', aspect='auto')
        axes[0].set_title('Input Scan Context')
        axes[0].set_xlabel('Sector')
        axes[0].set_ylabel('Ring')
        
        # Prediction probabilities
        axes[1].imshow(pred_probs, cmap='hot', aspect='auto', vmin=0, vmax=1)
        axes[1].set_title('Prediction Probabilities')
        axes[1].set_xlabel('Sector')
        
        # Binary mask
        axes[2].imshow(pred_mask, cmap='binary', aspect='auto')
        axes[2].set_title('Predicted Pedestrians')
        axes[2].set_xlabel('Sector')
        
        plt.tight_layout()
        plt.show()
    
    return pred_mask, pred_probs

if __name__ == "__main__":
    main()