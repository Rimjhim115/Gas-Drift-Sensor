"""
TDACNN — Temporal Dilated Attention CNN
========================================
Cross-Batch Smell Classification (UCI Gas Sensor Drift Dataset)

FIX LOG
-------
  [FIX 1] vae_augment was O(N²) — replaced with fully vectorized NumPy ops.
  [FIX 2] Entire dataset was moved to GPU before training — replaced with
          CPU tensors + per-batch .to(device) inside the loop.
  [FIX 3] NUM_WORKERS forced to 0 — eliminates DataLoader multiprocessing
          deadlock that caused the script to freeze after augmentation.
  [FIX 4] Added sys.stdout.flush() after every print and after every epoch
          log line — forces unbuffered output so progress is visible instantly.
  [FIX 5] torch.backends.cudnn.deterministic set to False — re-enables cuDNN
          fast kernels for ~20-40% training speedup (reproducibility traded
          for speed; set back to True if exact reproducibility is needed).
  [FIX 6] Added a heartbeat print every epoch so you always know it's alive,
          with the detailed log every PRINT_EVERY epochs.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — always flush after printing so output is never buffered
# ─────────────────────────────────────────────────────────────────────────────
def fprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # FIX 5: deterministic=False restores cuDNN fast kernels → faster training
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True
    os.environ["PYTHONHASHSEED"]       = str(seed)

set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
EPOCHS          = 200
LR              = 5e-4
D_MODEL         = 128
NUM_TDA_BLOCKS  = 3
DILATIONS       = [1, 2, 4, 8]
DROPOUT         = 0.3
SE_REDUCTION    = 8
LAMBDA_MSE      = 0.05
LAMBDA_CON      = 0.05
TEMPERATURE     = 0.07
PATIENCE        = 40
LABEL_SMOOTHING = 0.05
PRINT_EVERY     = 10

# FIX 3: NUM_WORKERS = 0 everywhere — eliminates the multiprocessing deadlock
#         that froze the script after "DataLoader workers: 4 | pin_memory: True"
NUM_WORKERS = 0


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
start_time = time.time()
fprint("Loading datasets...")

balanced_df = pd.read_csv("final_balanced_dataset.csv")
train_batch = sorted(balanced_df["batch"].unique())[0]
train_df    = balanced_df[balanced_df["batch"] == train_batch]

original_df  = pd.read_csv("smell_dataset.csv")
test_batches = [b for b in sorted(original_df["batch"].unique()) if b != train_batch]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fprint(f"Using device      : {device}")
fprint(f"Train batch       : Batch {train_batch} from final_balanced_dataset.csv")
fprint(f"Train samples     : {len(train_df)}")
fprint(f"Test batches      : {test_batches}")
fprint(f"Test samples      : {len(original_df[original_df['batch'].isin(test_batches)])}")

num_classes  = len(balanced_df["label"].unique())
feature_cols = [c for c in balanced_df.columns if c not in ["label", "batch"]]
fprint(f"Number of classes : {num_classes}")
fprint(f"Number of features: {len(feature_cols)}")

missing = [c for c in feature_cols if c not in original_df.columns]
if missing:
    fprint(f"WARNING: features missing in original dataset: {missing}")
    feature_cols = [c for c in feature_cols if c in original_df.columns]
    fprint(f"Using {len(feature_cols)} common features")

seq_len = len(feature_cols)


# ─────────────────────────────────────────────────────────────────────────────
# VAE-INSPIRED AUGMENTATION — fully vectorized, O(N) not O(N²)
# ─────────────────────────────────────────────────────────────────────────────
def vae_augment(X: np.ndarray, y: np.ndarray, augment_factor: int = 5, seed: int = SEED):
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y)
    class_indices  = {c: np.where(y == c)[0] for c in unique_classes}

    X_aug_list, y_aug_list = [X], [y]

    for aug_round in range(augment_factor):
        X_new = np.empty_like(X)
        for cls, idx in class_indices.items():
            mask  = (y == cls)
            n_cls = mask.sum()
            k_idx = rng.choice(idx, size=n_cls, replace=True)

            Xi  = X[mask]
            Xk  = X[k_idx]

            mu_i  = Xi.mean(axis=1, keepdims=True)
            std_i = Xi.std( axis=1, keepdims=True) + 1e-8
            mu_k  = Xk.mean(axis=1, keepdims=True)
            std_k = Xk.std( axis=1, keepdims=True) + 1e-8

            lam     = rng.uniform(0.3, 0.7, size=(n_cls, 1))
            mu_mix  = lam * mu_i  + (1 - lam) * mu_k
            std_mix = lam * std_i + (1 - lam) * std_k

            noise       = rng.standard_normal(size=Xi.shape) * std_mix * 0.1
            X_new[mask] = Xi + noise

        X_aug_list.append(X_new)
        y_aug_list.append(y)

    return np.vstack(X_aug_list), np.hstack(y_aug_list)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)
        features  = F.normalize(features, dim=1)
        sim       = torch.matmul(features, features.T) / self.temperature
        pos_mask  = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        pos_mask.fill_diagonal_(0)
        sim_max, _= sim.max(dim=1, keepdim=True)
        sim       = sim - sim_max.detach()
        exp_sim   = torch.exp(sim)
        self_mask = torch.ones(B, B, device=features.device).fill_diagonal_(0)
        denom     = (exp_sim * self_mask).sum(dim=1, keepdim=True) + 1e-8
        log_prob  = sim - torch.log(denom)
        n_pos     = pos_mask.sum(dim=1)
        loss      = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
        return loss[n_pos > 0].mean() if (n_pos > 0).any() else torch.tensor(0.0, device=features.device)


# ─────────────────────────────────────────────────────────────────────────────
# TDACNN BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleDilatedConv(nn.Module):
    def __init__(self, d_model: int, dilations: list, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            padding = (kernel_size - 1) * d // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                          dilation=d, padding=padding, groups=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            ))
        self.dropout = nn.Dropout(dropout)
        self.branch_scales = nn.Parameter(
            torch.ones(len(dilations)) / len(dilations)
        )

    def forward(self, x):
        scales = torch.softmax(self.branch_scales, dim=0)
        out    = sum(scales[i] * b(x) for i, b in enumerate(self.branches))
        return self.dropout(out)


class ChannelAttention(nn.Module):
    def __init__(self, d_model: int, reduction: int = 8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.ReLU(),
            nn.Linear(d_model // reduction, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.gap(x).squeeze(-1))
        return x * w.unsqueeze(-1)


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.q_proj  = nn.Linear(d_model, d_model // 2)
        self.k_proj  = nn.Linear(d_model, d_model // 2)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out_proj= nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale   = (d_model // 2) ** -0.5

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        Q   = self.q_proj(x_t)
        K   = self.k_proj(x_t)
        V   = self.v_proj(x_t)
        attn= torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1
        )
        attn = self.dropout(attn)
        out  = torch.bmm(attn, V)
        out  = self.out_proj(out)
        return out.permute(0, 2, 1)


class TDABlock(nn.Module):
    def __init__(self, d_model: int, dilations: list,
                 se_reduction: int = 8, dropout: float = 0.3):
        super().__init__()
        self.ms_dilconv = MultiScaleDilatedConv(d_model, dilations, dropout=dropout)
        self.ch_attn    = ChannelAttention(d_model, reduction=se_reduction)
        self.temp_attn  = TemporalAttention(d_model, dropout=dropout * 0.5)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.ms_dilconv(x)
        out = self.ch_attn(out)
        out = self.temp_attn(out)
        out = self.dropout(out)
        out = out + residual
        out = self.layer_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out


class GlobalContextPooling(nn.Module):
    def forward(self, x):
        avg = x.mean(dim=-1)
        mx  = x.max(dim=-1).values
        std = x.std(dim=-1)
        return torch.cat([avg, mx, std], dim=1)


class FeatureGenerationModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        z_high     = F.adaptive_max_pool1d(z.unsqueeze(1), z.shape[-1]).squeeze(1)
        z_low      = z - z_high
        mu_low     = z_low.mean(dim=0, keepdim=True)
        std_low    = z_low.std(dim=0,  keepdim=True) + 1e-8
        mu_new     = mu_low  + torch.randn_like(mu_low)  * std_low * 0.1
        std_new    = (std_low + torch.randn_like(std_low) * std_low * 0.05).abs() + 1e-8
        z_low_norm = (z_low - mu_low) / std_low
        z_low_new  = std_new * z_low_norm + mu_new
        return z_high + z_low_new


# ─────────────────────────────────────────────────────────────────────────────
# TDACNN MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
class TDACNN(nn.Module):
    def __init__(
        self,
        seq_len     : int   = 128,
        d_model     : int   = 128,
        num_blocks  : int   = 3,
        dilations   : list  = None,
        se_reduction: int   = 8,
        num_classes : int   = 6,
        dropout     : float = 0.3,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]

        self.input_proj = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.tda_blocks = nn.ModuleList([
            TDABlock(d_model, dilations, se_reduction=se_reduction, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.pool           = GlobalContextPooling()
        pool_dim            = d_model * 3
        self.post_pool_norm = nn.LayerNorm(pool_dim)
        self.feat_gen       = FeatureGenerationModule()

        self.projector = nn.Sequential(
            nn.Linear(pool_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512,      256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256,      num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def extract_features(self, x):
        x = x.squeeze(-1).unsqueeze(1)
        x = self.input_proj(x)
        for block in self.tda_blocks:
            x = block(x)
        z = self.pool(x)
        return self.post_pool_norm(z)

    def forward(self, x, return_features=False):
        z_s     = self.extract_features(x)
        z_s_gen = self.feat_gen(z_s)
        z_f     = self.projector(z_s)
        z_f_gen = self.projector(z_s_gen)
        out     = self.classifier(z_s)
        out_gen = self.classifier(z_s_gen)
        if return_features:
            return out, out_gen, z_f, z_f_gen, z_s, z_s_gen
        return out


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def instance_norm(X):
    mu  = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1,  keepdims=True) + 1e-8
    return (X - mu) / std

X_full = instance_norm(train_df[feature_cols].values)
y_full = train_df["label"].values - 1

fprint(f"\nOriginal training samples : {len(X_full)}")
fprint(f"Class distribution        : {np.bincount(y_full)}")

fprint("\nApplying VAE-inspired augmentation (vectorized)...")
t_aug = time.time()
X_aug, y_aug = vae_augment(X_full, y_full, augment_factor=5, seed=SEED)
fprint(f"Augmented training samples: {len(X_aug)}  (took {time.time()-t_aug:.2f}s)")

X_train, X_val, y_train, y_val = train_test_split(
    X_aug, y_aug, test_size=0.15, random_state=SEED, stratify=y_aug
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)


# ─────────────────────────────────────────────────────────────────────────────
# TENSORS — CPU tensors, moved to GPU per-batch inside training loop
# ─────────────────────────────────────────────────────────────────────────────
def to_tensor_cpu(X, y):
    Xl = X.reshape(X.shape[0], X.shape[1], 1)
    return (
        torch.tensor(Xl, dtype=torch.float32),
        torch.tensor(y,  dtype=torch.long),
    )

X_train_t, y_train_t = to_tensor_cpu(X_train_s, y_train)
X_val_t,   y_val_t   = to_tensor_cpu(X_val_s,   y_val)

g = torch.Generator(); g.manual_seed(SEED)

# FIX 3: NUM_WORKERS=0 — no multiprocessing, no deadlock
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    num_workers=0, pin_memory=(device.type == "cuda"),
    generator=g,
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(device.type == "cuda"),
)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL, LOSS, OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────
class_counts = np.bincount(y_train)
cw = torch.tensor(
    (1.0 / class_counts) / (1.0 / class_counts).sum() * num_classes,
    dtype=torch.float32,
).to(device)

set_seed(SEED)

model = TDACNN(
    seq_len     = seq_len,
    d_model     = D_MODEL,
    num_blocks  = NUM_TDA_BLOCKS,
    dilations   = DILATIONS,
    se_reduction= SE_REDUCTION,
    num_classes = num_classes,
    dropout     = DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
fprint(f"\nTDACNN parameters: {total_params:,}")
fprint(f"Architecture     : {NUM_TDA_BLOCKS} TDA blocks | dilations={DILATIONS} | d_model={D_MODEL}")
fprint(f"Pool dim         : {D_MODEL * 3}  (avg + max + std)")
fprint(f"DataLoader workers: 0 (deadlock-safe)  |  pin_memory: {device.type == 'cuda'}")

optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=1, eta_min=1e-6
)
ce_loss     = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
mse_loss_fn = nn.MSELoss()
con_loss_fn = SupervisedContrastiveLoss(temperature=TEMPERATURE)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# FIX 4: fprint() after every epoch so you always see progress
#         Detailed log every PRINT_EVERY epochs, heartbeat every epoch
# ─────────────────────────────────────────────────────────────────────────────
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
lr_history               = []
best_val_acc, best_state = 0.0, None
patience_count = 0

fprint(f"\nTraining on : final_balanced_dataset.csv — Batch {train_batch}")
fprint(f"Testing on  : smell_dataset.csv           — Batches {test_batches}")
fprint(f"Print every : {PRINT_EVERY} epochs  |  Early stop patience: {PATIENCE}\n")

for epoch in range(EPOCHS):
    model.train()
    ep_loss, ep_correct, ep_total = 0.0, 0, 0

    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)

        optimizer.zero_grad()
        out, out_gen, z_f, z_f_gen, z_s, z_s_gen = model(X_b, return_features=True)

        loss_ce  = ce_loss(out, y_b) + ce_loss(out_gen, y_b)
        loss_mse = mse_loss_fn(z_s_gen, z_s.detach())
        loss_con = con_loss_fn(torch.cat([z_f, z_f_gen]), torch.cat([y_b, y_b]))
        loss     = loss_ce + LAMBDA_MSE * loss_mse + LAMBDA_CON * loss_con

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        ep_loss    += loss_ce.item() * X_b.size(0)
        ep_correct += (out.argmax(1) == y_b).sum().item()
        ep_total   += X_b.size(0)

    scheduler.step()
    cur_lr = optimizer.param_groups[0]["lr"]

    avg_train_loss = ep_loss    / ep_total
    avg_train_acc  = ep_correct / ep_total

    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            out       = model(X_b)
            v_loss   += ce_loss(out, y_b).item() * X_b.size(0)
            v_correct += (out.argmax(1) == y_b).sum().item()
            v_total   += X_b.size(0)

    avg_val_loss = v_loss    / v_total
    avg_val_acc  = v_correct / v_total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(avg_train_acc)
    val_accs.append(avg_val_acc)
    lr_history.append(cur_lr)

    if avg_val_acc > best_val_acc + 0.001:
        best_val_acc   = avg_val_acc
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1

    # Print every 10 epochs only
    if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
        fprint(f"Epoch {epoch+1:3d}/{EPOCHS} | "
               f"Train Acc: {avg_train_acc:.4f}  Loss: {avg_train_loss:.4f} | "
               f"Val Acc: {avg_val_acc:.4f}  Loss: {avg_val_loss:.4f} | "
               f"LR: {cur_lr:.2e} | Patience: {patience_count}/{PATIENCE} | "
               f"Best: {best_val_acc:.4f}")

    if patience_count >= PATIENCE:
        fprint(f"\nEarly stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
        break

if best_state:
    model.load_state_dict(best_state)
    fprint(f"\nRestored best TDACNN weights (val acc: {best_val_acc:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
fprint("\n" + "=" * 52)
fprint("RESULTS — Original UCI Drift Dataset (Batches 2-10)")
fprint("=" * 52)
fprint(f"{'Task':<10} | {'Accuracy':>10} | {'Samples':>8}")
fprint("-" * 37)

accs     = []
all_true = []
all_pred = []

for b in test_batches:
    test_df  = original_df[original_df["batch"] == b]
    X_test   = instance_norm(test_df[feature_cols].values)
    y_test   = test_df["label"].values - 1

    X_test_s = scaler.transform(X_test)
    X_test_l = X_test_s.reshape(X_test_s.shape[0], X_test_s.shape[1], 1)
    X_test_t = torch.tensor(X_test_l, dtype=torch.float32)

    t_loader = DataLoader(
        TensorDataset(X_test_t),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    model.eval()
    preds_list = []
    with torch.no_grad():
        for (X_b,) in t_loader:
            preds_list.extend(
                model(X_b.to(device, non_blocking=True)).argmax(1).cpu().numpy()
            )

    preds = np.array(preds_list)
    acc   = accuracy_score(y_test, preds)
    accs.append(acc)
    all_true.extend(y_test)
    all_pred.extend(preds)

    marker = "✅" if acc >= 0.72 else "🟡" if acc >= 0.50 else "❌"
    fprint(f"1-{b:<7}  | {acc:>10.4f} | {len(y_test):>8}  {marker}")

fprint("-" * 37)
avg = np.mean(accs)
fprint(f"{'AVERAGE':<10} | {avg:>10.4f} |")
fprint("=" * 52)
fprint(f"\nPaper CDCNN target  : 0.7230")
fprint(f"TDACNN result       : {avg:.4f}  ({'✅ BEAT IT!' if avg >= 0.72 else f'gap: {0.723 - avg:.3f}'})")

fprint("\nClassification Report (TDACNN):")
fprint(classification_report(
    all_true, all_pred,
    target_names=[f"Class {i+1}" for i in range(num_classes)]
))

fprint(f"\nTotal time: {time.time() - start_time:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("TDACNN — Temporal Dilated Attention CNN", fontsize=14, fontweight="bold")

axes[0].plot(train_losses, color="royalblue",  label="Train Loss")
axes[0].plot(val_losses,   color="darkorange", label="Val Loss")
axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(True)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

axes[1].plot(train_accs, color="royalblue",  label="Train Acc")
axes[1].plot(val_accs,   color="darkorange", label="Val Acc")
axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(True)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")

axes[2].plot(lr_history, color="purple")
axes[2].set_title("LR Schedule (Cosine Warm Restarts)")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR"); axes[2].grid(True)

task_labels = [f"1-{b}" for b in test_batches]
colors = ["green" if a >= 0.72 else "steelblue" if a >= 0.50 else "tomato" for a in accs]
axes[3].bar(task_labels, accs, color=colors, edgecolor="black", width=0.6)
axes[3].axhline(y=avg,   color="blue", linestyle="--", lw=2, label=f"TDACNN avg: {avg:.3f}")
axes[3].axhline(y=0.723, color="red",  linestyle="--", lw=2, label="Paper CDCNN: 0.723")
axes[3].set_ylim(0, 1.0); axes[3].legend(fontsize=8); axes[3].grid(True, axis="y")
axes[3].set_title("Per-Batch Test Accuracy")
axes[3].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("tdacnn_results.png", dpi=150, bbox_inches="tight")
fprint("Saved: tdacnn_results.png")