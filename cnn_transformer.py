"""
CDCNN-Style CNN-Transformer Model — Cross-Batch Smell Classification
=====================================================================
Fixes over pure-Transformer version:
  1. CNN stem before Transformer:
       • 3 Conv1d layers extract local patterns + reduce sequence length
       • Gives Transformer meaningful tokens (not raw scalar features)
       • Adds inductive bias that pure Transformer lacks on small data
  2. [CLS] token pooling instead of mean-pooling (BERT-style)
  3. Warm-up + Cosine LR schedule instead of ReduceLROnPlateau
       • Transformer training is sensitive to LR; warm-up prevents early
         instability that caused class collapse in the pure-Transformer run
  4. Label smoothing in CrossEntropyLoss (prevents overconfident predictions
     that cause the 0.00 precision on Classes 4-6)
  5. Domain-adversarial style feature regularization via gradient reversal
     replaced with stronger augmentation (simpler, same effect)
  6. Mixup augmentation inside training loop for better generalization
  Everything else (contrastive loss, feature gen, reproducibility) unchanged.
"""

import os
import random
import math
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
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)

set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
EPOCHS          = 200
LR              = 3e-4         # lower LR — Transformer is sensitive
WARMUP_EPOCHS   = 10           # LR ramps from 0 → LR over first 10 epochs
CNN_CHANNELS    = [32, 64, 128]  # Conv1d output channels per layer
CNN_KERNEL      = 3
D_MODEL         = 128          # must equal CNN_CHANNELS[-1]
NHEAD           = 8            # 128 / 8 = 16 per head
NUM_LAYERS      = 2            # fewer layers → less overfitting on small data
DIM_FEEDFWD     = 256
DROPOUT         = 0.3          # reduced; CNN already regularizes
LABEL_SMOOTHING = 0.1          # prevents class collapse
LAMBDA_MSE      = 0.05
LAMBDA_CON      = 0.05
TEMPERATURE     = 0.07
PATIENCE        = 50           # more patience — cosine LR needs time
MIXUP_ALPHA     = 0.2          # mixup augmentation strength


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
start_time = time.time()
print("Loading datasets...")

balanced_df = pd.read_csv("final_balanced_dataset.csv")
train_batch = sorted(balanced_df["batch"].unique())[0]
train_df    = balanced_df[balanced_df["batch"] == train_batch]

original_df  = pd.read_csv("smell_dataset.csv")
test_batches = [b for b in sorted(original_df["batch"].unique()) if b != train_batch]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device      : {device}")
print(f"Train batch       : Batch {train_batch} from final_balanced_dataset.csv")
print(f"Train samples     : {len(train_df)}")
print(f"Test batches      : {test_batches}")
print(f"Test samples      : {len(original_df[original_df['batch'].isin(test_batches)])}")

num_classes  = len(balanced_df["label"].unique())
feature_cols = [c for c in balanced_df.columns if c not in ["label", "batch"]]
print(f"Number of classes : {num_classes}")
print(f"Number of features: {len(feature_cols)}")

missing = [c for c in feature_cols if c not in original_df.columns]
if missing:
    print(f"WARNING: features missing in original dataset: {missing}")
    feature_cols = [c for c in feature_cols if c in original_df.columns]
    print(f"Using {len(feature_cols)} common features")


# ─────────────────────────────────────────────────────────────────────────────
# VAE-INSPIRED AUGMENTATION  (unchanged, seeded)
# ─────────────────────────────────────────────────────────────────────────────
def vae_augment(X, y, augment_factor=5, seed=SEED):
    rng = np.random.default_rng(seed)
    X_aug_list = [X]
    y_aug_list = [y]
    for _ in range(augment_factor):
        X_new = np.zeros_like(X)
        for i in range(len(X)):
            same_class_idx = np.where(y == y[i])[0]
            k       = rng.choice(same_class_idx)
            lam     = rng.uniform(0.3, 0.7)
            mu_j    = X[i].mean();  std_j = X[i].std()  + 1e-8
            mu_k    = X[k].mean();  std_k = X[k].std()  + 1e-8
            mu_mix  = lam * mu_j  + (1 - lam) * mu_k
            std_mix = lam * std_j + (1 - lam) * std_k
            noise   = rng.normal(mu_mix, std_mix, size=X[i].shape)
            X_new[i]= X[i] + noise * 0.1
        X_aug_list.append(X_new)
        y_aug_list.append(y)
    return np.vstack(X_aug_list), np.hstack(y_aug_list)


# ─────────────────────────────────────────────────────────────────────────────
# MIXUP (applied per-batch during training)
# ─────────────────────────────────────────────────────────────────────────────
def mixup_batch(X, y, alpha=MIXUP_ALPHA):
    """Returns mixed inputs + both label vectors + lambda for loss computation."""
    if alpha <= 0:
        return X, y, y, 1.0
    lam   = np.random.beta(alpha, alpha)
    B     = X.size(0)
    idx   = torch.randperm(B, device=X.device)
    X_mix = lam * X + (1 - lam) * X[idx]
    return X_mix, y, y[idx], lam


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GENERATION MODULE  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureGenerationModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

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
# SUPERVISED CONTRASTIVE LOSS  (unchanged)
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
# CNN-TRANSFORMER MODEL
#
# Architecture:
#   Input  (B, seq=128, 1)
#     → CNN Stem: 3 × [Conv1d → BN → GELU → MaxPool]
#         Layer 1: 1  → 32  ch, kernel 3, pool 2  → seq 64
#         Layer 2: 32 → 64  ch, kernel 3, pool 2  → seq 32
#         Layer 3: 64 → 128 ch, kernel 3, pool 1  → seq 30
#     → Reshape to (B, ~30, 128)   [tokens for Transformer]
#     → Prepend learnable [CLS] token → (B, 31, 128)
#     → Positional Encoding
#     → N × TransformerEncoderLayer
#     → CLS token output → (B, 128)   [richer than mean pooling]
#     → FeatureGen + Projector + Classifier
#
# Why CNN stem works:
#   • Sensor features have local correlations (adjacent sensors respond similarly)
#   • CNN captures those local patterns in O(k·n) vs Transformer's O(n²)
#   • Reduces sequence length from 128 → ~30 so Transformer sees fewer tokens
#   • Provides strong inductive bias for small datasets (3600 samples)
# ─────────────────────────────────────────────────────────────────────────────
class CNNTransformer(nn.Module):
    def __init__(
        self,
        seq_len      : int   = 128,
        in_channels  : int   = 1,
        cnn_channels : list  = None,
        cnn_kernel   : int   = 3,
        d_model      : int   = 128,
        nhead        : int   = 8,
        num_layers   : int   = 2,
        dim_feedfwd  : int   = 256,
        num_classes  : int   = 6,
        dropout      : float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        assert d_model == cnn_channels[-1], \
            f"d_model ({d_model}) must equal cnn_channels[-1] ({cnn_channels[-1]})"

        # ── CNN Stem ──────────────────────────────────────────────────────────
        # Input shape:  (B, in_channels=1, seq_len=128)   [channels-first for Conv1d]
        # Output shape: (B, d_model=128,   reduced_seq)
        layers = []
        ch_in  = in_channels
        for i, ch_out in enumerate(cnn_channels):
            layers += [
                nn.Conv1d(ch_in, ch_out, kernel_size=cnn_kernel,
                          padding=cnn_kernel // 2),
                nn.BatchNorm1d(ch_out),
                nn.GELU(),
            ]
            # Pool on first 2 layers to reduce seq length; skip last to preserve
            if i < len(cnn_channels) - 1:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            ch_in = ch_out
        self.cnn_stem = nn.Sequential(*layers)

        # Compute reduced seq length after pooling
        # 2 MaxPool1d(2) → divide by 4
        self.reduced_seq = seq_len // 4

        # ── [CLS] token (learned, like BERT) ─────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional Encoding ───────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(
            d_model, dropout=dropout, max_len=self.reduced_seq + 1
        )

        # ── Transformer Encoder ───────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedfwd,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = False,   # Post-LN avoids the nested_tensor warning
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers = num_layers,
            norm       = nn.LayerNorm(d_model),
        )

        # ── Head modules (same as before) ─────────────────────────────────────
        feat_dim       = d_model
        self.feat_gen  = FeatureGenerationModule(feat_dim)

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x):
        # x: (B, seq_len, 1)
        B = x.size(0)

        # CNN stem expects (B, channels, seq)
        x = x.permute(0, 2, 1)                    # (B, 1, 128)
        x = self.cnn_stem(x)                       # (B, d_model, reduced_seq)
        x = x.permute(0, 2, 1)                     # (B, reduced_seq, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)           # (B, 1+reduced_seq, d_model)

        x   = self.pos_enc(x)
        x   = self.transformer(x)                  # (B, 1+reduced_seq, d_model)

        return x[:, 0, :]                           # CLS token → (B, d_model)

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
# PREPROCESSING & AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def instance_norm(X):
    mu  = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1,  keepdims=True) + 1e-8
    return (X - mu) / std

X_full = instance_norm(train_df[feature_cols].values)
y_full = train_df["label"].values - 1

print(f"\nOriginal training samples : {len(X_full)}")
print(f"Class distribution        : {np.bincount(y_full)}")

print("\nApplying VAE-inspired augmentation...")
X_aug, y_aug = vae_augment(X_full, y_full, augment_factor=5, seed=SEED)
print(f"Augmented training samples: {len(X_aug)}")

X_train, X_val, y_train, y_val = train_test_split(
    X_aug, y_aug, test_size=0.15, random_state=SEED, stratify=y_aug
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)


def to_tensor(X, y, dev):
    Xl = X.reshape(X.shape[0], X.shape[1], 1)
    return (torch.tensor(Xl, dtype=torch.float32).to(dev),
            torch.tensor(y,  dtype=torch.long).to(dev))

X_train_t, y_train_t = to_tensor(X_train_s, y_train, device)
X_val_t,   y_val_t   = to_tensor(X_val_s,   y_val,   device)

def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

g = torch.Generator(); g.manual_seed(SEED)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    worker_init_fn=seed_worker, generator=g,
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=BATCH_SIZE, shuffle=False,
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

seq_len = len(feature_cols)
model = CNNTransformer(
    seq_len      = seq_len,
    in_channels  = 1,
    cnn_channels = CNN_CHANNELS,
    cnn_kernel   = CNN_KERNEL,
    d_model      = D_MODEL,
    nhead        = NHEAD,
    num_layers   = NUM_LAYERS,
    dim_feedfwd  = DIM_FEEDFWD,
    num_classes  = num_classes,
    dropout      = DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters : {total_params:,}")
print(f"CNN reduced seq  : {model.reduced_seq}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# Cosine annealing with linear warm-up
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS          # linear ramp
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1 + math.cos(math.pi * progress)) # cosine decay

scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
ce_loss     = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
mse_loss_fn = nn.MSELoss()
con_loss_fn = SupervisedContrastiveLoss(temperature=TEMPERATURE)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
lr_history               = []
best_val_acc, best_state = 0.0, None
patience_count = 0

print(f"\nTraining on : final_balanced_dataset.csv — Batch {train_batch}")
print(f"Testing on  : smell_dataset.csv           — Batches {test_batches}\n")

for epoch in range(EPOCHS):
    model.train()
    ep_loss, ep_correct, ep_total = 0.0, 0, 0

    for X_b, y_b in train_loader:
        # Mixup augmentation
        X_mix, y_a, y_b2, lam = mixup_batch(X_b, y_b, alpha=MIXUP_ALPHA)

        optimizer.zero_grad()
        out, out_gen, z_f, z_f_gen, z_s, z_s_gen = model(X_mix, return_features=True)

        # Mixup loss: interpolate between two label sets
        loss_ce  = (lam * ce_loss(out, y_a) + (1 - lam) * ce_loss(out, y_b2) +
                    lam * ce_loss(out_gen, y_a) + (1 - lam) * ce_loss(out_gen, y_b2))
        loss_mse = mse_loss_fn(z_s_gen, z_s.detach())
        # Contrastive loss uses original labels (not mixed)
        loss_con = con_loss_fn(torch.cat([z_f, z_f_gen]), torch.cat([y_a, y_a]))
        loss     = loss_ce + LAMBDA_MSE * loss_mse + LAMBDA_CON * loss_con

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        ep_loss    += loss_ce.item() * X_b.size(0)
        ep_correct += (out.argmax(1) == y_a).sum().item()
        ep_total   += X_b.size(0)

    scheduler.step()

    avg_train_loss = ep_loss    / ep_total
    avg_train_acc  = ep_correct / ep_total

    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            out       = model(X_b)
            v_loss   += ce_loss(out, y_b).item() * X_b.size(0)
            v_correct += (out.argmax(1) == y_b).sum().item()
            v_total   += X_b.size(0)

    avg_val_loss = v_loss    / v_total
    avg_val_acc  = v_correct / v_total
    cur_lr       = optimizer.param_groups[0]["lr"]

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

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Acc: {avg_train_acc:.4f}  Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {avg_val_acc:.4f}  Loss: {avg_val_loss:.4f} | "
              f"LR: {cur_lr:.2e} | Patience: {patience_count}/{PATIENCE}")

    if patience_count >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
        break

if best_state:
    model.load_state_dict(best_state)
    print(f"Restored best weights (val acc: {best_val_acc:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 52)
print("RESULTS — Original UCI Drift Dataset (Batches 2-10)")
print("=" * 52)
print(f"{'Task':<10} | {'Accuracy':>10} | {'Samples':>8}")
print("-" * 37)

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

    t_loader = DataLoader(TensorDataset(X_test_t), batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds_list = []
    with torch.no_grad():
        for (X_b,) in t_loader:
            preds_list.extend(model(X_b.to(device)).argmax(1).cpu().numpy())

    preds = np.array(preds_list)
    acc   = accuracy_score(y_test, preds)
    accs.append(acc)
    all_true.extend(y_test)
    all_pred.extend(preds)

    marker = "✅" if acc >= 0.72 else "🟡" if acc >= 0.50 else "❌"
    print(f"1-{b:<7}  | {acc:>10.4f} | {len(y_test):>8}  {marker}")

print("-" * 37)
avg = np.mean(accs)
print(f"{'AVERAGE':<10} | {avg:>10.4f} |")
print("=" * 52)
print(f"\nPaper CDCNN target      : 0.7230")
print(f"Our CNN-Transformer     : {avg:.4f}  ({'✅ BEAT IT!' if avg >= 0.72 else f'gap: {0.723 - avg:.3f}'})")

print("\nClassification Report:")
print(classification_report(
    all_true, all_pred,
    target_names=[f"Class {i+1}" for i in range(num_classes)]
))

print(f"\nTotal time: {time.time() - start_time:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].plot(train_losses, color="royalblue",  label="Train Loss")
axes[0].plot(val_losses,   color="darkorange", label="Val Loss")
axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(True)

axes[1].plot(train_accs, color="royalblue",  label="Train Acc")
axes[1].plot(val_accs,   color="darkorange", label="Val Acc")
axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(True)

axes[2].plot(lr_history, color="purple")
axes[2].set_title("Learning Rate Schedule"); axes[2].grid(True)
axes[2].set_ylabel("LR"); axes[2].set_xlabel("Epoch")

task_labels = [f"1-{b}" for b in test_batches]
colors = ["green" if a >= 0.72 else "steelblue" if a >= 0.50 else "tomato" for a in accs]
axes[3].bar(task_labels, accs, color=colors, edgecolor="black", width=0.6)
axes[3].axhline(y=avg,   color="blue", linestyle="--", lw=2, label=f"Our avg: {avg:.3f}")
axes[3].axhline(y=0.723, color="red",  linestyle="--", lw=2, label="Paper CDCNN: 0.723")
axes[3].set_ylim(0, 1.0); axes[3].legend(); axes[3].grid(True, axis="y")
axes[3].set_title("Per-Batch Test Accuracy"); plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("cnn_transformer_results.png", dpi=150, bbox_inches="tight")
print("Saved: cnn_transformer_results.png")