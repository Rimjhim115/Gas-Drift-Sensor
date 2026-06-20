"""
CDCNN — Class-Discriminative CNN for Gas Sensor Drift Compensation
==================================================================
Faithful implementation of the CDCNN architecture from:
  "A Drift-Compensating Novel Deep Neural Network for Gas Sensor Drift"
  
Architecture:
  Conv1D blocks → Global pooling → Dense classifier
  + Supervised Contrastive Loss + Feature Generation Module
  + Drift-aware month feature
"""

import os
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
# BATCH → MONTH MAPPING
# ─────────────────────────────────────────────────────────────────────────────
BATCH_MONTH_MAP = {
    1:  1,
    2:  6,
    3:  12,
    4:  14,
    5:  16,
    6:  18,
    7:  21,
    8:  22,
    9:  27,
    10: 36,
}

DRIFT_FROM_BATCH1 = {b: BATCH_MONTH_MAP[b] - BATCH_MONTH_MAP[1] for b in BATCH_MONTH_MAP}


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 300
LR          = 0.0005
DROPOUT     = 0.4
LAMBDA_MSE  = 0.05
LAMBDA_CON  = 0.1
TEMPERATURE = 0.07
PATIENCE    = 50

# CNN architecture parameters
NUM_FILTERS   = [64, 128, 256]   # filters per conv block
KERNEL_SIZE   = 3
DENSE_UNITS   = [256, 128]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
start_time = time.time()
print("Loading datasets...")

balanced_df = pd.read_csv("final_balanced_dataset.csv")
original_df = pd.read_csv("smell_dataset.csv")

train_df     = balanced_df.copy()
test_batches = sorted([b for b in original_df["batch"].unique() if b != 1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device      : {device}")
print(f"Train batches     : ALL synthetic {sorted(train_df['batch'].unique())}")
print(f"Train samples     : {len(train_df)}")
print(f"Test batches      : {test_batches} (original dataset)")

num_classes  = len(balanced_df["label"].unique())
feature_cols = [c for c in balanced_df.columns if c not in ["label", "batch"]]

missing = [c for c in feature_cols if c not in original_df.columns]
if missing:
    print(f"WARNING: missing features: {missing}")
    feature_cols = [c for c in feature_cols if c in original_df.columns]

print(f"Number of classes : {num_classes}")
print(f"Sensor features   : {len(feature_cols)}")
print(f"Total features    : {len(feature_cols) + 1}  (sensors + month)")

print("\nDrift timeline from Batch 1:")
for b in sorted(BATCH_MONTH_MAP.keys()):
    print(f"  Batch {b:2d} → Month {BATCH_MONTH_MAP[b]:2d}  "
          f"(+{DRIFT_FROM_BATCH1[b]:2d} months)")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1D → BN → ReLU → Conv1D → BN → ReLU → MaxPool with residual."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1  = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1    = nn.BatchNorm1d(out_ch)
        self.conv2  = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2    = nn.BatchNorm1d(out_ch)
        self.pool   = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.drop   = nn.Dropout(dropout)
        # 1×1 conv for residual channel matching
        self.residual = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x   = F.relu(self.bn1(self.conv1(x)))
        x   = F.relu(self.bn2(self.conv2(x)))
        # Align residual length before adding
        if res.shape[-1] != x.shape[-1]:
            res = F.adaptive_avg_pool1d(res, x.shape[-1])
        x   = x + res
        x   = self.pool(x)
        return self.drop(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, L)
        gap    = x.mean(dim=-1)          # (B, C)
        scale  = self.fc(gap).unsqueeze(-1)  # (B, C, 1)
        return x * scale


class FeatureGenerationModule(nn.Module):
    """Generates drift-augmented features via latent perturbation."""
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
        return loss[n_pos > 0].mean() if (n_pos > 0).any() \
               else torch.tensor(0.0, device=features.device)


class CDCNN(nn.Module):
    """
    Class-Discriminative CNN for Sensor Drift Compensation.

    Input  : (B, 1, L)  — L = num_features (treated as 1D signal)
    Output : (B, num_classes)

    Architecture:
      3 × ConvBlock (64 → 128 → 256 filters, each with residual + channel attention)
      → Global Average Pool
      → Dense(256) → BN → ReLU → Dropout
      → Dense(128) → BN → ReLU → Dropout
      → Dense(num_classes)

    Auxiliary heads:
      Projector (for contrastive loss)
      FeatureGenerationModule (for drift robustness)
    """
    def __init__(self, input_len, num_classes=6,
                 num_filters=None, dense_units=None,
                 kernel_size=3, dropout=0.4):
        super().__init__()
        if num_filters  is None: num_filters  = [64, 128, 256]
        if dense_units  is None: dense_units  = [256, 128]

        # ── Conv blocks ──────────────────────────────────────────────────────
        self.conv_blocks = nn.ModuleList()
        self.ca_blocks   = nn.ModuleList()
        in_ch = 1
        for out_ch in num_filters:
            self.conv_blocks.append(ConvBlock(in_ch, out_ch, kernel_size, dropout=dropout * 0.5))
            self.ca_blocks.append(ChannelAttention(out_ch))
            in_ch = out_ch

        feat_dim = num_filters[-1]   # after global avg pool → (B, feat_dim)

        # ── Feature generation ───────────────────────────────────────────────
        self.feat_gen = FeatureGenerationModule()

        # ── Projector (for contrastive loss) ─────────────────────────────────
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )

        # ── Classifier ───────────────────────────────────────────────────────
        layers = []
        prev = feat_dim
        for units in dense_units:
            layers += [
                nn.Linear(prev, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = units
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

    def extract_features(self, x):
        # x: (B, 1, L)
        for conv, ca in zip(self.conv_blocks, self.ca_blocks):
            x = conv(x)
            x = ca(x)
        return x.mean(dim=-1)          # Global Average Pool → (B, feat_dim)

    def forward(self, x, return_features=False):
        z_s     = self.extract_features(x)   # (B, feat_dim)
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

# Build training matrix: instance-normalize sensors + append month
X_sensors      = train_df[feature_cols].values
X_sensors_norm = instance_norm(X_sensors)
months_train   = train_df["batch"].map(BATCH_MONTH_MAP).values / 36.0
X_full         = np.hstack([X_sensors_norm, months_train.reshape(-1, 1)])
y_full         = train_df["label"].values - 1

input_len = X_full.shape[1]   # sensors + 1 month feature

print(f"\nTraining samples : {len(X_full)}")
print(f"Class distribution: {np.bincount(y_full)}")
print(f"Input length      : {input_len}")

# No augmentation — data is already synthetic & balanced
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.15, random_state=SEED, stratify=y_full
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)


def to_tensor(X, y, dev):
    # Shape: (B, 1, L) — 1 channel, L features as sequence length
    Xl = X.reshape(X.shape[0], 1, X.shape[1])
    return (torch.tensor(Xl, dtype=torch.float32).to(dev),
            torch.tensor(y,  dtype=torch.long).to(dev))

X_train_t, y_train_t = to_tensor(X_train_s, y_train, device)
X_val_t,   y_val_t   = to_tensor(X_val_s,   y_val,   device)

def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

g = torch.Generator()
g.manual_seed(SEED)

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

model = CDCNN(
    input_len   = input_len,
    num_classes = num_classes,
    num_filters = NUM_FILTERS,
    dense_units = DENSE_UNITS,
    kernel_size = KERNEL_SIZE,
    dropout     = DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters  : {total_params:,}")

optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
ce_loss     = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
mse_loss_fn = nn.MSELoss()
con_loss_fn = SupervisedContrastiveLoss(temperature=TEMPERATURE)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_acc, best_state = 0.0, None
patience_count = 0

print(f"\nTraining CDCNN on ALL synthetic batches")
print(f"Testing on original smell_dataset.csv — Batches {test_batches}\n")

for epoch in range(EPOCHS):
    model.train()
    ep_loss, ep_correct, ep_total = 0.0, 0, 0

    for X_b, y_b in train_loader:
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

    avg_train_loss = ep_loss    / ep_total
    avg_train_acc  = ep_correct / ep_total

    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            out      = model(X_b)
            v_loss   += ce_loss(out, y_b).item() * X_b.size(0)
            v_correct += (out.argmax(1) == y_b).sum().item()
            v_total   += X_b.size(0)

    avg_val_loss = v_loss    / v_total
    avg_val_acc  = v_correct / v_total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(avg_train_acc)
    val_accs.append(avg_val_acc)

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
              f"Patience: {patience_count}/{PATIENCE}")

    if patience_count >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
        break

if best_state:
    model.load_state_dict(best_state)
    print(f"Restored best weights (val acc: {best_val_acc:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTS — Accuracy vs Drift from Batch 1 (Original UCI Dataset)")
print("=" * 65)
print(f"{'Batch':<7} | {'Months':>6} | {'Drift':>8} | {'Accuracy':>10} | {'Samples':>8}")
print("-" * 55)

accs     = []
drifts   = []
all_true = []
all_pred = []

for b in test_batches:
    test_df = original_df[original_df["batch"] == b]

    X_sensors_test = instance_norm(test_df[feature_cols].values)
    month_val      = BATCH_MONTH_MAP[b] / 36.0
    months_test    = np.full((len(X_sensors_test), 1), month_val)
    X_test         = np.hstack([X_sensors_test, months_test])

    y_test   = test_df["label"].values - 1
    X_test_s = scaler.transform(X_test)

    # Shape: (B, 1, L) for CNN
    X_test_l = X_test_s.reshape(X_test_s.shape[0], 1, X_test_s.shape[1])
    X_test_t = torch.tensor(X_test_l, dtype=torch.float32)

    t_loader = DataLoader(TensorDataset(X_test_t), batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds_list = []
    with torch.no_grad():
        for (X_b,) in t_loader:
            preds_list.extend(model(X_b.to(device)).argmax(1).cpu().numpy())

    preds      = np.array(preds_list)
    acc        = accuracy_score(y_test, preds)
    drift_mths = DRIFT_FROM_BATCH1[b]

    accs.append(acc)
    drifts.append(drift_mths)
    all_true.extend(y_test)
    all_pred.extend(preds)

    marker = "✅" if acc >= 0.72 else "🟡" if acc >= 0.50 else "❌"
    print(f"Batch {b:<2} | {BATCH_MONTH_MAP[b]:>6} | +{drift_mths:>6}mo | "
          f"{acc:>10.4f} | {len(y_test):>8}  {marker}")

print("-" * 55)
avg = np.mean(accs)
print(f"{'AVERAGE':<7} |        |          | {avg:>10.4f} |")
print("=" * 65)
print(f"\nPaper CDCNN target : 0.7230")
print(f"Our CDCNN result   : {avg:.4f}  "
      f"({'✅ BEAT IT!' if avg >= 0.72 else f'gap: {0.723 - avg:.4f}'})")

print("\nClassification Report:")
print(classification_report(
    all_true, all_pred,
    target_names=[f"Class {i+1}" for i in range(num_classes)]
))

print(f"\nTotal time: {time.time() - start_time:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, color="royalblue",  label="Train Loss")
axes[0].plot(val_losses,   color="darkorange", label="Val Loss")
axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(True)

axes[1].plot(train_accs, color="royalblue",  label="Train Acc")
axes[1].plot(val_accs,   color="darkorange", label="Val Acc")
axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(True)

colors   = ["green" if a >= 0.72 else "steelblue" if a >= 0.50 else "tomato" for a in accs]
x_labels = [f"B{b}\n+{DRIFT_FROM_BATCH1[b]}mo" for b in test_batches]
axes[2].bar(x_labels, accs, color=colors, edgecolor="black", width=0.6)
axes[2].axhline(y=avg,   color="blue", linestyle="--", lw=2, label=f"Our avg: {avg:.3f}")
axes[2].axhline(y=0.723, color="red",  linestyle="--", lw=2, label="Paper CDCNN: 0.723")
axes[2].set_ylim(0, 1.0)
axes[2].set_xlabel("Batch (months drifted from Batch 1)")
axes[2].set_ylabel("Accuracy")
axes[2].set_title("Accuracy vs Drift from Batch 1")
axes[2].legend(); axes[2].grid(True, axis="y")

plt.tight_layout()
plt.savefig("cdcnn_drift_results.png", dpi=150, bbox_inches="tight")
print("Saved: cdcnn_drift_results.png")