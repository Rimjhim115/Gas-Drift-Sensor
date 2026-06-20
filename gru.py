# """
# CDCNN-Style GRU Model — Drift-Aware, Train on ALL Synthetic Batches
# ====================================================================
# Changes from previous version:
#   1. Train on ALL batches from final_balanced_dataset.csv (not just batch 1)
#   2. Month added as an extra feature (drift time context)
#   3. Test on original smell_dataset.csv batches 2-10
#   4. Results show accuracy vs months drifted from batch 1
#   5. Removed VAE augmentation (training data already synthetic & balanced)
# """

# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import time
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset


# # ─────────────────────────────────────────────────────────────────────────────
# # REPRODUCIBILITY
# # ─────────────────────────────────────────────────────────────────────────────
# SEED = 42

# def set_seed(seed: int = SEED):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark     = False
#     os.environ["PYTHONHASHSEED"]       = str(seed)

# set_seed(SEED)


# # ─────────────────────────────────────────────────────────────────────────────
# # BATCH → MONTH MAPPING  (official UCI dataset timeline)
# # ─────────────────────────────────────────────────────────────────────────────
# BATCH_MONTH_MAP = {
#     1:  1,
#     2:  6,
#     3:  12,
#     4:  14,
#     5:  16,
#     6:  18,
#     7:  21,
#     8:  22,
#     9:  27,
#     10: 36,
# }

# DRIFT_FROM_BATCH1 = {b: BATCH_MONTH_MAP[b] - BATCH_MONTH_MAP[1] for b in BATCH_MONTH_MAP}


# # ─────────────────────────────────────────────────────────────────────────────
# # HYPERPARAMETERS
# # ─────────────────────────────────────────────────────────────────────────────
# BATCH_SIZE   = 64
# EPOCHS       = 200
# LR           = 0.001
# HIDDEN_SIZE  = 128
# NUM_LAYERS   = 3
# DROPOUT      = 0.5
# LAMBDA_MSE   = 0.05
# LAMBDA_CON   = 0.05
# TEMPERATURE  = 0.07
# PATIENCE     = 40


# # ─────────────────────────────────────────────────────────────────────────────
# # DATA LOADING
# # ─────────────────────────────────────────────────────────────────────────────
# start_time = time.time()
# print("Loading datasets...")

# balanced_df = pd.read_csv("final_balanced_dataset.csv")
# original_df = pd.read_csv("smell_dataset.csv")

# train_df    = balanced_df.copy()
# test_batches = sorted([b for b in original_df["batch"].unique() if b != 1])

# device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device      : {device}")
# print(f"Train batches     : ALL synthetic batches {sorted(train_df['batch'].unique())}")
# print(f"Train samples     : {len(train_df)}")
# print(f"Test batches      : {test_batches} (original dataset)")
# print(f"Test samples      : {len(original_df[original_df['batch'].isin(test_batches)])}")

# num_classes  = len(balanced_df["label"].unique())
# feature_cols = [c for c in balanced_df.columns if c not in ["label", "batch"]]

# missing = [c for c in feature_cols if c not in original_df.columns]
# if missing:
#     print(f"WARNING: features missing in original dataset: {missing}")
#     feature_cols = [c for c in feature_cols if c in original_df.columns]

# print(f"Number of classes : {num_classes}")
# print(f"Sensor features   : {len(feature_cols)}")
# print(f"Total features    : {len(feature_cols) + 1}  (sensors + month)")

# print("\nDrift timeline from Batch 1:")
# for b in sorted(BATCH_MONTH_MAP.keys()):
#     print(f"  Batch {b:2d} → Month {BATCH_MONTH_MAP[b]:2d}  "
#           f"(+{DRIFT_FROM_BATCH1[b]:2d} months from Batch 1)")


# # ─────────────────────────────────────────────────────────────────────────────
# # MODEL COMPONENTS
# # ─────────────────────────────────────────────────────────────────────────────
# class AttentionLayer(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.attn = nn.Linear(hidden_size * 2, 1)

#     def forward(self, gru_out):
#         weights = torch.softmax(self.attn(gru_out), dim=1)
#         return (weights * gru_out).sum(dim=1)


# class FeatureGenerationModule(nn.Module):
#     def __init__(self, feature_dim):
#         super().__init__()
#         self.feature_dim = feature_dim

#     def forward(self, z):
#         z_high     = F.adaptive_max_pool1d(z.unsqueeze(1), z.shape[-1]).squeeze(1)
#         z_low      = z - z_high
#         mu_low     = z_low.mean(dim=0, keepdim=True)
#         std_low    = z_low.std(dim=0,  keepdim=True) + 1e-8
#         mu_new     = mu_low  + torch.randn_like(mu_low)  * std_low * 0.1
#         std_new    = (std_low + torch.randn_like(std_low) * std_low * 0.05).abs() + 1e-8
#         z_low_norm = (z_low - mu_low) / std_low
#         z_low_new  = std_new * z_low_norm + mu_new
#         return z_high + z_low_new


# class SupervisedContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, features, labels):
#         B = features.shape[0]
#         if B < 2:
#             return torch.tensor(0.0, device=features.device)
#         features  = F.normalize(features, dim=1)
#         sim       = torch.matmul(features, features.T) / self.temperature
#         pos_mask  = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
#         pos_mask.fill_diagonal_(0)
#         sim_max, _= sim.max(dim=1, keepdim=True)
#         sim       = sim - sim_max.detach()
#         exp_sim   = torch.exp(sim)
#         self_mask = torch.ones(B, B, device=features.device).fill_diagonal_(0)
#         denom     = (exp_sim * self_mask).sum(dim=1, keepdim=True) + 1e-8
#         log_prob  = sim - torch.log(denom)
#         n_pos     = pos_mask.sum(dim=1)
#         loss      = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
#         return loss[n_pos > 0].mean() if (n_pos > 0).any() else torch.tensor(0.0, device=features.device)


# class CDCNNStyleGRU(nn.Module):
#     def __init__(self, input_size=1, hidden_size=128, num_layers=3,
#                  num_classes=6, dropout=0.3):
#         super().__init__()
#         self.gru = nn.GRU(
#             input_size   = input_size,
#             hidden_size  = hidden_size,
#             num_layers   = num_layers,
#             batch_first  = True,
#             dropout      = dropout if num_layers > 1 else 0.0,
#             bidirectional= True,
#         )
#         feat_dim        = hidden_size * 2
#         self.layer_norm = nn.LayerNorm(feat_dim)
#         self.attention  = AttentionLayer(hidden_size)
#         self.feat_gen   = FeatureGenerationModule(feat_dim)
#         self.projector  = nn.Sequential(
#             nn.Linear(feat_dim, 128), nn.ReLU(),
#             nn.Linear(128, 64),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(128, num_classes),
#         )

#     def extract_features(self, x):
#         out, _ = self.gru(x)
#         out    = self.layer_norm(out)
#         return self.attention(out)

#     def forward(self, x, return_features=False):
#         z_s     = self.extract_features(x)
#         z_s_gen = self.feat_gen(z_s)
#         z_f     = self.projector(z_s)
#         z_f_gen = self.projector(z_s_gen)
#         out     = self.classifier(z_s)
#         out_gen = self.classifier(z_s_gen)
#         if return_features:
#             return out, out_gen, z_f, z_f_gen, z_s, z_s_gen
#         return out


# # ─────────────────────────────────────────────────────────────────────────────
# # PREPROCESSING
# # ─────────────────────────────────────────────────────────────────────────────
# def instance_norm(X):
#     mu  = X.mean(axis=1, keepdims=True)
#     std = X.std(axis=1,  keepdims=True) + 1e-8
#     return (X - mu) / std

# # Instance-normalize sensor features, then append normalized month
# X_sensors      = train_df[feature_cols].values
# X_sensors_norm = instance_norm(X_sensors)
# months_train   = train_df["batch"].map(BATCH_MONTH_MAP).values / 36.0
# X_full         = np.hstack([X_sensors_norm, months_train.reshape(-1, 1)])
# y_full         = train_df["label"].values - 1

# print(f"\nTraining samples (all synthetic batches) : {len(X_full)}")
# print(f"Class distribution                       : {np.bincount(y_full)}")
# print(f"Input feature size (sensors + month)     : {X_full.shape[1]}")

# # No augmentation — data is already synthetic and balanced
# X_train, X_val, y_train, y_val = train_test_split(
#     X_full, y_full, test_size=0.15, random_state=SEED, stratify=y_full
# )

# scaler    = StandardScaler()
# X_train_s = scaler.fit_transform(X_train)
# X_val_s   = scaler.transform(X_val)


# def to_tensor(X, y, dev):
#     Xl = X.reshape(X.shape[0], X.shape[1], 1)
#     return (torch.tensor(Xl, dtype=torch.float32).to(dev),
#             torch.tensor(y,  dtype=torch.long).to(dev))

# X_train_t, y_train_t = to_tensor(X_train_s, y_train, device)
# X_val_t,   y_val_t   = to_tensor(X_val_s,   y_val,   device)

# def seed_worker(worker_id):
#     np.random.seed(SEED + worker_id)
#     random.seed(SEED + worker_id)

# g = torch.Generator()
# g.manual_seed(SEED)

# train_loader = DataLoader(
#     TensorDataset(X_train_t, y_train_t),
#     batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
#     worker_init_fn=seed_worker, generator=g,
# )
# val_loader = DataLoader(
#     TensorDataset(X_val_t, y_val_t),
#     batch_size=BATCH_SIZE, shuffle=False,
# )


# # ─────────────────────────────────────────────────────────────────────────────
# # MODEL, LOSS, OPTIMISER
# # ─────────────────────────────────────────────────────────────────────────────
# class_counts = np.bincount(y_train)
# cw = torch.tensor(
#     (1.0 / class_counts) / (1.0 / class_counts).sum() * num_classes,
#     dtype=torch.float32,
# ).to(device)

# set_seed(SEED)

# model = CDCNNStyleGRU(
#     input_size  = 1,
#     hidden_size = HIDDEN_SIZE,
#     num_layers  = NUM_LAYERS,
#     num_classes = num_classes,
#     dropout     = DROPOUT,
# ).to(device)

# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"\nModel parameters : {total_params:,}")

# optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
# ce_loss     = nn.CrossEntropyLoss(weight=cw)
# mse_loss_fn = nn.MSELoss()
# con_loss_fn = SupervisedContrastiveLoss(temperature=TEMPERATURE)


# # ─────────────────────────────────────────────────────────────────────────────
# # TRAINING LOOP
# # ─────────────────────────────────────────────────────────────────────────────
# train_losses, val_losses = [], []
# train_accs,   val_accs   = [], []
# best_val_acc, best_state = 0.0, None
# patience_count = 0

# print(f"\nTraining on : ALL synthetic batches from final_balanced_dataset.csv")
# print(f"Testing on  : original smell_dataset.csv — Batches {test_batches}\n")

# for epoch in range(EPOCHS):
#     model.train()
#     ep_loss, ep_correct, ep_total = 0.0, 0, 0

#     for X_b, y_b in train_loader:
#         optimizer.zero_grad()
#         out, out_gen, z_f, z_f_gen, z_s, z_s_gen = model(X_b, return_features=True)
#         loss_ce  = ce_loss(out, y_b) + ce_loss(out_gen, y_b)
#         loss_mse = mse_loss_fn(z_s_gen, z_s.detach())
#         loss_con = con_loss_fn(torch.cat([z_f, z_f_gen]), torch.cat([y_b, y_b]))
#         loss     = loss_ce + LAMBDA_MSE * loss_mse + LAMBDA_CON * loss_con

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()

#         ep_loss    += loss_ce.item() * X_b.size(0)
#         ep_correct += (out.argmax(1) == y_b).sum().item()
#         ep_total   += X_b.size(0)

#     avg_train_loss = ep_loss    / ep_total
#     avg_train_acc  = ep_correct / ep_total

#     model.eval()
#     v_loss, v_correct, v_total = 0.0, 0, 0
#     with torch.no_grad():
#         for X_b, y_b in val_loader:
#             out      = model(X_b)
#             v_loss   += ce_loss(out, y_b).item() * X_b.size(0)
#             v_correct += (out.argmax(1) == y_b).sum().item()
#             v_total   += X_b.size(0)

#     avg_val_loss = v_loss    / v_total
#     avg_val_acc  = v_correct / v_total

#     scheduler.step(avg_val_acc)
#     train_losses.append(avg_train_loss)
#     val_losses.append(avg_val_loss)
#     train_accs.append(avg_train_acc)
#     val_accs.append(avg_val_acc)

#     if avg_val_acc > best_val_acc + 0.001:
#         best_val_acc   = avg_val_acc
#         best_state     = {k: v.clone() for k, v in model.state_dict().items()}
#         patience_count = 0
#     else:
#         patience_count += 1

#     if (epoch + 1) % 20 == 0:
#         print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
#               f"Train Acc: {avg_train_acc:.4f}  Loss: {avg_train_loss:.4f} | "
#               f"Val Acc: {avg_val_acc:.4f}  Loss: {avg_val_loss:.4f} | "
#               f"Patience: {patience_count}/{PATIENCE}")

#     if patience_count >= PATIENCE:
#         print(f"\nEarly stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
#         break

# if best_state:
#     model.load_state_dict(best_state)
#     print(f"Restored best weights (val acc: {best_val_acc:.4f})")


# # ─────────────────────────────────────────────────────────────────────────────
# # EVALUATION
# # ─────────────────────────────────────────────────────────────────────────────
# print("\n" + "=" * 65)
# print("RESULTS — Accuracy vs Drift from Batch 1 (Original UCI Dataset)")
# print("=" * 65)
# print(f"{'Batch':<7} | {'Months':>6} | {'Drift':>8} | {'Accuracy':>10} | {'Samples':>8}")
# print("-" * 55)

# accs     = []
# drifts   = []
# all_true = []
# all_pred = []

# for b in test_batches:
#     test_df = original_df[original_df["batch"] == b]

#     X_sensors_test = instance_norm(test_df[feature_cols].values)
#     month_val      = BATCH_MONTH_MAP[b] / 36.0
#     months_test    = np.full((len(X_sensors_test), 1), month_val)
#     X_test         = np.hstack([X_sensors_test, months_test])

#     y_test   = test_df["label"].values - 1
#     X_test_s = scaler.transform(X_test)
#     X_test_l = X_test_s.reshape(X_test_s.shape[0], X_test_s.shape[1], 1)
#     X_test_t = torch.tensor(X_test_l, dtype=torch.float32)

#     t_loader = DataLoader(TensorDataset(X_test_t), batch_size=BATCH_SIZE, shuffle=False)

#     model.eval()
#     preds_list = []
#     with torch.no_grad():
#         for (X_b,) in t_loader:
#             preds_list.extend(model(X_b.to(device)).argmax(1).cpu().numpy())

#     preds      = np.array(preds_list)
#     acc        = accuracy_score(y_test, preds)
#     drift_mths = DRIFT_FROM_BATCH1[b]

#     accs.append(acc)
#     drifts.append(drift_mths)
#     all_true.extend(y_test)
#     all_pred.extend(preds)

#     marker = "✅" if acc >= 0.72 else "🟡" if acc >= 0.50 else "❌"
#     print(f"Batch {b:<2} | {BATCH_MONTH_MAP[b]:>6} | +{drift_mths:>6}mo | {acc:>10.4f} | {len(y_test):>8}  {marker}")

# print("-" * 55)
# avg = np.mean(accs)
# print(f"{'AVERAGE':<7} |        |          | {avg:>10.4f} |")
# print("=" * 65)
# print(f"\nPaper CDCNN target : 0.7230")
# print(f"Our GRU result     : {avg:.4f}  ({'✅ BEAT IT!' if avg >= 0.72 else f'gap: {0.723 - avg:.4f}'})")

# print("\nClassification Report:")
# print(classification_report(
#     all_true, all_pred,
#     target_names=[f"Class {i+1}" for i in range(num_classes)]
# ))

# print(f"\nTotal time: {time.time() - start_time:.1f}s")


# # ─────────────────────────────────────────────────────────────────────────────
# # PLOTS
# # ─────────────────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# axes[0].plot(train_losses, color="royalblue",  label="Train Loss")
# axes[0].plot(val_losses,   color="darkorange", label="Val Loss")
# axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(True)

# axes[1].plot(train_accs, color="royalblue",  label="Train Acc")
# axes[1].plot(val_accs,   color="darkorange", label="Val Acc")
# axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(True)

# colors   = ["green" if a >= 0.72 else "steelblue" if a >= 0.50 else "tomato" for a in accs]
# x_labels = [f"B{b}\n+{DRIFT_FROM_BATCH1[b]}mo" for b in test_batches]
# axes[2].bar(x_labels, accs, color=colors, edgecolor="black", width=0.6)
# axes[2].axhline(y=avg,   color="blue", linestyle="--", lw=2, label=f"Our avg: {avg:.3f}")
# axes[2].axhline(y=0.723, color="red",  linestyle="--", lw=2, label="Paper CDCNN: 0.723")
# axes[2].set_ylim(0, 1.0)
# axes[2].set_xlabel("Batch (months drifted from Batch 1)")
# axes[2].set_ylabel("Accuracy")
# axes[2].set_title("Accuracy vs Drift from Batch 1")
# axes[2].legend(); axes[2].grid(True, axis="y")

# plt.tight_layout()
# plt.savefig("gru_drift_results.png", dpi=150, bbox_inches="tight")
# print("Saved: gru_drift_results.png")
"""
CDCNN — Paper Protocol + VAE Augmentation (Chu et al., 2024)
=============================================================
Train : Synthetic Batch 1 ONLY  → VAE-augmented to ~7000 samples
Test  : Real Batches 2–10

Key fixes over previous version:
  1. VAE augmentation on Batch 1 (paper Section 3, Eq. 7) — most important fix
  2. FGM drift_scale increased from 0.35 → 1.2 (more aggressive domain shift)
  3. DANN removed (not in paper)
  4. Label smoothing reduced (was hurting generalization)
  5. StandardScaler applied consistently train→test

Paper Table 3 targets (CDCNN row):
  B2:0.8271  B3:0.7683  B4:0.7244  B5:0.7891  B6:0.9318
  B7:0.6182  B8:0.7152  B9:0.5664  B10:0.5663  Avg:0.7230
"""

import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ═══════════════════════════════════════════════════════════════════
# 1. SEED
# ═══════════════════════════════════════════════════════════════════
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)

set_seed()


# ═══════════════════════════════════════════════════════════════════
# 2. PATHS
# ═══════════════════════════════════════════════════════════════════
SYNTHETIC_CSV = "/kaggle/input/models/rimjhimbhura/gas-drift-sensors/pytorch/default/1/smell-datasets-main/smell-datasets-main/final_balanced_dataset.csv"
REAL_CSV      = "/kaggle/input/datasets/rimjhimbhura/smell-dataset/New folder/smell_dataset.csv"


# ═══════════════════════════════════════════════════════════════════
# 3. HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════
BATCH_MONTH_MAP = {1:1, 2:6, 3:12, 4:14, 5:16, 6:18, 7:21, 8:22, 9:27, 10:36}
TOTAL_MONTHS    = 36

TRAIN_BATCH  = 1
TEST_BATCHES = list(range(2, 11))

BATCH_SIZE   = 64
EPOCHS       = 400
LR           = 3e-4
DROPOUT      = 0.40
LAMBDA_MSE   = 0.05
LAMBDA_CON   = 0.10
TEMPERATURE  = 0.07
PATIENCE     = 60
MIXUP_ALPHA  = 0.2
FGM_SCALE    = 1.2        # was 0.35 — more aggressive drift simulation
VAE_N_AUG    = 15         # number of augmented copies of Batch 1
LABEL_SMOOTH = 0.05       # was 0.08

CNN_FILTERS  = [64, 128, 256]
KERNEL_SIZE  = 3
GRU_HIDDEN   = 128
GRU_LAYERS   = 2
DENSE_UNITS  = [256, 128]

PAPER_RESULTS = {
    2: 0.8271, 3: 0.7683, 4: 0.7244, 5: 0.7891,
    6: 0.9318, 7: 0.6182, 8: 0.7152, 9: 0.5664, 10: 0.5663,
}


# ═══════════════════════════════════════════════════════════════════
# 4. HELPERS
# ═══════════════════════════════════════════════════════════════════
def robust_instance_norm(X: np.ndarray, clip_sigma: float = 5.0) -> np.ndarray:
    """
    Instance normalize but clip extreme values.
    Prevents near-zero-variance samples from exploding after division.
    """
    mu  = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    X_norm = (X - mu) / std
    # Clip to ±clip_sigma standard deviations
    X_norm = np.clip(X_norm, -clip_sigma, clip_sigma)
    return X_norm

def append_month(X: np.ndarray, batch_id: int) -> np.ndarray:
    m = BATCH_MONTH_MAP.get(int(batch_id), int(batch_id)) / TOTAL_MONTHS
    return np.hstack([X, np.full((len(X), 1), m, dtype=np.float32)])


def append_month_array(X: np.ndarray, batch_ids: np.ndarray) -> np.ndarray:
    months = np.array(
        [BATCH_MONTH_MAP.get(int(b), int(b)) / TOTAL_MONTHS for b in batch_ids],
        dtype=np.float32).reshape(-1, 1)
    return np.hstack([X, months])


def vae_augment(X: np.ndarray, y: np.ndarray,
                n_aug: int = 15, noise_scale: float = 0.08) -> tuple:
    """
    Paper Section 3, Eq. 7 — VAE-inspired augmentation.
    For each augmentation round:
      1. Randomly pair each sample j with another sample k
      2. Mix their per-sample Gaussian statistics with weight λ
      3. Sample noise from mixed Gaussian and add to original signal

    G_j = f(N(λ*μ_j + (1-λ)*μ_k,  λ*δ_j + (1-λ)*δ_k)) + X_j
    """
    X_list, y_list = [X], [y]
    rng = np.random.RandomState(SEED)

    for aug_i in range(n_aug):
        idx = rng.permutation(len(X))
        lam = rng.uniform(0.3, 0.7)

        # Per-sample statistics (Eq. 5 & 6)
        mu_j  = X.mean(axis=1, keepdims=True)
        std_j = X.std(axis=1,  keepdims=True) + 1e-8
        mu_k  = X[idx].mean(axis=1, keepdims=True)
        std_k = X[idx].std(axis=1,  keepdims=True) + 1e-8

        # Mixed Gaussian parameters (Eq. 7 numerator)
        mu_mix  = lam * mu_j  + (1 - lam) * mu_k
        std_mix = lam * std_j + (1 - lam) * std_k

        # Sample from mixed Gaussian and add to original
        noise   = rng.randn(*X.shape).astype(np.float32)
        X_new   = X + (noise * std_mix + mu_mix * 0.01) * noise_scale
        X_list.append(X_new)
        y_list.append(y)

    return np.vstack(X_list), np.concatenate(y_list)


def mixup_batch(X: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return X, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(X.size(0), device=X.device)
    return lam * X + (1 - lam) * X[idx], y, y[idx], lam


def to_cnn_tensor(X: np.ndarray, y: np.ndarray, dev: torch.device):
    Xt = torch.tensor(X.reshape(X.shape[0], 1, X.shape[1]),
                      dtype=torch.float32).to(dev)
    yt = torch.tensor(y, dtype=torch.long).to(dev)
    return Xt, yt


def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


# ═══════════════════════════════════════════════════════════════════
# 5. LOAD & PREPARE DATA
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("  CDCNN + VAE Aug  |  Train B1 Synthetic  →  Test B2-10 Real")
print("=" * 65)

synth_df     = pd.read_csv(SYNTHETIC_CSV)
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes  = int(synth_df["label"].nunique())
feature_cols = [c for c in synth_df.columns if c not in ("label", "batch")]
N_SENSOR_F   = len(feature_cols)
N_FEATURES   = N_SENSOR_F + 1

print(f"\n  Device          : {device}")
print(f"  Classes         : {num_classes}")
print(f"  Sensor features : {N_SENSOR_F}")
print(f"  Total features  : {N_FEATURES}")

# ── Step 1: Load synthetic Batch 1 ────────────────────────────────
print(f"\n  Loading synthetic Batch 1…")
df_b1  = synth_df[synth_df["batch"] == TRAIN_BATCH]
X_raw  = robust_instance_norm(df_b1[feature_cols].values.astype(np.float32))
X_raw  = append_month(X_raw, TRAIN_BATCH)
y_raw  = (df_b1["label"].values - 1).astype(np.int64)
print(f"  Raw Batch 1 : {len(y_raw)} samples  "
      f"classes={np.bincount(y_raw).tolist()}")

# ── Step 2: VAE augmentation (paper Section 3) ────────────────────
print(f"\n  Applying VAE augmentation (n_aug={VAE_N_AUG})…")
X_aug, y_aug = vae_augment(X_raw, y_raw, n_aug=VAE_N_AUG)
print(f"  After augmentation : {len(X_aug)} samples  "
      f"classes={np.bincount(y_aug).tolist()}")

# ── Step 3: Train/Val split ───────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_aug, y_aug, test_size=0.15, random_state=SEED, stratify=y_aug)
print(f"  Train : {len(X_train)}   Val : {len(X_val)}")

# ── Step 4: StandardScaler fit on train, apply to val & test ──────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_val_s   = scaler.transform(X_val).astype(np.float32)

g = torch.Generator(); g.manual_seed(SEED)
X_tr_t, y_tr_t = to_cnn_tensor(X_train_s, y_train, device)
X_va_t, y_va_t = to_cnn_tensor(X_val_s,   y_val,   device)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(
    TensorDataset(X_va_t, y_va_t),
    batch_size=BATCH_SIZE, shuffle=False)

# ── Step 5: Load real test data (batches 2–10) ────────────────────
print("\n  Loading real sensor data (test batches 2–10)…")
real_df           = pd.read_csv(REAL_CSV)
feature_cols_real = [str(i) for i in range(128)]
real_test_df      = real_df[real_df["batch"].isin(TEST_BATCHES)].copy()

X_real        = real_test_df[feature_cols_real].values.astype(np.float32)
y_real        = (real_test_df["label"].values - 1).astype(np.int64)
b_real        = real_test_df["batch"].values.astype(np.int64)

X_real_normed = robust_instance_norm(X_real)
X_real_normed = append_month_array(X_real_normed, b_real)
X_real_scaled = scaler.transform(X_real_normed).astype(np.float32)

print(f"  Total real : {len(X_real)}  classes={np.bincount(y_real).tolist()}")

assert X_train_s.shape[1] == X_real_scaled.shape[1], \
    f"Feature mismatch: train={X_train_s.shape[1]} test={X_real_scaled.shape[1]}"
print(f"  Feature dim check passed : {X_train_s.shape[1]}")


# ═══════════════════════════════════════════════════════════════════
# 6. MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dropout=0.2):
        super().__init__()
        p = kernel // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=p)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=p)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.pool  = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        self.drop  = nn.Dropout(dropout)
        self.skip  = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1),
                                    nn.BatchNorm1d(out_ch))
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        res = self.skip(x)
        x   = F.relu(self.bn1(self.conv1(x)))
        x   = F.relu(self.bn2(self.conv2(x)))
        if res.shape[-1] != x.shape[-1]:
            res = F.adaptive_avg_pool1d(res, x.shape[-1])
        return self.drop(self.pool(x + res))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x.mean(dim=-1)).unsqueeze(-1)


class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, 1)
    def forward(self, gru_out):
        w = torch.softmax(self.attn(gru_out), dim=1)
        return (w * gru_out).sum(dim=1)


class FeatureGenerationModule(nn.Module):
    """
    Paper Section 4, Eq. 14–16.
    Decomposes features into high/low frequency,
    perturbs low-frequency statistics to simulate unseen drift domains.
    drift_scale=1.2 generates more aggressive domain shifts than 0.35.
    """
    def __init__(self, drift_scale=1.2):
        super().__init__()
        self.scale = drift_scale

    def forward(self, z):
        z_high    = F.adaptive_max_pool1d(z.unsqueeze(1), z.shape[-1]).squeeze(1)
        z_low     = z - z_high
        mu        = z_low.mean(dim=0, keepdim=True)
        std       = z_low.std(dim=0,  keepdim=True) + 1e-8
        mu_new    = mu  + torch.randn_like(mu)  * std * self.scale
        std_new   = (std + torch.randn_like(std) * std * self.scale * 0.5).abs() + 1e-8
        z_low_new = std_new * ((z_low - mu) / std) + mu_new
        return z_high + z_low_new


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)
        f        = F.normalize(features, dim=1)
        sim      = torch.matmul(f, f.T) / self.T
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        pos_mask.fill_diagonal_(0)
        sim      = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim  = torch.exp(sim)
        denom    = (exp_sim * torch.ones_like(exp_sim).fill_diagonal_(0)
                    ).sum(dim=1, keepdim=True) + 1e-8
        log_p    = sim - torch.log(denom)
        n_pos    = pos_mask.sum(dim=1)
        loss     = -(pos_mask * log_p).sum(dim=1) / (n_pos + 1e-8)
        valid    = n_pos > 0
        return loss[valid].mean() if valid.any() \
               else torch.tensor(0.0, device=f.device)


# ═══════════════════════════════════════════════════════════════════
# 7. CDCNN MODEL
# ═══════════════════════════════════════════════════════════════════
class CDCNN(nn.Module):
    """
    Paper Fig. 2 architecture.
    Loss = L_ce + λ_MSE * L_MSE + λ_con * L_con   (Eq. 4)
    """
    def __init__(self, input_len, num_classes=6, cnn_filters=None,
                 kernel_size=3, gru_hidden=128, gru_layers=2,
                 dense_units=None, dropout=0.40):
        super().__init__()
        if cnn_filters is None: cnn_filters = [64, 128, 256]
        if dense_units  is None: dense_units  = [256, 128]

        self.conv_blocks = nn.ModuleList()
        self.ca_blocks   = nn.ModuleList()
        in_ch = 1
        for out_ch in cnn_filters:
            self.conv_blocks.append(
                ConvBlock(in_ch, out_ch, kernel_size, dropout=dropout * 0.5))
            self.ca_blocks.append(ChannelAttention(out_ch))
            in_ch = out_ch

        self.gru = nn.GRU(
            input_size    = cnn_filters[-1],
            hidden_size   = gru_hidden,
            num_layers    = gru_layers,
            batch_first   = True,
            dropout       = dropout if gru_layers > 1 else 0.0,
            bidirectional = True)
        feat_dim        = gru_hidden * 2
        self.layer_norm = nn.LayerNorm(feat_dim)
        self.attention  = TemporalAttention(gru_hidden)
        self.feat_gen   = FeatureGenerationModule(drift_scale=FGM_SCALE)
        self.projector  = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Linear(128, 64))

        layers, prev = [], feat_dim
        for units in dense_units:
            layers += [nn.Linear(prev, units), nn.BatchNorm1d(units),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = units
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

    def _cnn_encode(self, x):
        for cb, ca in zip(self.conv_blocks, self.ca_blocks):
            x = ca(cb(x))
        return x.permute(0, 2, 1)

    def extract_features(self, x):
        gru_out, _ = self.gru(self._cnn_encode(x))
        return self.attention(self.layer_norm(gru_out))

    def forward(self, x, return_all=False):
        z_s     = self.extract_features(x)
        z_s_gen = self.feat_gen(z_s)
        z_f     = self.projector(z_s)
        z_f_gen = self.projector(z_s_gen)
        out     = self.classifier(z_s)
        out_gen = self.classifier(z_s_gen)
        if return_all:
            return out, out_gen, z_f, z_f_gen, z_s, z_s_gen
        return out


# ═══════════════════════════════════════════════════════════════════
# 8. INSTANTIATE
# ═══════════════════════════════════════════════════════════════════
set_seed()
model = CDCNN(
    input_len   = N_FEATURES,
    num_classes = num_classes,
    cnn_filters = CNN_FILTERS,
    kernel_size = KERNEL_SIZE,
    gru_hidden  = GRU_HIDDEN,
    gru_layers  = GRU_LAYERS,
    dense_units = DENSE_UNITS,
    dropout     = DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  CDCNN parameters : {total_params:,}")

class_counts = np.bincount(y_train)
cw = torch.tensor(
    (1.0 / class_counts) / (1.0 / class_counts).sum() * num_classes,
    dtype=torch.float32).to(device)

optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                  optimizer, T_0=80, T_mult=2, eta_min=1e-6)
ce_loss_fn  = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTH)
mse_loss_fn = nn.MSELoss()
con_loss_fn = SupervisedContrastiveLoss(temperature=TEMPERATURE)


# ═══════════════════════════════════════════════════════════════════
# 9. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  TRAINING  |  Synthetic B1 + VAE Aug  |  FGM scale={FGM_SCALE}")
print(f"{'═'*65}")

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_acc, best_state, patience_count = 0.0, None, 0
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    ep_loss, ep_correct, ep_total = 0.0, 0, 0

    for X_b, y_b in train_loader:
        optimizer.zero_grad()

        X_mix, y_a, y_b2, lam = mixup_batch(X_b, y_b, MIXUP_ALPHA)

        out, out_gen, z_f, z_f_gen, z_s, z_s_gen = \
            model(X_mix, return_all=True)

        # Paper Eq. 4
        loss_ce = (lam       * ce_loss_fn(out,     y_a)  +
                   (1 - lam) * ce_loss_fn(out,     y_b2) +
                   lam       * ce_loss_fn(out_gen, y_a)  +
                   (1 - lam) * ce_loss_fn(out_gen, y_b2))

        loss_mse = LAMBDA_MSE * mse_loss_fn(z_s_gen, z_s.detach())

        loss_con = LAMBDA_CON * con_loss_fn(
            torch.cat([z_f, z_f_gen]),
            torch.cat([y_a, y_a]))

        loss = loss_ce + loss_mse + loss_con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            out_clean = model(X_b)
        ep_loss    += loss_ce.item() * X_b.size(0)
        ep_correct += (out_clean.argmax(1) == y_b).sum().item()
        ep_total   += X_b.size(0)

    scheduler.step()
    avg_train_loss = ep_loss    / ep_total
    avg_train_acc  = ep_correct / ep_total

    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            out        = model(X_b)
            v_loss    += ce_loss_fn(out, y_b).item() * X_b.size(0)
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
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train {avg_train_acc:.4f}  Loss {avg_train_loss:.4f} | "
              f"Val {avg_val_acc:.4f}  Loss {avg_val_loss:.4f} | "
              f"Pat {patience_count}/{PATIENCE}")

    if patience_count >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch+1}  "
              f"(best val acc = {best_val_acc:.4f})")
        break

if best_state:
    model.load_state_dict(best_state)
    print(f"  Restored best weights  (val acc = {best_val_acc:.4f})")

print(f"\n  Training time : {time.time() - start_time:.1f}s")


# ═══════════════════════════════════════════════════════════════════
# 10. TESTING — Real Batches 2–10
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print("  TESTING PHASE  —  Real Batches 2–10")
print(f"{'═'*65}")
print(f"\n  {'Batch':<8} | {'Month':>5} | {'Samples':>8} | "
      f"{'Ours':>8} | {'Paper':>8} | {'Gap':>8}")
print(f"  {'-'*60}")

model.eval()
all_true, all_pred = [], []
real_batch_accs    = {}

for b in TEST_BATCHES:
    mask = (b_real == b)
    X_b  = X_real_scaled[mask]
    y_b  = y_real[mask]

    X_t    = torch.tensor(X_b.reshape(X_b.shape[0], 1, X_b.shape[1]),
                          dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=64, shuffle=False)
    preds  = []
    with torch.no_grad():
        for (Xb,) in loader:
            preds.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
    preds = np.array(preds)

    acc   = accuracy_score(y_b, preds)
    paper = PAPER_RESULTS[b]
    gap   = acc - paper
    real_batch_accs[b] = acc
    all_true.extend(y_b.tolist())
    all_pred.extend(preds.tolist())

    month  = BATCH_MONTH_MAP[b]
    status = "✅" if gap >= 0 else f"❌ {gap:+.4f}"
    print(f"  Batch {b:<3} | {month:>5} | {mask.sum():>8} | "
          f"{acc:>8.4f} | {paper:>8.4f} | {status}")

print(f"  {'-'*60}")
overall_acc = accuracy_score(all_true, all_pred)
paper_avg   = np.mean(list(PAPER_RESULTS.values()))
gap_total   = overall_acc - paper_avg

print(f"  {'AVERAGE':<8} |       | {len(all_true):>8} | "
      f"{overall_acc:>8.4f} | {paper_avg:>8.4f} | "
      f"{'✅ +' if gap_total >= 0 else '❌ '}{abs(gap_total):.4f}")

print(f"\n  Paper CDCNN : {paper_avg:.4f}")
print(f"  Ours        : {overall_acc:.4f}  "
      f"({'✅ BEAT IT!' if overall_acc >= paper_avg else f'gap = {paper_avg - overall_acc:.4f}'})")

print("\nClassification Report (Real Batches 2–10):")
print(classification_report(
    all_true, all_pred,
    target_names=[f"Class {i+1}" for i in range(num_classes)]))


# ═══════════════════════════════════════════════════════════════════
# 11. VISUALISATION
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle(
    f"CDCNN + VAE Aug  |  Train: Synthetic B1 ({VAE_N_AUG}x aug)  →  Test: Real B2-10\n"
    f"Ours: {overall_acc:.4f}   Paper: {paper_avg:.4f}   Gap: {gap_total:+.4f}",
    fontsize=13, fontweight="bold", y=1.02)

ax = axes[0]
ax.plot(train_losses, color="royalblue",  lw=1.5, label="Train Loss")
ax.plot(val_losses,   color="darkorange", lw=1.5, label="Val Loss")
ax.set_title("Loss Curve"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(train_accs, color="royalblue",  lw=1.5, label="Train Acc")
ax.plot(val_accs,   color="darkorange", lw=1.5, label="Val Acc")
ax.set_title("Accuracy Curve"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)

ax      = axes[2]
batches = sorted(real_batch_accs.keys())
our_acc = [real_batch_accs[b] for b in batches]
pap_acc = [PAPER_RESULTS[b]   for b in batches]
months_ = [BATCH_MONTH_MAP[b] for b in batches]
x_pos   = np.arange(len(batches))
w       = 0.35

b1 = ax.bar(x_pos - w/2, our_acc, w, label="Ours",
            color="royalblue", edgecolor="black", lw=0.7)
b2 = ax.bar(x_pos + w/2, pap_acc, w, label="Paper",
            color="darkorange", edgecolor="black", lw=0.7, alpha=0.8)

for bar, acc in zip(b1, our_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{acc:.3f}", ha="center", va="bottom",
            fontsize=7, fontweight="bold", color="royalblue")
for bar, acc in zip(b2, pap_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{acc:.3f}", ha="center", va="bottom", fontsize=7, color="darkorange")

ax.axhline(overall_acc, color="royalblue", ls="--", lw=1.5,
           label=f"Ours avg: {overall_acc:.3f}")
ax.axhline(paper_avg,   color="darkorange", ls="--", lw=1.5,
           label=f"Paper avg: {paper_avg:.3f}")

ax.set_xticks(x_pos)
ax.set_xticklabels([f"B{b}\nM{months_[i]}" for i, b in enumerate(batches)], fontsize=8)
ax.set_ylim(0, 1.10); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.set_xlabel("Real Batch"); ax.set_ylabel("Accuracy")
ax.set_title("Per-Batch Accuracy vs Paper")

plt.tight_layout()
plt.savefig("cdcnn_vae_results.png", dpi=150, bbox_inches="tight")
print(f"\n  Plot saved → cdcnn_vae_results.png")
print(f"  Total elapsed : {time.time() - start_time:.1f}s")