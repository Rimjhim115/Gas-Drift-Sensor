import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset



start_time = time.time()
torch.manual_seed(42)
np.random.seed(42)


BATCH_SIZE      = 128
EPOCHS          = 400
LR              = 2e-4
PATIENCE        = 60
HIDDEN          = 192          
DROPOUT         = 0.55        
LABEL_SMOOTH    = 0.15        
WEIGHT_DECAY    = 5e-3        

DRIFT_SCALE_MIN = 0.2         
DRIFT_SCALE_MAX = 1.0          
DRIFT_SHIFT_MIN = 0.3
DRIFT_SHIFT_MAX = 1.5          
FEAT_DROPOUT_P  = 0.10        

print("Loading datasets...")
train_source_df = pd.read_csv("final_balanced_dataset.csv")
test_source_df  = pd.read_csv("smell_dataset.csv")

train_df     = train_source_df[train_source_df["batch"] == 1].copy()
feature_cols = [c for c in train_source_df.columns if c not in ["label", "batch"]]
num_features = len(feature_cols)
num_classes  = len(train_source_df["label"].unique())
test_batches = sorted([b for b in test_source_df["batch"].unique() if b != 1])

missing = [c for c in feature_cols if c not in test_source_df.columns]
if missing:
    feature_cols = [c for c in feature_cols if c in test_source_df.columns]
    num_features = len(feature_cols)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device         : {device}")
print(f"Features       : {num_features}")
print(f"Classes        : {num_classes}")
print(f"Train samples  : {len(train_df)}  (batch 1, synthetic)")
print(f"Test batches   : {test_batches}  (smell_dataset.csv)")


def build_drift_robust_features(X):
    mu     = X.mean(axis=1, keepdims=True)
    std    = X.std(axis=1,  keepdims=True) + 1e-8
    X_in   = (X - mu) / std

    ranks  = np.argsort(np.argsort(X, axis=1), axis=1).astype(np.float32)
    X_rank = ranks / (X.shape[1] - 1)

    return np.concatenate([X_in, X_rank], axis=1)

def apply_progressive_drift(X_batch, epoch, total_epochs):
    B, F   = X_batch.shape
    device = X_batch.device

    progress = min(epoch / (total_epochs * 0.7), 1.0)

    scale_std = DRIFT_SCALE_MIN + (DRIFT_SCALE_MAX - DRIFT_SCALE_MIN) * progress
    shift_std = DRIFT_SHIFT_MIN + (DRIFT_SHIFT_MAX - DRIFT_SHIFT_MIN) * progress

    gain  = 1.0 + torch.randn(B, F, device=device) * scale_std
    X_aug = X_batch * gain

    indiv_shift  = torch.randn(B, F, device=device) * shift_std
    corr_shift   = torch.randn(B, 1, device=device).expand(B, F) * shift_std * 0.5
    X_aug = X_aug + indiv_shift + corr_shift

    mask  = torch.bernoulli(
        torch.full((B, F), 1 - FEAT_DROPOUT_P, device=device)
    )
    X_aug = X_aug * mask


    X_aug = X_aug + torch.randn_like(X_aug) * 0.1

    return X_aug


class DriftRobustMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),

            # Block 2
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),

            # Block 3 — bottleneck
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.classifier = nn.Linear(hidden_size // 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.net(x))


def mixup(X, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(X.size(0))
    return lam * X + (1 - lam) * X[idx], y, y[idx], lam

def mixup_loss(criterion, out, ya, yb, lam):
    return lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)


X_raw  = train_df[feature_cols].values
y_full = train_df["label"].values - 1

print(f"\nBuilding features...")
X_full = build_drift_robust_features(X_raw)
print(f"Feature dim : {X_full.shape[1]}")
print(f"Class dist  : {np.bincount(y_full)}")

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full,
    test_size=0.15,
    random_state=42,
    stratify=y_full
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

def make_drifted_val(X_val_t, drift_scale=0.8, drift_shift=1.2):
    B, F   = X_val_t.shape
    gain   = 1.0 + torch.randn(B, F) * drift_scale
    shift  = torch.randn(B, F) * drift_shift
    corr   = torch.randn(B, 1).expand(B, F) * drift_shift * 0.5
    mask   = torch.bernoulli(torch.full((B, F), 0.92))
    return (X_val_t * gain + shift + corr) * mask

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train,   dtype=torch.long)
X_val_t   = torch.tensor(X_val_s,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,     dtype=torch.long)

# Drifted val tensors (pre-computed, fixed for consistency)
X_val_drifted = make_drifted_val(X_val_t)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = DataLoader(
    TensorDataset(X_val_drifted, y_val_t),
    batch_size=BATCH_SIZE, shuffle=False
)


input_dim = X_full.shape[1]

model = DriftRobustMLP(
    input_size=input_dim,
    hidden_size=HIDDEN,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model params: {total_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR * 8,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.05,
    anneal_strategy='cos',
    final_div_factor=500
)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_acc, best_state = 0.0, None
patience_count = 0

print(f"\nTraining with progressive drift augmentation...")
print("="*72)

for epoch in range(EPOCHS):
    model.train()
    ep_loss, ep_correct, ep_total = 0.0, 0, 0

    for X_b, y_b in train_loader:
        X_b = X_b.to(device)
        y_b = y_b.to(device)

        # Progressive drift: magnitude increases with epoch
        X_aug = apply_progressive_drift(X_b, epoch, EPOCHS)

        use_mixup = np.random.random() < 0.5
        if use_mixup:
            X_mix, ya, yb2, lam = mixup(X_aug, y_b, alpha=0.4)
            optimizer.zero_grad()
            out  = model(X_mix)
            loss = mixup_loss(criterion, out, ya, yb2, lam)
        else:
            optimizer.zero_grad()
            out  = model(X_aug)
            loss = criterion(out, y_b)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        ep_loss += loss.item() * X_b.size(0)
        with torch.no_grad():
            # Measure train acc on moderately-drifted samples
            # (not clean, not extreme — middle ground)
            X_mid = apply_progressive_drift(X_b, epoch // 2, EPOCHS)
            ep_correct += (model(X_mid).argmax(1) == y_b).sum().item()
        ep_total += X_b.size(0)

    avg_train_loss = ep_loss    / ep_total
    avg_train_acc  = ep_correct / ep_total

    # Validate on drifted val set
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out       = model(X_b)
            v_loss   += criterion(out, y_b).item() * X_b.size(0)
            v_correct+= (out.argmax(1) == y_b).sum().item()
            v_total  += X_b.size(0)

    avg_val_loss = v_loss    / v_total
    avg_val_acc  = v_correct / v_total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(avg_train_acc)
    val_accs.append(avg_val_acc)

    if avg_val_acc > best_val_acc + 0.0005:
        best_val_acc   = avg_val_acc
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1

    if (epoch + 1) % 25 == 0:
        progress = min(epoch / (EPOCHS * 0.7), 1.0)
        cur_scale = DRIFT_SCALE_MIN + (DRIFT_SCALE_MAX - DRIFT_SCALE_MIN) * progress
        print(f"Ep {epoch+1:3d}/{EPOCHS} | "
              f"Train Acc:{avg_train_acc:.3f} Loss:{avg_train_loss:.3f} | "
              f"Val Acc:{avg_val_acc:.3f} Loss:{avg_val_loss:.3f} | "
              f"Drift σ:{cur_scale:.2f} | Pat:{patience_count}/{PATIENCE}")

    if patience_count >= PATIENCE:
        print(f"\nEarly stop at epoch {epoch+1} (best val: {best_val_acc:.4f})")
        break

if best_state:
    model.load_state_dict(best_state)
    print(f"Restored best weights (val acc: {best_val_acc:.4f})")

def predict_batch_robust(model, X_raw, device, n_tta=15, noise=0.03):
    X_eng = build_drift_robust_features(X_raw)
    X_s   = scaler.transform(X_eng)
    X_t   = torch.tensor(X_s, dtype=torch.float32)

    model.eval()
    all_probs = []
    with torch.no_grad():
        # Clean pass
        out = model(X_t.to(device))
        all_probs.append(F.softmax(out, dim=1).cpu())

        # TTA: more passes with varied noise levels
        for i in range(n_tta - 1):
            noise_level = noise * (1 + i * 0.1)   # escalating noise
            noisy = X_t + torch.randn_like(X_t) * noise_level
            out   = model(noisy.to(device))
            all_probs.append(F.softmax(out, dim=1).cpu())

    avg_probs = torch.stack(all_probs).mean(0)
    return avg_probs.argmax(1).numpy()


print("\n" + "="*58)
print(f"TEST: smell_dataset.csv  (real sensor data, batches 2–10)")
print("="*58)
print(f"{'Batch':<10} | {'Accuracy':>10} | {'Samples':>8}")
print("-"*38)

accs     = []
all_true = []
all_pred = []

for b in test_batches:
    test_df = test_source_df[test_source_df["batch"] == b].copy()
    if len(test_df) == 0:
        print(f"Batch {b:<5}  | {'N/A':>10} | {'0':>8}  ⚠️")
        continue

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values - 1

    preds  = predict_batch_robust(model, X_test, device)
    acc    = accuracy_score(y_test, preds)
    accs.append(acc)
    all_true.extend(y_test)
    all_pred.extend(preds)

    marker = "✅" if acc >= 0.72 else "🟡" if acc >= 0.55 else "❌"
    print(f"Batch {b:<5}  | {acc:>10.4f} | {len(y_test):>8}  {marker}")

print("-"*38)
avg = np.mean(accs)
print(f"{'AVERAGE':<10} | {avg:>10.4f} |")
print("="*58)
print(f"\nPaper CDCNN  : 0.7230")
print(f"Our result   : {avg:.4f}  ({'✅ BEAT IT!' if avg >= 0.72 else f'gap: {0.723 - avg:.3f}'})")

print("\nClassification Report:")
print(classification_report(
    all_true, all_pred,
    target_names=[f"Gas{i+1}" for i in range(num_classes)]
))
print(f"\nTotal time: {time.time() - start_time:.1f}s")



fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(train_losses, color="royalblue",  label="Train Loss")
axes[0].plot(val_losses,   color="darkorange", label="Val Loss (drifted)")
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(train_accs, color="royalblue",  label="Train Acc")
axes[1].plot(val_accs,   color="darkorange", label="Val Acc (drifted)")
axes[1].set_title("Accuracy Curve")
axes[1].set_xlabel("Epoch")
axes[1].legend(); axes[1].grid(True)

colors = ["green" if a >= 0.72 else "steelblue" if a >= 0.55 else "tomato"
          for a in accs]
axes[2].bar([f"Batch {b}" for b in test_batches[:len(accs)]],
            accs, color=colors, edgecolor="black", width=0.6)
axes[2].axhline(y=avg,   color="blue", linestyle="--", lw=2,
                label=f"Our avg: {avg:.3f}")
axes[2].axhline(y=0.723, color="red",  linestyle="--", lw=2,
                label="Paper: 0.723")
axes[2].set_ylim(0, 1.0)
axes[2].set_title("Per-Batch Test Accuracy\n(smell_dataset.csv)")
axes[2].legend(); axes[2].grid(True, axis="y")
plt.xticks(rotation=45)

plt.suptitle(
    "Train: final_balanced_dataset.csv (batch 1)  |  "
    "Test: smell_dataset.csv (batches 2–10)",
    fontsize=10, y=1.02
)
plt.tight_layout()
plt.savefig("final_results_v6.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: final_results_v6.png")






