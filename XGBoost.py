import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ================================================================
# XGBoost SMALL — DRIFT ROBUST GAS SENSOR CLASSIFICATION
#
# Problem with previous version:
#   - 1000 trees × depth 5 = too complex for 3600 samples
#   - Model memorizes batch 1 → fails on batches 2-10
#
# Fix:
#   - Reduce trees: 1000 → 200
#   - Reduce depth: 5 → 3  (shallow trees generalize better)
#   - Fewer drifted copies: 15 → 8 (less noisy training data)
#   - Stronger regularization
# ================================================================

start_time = time.time()
np.random.seed(42)

# ================================================================
# LOAD DATA
# ================================================================
print("Loading datasets...")
train_source_df = pd.read_csv("final_balanced_dataset.csv")
test_source_df  = pd.read_csv("smell_dataset.csv")

train_df     = train_source_df[train_source_df["batch"] == 1].copy()
feature_cols = [c for c in train_source_df.columns if c not in ["label", "batch"]]
num_features = len(feature_cols)
num_classes  = len(train_source_df["label"].unique())
test_batches = sorted([b for b in test_source_df["batch"].unique() if b != 1])

# Column alignment guard
missing = [c for c in feature_cols if c not in test_source_df.columns]
if missing:
    feature_cols = [c for c in feature_cols if c in test_source_df.columns]
    num_features = len(feature_cols)

device      = "cuda" if __import__("torch").cuda.is_available() else "cpu"
tree_method = "gpu_hist" if device == "cuda" else "hist"

print(f"Device         : {device}")
print(f"Features       : {num_features}")
print(f"Classes        : {num_classes}")
print(f"Train samples  : {len(train_df)}  (batch 1 only)")
print(f"Test batches   : {test_batches}")


# ================================================================
# DRIFT-ROBUST FEATURE ENGINEERING
# Keep it simple — only 2 views (instance norm + rank)
# Adding more views = more overfit risk on small data
# ================================================================
def build_drift_robust_features(X):
    # View 1: instance normalize
    mu     = X.mean(axis=1, keepdims=True)
    std    = X.std(axis=1,  keepdims=True) + 1e-8
    X_in   = (X - mu) / std

    # View 2: rank percentile [0,1] — immune to monotonic drift
    ranks  = np.argsort(np.argsort(X, axis=1), axis=1).astype(np.float32)
    X_rank = ranks / (X.shape[1] - 1)

    return np.concatenate([X_in, X_rank], axis=1)


# ================================================================
# SMALLER DRIFT AUGMENTATION
# Only 8 copies instead of 15 — enough diversity without drowning
# the small dataset in noisy synthetic samples
# ================================================================
def generate_drifted_copies(X, y, n_copies=8):
    all_X = [X]
    all_y = [y]

    for i in range(n_copies):
        progress    = (i + 1) / n_copies

        # Milder drift range than before
        scale_std   = 0.1 + 0.5 * progress   # 0.1 → 0.6
        shift_std   = 0.2 + 0.8 * progress   # 0.2 → 1.0
        feat_drop_p = 0.02 + 0.05 * progress # 2%  → 7%

        # Gain drift
        gain  = 1.0 + np.random.randn(*X.shape) * scale_std
        X_aug = X * gain

        # Offset drift
        indiv = np.random.randn(*X.shape) * shift_std
        corr  = np.random.randn(X.shape[0], 1) * shift_std * 0.5
        X_aug = X_aug + indiv + corr

        # Sensor dropout
        mask  = np.random.binomial(
            1, 1 - feat_drop_p, X.shape
        ).astype(np.float32)
        X_aug = X_aug * mask

        # Light noise
        X_aug = X_aug + np.random.randn(*X.shape) * 0.05

        all_X.append(X_aug)
        all_y.append(y)

    return np.vstack(all_X), np.concatenate(all_y)


# ================================================================
# PREPARE DATA
# ================================================================
print("\nBuilding features...")
X_raw  = train_df[feature_cols].values
y_full = train_df["label"].values - 1

X_full = build_drift_robust_features(X_raw)
print(f"Feature dim              : {X_full.shape[1]}")
print(f"Class dist               : {np.bincount(y_full)}")

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full,
    test_size=0.15,
    random_state=42,
    stratify=y_full
)

print(f"\nGenerating drifted copies...")
X_train_aug, y_train_aug = generate_drifted_copies(
    X_train, y_train, n_copies=8
)
print(f"Train samples after aug  : {len(X_train_aug)}")
print(f"Class dist after aug     : {np.bincount(y_train_aug)}")

scaler        = StandardScaler()
X_train_aug_s = scaler.fit_transform(X_train_aug)
X_val_s       = scaler.transform(X_val)

# Drifted val set for realistic early stopping signal
def make_drifted_val(X, scale=0.6, shift=0.8):
    gain = 1.0 + np.random.randn(*X.shape) * scale
    shft = np.random.randn(*X.shape) * shift
    corr = np.random.randn(X.shape[0], 1) * shift * 0.5
    mask = np.random.binomial(1, 0.93, X.shape).astype(np.float32)
    return (X * gain + shft + corr) * mask

X_val_drifted = make_drifted_val(X_val_s)


# ================================================================
# XGBOOST — SMALL MODEL
#
# Key changes vs previous version:
#   n_estimators  : 1000 → 200   (main overfit fix)
#   max_depth     :    5 →   3   (shallower = more general)
#   learning_rate : 0.05 → 0.1   (faster convergence on small data)
#   min_child_weight: 5  →  10   (each leaf needs more samples)
#   reg_alpha     :  0.1 → 0.5   (stronger L1)
#   reg_lambda    :  1.5 → 2.0   (stronger L2)
# ================================================================
print("\nInitializing XGBoost (small model)...")

model = XGBClassifier(
    n_estimators          = 200,       # ← reduced from 1000
    max_depth             = 3,         # ← reduced from 5
    learning_rate         = 0.1,       # ← increased (small data converges faster)
    subsample             = 0.7,       # 70% samples per tree
    colsample_bytree      = 0.6,       # 60% features per tree
    colsample_bylevel     = 0.6,
    reg_alpha             = 0.5,       # ← stronger L1
    reg_lambda            = 2.0,       # ← stronger L2
    min_child_weight      = 10,        # ← increased (more samples per leaf)
    gamma                 = 0.2,       # ← increased min split gain
    tree_method           = tree_method,
    num_class             = num_classes,
    objective             = "multi:softprob",
    eval_metric           = "mlogloss",
    random_state          = 42,
    n_jobs                = -1,
    verbosity             = 1,
    early_stopping_rounds = 30,        # ← reduced (smaller model converges faster)
)


# ================================================================
# TRAIN
# ================================================================
print("Training...")
print("="*60)

model.fit(
    X_train_aug_s, y_train_aug,
    eval_set = [(X_train_aug_s, y_train_aug),
                (X_val_drifted, y_val)],
    verbose  = 25,
)

print(f"\nBest round     : {model.best_iteration}")
print(f"Best val score : {model.best_score:.4f}")
print(f"Val accuracy   : {accuracy_score(y_val, model.predict(X_val_drifted)):.4f}")


# ================================================================
# TEST
# ================================================================
def predict_batch_robust(model, X_raw):
    X_eng = build_drift_robust_features(X_raw)
    X_s   = scaler.transform(X_eng)
    return model.predict(X_s)


print("\n" + "="*58)
print("TEST: smell_dataset.csv  (real sensor data, batches 2–10)")
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

    preds  = predict_batch_robust(model, X_test)
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


# ================================================================
# PLOTS
# ================================================================
importance    = model.feature_importances_
half          = X_full.shape[1] // 2
inst_norm_imp = importance[:half]
rank_imp      = importance[half:]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Per-batch accuracy
colors = ["green" if a >= 0.72 else "steelblue" if a >= 0.55 else "tomato"
          for a in accs]
axes[0].bar(
    [f"Batch {b}" for b in test_batches[:len(accs)]],
    accs, color=colors, edgecolor="black", width=0.6
)
axes[0].axhline(y=avg,   color="blue", linestyle="--", lw=2,
                label=f"Our avg: {avg:.3f}")
axes[0].axhline(y=0.723, color="red",  linestyle="--", lw=2,
                label="Paper: 0.723")
axes[0].set_ylim(0, 1.0)
axes[0].set_title("Per-Batch Test Accuracy\n(smell_dataset.csv)")
axes[0].legend(); axes[0].grid(True, axis="y")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

# Top 20 instance-norm importances
top20_idx = np.argsort(inst_norm_imp)[-20:]
axes[1].barh(range(20), inst_norm_imp[top20_idx],
             color="royalblue", edgecolor="black")
axes[1].set_yticks(range(20))
axes[1].set_yticklabels([f"Sensor {i}" for i in top20_idx])
axes[1].set_title("Top 20 Feature Importances\n(instance-norm view)")
axes[1].grid(True, axis="x")

# Top 20 rank importances
top20_rank = np.argsort(rank_imp)[-20:]
axes[2].barh(range(20), rank_imp[top20_rank],
             color="darkorange", edgecolor="black")
axes[2].set_yticks(range(20))
axes[2].set_yticklabels([f"Sensor {i}" for i in top20_rank])
axes[2].set_title("Top 20 Feature Importances\n(rank view)")
axes[2].grid(True, axis="x")

plt.suptitle(
    "XGBoost (small) | Train: final_balanced_dataset.csv (batch 1)  |  "
    "Test: smell_dataset.csv (batches 2–10)",
    fontsize=10, y=1.02
)
plt.tight_layout()
plt.savefig("final_results_xgboost.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: final_results_xgboost.png")