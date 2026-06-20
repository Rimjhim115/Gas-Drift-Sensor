import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# CONFIG
# ============================================================
EPOCHS = 120
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# LOAD CLEAN DATA (IMPORTANT)
# ============================================================
df = pd.read_csv("smell_dataset.csv")   # ✅ ONLY REAL DATA

feature_cols = [c for c in df.columns if c not in ["label", "batch"]]
num_classes = len(df["label"].unique())

# Train only on batch 1
train_df = df[df["batch"] == 1]
test_batches = sorted(df["batch"].unique())[1:]

# ============================================================
# INSTANCE NORMALIZATION (CRITICAL)
# ============================================================
def instance_norm(X):
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

# ============================================================
# DRIFT AUGMENTATION
# ============================================================
def augment(X):
    noise = torch.randn_like(X) * 0.05
    scale = 1 + torch.randn_like(X) * 0.2
    shift = torch.randn_like(X) * 0.5
    return X * scale + shift + noise

# ============================================================
# MODEL
# ============================================================
class CleanModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 64)
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        emb = self.encoder(x)
        out = self.classifier(emb)
        return out, emb

# ============================================================
# DATA PREP
# ============================================================
X_train = train_df[feature_cols].values
y_train = train_df["label"].values - 1

# Normalize per sample
X_train = instance_norm(X_train)

# Scale globally
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

# ============================================================
# MODEL INIT
# ============================================================
model = CleanModel(X_train.shape[1], num_classes).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ============================================================
# TRAINING
# ============================================================
print("\nTraining on REAL data only...")
print("="*50)

for epoch in range(EPOCHS):
    model.train()

    X_aug = augment(X_train_t)

    optimizer.zero_grad()

    outputs, emb = model(X_aug)
    loss = criterion(outputs, y_train_t)

    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        preds = outputs.argmax(1)
        acc = (preds == y_train_t).float().mean().item()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")

# ============================================================
# PROTOTYPE COMPUTATION
# ============================================================
def compute_prototypes(model, X, y):
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        emb = model.encoder(X_t).cpu().numpy()

    prototypes = []
    for c in range(num_classes):
        proto = emb[y == c].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes.append(proto)

    return np.array(prototypes)

prototypes = compute_prototypes(model, X_train, y_train)

# ============================================================
# PREDICT
# ============================================================
def predict(X_raw):
    X = instance_norm(X_raw)
    X = scaler.transform(X)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        emb = model.encoder(X_t).cpu().numpy()

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(emb, prototypes.T)

    return np.argmax(sims, axis=1)

# ============================================================
# EVALUATION
# ============================================================
print("\n" + "="*50)
print(f"{'Task':<10} | {'Accuracy':>10}")
print("-"*30)

accs = []
all_true = []
all_pred = []

for b in test_batches:
    test_df = df[df["batch"] == b]

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values - 1

    preds = predict(X_test)

    acc = accuracy_score(y_test, preds)
    accs.append(acc)

    all_true.extend(y_test)
    all_pred.extend(preds)

    print(f"1-{b:<7} | {acc:.4f}")

print("-"*30)
print(f"Average  | {np.mean(accs):.4f}")

print("\nClassification Report:")
print(classification_report(all_true, all_pred))