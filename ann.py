import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

start_time = time.time()

print("Loading dataset...")
df = pd.read_csv("final_balanced_dataset.csv")

batches = sorted(df["batch"].unique())
train_batch = batches[0]
test_batches = batches[1:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
class DeepANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

train_df = df[df["batch"] == train_batch]

X_full = train_df.drop(["label", "batch"], axis=1).values
y_full = train_df["label"].values - 1

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

pca = PCA(n_components=50)
X_train_p = pca.fit_transform(X_train_s)
X_val_p = pca.transform(X_val_s)

X_train_t = torch.tensor(X_train_p, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

X_val_t = torch.tensor(X_val_p, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)


model = DeepANN(
    input_size=X_train_p.shape[1],
    num_classes=len(np.unique(y_full))
).to(device)

print("\nMODEL SUMMARY")
print(model)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    betas=(0.85, 0.995)
)

# 🔥 Scheduler (important)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

loss_fn = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accs, val_accs = [], []

# -------------------------------
# TRAINING
# -------------------------------
print(f"\nTraining on Batch {train_batch}...")

for epoch in range(200):

    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_t)
    loss = loss_fn(outputs, y_train_t)

    loss.backward()
    optimizer.step()

    preds = torch.argmax(outputs, axis=1)
    train_acc = (preds == y_train_t).float().mean().item()

    # VALIDATION
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = loss_fn(val_outputs, y_val_t)

        val_preds = torch.argmax(val_outputs, axis=1)
        val_acc = (val_preds == y_val_t).float().mean().item()

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

# -------------------------------
# TESTING
# -------------------------------
print("\nTask     | Accuracy")
print("-" * 25)

accs = []

for b in test_batches:
    test_df = df[df["batch"] == b]

    X_test = test_df.drop(["label", "batch"], axis=1).values
    y_test = test_df["label"].values - 1

    X_test_s = scaler.transform(X_test)
    X_test_p = pca.transform(X_test_s)

    X_test_t = torch.tensor(X_test_p, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test_t), axis=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    accs.append(acc)

    print(f"{train_batch}-{b:<6} | {acc:.4f}")

print("-" * 25)
print(f"Average | {np.mean(accs):.4f}")

# -------------------------------
# TIME
# -------------------------------
end_time = time.time()
print(f"\nTraining Time: {end_time - start_time:.2f} seconds")

# -------------------------------
# GRAPH
# -------------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy")

plt.tight_layout()
plt.show()