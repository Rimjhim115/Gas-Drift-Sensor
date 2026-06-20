import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("Loading dataset...")
df = pd.read_csv("final_balanced_dataset.csv")

train_df = df[df["batch"] == 1]

X_train = train_df.drop(["label","batch"], axis=1).values
y_train = train_df["label"].values - 1   


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

class SimpleCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64 * input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = x.unsqueeze(1)   # (batch, 1, features)
        return self.net(x)


input_dim = X_train.shape[1]

model = SimpleCNN(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training model...")

epochs = 200

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")
print("\n=== RESULTS (Batch 1 → Others) ===\n")

batches = sorted(df["batch"].unique())

for batch in batches:

    if batch == 1:
        continue

    test_df = df[df["batch"] == batch]

    X_test = test_df.drop(["label","batch"], axis=1).values
    y_test = test_df["label"].values - 1

    X_test = scaler.transform(X_test)

    X_test = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)

    acc = accuracy_score(y_test, preds.numpy())

    print(f"Batch 1 → Batch {batch} Accuracy: {acc:.4f}")