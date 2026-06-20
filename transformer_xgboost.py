import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

print("Loading dataset...")
df = pd.read_csv("final_balanced_dataset.csv")

train_df = df[df["batch"] == 1]

X_train = train_df.drop(["label","batch"], axis=1).values
y_train = train_df["label"].values - 1  

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.embedding = nn.Linear(1, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)


print("\nTraining Transformer...")

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

input_dim = X_train_tensor.shape[1]

transformer_model = TransformerModel(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)

epochs = 30

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = transformer_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")


print("\nTraining XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train_scaled, y_train)


print("\n=== RESULTS (Batch 1 → Others) ===\n")

batches = sorted(df["batch"].unique())

for batch in batches:
    if batch == 1:
        continue

    test_df = df[df["batch"] == batch]

    X_test = test_df.drop(["label","batch"], axis=1).values
    y_test = test_df["label"].values - 1

    X_test_scaled = StandardScaler().fit_transform(X_test)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    with torch.no_grad():
        outputs = transformer_model(X_test_tensor)
        _, preds_trans = torch.max(outputs, 1)


    acc_trans = accuracy_score(y_test, preds_trans.numpy())

    preds_xgb = xgb_model.predict(X_test_scaled)
    acc_xgb = accuracy_score(y_test, preds_xgb)

  
    print(f"Batch 1 → Batch {batch}")
    print(f"   Transformer Accuracy: {acc_trans:.4f}")
    print(f"   XGBoost Accuracy:     {acc_xgb:.4f}\n")
# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score

# from xgboost import XGBClassifier

# print("Loading dataset...")
# df = pd.read_csv("final_balanced_dataset.csv")

# train_df = df[df["batch"] == 1]

# X_train = train_df.drop(["label","batch"], axis=1).values
# y_train = train_df["label"].values - 1


# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

# pca = PCA(n_components=50)
# X_train_scaled = pca.fit_transform(X_train_scaled)


# class TransformerModel(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()

#         self.embedding = nn.Linear(1, 64)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=64,
#             nhead=8,
#             dim_feedforward=128,
#             batch_first=True,
#             dropout=0.2
#         )

#         self.transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=3
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * input_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 6)
#         )

#     def forward(self, x):
#         x = x.unsqueeze(-1)
#         x = self.embedding(x)
#         x = self.transformer(x)
#         return self.fc(x)


# print("\nTraining Transformer...")

# X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# input_dim = X_train_tensor.shape[1]

# transformer_model = TransformerModel(input_dim)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.0005)

# epochs = 60

# for epoch in range(epochs):
#     transformer_model.train()

#     optimizer.zero_grad()

#     outputs = transformer_model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)

#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 1.0)
#     optimizer.step()

#     if (epoch+1) % 10 == 0:
#         print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")


# print("\nTraining XGBoost...")

# xgb_model = XGBClassifier(
#     n_estimators=600,
#     max_depth=10,
#     learning_rate=0.03,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     reg_lambda=1.5,
#     reg_alpha=0.5,
#     eval_metric='mlogloss'
# )

# xgb_model.fit(X_train_scaled, y_train)



# print("\n=== FINAL RESULTS (Batch 1 → Others) ===\n")

# batches = sorted(df["batch"].unique())

# results = []

# for batch in batches:
#     if batch == 1:
#         continue

#     test_df = df[df["batch"] == batch]

#     X_test = test_df.drop(["label","batch"], axis=1).values
#     y_test = test_df["label"].values - 1

#     X_test_scaled = scaler.transform(X_test)
#     X_test_scaled = pca.transform(X_test_scaled)

#     X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

#     transformer_model.eval()
#     with torch.no_grad():
#         outputs = transformer_model(X_test_tensor)
#         _, preds_trans = torch.max(outputs, 1)

#     acc_trans = accuracy_score(y_test, preds_trans.numpy())

#     preds_xgb = xgb_model.predict(X_test_scaled)
#     acc_xgb = accuracy_score(y_test, preds_xgb)

#     print(f"Batch 1 → Batch {batch}")
#     print(f"   Transformer Accuracy: {acc_trans:.4f}")
#     print(f"   XGBoost Accuracy:     {acc_xgb:.4f}\n")

#     results.append((batch, acc_trans, acc_xgb))


# avg_trans = np.mean([r[1] for r in results])
# avg_xgb = np.mean([r[2] for r in results])

# print("=== AVERAGE ACCURACY ===")
# print(f"Transformer Avg: {avg_trans:.4f}")
# print(f"XGBoost Avg:     {avg_xgb:.4f}")