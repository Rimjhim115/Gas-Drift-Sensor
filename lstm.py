# # import pandas as pd
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.metrics import accuracy_score

# # print("Loading dataset...")
# # df = pd.read_csv("final_balanced_dataset.csv")

# # batches = sorted(df["batch"].unique())

# # train_batch = batches[0]
# # test_batches = batches[1:]

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print("Using device:", device)

# # class LSTMClassifier(nn.Module):
# #     def __init__(self, input_size=1, hidden_size=128, num_layers=3, num_classes=6):
# #         super().__init__()

# #         self.lstm = nn.LSTM(
# #             input_size=input_size,
# #             hidden_size=hidden_size,
# #             num_layers=num_layers,
# #             batch_first=True,
# #             dropout=0.3,
# #             bidirectional=True   # 🔥 important
# #         )

# #         self.fc = nn.Sequential(
# #             nn.Linear(hidden_size * 2, 128),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(128, num_classes)
# #         )

# #     def forward(self, x):
# #         out, _ = self.lstm(x)
# #         out = out[:, -1, :]
# #         out = self.fc(out)
# #         return out


# # train_df = df[df["batch"] == train_batch]

# # X_train = train_df.drop(["label", "batch"], axis=1).values
# # y_train = train_df["label"].values - 1

# # scaler = StandardScaler()
# # X_train_s = scaler.fit_transform(X_train)

# # X_train_lstm = X_train_s.reshape(X_train_s.shape[0], X_train_s.shape[1], 1)

# # X_train_t = torch.tensor(X_train_lstm, dtype=torch.float32).to(device)
# # y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

# # model = LSTMClassifier(
# #     input_size=1,
# #     hidden_size=128,
# #     num_layers=3,
# #     num_classes=len(np.unique(y_train))
# # ).to(device)

# # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# # loss_fn = nn.CrossEntropyLoss()

# # print(f"\nTraining on Batch {train_batch}...")

# # model.train()
# # for epoch in range(40):   
# #     optimizer.zero_grad()

# #     outputs = model(X_train_t)
# #     loss = loss_fn(outputs, y_train_t)

# #     loss.backward()

# #     # 🔥 prevents exploding gradients
# #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# #     optimizer.step()

# #     if (epoch+1) % 5 == 0:
# #         print(f"Epoch {epoch+1}/40, Loss: {loss.item():.4f}")

# # # -------------------------------
# # # TESTING
# # # -------------------------------
# # print("\nTask     | Accuracy")
# # print("-" * 25)

# # accs = []

# # for b in test_batches:
# #     test_df = df[df["batch"] == b]

# #     X_test = test_df.drop(["label", "batch"], axis=1).values
# #     y_test = test_df["label"].values - 1

# #     X_test_s = scaler.transform(X_test)
# #     X_test_lstm = X_test_s.reshape(X_test_s.shape[0], X_test_s.shape[1], 1)

# #     X_test_t = torch.tensor(X_test_lstm, dtype=torch.float32).to(device)

# #     model.eval()
# #     with torch.no_grad():
# #         preds = torch.argmax(model(X_test_t), axis=1).cpu().numpy()

# #     acc = accuracy_score(y_test, preds)
# #     accs.append(acc)

# #     print(f"{train_batch}-{b:<6} | {acc:.4f}")

# # print("-" * 25)
# # print(f"Average | {np.mean(accs):.4f}")
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import time
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset


# start_time = time.time()

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

# print("Loading datasets...")

# # Training data — your synthetic balanced dataset (batch 1 only)
# balanced_df  = pd.read_csv("final_balanced_dataset.csv")
# train_batch  = sorted(balanced_df["batch"].unique())[0]
# train_df     = balanced_df[balanced_df["batch"] == train_batch]

# original_df  = pd.read_csv("smell_dataset.csv")
# test_batches = sorted(original_df["batch"].unique())

# test_batches = [b for b in test_batches if b != train_batch]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device     : {device}")
# print(f"Train batch      : Batch {train_batch} from final_balanced_dataset.csv")
# print(f"Train samples    : {len(train_df)}")
# print(f"Test batches     : {test_batches} from original dataset")
# print(f"Test samples     : {len(original_df[original_df['batch'].isin(test_batches)])}")

# num_classes  = len(balanced_df["label"].unique())
# print(f"Number of classes: {num_classes}")

# # Feature columns — use same features from both files
# feature_cols = [c for c in balanced_df.columns if c not in ["label", "batch"]]
# print(f"Number of features: {len(feature_cols)}")

# # Verify original dataset has same feature columns
# missing = [c for c in feature_cols if c not in original_df.columns]
# if missing:
#     print(f"WARNING: These features missing in original dataset: {missing}")
#     print("Using common features only...")
#     feature_cols = [c for c in feature_cols if c in original_df.columns]
#     print(f"Using {len(feature_cols)} common features")


# def vae_augment(X, y, augment_factor=5):
#     X_aug_list = [X]
#     y_aug_list = [y]
#     for _ in range(augment_factor):
#         X_new = np.zeros_like(X)
#         for i in range(len(X)):
#             same_class_idx = np.where(y == y[i])[0]
#             k       = np.random.choice(same_class_idx)
#             lam     = np.random.uniform(0.3, 0.7)
#             mu_j    = X[i].mean();  std_j = X[i].std()  + 1e-8
#             mu_k    = X[k].mean();  std_k = X[k].std()  + 1e-8
#             mu_mix  = lam * mu_j  + (1 - lam) * mu_k
#             std_mix = lam * std_j + (1 - lam) * std_k
#             noise   = np.random.normal(mu_mix, std_mix, size=X[i].shape)
#             X_new[i]= X[i] + noise * 0.1
#         X_aug_list.append(X_new)
#         y_aug_list.append(y)
#     return np.vstack(X_aug_list), np.hstack(y_aug_list)



# class AttentionLayer(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.attn = nn.Linear(hidden_size * 2, 1)

#     def forward(self, lstm_out):
#         weights = torch.softmax(self.attn(lstm_out), dim=1)
#         return (weights * lstm_out).sum(dim=1)



# class FeatureGenerationModule(nn.Module):
#     def __init__(self, feature_dim):
#         super().__init__()
#         self.feature_dim = feature_dim

#     def forward(self, z):
#         z_high       = F.adaptive_max_pool1d(z.unsqueeze(1), z.shape[-1]).squeeze(1)
#         z_low        = z - z_high
#         mu_low       = z_low.mean(dim=0, keepdim=True)
#         std_low      = z_low.std(dim=0,  keepdim=True) + 1e-8
#         mu_new       = mu_low  + torch.randn_like(mu_low)  * std_low * 0.1
#         std_new      = (std_low + torch.randn_like(std_low) * std_low * 0.05).abs() + 1e-8
#         z_low_norm   = (z_low - mu_low) / std_low
#         z_low_new    = std_new * z_low_norm + mu_new
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
#         sim_max,_ = sim.max(dim=1, keepdim=True)
#         sim       = sim - sim_max.detach()
#         exp_sim   = torch.exp(sim)
#         self_mask = torch.ones(B, B, device=features.device).fill_diagonal_(0)
#         denom     = (exp_sim * self_mask).sum(dim=1, keepdim=True) + 1e-8
#         log_prob  = sim - torch.log(denom)
#         n_pos     = pos_mask.sum(dim=1)
#         loss      = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
#         return loss[n_pos > 0].mean() if (n_pos > 0).any() else torch.tensor(0.0, device=features.device)



# class CDCNNStyleLSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_size=128, num_layers=3,
#                  num_classes=6, dropout=0.3):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                             num_layers=num_layers, batch_first=True,
#                             dropout=dropout, bidirectional=True)
#         feat_dim        = hidden_size * 2
#         self.layer_norm = nn.LayerNorm(feat_dim)
#         self.attention  = AttentionLayer(hidden_size)
#         self.feat_gen   = FeatureGenerationModule(feat_dim)
#         self.projector  = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(),
#                                         nn.Linear(128, 64))
#         self.classifier = nn.Sequential(
#             nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(128, num_classes)
#         )

#     def extract_features(self, x):
#         out, _ = self.lstm(x)
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



# X_full = train_df[feature_cols].values
# y_full = train_df["label"].values - 1


# def instance_norm(X):
#     mu  = X.mean(axis=1, keepdims=True)
#     std = X.std(axis=1,  keepdims=True) + 1e-8
#     return (X - mu) / std

# X_full = instance_norm(X_full)

# print(f"\nOriginal training samples : {len(X_full)}")
# print(f"Class distribution        : {np.bincount(y_full)}")

# print("\nApplying VAE-inspired augmentation...")
# X_aug, y_aug = vae_augment(X_full, y_full, augment_factor=5)
# print(f"Augmented training samples: {len(X_aug)}")

# X_train, X_val, y_train, y_val = train_test_split(
#     X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug
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

# train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
#                           batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
# val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),
#                           batch_size=BATCH_SIZE, shuffle=False)

# class_counts = np.bincount(y_train)
# cw = torch.tensor(
#     (1.0/class_counts) / (1.0/class_counts).sum() * num_classes,
#     dtype=torch.float32
# ).to(device)


# model = CDCNNStyleLSTM(input_size=1, hidden_size=HIDDEN_SIZE,
#                        num_layers=NUM_LAYERS, num_classes=num_classes,
#                        dropout=DROPOUT).to(device)

# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"\nModel parameters: {total_params:,}")

# optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='max', factor=0.5, patience=10  # reduce when val acc stops improving
# )
# ce_loss     = nn.CrossEntropyLoss(weight=cw)
# mse_loss_fn = nn.MSELoss()
# con_loss_fn = SupervisedContrastiveLoss(temperature=TEMPERATURE)


# train_losses, val_losses = [], []
# train_accs,   val_accs   = [], []
# best_val_acc, best_state = 0.0, None
# patience_count = 0

# print(f"\nTraining on: final_balanced_dataset.csv — Batch {train_batch}")
# print(f"Testing on : original_dataset.csv       — Batches {test_batches}\n")

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
#             v_loss  += ce_loss(out, y_b).item() * X_b.size(0)
#             v_correct+= (out.argmax(1) == y_b).sum().item()
#             v_total  += X_b.size(0)

#     avg_val_loss = v_loss    / v_total
#     avg_val_acc  = v_correct / v_total

#     scheduler.step(avg_val_acc)  # ReduceLROnPlateau needs the metric

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

#     if (epoch+1) % 20 == 0:
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

# print("\n" + "="*50)
# print("RESULTS — Original UCI Drift Dataset (Batches 2-10)")
# print("="*50)
# print(f"{'Task':<10} | {'Accuracy':>10} | {'Samples':>8}")
# print("-"*35)

# accs     = []
# all_true = []
# all_pred = []

# for b in test_batches:
#     test_df = original_df[original_df["batch"] == b]
#     X_test  = test_df[feature_cols].values
#     y_test  = test_df["label"].values - 1

#     X_test  = instance_norm(X_test)

#     X_test_s = scaler.transform(X_test)
#     X_test_l = X_test_s.reshape(X_test_s.shape[0], X_test_s.shape[1], 1)
#     X_test_t = torch.tensor(X_test_l, dtype=torch.float32)

#     t_loader = DataLoader(TensorDataset(X_test_t), batch_size=BATCH_SIZE, shuffle=False)

#     model.eval()
#     preds_list = []
#     with torch.no_grad():
#         for (X_b,) in t_loader:
#             preds_list.extend(model(X_b.to(device)).argmax(1).cpu().numpy())

#     preds = np.array(preds_list)
#     acc   = accuracy_score(y_test, preds)
#     accs.append(acc)
#     all_true.extend(y_test)
#     all_pred.extend(preds)

#     marker = "✅" if acc >= 0.72 else "🟡" if acc >= 0.50 else "❌"
#     print(f"1-{b:<7}  | {acc:>10.4f} | {len(y_test):>8}  {marker}")

# print("-"*35)
# avg = np.mean(accs)
# print(f"{'AVERAGE':<10} | {avg:>10.4f} |")
# print("="*50)
# print(f"\nPaper CDCNN target : 0.7230")
# print(f"Our LSTM result    : {avg:.4f}  ({'✅ BEAT IT!' if avg >= 0.72 else f'gap: {0.723-avg:.3f}'})")

# print("\nClassification Report:")
# print(classification_report(all_true, all_pred,
#       target_names=[f"Class {i+1}" for i in range(num_classes)]))

# print(f"\nTotal time: {time.time()-start_time:.1f}s")

# fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# axes[0].plot(train_losses, color="royalblue",  label="Train Loss")
# axes[0].plot(val_losses,   color="darkorange", label="Val Loss")
# axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(True)

# axes[1].plot(train_accs, color="royalblue",  label="Train Acc")
# axes[1].plot(val_accs,   color="darkorange", label="Val Acc")
# axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(True)

# task_labels = [f"1-{b}" for b in test_batches]
# colors = ["green" if a >= 0.72 else "steelblue" if a >= 0.50 else "tomato" for a in accs]
# axes[2].bar(task_labels, accs, color=colors, edgecolor="black", width=0.6)
# axes[2].axhline(y=avg,   color="blue", linestyle="--", lw=2, label=f"Our avg: {avg:.3f}")
# axes[2].axhline(y=0.723, color="red",  linestyle="--", lw=2, label="Paper CDCNN: 0.723")
# axes[2].set_ylim(0, 1.0); axes[2].legend(); axes[2].grid(True, axis="y")
# axes[2].set_title("Per-Batch Test Accuracy (Real Drift Data)")
# plt.xticks(rotation=45)

# plt.tight_layout()
# plt.savefig("cdcnn_lstm_results.png", dpi=150, bbox_inches="tight")
# plt.show()
# print("Saved: cdcnn_lstm_results.png")
