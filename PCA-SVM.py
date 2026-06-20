# import pandas as pd
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# import warnings
# warnings.filterwarnings("ignore")

# print("Loading dataset...")
# df = pd.read_csv("smell_dataset.csv")
# batches = sorted(df["batch"].unique())
# train_batch = batches[0]
# test_batches = batches[1:]

# train_df = df[df["batch"] == train_batch]
# X_train = train_df.drop(["label", "batch"], axis=1).values
# y_train = train_df["label"].values

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

# pca = PCA(n_components=0.95, random_state=42)
# X_train_pca = pca.fit_transform(X_train_scaled)

# model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
# model.fit(X_train_pca, y_train)

# print("\nTask     | Accuracy")
# print("-" * 22)
# accs = []
# for b in test_batches:
#     b_df = df[df["batch"] == b]
#     X_test = pca.transform(scaler.transform(b_df.drop(["label", "batch"], axis=1).values))
#     y_test = b_df["label"].values
#     acc = accuracy_score(y_test, model.predict(X_test))
#     accs.append(acc)
#     print(f"1-{b:<6}  | {acc:.4f}")
# print("-" * 22)
# print(f"Average  | {np.mean(accs):.4f}")
