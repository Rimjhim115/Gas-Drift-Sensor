import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

original = pd.read_csv("smell_dataset.csv")
synthetic = pd.read_csv("final_balanced_dataset.csv")


batch_id = 10
orig_batch = original[original["batch"] == batch_id]
syn_batch = synthetic[synthetic["batch"] == batch_id]

X_orig = orig_batch.drop(columns=["batch", "label"])
X_syn = syn_batch.drop(columns=["batch", "label"])

X = pd.concat([X_orig, X_syn])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

orig_pca = X_pca[:len(X_orig)]
syn_pca = X_pca[len(X_orig):]

plt.figure(figsize=(8,6))

plt.scatter(orig_pca[:,0], orig_pca[:,1],
            label="Original Data",
            alpha=0.8,
            marker='o',
            s=40)

# plot synthetic second
plt.scatter(syn_pca[:,0], syn_pca[:,1],
            label="New Data",
            alpha=0.4,
            marker='x',
            s=40)

plt.title("PCA Comparison: Original vs New Data (Batch 10)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend()
plt.show()