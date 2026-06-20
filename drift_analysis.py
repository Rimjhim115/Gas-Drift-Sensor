import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy

print("Loading dataset...")
df = pd.read_csv("final_balanced_dataset.csv")

features = df.drop(["label"], axis=1)

batches = sorted(df["batch"].unique())

train_batch = batches[0]
test_batches = batches[1:]

train_data = df[df["batch"] == train_batch]
test_data = df[df["batch"] != train_batch]

print(f"\nTraining batch: {train_batch} ({len(train_data)} samples)")
print(f"Testing batches: {test_batches} ({len(test_data)} samples)")

print("\n=== DRIFT ANALYSIS ===")

drift_results = []

train_features = train_data.drop(["label", "batch"], axis=1)
mean_train = train_features.mean().values

for b in test_batches:
    b_features = features[df["batch"] == b].drop("batch", axis=1)
    mean_b = b_features.mean().values

    drift = np.linalg.norm(mean_train - mean_b)
    drift_results.append((train_batch, b, drift))

    print(f"Train (Batch {train_batch}) vs Test Batch {b} Drift: {drift:.4f}")

drift_df = pd.DataFrame(drift_results, columns=["Batch1", "Batch2", "Drift"])
drift_df.to_csv("drift_results.csv", index=False)

print("\n=== GAS INTENSITY ===")

intensity_train = train_data.groupby(["batch", "label"]).mean()
intensity_test = test_data.groupby(["batch", "label"]).mean()

intensity_train.to_csv("gas_intensity_train.csv")
intensity_test.to_csv("gas_intensity_test.csv")

print("Saved drift_results.csv, gas_intensity_train.csv, and gas_intensity_test.csv")