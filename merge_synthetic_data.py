import pandas as pd
import numpy as np

print("Loading datasets...")

real = pd.read_csv("smell_dataset.csv")
synthetic = pd.read_csv("synthetic_smell_data.csv")

# only use 30% synthetic samples
synthetic = synthetic.sample(frac=0.3, random_state=42)

# keep labels if already generated
if "label" not in synthetic.columns:
    synthetic["label"] = real["label"].sample(len(synthetic), replace=True).values

# assign batches PROPORTIONALLY (fixes 21,600 → should become ~36,000+)
batch_counts = real["batch"].value_counts(normalize=True)
synthetic["batch"] = np.random.choice(
    batch_counts.index,
    size=len(synthetic),
    replace=True,
    p=batch_counts.values
)

balanced = pd.concat([real, synthetic])

balanced.to_csv("balanced_smell_dataset.csv", index=False)

print("Balanced dataset created")
print("Total rows:", len(balanced))