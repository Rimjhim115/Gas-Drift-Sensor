import pandas as pd

print("Loading dataset...")

df = pd.read_csv("balanced_smell_dataset.csv")

TARGET_BATCH_SIZE = 3600
NUM_LABELS = 6

balanced_batches = []

for batch_id, batch_df in df.groupby("batch"):

    print(f"Processing batch {batch_id}")

    per_label_target = TARGET_BATCH_SIZE // NUM_LABELS

    new_batch = []

    for label in range(1, NUM_LABELS+1):

        label_df = batch_df[batch_df["label"] == label]

        current_count = len(label_df)

        if current_count > per_label_target:
            label_df = label_df.sample(per_label_target)

        elif current_count < per_label_target:
            needed = per_label_target - current_count
            extra = label_df.sample(needed, replace=True)
            label_df = pd.concat([label_df, extra])

        new_batch.append(label_df)

    new_batch = pd.concat(new_batch)

    balanced_batches.append(new_batch)

balanced_df = pd.concat(balanced_batches)

balanced_df.to_csv("final_balanced_dataset.csv", index=False)

print("\nBalanced dataset created!")

print("\nSamples per batch:")
print(balanced_df.groupby("batch").size())

print("\nGas distribution per batch:")
print(balanced_df.groupby(["batch","label"]).size().unstack())