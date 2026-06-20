# import os
# import pandas as pd

# DATA_DIR = "archive"
# OUTPUT_CSV = "smell_dataset.csv"

# all_data = []

# for root, dirs, files in os.walk(DATA_DIR):
#     for file in files:
#         if file.endswith(".dat"):
#             file_path = os.path.join(root, file)
#             print("Reading:", file_path)

#             rows = []
#             labels = []

#             with open(file_path, "r") as f:
#                 for line in f:
#                     parts = line.strip().split()

#                     # First value = class label
#                     class_label = int(parts[0])
#                     labels.append(class_label)

#                     # Remaining values = sensor readings
#                     sensor_values = []
#                     for item in parts[1:]:
#                         if ":" in item:
#                             sensor_values.append(float(item.split(":")[1]))

#                     rows.append(sensor_values)

#             df = pd.DataFrame(rows)
#             df["label"] = labels

#             all_data.append(df)

# if not all_data:
#     raise ValueError("❌ No data loaded. Check dataset path.")

# final_df = pd.concat(all_data, ignore_index=True)
# final_df.to_csv(OUTPUT_CSV, index=False)

# print("✅ Dataset created successfully!")
# print("Shape:", final_df.shape)
import os
import pandas as pd

DATA_DIR = "archive"
OUTPUT_CSV = "smell_dataset.csv"

all_data = []

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".dat"):

            file_path = os.path.join(root, file)
            print("Reading:", file_path)

            # Extract batch number from filename
            batch_id = int(file.split("batch")[1].split(".")[0])

            rows = []
            labels = []

            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    label = int(parts[0])
                    labels.append(label)

                    sensor_values = []
                    for item in parts[1:]:
                        if ":" in item:
                            sensor_values.append(float(item.split(":")[1]))

                    rows.append(sensor_values)

            df = pd.DataFrame(rows)
            df["label"] = labels
            df["batch"] = batch_id

            all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

final_df.to_csv(OUTPUT_CSV, index=False)

print("Dataset created")
print("Shape:", final_df.shape)