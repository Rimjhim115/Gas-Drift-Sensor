# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# print("Loading model...")
# checkpoint = torch.load("tabddpm_model.pth", weights_only=False)

# scaler = checkpoint["scaler"]

# df = pd.read_csv("smell_dataset.csv")

# sensor_cols = df.drop(columns=["label","batch"]).columns
# input_dim = len(sensor_cols)

# class TabDDPM(nn.Module):
#     def __init__(self, dim):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Linear(dim,256),
#             nn.ReLU(),
#             nn.Linear(256,256),
#             nn.ReLU(),
#             nn.Linear(256,dim)
#         )

#     def forward(self,x):
#         return self.net(x)


# model = TabDDPM(input_dim)
# model.load_state_dict(checkpoint["model"])

# model.eval()

# print("Generating synthetic samples...")

# num_samples = 30000

# noise = torch.randn(num_samples,input_dim)

# synthetic = model(noise).detach().numpy()

# synthetic = scaler.inverse_transform(synthetic)

# syn_df = pd.DataFrame(synthetic, columns=sensor_cols)

# syn_df.to_csv("synthetic_smell_data.csv", index=False)

# print("Synthetic dataset generated")




import torch
import torch.nn as nn
import pandas as pd
import numpy as np

print("Loading model...")

checkpoint = torch.load("tabddpm_model.pth", weights_only=False)

scaler = checkpoint["scaler"]

df = pd.read_csv("smell_dataset.csv")

sensor_cols = df.drop(columns=["label","batch"], errors="ignore").columns
input_dim = len(sensor_cols)

class TabDDPM(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,dim)
        )

    def forward(self,x):
        return self.net(x)

model = TabDDPM(input_dim)
model.load_state_dict(checkpoint["model"])

model.eval()

print("Generating synthetic samples...")

num_samples = 10000

noise = torch.randn(num_samples,input_dim)

with torch.no_grad():
    synthetic = model(noise).numpy()

synthetic = scaler.inverse_transform(synthetic)

syn_df = pd.DataFrame(synthetic, columns=sensor_cols)

# assign labels using nearest neighbor from real data
real_features = df[sensor_cols].values

from sklearn.neighbors import NearestNeighbors

nn_model = NearestNeighbors(n_neighbors=1)
nn_model.fit(real_features)

_, indices = nn_model.kneighbors(synthetic)

syn_df["label"] = df.iloc[indices.flatten()]["label"].values

syn_df.to_csv("synthetic_smell_data.csv", index=False)

print("Synthetic dataset generated")