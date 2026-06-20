import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

print("Loading dataset...")

df = pd.read_csv("smell_dataset.csv")

X = df.drop(columns=["label", "batch"]).values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)

input_dim = X.shape[1]

class TabDDPM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,dim)
        )

    def forward(self,x):
        return self.net(x)


model = TabDDPM(input_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs = 100

print("Training diffusion model...")

for epoch in range(epochs):

    noise = torch.randn_like(X)
    noisy_data = X + noise*0.1

    pred_noise = model(noisy_data)

    loss = loss_fn(pred_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

torch.save({
    "model": model.state_dict(),
    "scaler": scaler
}, "tabddpm_model.pth")

print("Model saved")