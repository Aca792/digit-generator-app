import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Load model architecture (same as used in training)
class Encoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(784 + 10, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)

    def forward(self, x, y):
        h = torch.cat([x, y], dim=1)
        h = torch.relu(self.fc1(h))
        return self.fc21(h), self.fc22(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc3 = nn.Linear(latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        h = torch.relu(self.fc3(h))
        return torch.sigmoid(self.fc4(h))

class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

# Load model
device = torch.device("cpu")
model = CVAE()
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
model.eval()

# Web app
st.title("🧠 Handwritten Digit Generator")
digit = st.selectbox("Choose a digit (0-9)", list(range(10)))

# Generate 5 images
def generate_images(digit):
    with torch.no_grad():
        y = torch.eye(10)[digit].repeat(5, 1)
        z = torch.randn(5, model.latent_dim)
        output = model.decoder(z, y).view(-1, 28, 28).numpy()
    return output

if st.button("Generate"):
    imgs = generate_images(digit)
    cols = st.columns(5)
    for i in range(5):
        cols[i].image(imgs[i], width=100, clamp=True, channels="L", caption=f"{digit}")
