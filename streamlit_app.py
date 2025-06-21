import streamlit as st
import torch
import torch.nn as nn

# Model definitions (same as before)
class Encoder(nn.Module):
    def __init__(self, latent_dim=50):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(784 + 10, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 400),
            nn.ReLU()
        )
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)

    def forward(self, x, y):
        h = torch.cat([x, y], dim=1)
        h = self.fc1(h)
        return self.fc21(h), self.fc22(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=50):
        super().__init__()
        self.fc3 = nn.Sequential(
            nn.Linear(latent_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        return self.fc3(h)

class CVAE(nn.Module):
    def __init__(self, latent_dim=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

# Load model
device = torch.device("cpu")
model = CVAE(latent_dim=50)
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
model.eval()

st.set_page_config(page_title="Handwritten Digit Image Generator")

# Title and description
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

# Dropdown to select digit
digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button("Generate Images"):
    with st.spinner("Generating..."):
        with torch.no_grad():
            y = torch.eye(10)[digit].repeat(5, 1)
            z = torch.randn(5, model.latent_dim)
            outputs = model.decoder(z, y).view(-1, 28, 28).numpy()

    st.markdown(f"### Generated images of digit {digit}")

    cols = st.columns(5)
    for i, col in enumerate(cols):
        col.image(outputs[i], width=80)
        col.caption(f"Sample {i+1}")
