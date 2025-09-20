import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class OASISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = sorted(glob(os.path.join(root_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label


# ------------------------------
# 1. Config
# ------------------------------
DATA_DIR = "./OASIS"  # Adjust if needed
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
LATENT_DIM = 16  # try 2 if you want a directly visualisable latent space
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 2. Dataset + Dataloaders
# ------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure single channel
    transforms.Resize((128, 128)),                # resize to fixed size
    transforms.ToTensor(),                        # [0,255] -> [0,1]
])

train_dataset = OASISDataset(os.path.join(DATA_DIR, "keras_png_slices_train"), transform=transform)
val_dataset   = OASISDataset(os.path.join(DATA_DIR, "keras_png_slices_validate"), transform=transform)
test_dataset  = OASISDataset(os.path.join(DATA_DIR, "keras_png_slices_test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------
# 3. Define VAE
# ------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # -> 32 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # -> 64 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # -> 128 x 16 x 16
            nn.ReLU(),
        )
        
        self.flatten_dim = 128 * 16 * 16
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # -> 64 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 32 x 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # -> 1 x 128 x 128
            nn.Sigmoid(),
        )
    
    """
    The reparameterization trick is essential in VAEs because it lets you sample latent vectors from a distribution while still allowing backpropagation.
    This makes training possible and gives the model a smooth, generative latent space instead of just fixed encodings.
    """
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x = self.fc_dec(z).view(-1, 128, 16, 16)
        x = self.decoder(x)
        return x, mu, logvar

# ------------------------------
# 4. Loss function
# ------------------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# ------------------------------
# 5. Training Loop
# ------------------------------
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0
    for x, _ in train_loader:
        x = x.to(DEVICE)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = vae_loss(recon_x, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f"Epoch {epoch}, Train Loss: {train_loss/len(train_loader.dataset):.4f}")

# ------------------------------
# 6. Reconstructions
# ------------------------------
model.eval()
with torch.no_grad():
    imgs, _ = next(iter(test_loader))
    imgs = imgs.to(DEVICE)
    recon, _, _ = model(imgs)
    comparison = torch.cat([imgs[:8], recon[:8]])
    utils.save_image(comparison.cpu(), "reconstructions.png", nrow=8)

print("Saved reconstructions → reconstructions.png")

# ------------------------------
# 7. Latent Space t-SNE Projection
# ------------------------------
latents = []
labels = []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        _, mu, logvar = model(x)
        z = mu
        latents.append(z.cpu().numpy())
        labels.append(y.numpy())

latents = np.concatenate(latents, axis=0)
labels = np.concatenate(labels, axis=0)

print("Running t-SNE on latent space...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
latents_2d = tsne.fit_transform(latents)

plt.figure(figsize=(8,6))
scatter = plt.scatter(latents_2d[:,0], latents_2d[:,1], c=labels, cmap="tab10", s=5)
plt.colorbar(scatter, label="Class (dummy labels from ImageFolder)")
plt.title("Latent Space Projection (t-SNE)")
plt.savefig("latent_tsne.png", dpi=150)
plt.close()

print("Saved latent space projection → latent_tsne.png")

# ------------------------------
# 8. Latent Manifold Grid
# ------------------------------
def sample_latent_grid(model, n=10, latent_dim=LATENT_DIM):
    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)
    samples = []
    with torch.no_grad():
        for yi in grid_y:
            row = []
            for xi in grid_x:
                z = torch.zeros(1, latent_dim).to(DEVICE)
                if latent_dim >= 2:
                    z[0,0], z[0,1] = xi, yi
                sample = model.decoder(model.fc_dec(z).view(-1, 128, 16, 16))
                row.append(sample.cpu())
            samples.append(torch.cat(row, dim=0))
    return torch.cat(samples, dim=0)

grid_samples = sample_latent_grid(model, n=10, latent_dim=LATENT_DIM)
utils.save_image(grid_samples, "latent_manifold.png", nrow=10, normalize=True)

print("Saved latent manifold sampling grid → latent_manifold.png")
