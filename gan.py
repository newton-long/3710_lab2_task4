import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
#  Custom Dataset for OASIS PNGs
# -----------------------------
class OASISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Collect all PNG file paths in the directory
        self.files = glob.glob(os.path.join(root_dir, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Open image, force grayscale ("L" = 1 channel)
        img = Image.open(self.files[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        # Return dummy label (GANs don’t need class labels)
        return img, 0

def get_dataloader(data_dir, image_size=64, batch_size=64, num_workers=2):
    # Preprocessing:
    # 1. Resize to 64x64
    # 2. Convert to tensor [0,1]
    # 3. Normalize to [-1,1] (since Generator uses tanh)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = OASISDataset(root_dir=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# -----------------------------
#  Generator (G)
# -----------------------------
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        """
        nz  = size of latent vector (random noise input)
        ngf = base number of feature maps
        nc  = number of channels in output (1 for grayscale)
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input Z: [nz x 1 x 1] → [ngf*8 x 4 x 4]
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # [ngf*8 x 4 x 4] → [ngf*4 x 8 x 8]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # [ngf*4 x 8 x 8] → [ngf*2 x 16 x 16]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # [ngf*2 x 16 x 16] → [ngf x 32 x 32]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # [ngf x 32 x 32] → [nc x 64 x 64]
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Outputs in [-1,1]
        )

    def forward(self, x):
        return self.main(x)

# -----------------------------
#  Discriminator (D)
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        """
        nc  = input channels (1 for grayscale)
        ndf = base number of feature maps
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # [1 x 64 x 64] → [ndf x 32 x 32]
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # [ndf x 32 x 32] → [ndf*2 x 16 x 16]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # [ndf*2 x 16 x 16] → [ndf*4 x 8 x 8]
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # [ndf*4 x 8 x 8] → [ndf*8 x 4 x 4]
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # [ndf*8 x 4 x 4] → [1]
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Probability of "real"
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# -----------------------------
#  Training Loop
# -----------------------------
def train_gan(data_dir, epochs=10, batch_size=64, lr=0.0002, nz=100, device="cuda"):
    dataloader = get_dataloader(data_dir, batch_size=batch_size)

    # Init models
    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)

    # Loss = Binary Cross Entropy (real vs fake)
    criterion = nn.BCELoss()

    # Optimizers (Adam is standard for DCGAN)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    # Fixed noise → lets us see progress at same seeds each epoch
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Track loss history
    G_losses, D_losses = [], []

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            # ---------------------
            # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # ---------------------
            netD.zero_grad()
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            # Labels: 1 for real, 0 for fake
            labels_real = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            labels_fake = torch.full((b_size,), 0.0, dtype=torch.float, device=device)

            # Real pass
            output_real = netD(real_imgs)
            lossD_real = criterion(output_real, labels_real)

            # Fake pass
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_imgs = netG(noise)
            output_fake = netD(fake_imgs.detach())
            lossD_fake = criterion(output_fake, labels_fake)

            # Total D loss
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # ---------------------
            # Train Generator: maximize log(D(G(z)))
            # ---------------------
            netG.zero_grad()
            output = netD(fake_imgs)
            # Trick D: pretend fakes are real (label=1)
            lossG = criterion(output, labels_real)
            lossG.backward()
            optimizerG.step()

            # Save losses for plotting
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {i}/{len(dataloader)} "
                      f"Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

        # ---------------------
        # Save generated samples every epoch
        # ---------------------
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        grid = torchvision.utils.make_grid(fake[:16], nrow=4, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.title(f"Epoch {epoch+1}")
        plt.axis("off")
        plt.savefig(f"gan_epoch_{epoch+1}.png")
        plt.close()

    # ---------------------
    # Save loss curves
    # ---------------------
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("gan_losses.png")
    plt.close()

    return netG, netD

# -----------------------------
#  Run Training
# -----------------------------
if __name__ == "__main__":
    data_dir = "./OASIS/keras_png_slices_train"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    netG, netD = train_gan(data_dir, epochs=10, batch_size=64, device=device)
