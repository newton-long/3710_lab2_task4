import os
import numpy as np
from glob import glob, os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch.nn.functional as F

import matplotlib.pyplot as plt


# ------------------
# CONFIG
# ------------------
DATA_DIR = "./OASIS"   # adjust if needed
IMG_SIZE = 128
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def visualize_predictions(model, dataset, device=DEVICE, num_samples=3):
    model.eval()
    fig, axs = plt.subplots(num_samples, 3, figsize=(9, 3*num_samples))

    for i in range(num_samples):
        img, mask = dataset[i]
        img_t = img.unsqueeze(0).to(device)  # add batch dim

        with torch.no_grad():
            logits = model(img_t)
            pred = torch.argmax(logits, dim=1).squeeze().cpu()

        axs[i,0].imshow(img.squeeze(), cmap="gray")
        axs[i,0].set_title("Input MRI")
        axs[i,0].axis("off")

        axs[i,1].imshow(mask, cmap="gray")
        axs[i,1].set_title("Ground Truth Mask")
        axs[i,1].axis("off")

        axs[i,2].imshow(pred, cmap="gray")
        axs[i,2].set_title("Predicted Mask")
        axs[i,2].axis("off")

    plt.tight_layout()
    plt.show()

# ------------------
# DATASET
# ------------------
class OASISSegDataset(Dataset):
    """
    Dataset for OASIS segmentation.
    Loads MRI slices and corresponding masks, resizes them,
    and remaps mask pixel values {0,85,170,255} → {0,1,2,3}.
    """
    def __init__(self, img_dir, mask_dir, img_size=128):
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.img_size = img_size

        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No PNGs found in {img_dir}")
        if len(self.mask_paths) == 0:
            raise FileNotFoundError(f"No PNGs found in {mask_dir}")

        if len(self.img_paths) != len(self.mask_paths):
            print(f"[WARN] Unequal counts: {len(self.img_paths)} imgs vs {len(self.mask_paths)} masks")

        # Image transform (grayscale, resize, tensor)
        self.img_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        # Mask transform (resize only, keep as integer labels)
        self.mask_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image and mask
        img = Image.open(self.img_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Apply transforms
        img = self.img_transform(img)
        mask = self.mask_resize(mask)

        # Convert to numpy for label remapping
        mask = np.array(mask, dtype=np.int64)

        # Remap {0,85,170,255} → {0,1,2,3}
        mask = mask // 85
        mask = torch.from_numpy(mask)

        return img, mask


# ----------------------------------- the model
# UNET MODEL
# ------------------
class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) x2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_ch=32):
        super().__init__()
        self.inc   = DoubleConv(in_channels, base_ch)          # 1 → 32
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch*2))   # 32 → 64
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch*2, base_ch*4)) # 64 → 128
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch*4, base_ch*8)) # 128 → 256

        self.bottleneck = DoubleConv(base_ch*8, base_ch*16)    # 256 → 512

        self.up3 = nn.ConvTranspose2d(base_ch*16, base_ch*8, kernel_size=2, stride=2) # 512 → 256
        self.dec3 = DoubleConv(base_ch*8 + base_ch*4, base_ch*8)  # 256+128=384 → 256

        self.up2 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2) # 256 → 128
        self.dec2 = DoubleConv(base_ch*4 + base_ch*2, base_ch*4)  # 128+64=192 → 128

        self.up1 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2) # 128 → 64
        self.dec1 = DoubleConv(base_ch*2 + base_ch, base_ch*2)   # 64+32=96 → 64

        self.outc = nn.Conv2d(base_ch*2, num_classes, kernel_size=1)  # 64 → num_classes

    def forward(self, x):
        x1 = self.inc(x)           
        x2 = self.down1(x1)        
        x3 = self.down2(x2)        
        x4 = self.down3(x3)        

        xb = self.bottleneck(x4)   

        u3 = self.up3(xb)          
        d3 = self.dec3(torch.cat([u3, x3], dim=1))
        u2 = self.up2(d3)          
        d2 = self.dec2(torch.cat([u2, x2], dim=1))
        u1 = self.up1(d2)          
        d1 = self.dec1(torch.cat([u1, x1], dim=1))

        out = self.outc(d1)        # logits (B, num_classes, H, W)
        return out


# ----- Loss functions -----
ce_loss_fn = nn.CrossEntropyLoss()

def dice_coefficient(logits, targets, eps=1e-6):
    """Dice coefficient for binary/multi-class segmentation."""
    num_classes = logits.shape[1]
    preds = torch.argmax(logits, dim=1)  # (B,H,W)
    
    dice_scores = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        targ_c = (targets == c).float()
        intersection = (pred_c * targ_c).sum()
        union = pred_c.sum() + targ_c.sum()
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice.item())
    return dice_scores

# ------------------
# SANITY TEST + TRAINING + TEST EVAL + VISUALISATION
# ------------------
if __name__ == "__main__":
    
    # mask_files = glob.glob("./OASIS/keras_png_slices_seg_train/*.png")
    # sample_mask = np.array(Image.open(mask_files[0]))
    # print("Unique values in sample mask:", np.unique(sample_mask))
    
    # ---- TRAIN DATA ----
    train_imgs = os.path.join(DATA_DIR, "keras_png_slices_train")
    train_msks = os.path.join(DATA_DIR, "keras_png_slices_seg_train")

    ds = OASISSegDataset(train_imgs, train_msks, img_size=IMG_SIZE)
    print(f"Dataset size: {len(ds)} samples")

    train_loader = DataLoader(ds, batch_size=8, shuffle=True)

    # ---- MODEL SETUP ----
    model = UNet(in_channels=1, num_classes=4).to(DEVICE)
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # quick sanity check
    imgs, masks = next(iter(train_loader))
    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
    logits = model(imgs)
    print("Logits shape:", logits.shape)
    print("Initial loss:", ce_loss_fn(logits, masks).item())

    # ---- TRAINING LOOP ----
    EPOCHS = 2   # adjust to 20–30 later if you want better results
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        dice_scores_epoch = []

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            # Forward
            logits = model(imgs)
            loss = ce_loss_fn(logits, masks)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Dice
            dice_scores = dice_coefficient(logits, masks)
            dice_scores_epoch.append(dice_scores)

        avg_loss = total_loss / len(train_loader)
        avg_dice = np.mean(dice_scores_epoch, axis=0)

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Dice={avg_dice}")

    # ---- TEST DATA EVAL ----
    test_imgs = os.path.join(DATA_DIR, "keras_png_slices_test")
    test_msks = os.path.join(DATA_DIR, "keras_png_slices_seg_test")
    test_ds = OASISSegDataset(test_imgs, test_msks, img_size=IMG_SIZE)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model.eval()
    test_dice_scores = []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            dice = dice_coefficient(logits, masks)
            test_dice_scores.append(dice)

    avg_test_dice = np.mean(test_dice_scores, axis=0)
    print(f"\nTest Dice (background, brain): {avg_test_dice}")

    # ---- TEST VISUALISATION ----
    print("Visualising predictions on test set...")
    visualize_predictions(model, test_ds, device=DEVICE, num_samples=3)

