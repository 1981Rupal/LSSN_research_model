import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from lssn_model import LSSN_UNet
from loss import InvarianceLoss
import os

# --- Configuration ---
BATCH_SIZE = 2
IMAGE_SIZE = 64
CHANNELS = 3
LR = 1e-4
EPOCHS = 2
LAMBDA_INV = 1.0
LAMBDA_COSINE = 0.5 # New innovative loss weight
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dummy Dataset ---
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random image
        img = torch.randn(CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        # Random text embedding (simulating CLIP text)
        text_emb = torch.randn(77, 768)
        # Random image embedding (simulating CLIP image)
        img_emb = torch.randn(4, 768) # e.g. 4 tokens
        return img, text_emb, img_emb

# --- Training Script ---
def train():
    print(f"Initializing LSSN on {DEVICE}...")
    
    # Model
    model = LSSN_UNet(
        in_channels=CHANNELS, 
        out_channels=CHANNELS, 
        model_channels=64, # Small for testing
        num_res_blocks=1, 
        channel_mult=(1, 2)
    ).to(DEVICE)
    
    # Loss: Hybrid Invariance Loss (L2 + Cosine)
    inv_criterion = InvarianceLoss(lambda_inv=LAMBDA_INV, lambda_cosine=LAMBDA_COSINE)
    mse_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # Data
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Starting training loop...")
    
    for epoch in range(EPOCHS):
        for i, (clean_img, c_text, c_img_cond) in enumerate(dataloader):
            clean_img = clean_img.to(DEVICE)
            c_text = c_text.to(DEVICE)
            c_img_cond = c_img_cond.to(current_device := DEVICE) 
            
            b = clean_img.shape[0]
            
            # 1. Sample Timesteps
            t = torch.randint(0, 1000, (b,), device=DEVICE).long()
            
            # 2. Add Noise (Simple linear schedule approximation for demo)
            noise = torch.randn_like(clean_img)
            # Alpha/Beta schedule would go here. We'll just assume scalar mix for demo.
            alpha = 0.5 
            noisy_img = alpha * clean_img + (1 - alpha) * noise
            
            # 3. Text Path (Image conditioning zeroed)
            # Need to match shape of c_img_cond but zeros
            zero_image_cond = torch.zeros_like(c_img_cond)
            pred_noise_text, feats_text = model(
                noisy_img, t, c_text, zero_image_cond, return_features=True
            )
            
            # 4. Image Path (Text conditioning zeroed)
            zero_text_cond = torch.zeros_like(c_text)
            pred_noise_image, feats_image = model(
                noisy_img, t, zero_text_cond, c_img_cond, return_features=True
            )
            
            # 5. Compute Losses
            # Diffusion Loss (Standard MSE on noise prediction)
            loss_diff = mse_criterion(pred_noise_text, noise) + mse_criterion(pred_noise_image, noise)
            
            # Invariance Loss
            loss_inv = inv_criterion(feats_text, feats_image)
            
            total_loss = loss_diff + loss_inv
            
            # 6. Optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} | Step {i} | Total: {total_loss.item():.4f} | Diff: {loss_diff.item():.4f} | Inv: {loss_inv.item():.4f}")

    print("Training finished successfully.")
    
    # Serialize model
    torch.save(model.state_dict(), "lssn_model.pth")
    print("Model saved to lssn_model.pth")

if __name__ == "__main__":
    train()
