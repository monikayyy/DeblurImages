
import torch
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import config as config
from torchvision.utils import save_image

import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

def visualize_val_batch(blur_batch, pred_batch, sharp_batch, num_images=4, title_prefix=""):

    if not all(isinstance(t, torch.Tensor) for t in [blur_batch, pred_batch, sharp_batch]):
        print("Warning: All input batches must be PyTorch Tensors.")
        return

    if not (blur_batch.ndim == 4 and pred_batch.ndim == 4 and sharp_batch.ndim == 4):
        print("Warning: All input batches must be 4D tensors (B, C, H, W).")
        return

    num_to_show = min(num_images, blur_batch.shape[0], pred_batch.shape[0], sharp_batch.shape[0])

    if num_to_show == 0:
        print("Warning: No images to show (batch size might be 0, num_images=0, or mismatched batch sizes).")
        return

    # Detach tensors from the computation graph and select the subset to show
    blur_imgs_t = blur_batch[:num_to_show].detach()
    pred_imgs_t = pred_batch[:num_to_show].detach()
    sharp_imgs_t = sharp_batch[:num_to_show].detach()

    # Clamp image values to [0, 1] for proper display
    # (Important if model outputs are not strictly in this range)
    blur_imgs_t = torch.clamp(blur_imgs_t, 0, 1)
    pred_imgs_t = torch.clamp(pred_imgs_t, 0, 1)
    sharp_imgs_t = torch.clamp(sharp_imgs_t, 0, 1)

    # Create subplots: num_to_show rows, 3 columns (Input, Predicted, Ground Truth)
    # Adjust figsize as needed
    fig, axes = plt.subplots(num_to_show, 3, figsize=(12, num_to_show * 4))

    # If num_to_show is 1, axes is a 1D array, so we need to handle it
    if num_to_show == 1:
        axes = axes.reshape(1, -1) # Reshape to (1, 3) to make indexing consistent

    for i in range(num_to_show):
        # Convert tensors to NumPy arrays for matplotlib
        # Permute from (C, H, W) to (H, W, C) and move to CPU
        blur_np = blur_imgs_t[i].cpu().permute(1, 2, 0).numpy()
        pred_np = pred_imgs_t[i].cpu().permute(1, 2, 0).numpy()
        sharp_np = sharp_imgs_t[i].cpu().permute(1, 2, 0).numpy()

        # --- Column 0: Input Blurry ---
        axes[i, 0].imshow(blur_np)
        axes[i, 0].set_title(f"Input Blurry {i+1}")
        axes[i, 0].axis('off')

        # --- Column 1: Predicted Sharp ---
        axes[i, 1].imshow(pred_np)
        axes[i, 1].set_title(f"Predicted Sharp {i+1}")
        axes[i, 1].axis('off')

        # --- Column 2: Ground Truth Sharp ---
        axes[i, 2].imshow(sharp_np)
        axes[i, 2].set_title(f"Ground Truth Sharp {i+1}")
        axes[i, 2].axis('off')

    plt.suptitle(title_prefix, fontsize=14, y=1.0) # y=1.0 might be slightly high, adjust if needed
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # rect to make space for suptitle
    plt.show()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """Saves model, optimizer state, etc., to a file."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"==> Saved checkpoint to {filepath}")

def load_checkpoint(model, optimizer, filename):
    """Loads model and optimizer state from a file."""
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    print(f"==> Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def calculate_metrics(y_true, y_pred):
    """
    Calculates PSNR and SSIM between two images.
    Images are tensors [N, C, H, W] in range [-1, 1].
    """
    # Denormalize from [-1, 1] to [0, 1]
    y_true = (y_true * 0.5 + 0.5).clamp(0, 1)
    y_pred = (y_pred * 0.5 + 0.5).clamp(0, 1)

    # Convert to numpy [N, H, W, C] for scikit-image
    y_true_np = y_true.permute(0, 2, 3, 1).cpu().numpy()
    y_pred_np = y_pred.permute(0, 2, 3, 1).cpu().numpy()

    batch_psnr = 0.0
    batch_ssim = 0.0
    
    for i in range(y_true_np.shape[0]):
        batch_psnr += psnr(y_true_np[i], y_pred_np[i], data_range=1.0)
        batch_ssim += ssim(y_true_np[i], y_pred_np[i], data_range=1.0, channel_axis=-1, win_size=7)

    return batch_psnr / y_true_np.shape[0], batch_ssim / y_true_np.shape[0]

def save_validation_results(blurry, sharp, deblurred, epoch, model_name):
    """Saves a grid of images: blurry, deblurred, and sharp."""
    # Denormalize from [-1, 1] to [0, 1] for saving
    blurry = blurry * 0.5 + 0.5
    sharp = sharp * 0.5 + 0.5
    deblurred = deblurred * 0.5 + 0.5
    
    # Concatenate images horizontally
    comparison = torch.cat([blurry, deblurred, sharp], dim=-1)
    
    # Create results directory
    result_path = os.path.join(config.RESULTS_DIR, model_name)
    os.makedirs(result_path, exist_ok=True)
    
    # Save the image
    filename = f"epoch_{epoch}.png"
    save_image(comparison, os.path.join(result_path, filename))

