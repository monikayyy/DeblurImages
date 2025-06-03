
import torch
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import os

import config as config
from model import get_model

import torchvision.transforms.functional as TF
import numpy as np

import matplotlib.pyplot as plt

def visualize_validation_batch(blur_batch, pred_batch, sharp_batch, num_images=4, title_prefix=""):

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


def deblur_image(model, image_path, output_path):
    """Loads an image, processes it through the model, and saves the output."""
    
    # Define transformations (should match validation transform without normalization for saving)
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    
    # Process with the model
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # De-normalize from [-1, 1] to [0, 1] for saving
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)
    
    # Save the deblurred image
    save_image(output_tensor, output_path)
    print(f"Deblurred image saved to {output_path}")

def main(args):
    """Main function to load model and run inference."""
    model = get_model(args.model).to(config.DEVICE)
    
    # We only need to load the model's state_dict for inference
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, args.checkpoint)
    print(f"==> Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    
    # Prepare output path
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    base_name = os.path.basename(args.image)
    output_filename = f"{os.path.splitext(base_name)[0]}_deblurred.png"
    output_path = os.path.join(config.RESULTS_DIR, output_filename)
    
    deblur_image(model, args.image, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deblur a single image.")
    parser.add_argument("-m", "--model", type=str, required=True, choices=["Restormer", "DeepBlur"], help="Model architecture used for the checkpoint.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Name of the model checkpoint file in the saved_models folder.")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to the blurry input image.")
    args = parser.parse_args()
    main(args)

