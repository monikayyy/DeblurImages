
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from timm.models.vision_transformer import PatchEmbed, Block as TimmBlock
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
from joblib import Parallel, delayed
import multiprocessing
from functools import partial


import cv2
from natsort import natsorted
import os
import time
import datetime
import gc
import importlib
import config


from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    LR_DECAY_EPOCHS,
    LR_FINAL,
    WEIGHT_DECAY,
    BATCH_SIZE,
    NUM_WORKERS,
    OUTPUT_DIR,
    PRETRAINED_WEIGHTS_PATH,
    OPT)
from model import get_model
from losses.perceptual_loss import ReconstructPerceptualLoss
from dataset import get_data_loaders
from losses import pos_embed

from utils import (
    save_checkpoint,
    load_checkpoint,
    visualize_val_batch,
    calculate_metrics,
    save_validation_results
)

def train_one_epoch(model, optimizer, criterion, progress_bar):
    """Performs one full training pass over the dataset."""
    # loop = tqdm(loader, leave=True)
    # loop.set_description(f"Training")

    epoch_loss_total = 0.0
    epoch_loss_l1 = 0.0
    epoch_loss_perc = 0.0

    for batch_idx, (blur_imgs, sharp_imgs) in enumerate(progress_bar):

        gt_img = sharp_imgs.to(DEVICE)
        b_img = blur_imgs.to(DEVICE)

        sharp_gt_half = F.interpolate(gt_img, scale_factor=0.5, mode='bilinear', align_corners=False)
        sharp_gt_quarter = F.interpolate(sharp_gt_half, scale_factor=0.5, mode='bilinear', align_corners=False)

        fine_out, mid_out, coarse_out = model(b_img)

        sharp_gt_half_224 = F.interpolate(sharp_gt_half, size = 224, mode='bilinear', align_corners=False)
        sharp_gt_quarter_224 = F.interpolate(sharp_gt_quarter, size = 224, mode='bilinear', align_corners=False)

        mid_out_224 = F.interpolate(mid_out, size = 224, mode='bilinear', align_corners=False)
        coarse_out_224 = F.interpolate(coarse_out, size = 224, mode='bilinear', align_corners=False)

        # Uses ReconstructLoss's forward
        loss_fine = criterion(fine_out, gt_img)
        loss_mid = criterion(mid_out_224, sharp_gt_half_224)
        loss_coarse = criterion(coarse_out_224, sharp_gt_quarter_224)
        # Combining the fine, mid and coarse losses
        losses = loss_fine["total_loss"] + loss_mid["total_loss"] + loss_coarse["total_loss"]

        # 4. Backpropagation
        grad_loss = loss_fine["total_loss"] + loss_mid["total_loss"] + loss_coarse["total_loss"]
        optimizer.zero_grad()
        grad_loss.backward()
        optimizer.step()

        # Logging
        epoch_loss_total += grad_loss.item()
        epoch_loss_l1 += loss_fine.get('l1', torch.tensor(0.0)).item() + loss_mid.get('l1', torch.tensor(0.0)).item() + loss_coarse.get('l1', torch.tensor(0.0)).item()
        epoch_loss_perc += loss_fine.get('Perceptual', torch.tensor(0.0)).item() + loss_mid.get('Perceptual', torch.tensor(0.0)).item() + loss_coarse.get('Perceptual', torch.tensor(0.0)).item()

        progress_bar.set_postfix(loss=f"{grad_loss.item():.4f}")
        
    
    return epoch_loss_total, epoch_loss_l1, epoch_loss_perc


@torch.no_grad()
def validate_epoch(model, dataloader, device, criterion, visualize=False, num_images_to_show=4, epoch_num=None):
    model.eval()
    # Initialize validation loss
    val_loss = 0.0
    visualized_this_epoch = not visualize
    # Handle empty dataloader
    if not dataloader:
        print("Validation dataloader is empty.")
        model.train()
        return 0.0

    progress_bar = tqdm(dataloader, desc=f"Validating Epoch {epoch_num if epoch_num is not None else 'N/A'}", leave=False)
    for batch_idx, (blur_imgs, sharp_imgs) in enumerate(progress_bar):
        blur_imgs = blur_imgs.to(device)
        sharp_imgs = sharp_imgs.to(device)

        sharp_gt_half = F.interpolate(sharp_imgs, scale_factor=0.5, mode='bilinear', align_corners=False)
        sharp_gt_quarter = F.interpolate(sharp_gt_half, scale_factor=0.5, mode='bilinear', align_corners=False)

        fine_out, mid_out, coarse_out = model(blur_imgs)

        sharp_gt_half_224 = F.interpolate(sharp_gt_half, size = 224, mode='bilinear', align_corners=False)
        sharp_gt_quarter_224 = F.interpolate(sharp_gt_quarter, size = 224, mode='bilinear', align_corners=False)

        mid_out_224 = F.interpolate(mid_out, size = 224, mode='bilinear', align_corners=False)
        coarse_out_224 = F.interpolate(coarse_out, size = 224, mode='bilinear', align_corners=False)

        if not visualized_this_epoch:
            print(f"\nVisualizing Validation Batch {batch_idx} (Epoch {epoch_num if epoch_num is not None else 'N/A'})...")
            title = f"Validation - Epoch {epoch_num}" if epoch_num is not None else "Validation"
            visualize_val_batch(blur_imgs.cpu(),
                                       fine_out.cpu(),
                                       sharp_imgs.cpu(),
                                       num_images=num_images_to_show,
                                       title_prefix=title)
            visualized_this_epoch = True # Ensure visualization happens only once per epoch call

        # Uses ReconstructLoss's forward
        loss_fine = criterion(fine_out, sharp_imgs)
        loss_mid = criterion(mid_out_224, sharp_gt_half_224)
        loss_coarse = criterion(coarse_out_224, sharp_gt_quarter_224)
        # Combining the fine, mid and coarse losses
        losses = loss_fine["total_loss"] + loss_mid["total_loss"] + loss_coarse["total_loss"]

        current_batch_loss = losses.item()
        val_loss += current_batch_loss

        # Display current batch loss and running average epoch loss
        progress_bar.set_postfix(
            BatchLoss=f"{current_batch_loss:.4f}",
            AvgEpochLoss=f"{val_loss / (batch_idx + 1):.4f}"
        )

        del fine_out, mid_out, coarse_out, mid_out_224, coarse_out_224, sharp_gt_half, sharp_gt_quarter, sharp_gt_half_224, sharp_gt_quarter_224


    if len(dataloader) > 0:
        avg_val_loss = val_loss / len(dataloader)
    else:
        avg_val_loss = 0.0

    try:
        del blur_imgs, sharp_imgs
    except NameError:
        pass
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    model.train()

    return avg_val_loss

def lr_lambda(epoch):
    num_halvings = epoch // LR_DECAY_EPOCHS
    lr_multiplier = 0.5 ** num_halvings
    final_multiplier = LR_FINAL / config.EFFECTIVE_LR # Use the actual starting LR
    return max(lr_multiplier, final_multiplier)


def main():
    """Main function to run the training and validation loop."""
    criterion = ReconstructPerceptualLoss(OPT)
    # criterion.opt = OPT
    criterion.pretrain_mae = criterion.pretrain_mae.to(torch.device(DEVICE))

    model = get_model(config.MODEL_NAME).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # effective_lr = config.EFFECTIVE_LR 
    optimizer = optim.Adam(model.parameters(), lr=config.EFFECTIVE_LR, weight_decay=WEIGHT_DECAY)
    print(f"Optimizer: Adam, LR: {config.EFFECTIVE_LR:.1e}, Weight Decay: {WEIGHT_DECAY:.1e}")

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"LR Scheduler: Halve every {LR_DECAY_EPOCHS} epochs, min LR {LR_FINAL:.1e}")


    if config.LOAD_MODEL:
        load_checkpoint(model, optimizer, f"{config.MODEL_NAME}_checkpoint.pth.tar")

    train_loader, val_loader = get_data_loaders()

    start_time = time.time()
    best_val_loss = 0.0
    epoch_loss_total = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        epoch_loss_total, epoch_loss_l1, epoch_loss_perc = train_one_epoch(model, optimizer, criterion, progress_bar)
        
        # Update learning rate
        scheduler.step()

        avg_loss_total = epoch_loss_total / len(train_loader)
        avg_loss_l1 = epoch_loss_l1 / len(train_loader)
        avg_loss_perc = epoch_loss_perc / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        avg_val_loss = validate_epoch(model, val_loader, DEVICE, criterion, visualize=True, num_images_to_show=4, epoch_num=None)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            config.SAVE_BEST_PATH = True
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - LR: {current_lr:.6f} - Loss: {avg_loss_total:.4f} (L1: {avg_loss_l1:.4f}, Perc: {avg_loss_perc:.4f}) || Validation Loss: {avg_val_loss:.4f} *** Best Model Saved ***")
        else:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - LR: {current_lr:.6f} - Loss: {avg_loss_total:.4f} (L1: {avg_loss_l1:.4f}, Perc: {avg_loss_perc:.4f}) || Validation Loss: {avg_val_loss:.4f}")


        if config.SAVE_MODEL:
            current_checkpoint_path = os.path.join(OUTPUT_DIR, f"{config.MODEL_NAME}_checkpoint.pth")
            save_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss_total,
                'val_loss': avg_val_loss, }
            torch.save(save_data, current_checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\n--- Training Finished ---")
    print(f"Total Training Time: {total_time_str}")
    status = "transferred_weights" if config.LOAD_PRETRAINED and os.path.exists(PRETRAINED_WEIGHTS_PATH) else "scratch"
    
    print(f"Final model saved to: {final_model_path}")
    if config.SAVE_BEST_PATH:
        model_best_path = os.path.join(OUTPUT_DIR, f"{config.MODEL_NAME}_{status}_best.pth")
        torch.save(model.state_dict(), model_best_path)
        print(f"Best model (Validation Perceptual Loss: {best_val_loss:.4f}) saved to: {model_best_path}")
    else:
        final_model_path = os.path.join(OUTPUT_DIR, f"{config.MODEL_NAME}_{status}_epoch{NUM_EPOCHS}_final.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Best model not saved (Validation Perceptual Loss did not improve beyond initial {best_val_loss:.4f}).")

    # if 'dataset' in locals() and hasattr(dataset, 'close'): dataset.close()
    # print("Done.")


if __name__ == '__main__':
    main()


#Training DeepDeblur on Perceptual Loss


# LOAD_PRETRAINED = True
# run_validation = True

# model = DeepDeblurMS()
