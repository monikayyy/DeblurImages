import argparse, glob, os, random
from pathlib import Path
from PIL import Image
import config

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PairDataset(Dataset):
    """
    Generic loader for <root>/<split>/<blur|sharp>/*.png
    Works for both train and val because transforms are injected.
    """
    def __init__(self, root, split, transform=None):
        self.root_dir = os.path.join(root, split)
        self.blur_dir  = os.path.join(self.root_dir, "blur")
        self.sharp_dir = os.path.join(self.root_dir, "sharp")

        search_pattern = os.path.join(self.blur_dir, '**', '*.png')
        self.blur_paths = sorted(glob.glob(search_pattern, recursive=True))
        self.transform = transform

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        blur_path = self.blur_paths[idx]
        
        # Create the sharp path by replacing the 'blur' part of the full path
        sharp_path = blur_path.replace(os.sep + 'blur' + os.sep, os.sep + 'sharp' + os.sep)

        if not os.path.exists(sharp_path):
            raise FileNotFoundError(f"Sharp image not found for blur image: {blur_path}\nLooked for: {sharp_path}")
            
        blur_img  = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")
        
        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
            
        return blur_img, sharp_img
    
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if hasattr(dataset, '_open_zips'):
         dataset._open_zips()



print("DataLoader created.")

def get_data_loaders():
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
                ])
    
    root = "/Users/monikayadav/Downloads/566/DEBLUR/data/"
    train_ds = PairDataset(root, 'train', transform=transform)
    val_ds   = PairDataset(root, 'val',   transform=transform)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")

    train_loader = DataLoader(
                    train_ds,
                    batch_size=config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=config.NUM_WORKERS,
                    # pin_memory=True, # Use True if DEVICE is 'cuda'
                    drop_last=True,
                    worker_init_fn=worker_init_fn
                    # persistent_workers=True if config.NUM_WORKERS > 0 else False # Can speed up epoch start
                )
    val_loader = DataLoader(
                        val_ds,
                        batch_size=config.VAL_BATCH_SIZE,
                        shuffle=True,
                        num_workers=config.NUM_WORKERS,
                        # pin_memory=True, # Use True if DEVICE is 'cuda'
                        drop_last=True,
                        worker_init_fn=worker_init_fn
                        # persistent_workers=True if config.NUM_WORKERS > 0 else False # Can speed up epoch start
                    )
    return train_loader, val_loader
