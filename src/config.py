
import os
import torch


IMG_SIZE = 224
VAL_BATCH_SIZE = 15
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR_DECAY_EPOCHS = 40
LR_FINAL = 2e-5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
NUM_WORKERS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# --- Directories ---
BLUR_ZIP_PATH = '/Users/monikayadav/Downloads/566/DEBLUR/data/REDS/train_ablur.zip'
SHARP_ZIP_PATH = '/Users/monikayadav/Downloads/566/DEBLUR/data/REDS/train_sharp.zip'
VAL_BLUR_ZIP_PATH = '/Users/monikayadav/Downloads/566/DEBLUR/data/REDS/val_blur.zip'
VAL_SHARP_ZIP_PATH = '/Users/monikayadav/Downloads/566/DEBLUR/data/REDS/val_sharp.zip'
TEST_BLUR_ZIP_PATH = '/Users/monikayadav/Downloads/566/DEBLUR/data/REDS/test_blur.zip'
OUTPUT_DIR = '/results/mae_vit_output'

#Pretrained MAE vision transformer for feature extraction
PRETRAINED_WEIGHTS_PATH = '/Users/monikayadav/Downloads/566/DEBLUR/pytorch_model.bin'

CHECKPOINT_DIR = "saved_models/"
RESULTS_DIR = "results/"
OPT = {'image_size': IMG_SIZE, 'pretrain_mae': PRETRAINED_WEIGHTS_PATH, 'device': DEVICE}


# --- Model Configurations for Ablation Study ---

LOAD_MODEL = False
LOAD_PRETRAINED = True
MODEL_NAME = "DeepBlur"  # Options: "Restormer", "DeepBlur"
EFFECTIVE_LR = 5e-5 if LOAD_PRETRAINED and os.path.exists(PRETRAINED_WEIGHTS_PATH) else LEARNING_RATE
SAVE_BEST_PATH = False