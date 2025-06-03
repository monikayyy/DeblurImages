
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as config
from models.DeepDeblurMS import DeepDeblurMS


# --- Factory function to get the model ---
def get_model(model_name):
    """Returns the specified model based on the name."""
    if model_name.lower() == "restormer":
        print("==> Initializing Restormer model")
        # return Restormer(**config.RESTORMER_CONFIG)
    elif model_name.lower() == "deepblur":
        print("==> Initializing DeepBlur model")
        return DeepDeblurMS()
    # Add other models for your ablation study here
    # elif model_name.lower() == "vit_deblur":
    #     return ViTDeblur(**config.VIT_DEBLUR_CONFIG)
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

