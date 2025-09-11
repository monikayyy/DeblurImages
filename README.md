# Perceptual-Loss–Driven Deblurring
Experiments & code for adding a ViT–based local perceptual loss to lightweight and transformer-based image-deblurring networks.

***

## What’s here?
This sub-repo contains everything I did for the **Perceptual Loss** study that became part of our larger video-restoration project:

1. Implemented a **local MAE perceptual loss** (LPL) that mixes  
   -  L1 pixel loss  
   -  MAE computed on Vision-Transformer (ViT) feature tokens  
2. Integrated the loss into  
   -  DeepDeblur (≈1 M params)  
   -  Restormer (Zamir et al., 2022)  
3. Re-trained both networks on the **REDS** dataset and logged PSNR / SSIM / LPIPS.  
4. Analyzed the trade-off between distortion metrics and perceptual quality.

***

## Motivation
Traditional ℓ1 / ℓ2 losses favor pixel-wise fidelity but often oversmooth textures.  
Following Zamir et al. (Restormer), I compare blurred & restored images in a **representation space** learned by an ImageNet-pre-trained ViT (ViT-B/16). By minimizing token-wise MAE on 8×8 local windows **plus** the usual L1, the networks are nudged toward structure- and semantics-aware reconstructions.

***

## 📊 Main Results (REDS val set)

| Model | Loss | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|------|--------|--------|---------|
| DeepDeblur | L1 | 31.2150 | **0.9446** | **0.0547** |
| DeepDeblur | L1 + ViT-LPL | **31.5268** | 0.9357 | 0.0964 |
| Restormer  | L1 + ViT-LPL | 32.3452 | 0.9394 | 0.0882 |

Take-aways  
-  Perceptual loss boosts PSNR and visual realism, but slightly harms SSIM.  
-  Lightweight DeepDeblur benefited more in LPIPS than heavy Restormer in this setting.  

***
***

## License
MIT – feel free to build on top, but please give credit.
