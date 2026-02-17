# Latent-Space Synchronization Network (LSSN)

This repository contains the implementation of the Latent-Space Synchronization Network (LSSN) as proposed in the research project "AI_HONS_PR1".

## Overview

LSSN addresses Cross-Modal Inconsistency in multimodal generative AI by enforcing Latent Trajectory Invariance (LTI). It uses a Synchronization Module (SM) and an Invariance Regularization Loss (Linv) to align latent features conditioned on text and image inputs.

## Project Structure

- `lssn_model.py`: Defines the `LSSN_UNet` architecture.
- `lssn_modules.py`: Contains the `SynchronizationModule` and other custom layers.
- `loss.py`: Implements the `InvarianceLoss` (Linv).
- `train_lssn.py`: Training script with dummy data loaders and the dual-path training loop.

## Installation

1. Install PyTorch:
   ```bash
   pip install torch torchvision
   ```

## Usage

### Training

To run a test training loop with dummy data:

```bash
python train_lssn.py
```

Arguments such as batch size and learning rate can be modified in `train_lssn.py`.

### Architecture Details

- **Dual-Path Conditioning**: The model processes text and image conditions in separate passes (or masked passes) during training to compute the invariance loss.
- **Synchronization Module**: Performs symmetric cross-attention where latent features attend to both text and image embeddings.
- **Invariance Loss**: Computes the L2 distance between intermediate feature maps of the Text-Conditioned path and the Image-Conditioned path.

## Citation

If you use this code for your research, please cite the project "AI_HONS_PR1".
