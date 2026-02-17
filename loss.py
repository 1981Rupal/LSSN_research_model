import torch
import torch.nn as nn
import torch.nn.functional as F

class InvarianceLoss(nn.Module):
    """
    Invariance Regularization Loss (Linv).
    Computes a hybrid loss combining L2 distance and Cosine Similarity
    between latent feature maps from the text path and the image path.
    Innovation: Ensures both magnitude and direction alignment in latent space.
    Shape of features: List of (B, C, H, W) tensors.
    """
    def __init__(self, lambda_inv=1.0, lambda_cosine=1.0):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_cosine = lambda_cosine

    def forward(self, features_text, features_image):
        """
        features_text: List of feature maps from the text-conditioned pass.
        features_image: List of feature maps from the image-conditioned pass.
        Returns: Scalar loss.
        """
        mse_loss = 0.0
        cosine_loss = 0.0
        
        # Ensure we have matching features
        if len(features_text) != len(features_image):
            raise ValueError("Feature lists from text and image paths must have the same length.")

        for ft, fi in zip(features_text, features_image):
            # L2 distance (MSE)
            mse_loss += F.mse_loss(ft, fi)
            
            # Cosine Direction Loss
            # Flatten spatial dimensions to (B, C, H*W) or keep (B, C, H, W)
            # F.cosine_similarity works on a dim. Let's align channel vectors at each spatial location.
            # features: (B, C, H, W) -> similarity along dim 1 (C) -> (B, H, W) -> mean
            sim = F.cosine_similarity(ft, fi, dim=1).mean()
            cosine_loss += (1.0 - sim)

        total_loss = (self.lambda_inv * mse_loss) + (self.lambda_cosine * cosine_loss)
        return total_loss
