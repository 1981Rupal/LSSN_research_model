import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SynchronizationModule(nn.Module):
    """
    Synchronization Module (SM) for LSSN.
    Intergrates information from both text and image modalities into the latent features.
    It performs symmetric cross-attention where the latent features attend to both
    text and image conditioning inputs.
    """
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.0, context_dim=768):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_text = nn.Linear(context_dim, inner_dim, bias=False) 
        self.to_v_text = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_image = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_image = nn.Linear(context_dim, inner_dim, bias=False)

        # Learnable gating parameter for Gated Fusion
        self.gate = nn.Parameter(torch.tensor([0.0])) 

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, c_text, c_image):
        """
        x: Latent features (Batch, Sequence/Pixels, Dim)
        c_text: Text conditioning (Batch, Seq_len, Context_Dim)
        c_image: Image conditioning (Batch, Seq_len, Context_Dim)
        """
        h = self.num_heads
        q = self.to_q(x)

        # Cross Attention with Text
        k_text = self.to_k_text(c_text)
        v_text = self.to_v_text(c_text)
        
        # Cross Attention with Image
        k_image = self.to_k_image(c_image)
        v_image = self.to_v_image(c_image)

        # Split heads
        q, k_text, v_text, k_image, v_image = map(
            lambda t: t.view(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2),
            (q, k_text, v_text, k_image, v_image)
        )

        # Attention text
        dots_text = (q @ k_text.transpose(-2, -1)) * self.scale
        attn_text = dots_text.softmax(dim=-1)
        out_text = attn_text @ v_text

        # Attention image
        dots_image = (q @ k_image.transpose(-2, -1)) * self.scale
        attn_image = dots_image.softmax(dim=-1)
        out_image = attn_image @ v_image

        # Symmetric combination with Learnable Gated Fusion
        # Innovative: Learn to weight modalities dynamically to minimize bias.
        alpha = torch.sigmoid(self.gate) 
        out = alpha * out_text + (1 - alpha) * out_image

        # Reshape and project out
        out = out.transpose(1, 2).reshape(out.shape[0], -1, out.shape[-1] * h)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_head, dropout=0.0, context_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SynchronizationModule(dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout, context_dim=context_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * 4, dropout=dropout)

    def forward(self, x, c_text, c_image):
        # Apply SM
        x = x + self.attn(self.norm1(x), c_text, c_image)
        x = x + self.ff(self.norm2(x))
        return x
