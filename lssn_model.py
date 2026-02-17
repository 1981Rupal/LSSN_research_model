import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lssn_modules import BasicTransformerBlock

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = h + self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)

class SpatialTransformer(nn.Module):
    def __init__(self, channels, num_heads, dim_head, depth=1, dropout=0.0, context_dim=768):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # Using BasicTransformerBlock which contains SynchronizationModule
            self.layers.append(BasicTransformerBlock(channels, num_heads, dim_head, dropout, context_dim=context_dim))
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False) 

    def forward(self, x, c_text, c_image):
        b, c, h, w = x.shape
        x_in = x
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        
        for layer in self.layers:
            x = layer(x, c_text, c_image) 
            
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x_in + self.proj_out(x)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
    def forward(self, x):
        return self.op(x)

class LSSN_UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, model_channels=320, num_res_blocks=2, 
                 channel_mult=(1, 2, 4), num_heads=8, dim_head=64, context_dim=768):
        super().__init__()
        self.context_dim = context_dim
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # Downsampling
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResnetBlock(ch, mult * model_channels, time_embed_dim)]
                ch = mult * model_channels
                if ds in [2, 4, 8]:
                   layers.append(SpatialTransformer(ch, num_heads, dim_head, context_dim=context_dim))
                
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch))
                input_block_chans.append(ch)
                ds *= 2

        # Middle
        self.middle_block = nn.ModuleList([
            ResnetBlock(ch, ch, time_embed_dim),
            SpatialTransformer(ch, num_heads, dim_head, context_dim=context_dim),
            ResnetBlock(ch, ch, time_embed_dim)
        ])

        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResnetBlock(ch + ich, mult * model_channels, time_embed_dim)]
                ch = mult * model_channels
                if ds in [2, 4, 8]:
                    layers.append(SpatialTransformer(ch, num_heads, dim_head, context_dim=context_dim))
                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(nn.ModuleList(layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, c_text, c_image, return_features=False):
        """
        x: (B, C, H, W)
        timesteps: (B,)
        c_text: (B, Seq, Context_Dim)
        c_image: (B, Seq, Context_Dim)
        return_features: If True, returns a list of intermediate features from SpatialTransformers
        """
        t_emb = self.time_embed(self.get_timestep_embedding(timesteps, self.input_blocks[0].out_channels))
        
        hs = []
        h = x
        features = []
        
        # Input Blocks
        for module in self.input_blocks:
            if isinstance(module, nn.Conv2d):
                h = module(h)
            elif isinstance(module, Downsample):
                h = module(h)
            elif isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResnetBlock):
                        h = layer(h, t_emb)
                    elif isinstance(layer, SpatialTransformer):
                        h = layer(h, c_text, c_image)
                        if return_features:
                            features.append(h)
            hs.append(h)
            
        # Middle Block
        for module in self.middle_block:
             if isinstance(module, ResnetBlock):
                 h = module(h, t_emb)
             elif isinstance(module, SpatialTransformer):
                 h = module(h, c_text, c_image)
                 if return_features:
                     features.append(h)
        
        # Output Blocks
        for module in self.output_blocks:
             h_cat = hs.pop()
             h = torch.cat([h, h_cat], dim=1)
             for layer in module:
                 if isinstance(layer, ResnetBlock):
                     h = layer(h, t_emb)
                 elif isinstance(layer, SpatialTransformer):
                     h = layer(h, c_text, c_image)
                     if return_features:
                         features.append(h)
                 elif isinstance(layer, Upsample):
                     h = layer(h)

        output = self.out(h)
        if return_features:
            return output, features
        return output

    def get_timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
