"""
PredFormer_Triplet_STS Model Architecture

Comments written by DeepSeek AI :)

This implementation provides a transformer-based video encoding model that processes video frames 
using a novel spatial-temporal-spatial (STS) attention mechanism. The model is designed for 
video prediction tasks, where it learns to encode and predict future frames based on input sequences.

Key Components:
1. SwiGLU: A gated linear unit with Swish activation for improved feature transformation
2. GatedTransformer: A transformer block incorporating SwiGLU and attention mechanisms
3. PredFormerLayer: The core STS (Spatial-Temporal-Spatial) attention layer
4. PredFormer_Model: The complete video prediction model architecture

The model processes video frames by:
1. Dividing frames into patches
2. Applying spatial attention to understand intra-frame relationships
3. Applying temporal attention to understand inter-frame relationships
4. Applying spatial attention again to refine features
5. Reconstructing the predicted frames

The architecture is particularly designed for human motion prediction tasks (e.g., Human3.6M dataset).
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat  # For tensor reshaping operations
from einops.layers.torch import Rearrange
import numpy as np
import os
from fvcore.nn import FlopCountAnalysis, flop_count_table  # For FLOPs calculation
from timm.models.layers import DropPath, to_2tuple, trunc_normal_  # From timm library
from modules.PredFormer_modules import Attention, PreNorm, FeedForward  # Custom modules
import math


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) activation block.
    A variant of GLU that uses Swish activation for the gating mechanism.
    
    Args:
        in_features: Input dimension
        hidden_features: Hidden layer dimension (defaults to in_features)
        out_features: Output dimension (defaults to in_features)
        act_layer: Activation layer (default: nn.SiLU/Swish)
        norm_layer: Normalization layer (optional)
        bias: Whether to use bias in linear layers
        drop: Dropout rate
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        # Initialize dimensions
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)  # Separate bias for both linear layers
        drop_probs = to_2tuple(drop)  # Separate dropout for both paths

        # Gating branch
        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        # Value branch
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()  # Swish activation
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        """Initialize weights with specific schemes for better training"""
        nn.init.ones_(self.fc1_g.bias)  # Initialize gate bias to 1
        nn.init.normal_(self.fc1_g.weight, std=1e-6)  # Small initialization for gate weights

    def forward(self, x):
        """Forward pass with gating mechanism"""
        x_gate = self.fc1_g(x)  # Gate path
        x = self.fc1_x(x)  # Value path
        x = self.act(x_gate) * x  # Gating operation (element-wise multiplication)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class GatedTransformer(nn.Module):
    """
    Transformer block with SwiGLU and attention mechanisms.
    Combines self-attention with gated feed-forward network.
    
    Args:
        dim: Input dimension
        depth: Number of transformer blocks
        heads: Number of attention heads
        dim_head: Dimension of each attention head
        mlp_dim: Hidden dimension for MLP
        dropout: Dropout rate
        attn_dropout: Attention dropout rate
        drop_path: Stochastic depth rate
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)  # Final layer normalization
        
        # Create transformer blocks
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),  # Stochastic depth
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
        self.apply(self._init_weights)  # Initialize weights

    def _init_weights(self, m):
        """Weight initialization following ViT conventions"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # Truncated normal initialization
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Zero bias
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)       
            
    def forward(self, x):
        """Forward pass through transformer blocks"""
        for attn, ff, drop_path1, drop_path2 in self.layers:
            # Attention with residual connection and drop path
            x = x + drop_path1(attn(x))
            # Feed forward with residual connection and drop path
            x = x + drop_path2(ff(x))
        return self.norm(x)  # Final normalization


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class PredFormerLayer(nn.Module):
    """
    Core PredFormer layer with Spatial-Temporal-Spatial (STS) attention.
    Processes video data through alternating spatial and temporal attention.
    
    Args:
        dim: Input dimension
        depth: Number of transformer blocks in each branch
        heads: Number of attention heads
        dim_head: Dimension of each attention head
        mlp_dim: Hidden dimension for MLP
        dropout: Dropout rate
        attn_dropout: Attention dropout rate
        drop_path: Stochastic depth rate
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super(PredFormerLayer, self).__init__()

        # Three transformer branches: Spatial -> Temporal -> Spatial
        self.space_transformer_first = GatedTransformer(dim, depth, heads, dim_head, 
                                                   mlp_dim, dropout, attn_dropout, drop_path)
        self.temporal_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                                mlp_dim, dropout, attn_dropout, drop_path)
        self.space_transformer_second = GatedTransformer(dim, depth, heads, dim_head, 
                                                    mlp_dim, dropout, attn_dropout, drop_path)

    def forward(self, x):
        """STS attention forward pass"""
        b, t, n, _ = x.shape  # batch, time, num_patches, dimension
        x_s, x_ori = x, x  # Keep original for residual connection
        
        # First spatial attention branch
        x_s = rearrange(x_s, 'b t n d -> (b t) n d')  # Flatten batch and time
        x_s = self.space_transformer_first(x_s)
        
        # Temporal attention branch
        x_st = rearrange(x_s, '(b t) ... -> b t ...', b=b)  # Unflatten batch
        x_st = x_st.permute(0, 2, 1, 3)  # [b, n, t, d] - prepare for temporal attention      
        x_st = rearrange(x_st, 'b n t d -> (b n) t d')  # Flatten batch and patches
        x_st = self.temporal_transformer(x_st)
        
        # Second spatial attention branch     
        x_st = rearrange(x_st, '(b n) t d -> b n t d', b=b)
        x_st = rearrange(x_st, 'b n t d -> b t n d') 
        
        x_sts = rearrange(x_st, 'b t n d -> (b t) n d') 
        x_sts = self.space_transformer_second(x_sts)

        # Final output     
        x_sts = rearrange(x_sts, '(b t) n d -> b t n d', b=b)
  
        # Residual connection (commented out - only used for Human3.6M)
        # x_sts += x_ori
        
        return x_sts


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sinusoidal_embedding(n_channels, dim):
    """
    Creates sinusoidal positional embeddings.
    Standard transformer positional encoding adapted for video patches.
    
    Args:
        n_channels: Number of positions to encode
        dim: Dimension of the embedding
    Returns:
        Positional embeddings tensor
    """
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])  # Even indices: sine
    pe[:, 1::2] = torch.cos(pe[:, 1::2])  # Odd indices: cosine
    return rearrange(pe, '... -> 1 ...')  # Add batch dimension
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class PredFormer_Model(nn.Module):
    """
    Complete PredFormer model for video prediction.
    Processes input video frames through patch embedding and STS attention layers.
    
    Args:
        in_shape: Input shape tuple (T, C, H, W)
        model_config: Dictionary containing model configuration:
            - patch_size: Size of image patches
            - dim: Embedding dimension
            - heads: Number of attention heads
            - dim_head: Dimension per attention head
            - dropout: Dropout rate
            - attn_dropout: Attention dropout rate
            - drop_path: Stochastic depth rate
            - scale_dim: MLP expansion factor
            - Ndepth: Number of PredFormer layers
            - depth: Depth of each transformer block
    """
    def __init__(self, in_shape, model_config, **kwargs):
        super().__init__()
        # Input configuration
        self.image_height = in_shape[2]
        self.image_width = in_shape[3]
        self.patch_size = model_config['patch_size']
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = in_shape[0]
        self.dim = model_config['dim']
        self.num_channels = in_shape[1]
        self.num_classes = self.num_channels  # For reconstruction
        
        # Attention configuration
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        
        # Regularization
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        
        # Architecture configuration
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']  # Number of PredFormer layers
        self.depth = model_config['depth']  # Depth of each transformer block
        
        # Input validation
        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'
        
        # Patch embedding
        self.patch_dim = self.num_channels * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            # Split into patches and flatten
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', 
                     p1=self.patch_size, p2=self.patch_size),
            # Linear projection to embedding dimension
            nn.Linear(self.patch_dim, self.dim),
        )
        
        # Positional embedding (fixed sinusoidal)
        self.pos_embedding = nn.Parameter(
            sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
            requires_grad=False
        ).view(1, self.num_frames_in, self.num_patches, self.dim)
        
        # Stack of PredFormer layers
        self.blocks = nn.ModuleList([
            PredFormerLayer(self.dim, self.depth, self.heads, self.dim_head, 
                          self.dim * self.scale_dim, self.dropout, 
                          self.attn_dropout, self.drop_path)
            for i in range(self.Ndepth)
        ])
        
        # Reconstruction head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)
        ) 
                
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def forward(self, x):
        """Full model forward pass"""
        B, T, C, H, W = x.shape
        
        # 1. Patch Embedding
        x = self.to_patch_embedding(x)
        
        # 2. Positional Embedding
        x += self.pos_embedding.to(x.device)
        
        # 3. PredFormer Encoder (STS attention layers)
        for blk in self.blocks:
            x = blk(x)
        
        # 4. MLP head for reconstruction        
        x = self.mlp_head(x.reshape(-1, self.dim))
        
        # 5. Reshape back to video format
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        
        return x


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class FlexiblePredFormer(nn.Module):
    """
    Modified PredFormer model with flexible input/output channels.
    
    Args:
        in_shape: Input shape tuple (T, C_in, H, W)
        out_channels: Number of output channels (C_out)
        model_config: Dictionary containing model configuration
    """
    def __init__(self, in_shape, out_channels, model_config, **kwargs):
        super().__init__()
        # Input configuration
        self.image_height = in_shape[2]
        self.image_width = in_shape[3]
        self.patch_size = model_config['patch_size']
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = in_shape[0]
        self.in_channels = in_shape[1]
        self.out_channels = out_channels
        self.dim = model_config['dim']
        
        # Attention configuration
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        
        # Regularization
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        
        # Architecture configuration
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']
        self.depth = model_config['depth']
        
        # Input validation
        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by patch size.'
        
        # ENCODER: Input to latent space
        self.patch_dim_in = self.in_channels * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', 
                     p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim_in, self.dim),
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(
            sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
            requires_grad=False
        ).view(1, self.num_frames_in, self.num_patches, self.dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            PredFormerLayer(self.dim, self.depth, self.heads, self.dim_head, 
                          self.dim * self.scale_dim, self.dropout, 
                          self.attn_dropout, self.drop_path)
            for _ in range(self.Ndepth)
        ])
        
        # DECODER: Latent space to output
        self.patch_dim_out = self.out_channels * self.patch_size ** 2
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_dim_out)
        )
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def forward(self, x):
        """Forward pass with flexible channels"""
        B, T, C_in, H, W = x.shape
        
        # 1. ENCODE: Patch embedding
        x = self.to_patch_embedding(x)
        
        # 2. Add positional embedding
        x += self.pos_embedding.to(x.device)
        
        # 3. Process through STS attention layers
        for blk in self.blocks:
            x = blk(x)
        
        # 4. DECODE: Project to output dimension
        x = self.mlp_head(x.reshape(-1, self.dim))
        
        # 5. Reshape to output video format
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, 
                  self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, self.out_channels, H, W)
        
        return x



# Example usage:
if __name__ == "__main__":
    # Configuration - same as original but with flexible channels
    model_config = {
        'patch_size': 8,
        'dim': 128,
        'heads': 4,
        'dim_head': 32,
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'drop_path': 0.1,
        'scale_dim': 4,
        'Ndepth': 3,
        'depth': 2
    }
    
    # Example with 3 input channels and 1 output channel
    in_shape = (10, 3, 256, 256)  # (T, C_in, H, W)
    out_channels = 1
    
    model = FlexiblePredFormer(in_shape, out_channels, model_config)
    
    # Test forward pass
    x = torch.randn(2, 10, 3, 256, 256)  # (batch, time, C_in, H, W)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # Should be (2, 10, 1, 256, 256)