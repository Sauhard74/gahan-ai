"""
Vision Transformer (ViT-Large) backbone with spatial attention.
Optimized for cutting behavior detection with ROI focus.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from typing import Tuple, Optional

class SpatialAttention(nn.Module):
    """Spatial attention module to focus on ROI regions."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature maps
            
        Returns:
            Attention-weighted features
        """
        b, c, h, w = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        # Apply attention
        out = x * channel_att.expand_as(x)
        
        return out

class ViTBackbone(nn.Module):
    """
    ViT-Large backbone with spatial attention for cutting detection.
    """
    
    def __init__(self, model_name: str = "google/vit-large-patch16-224", 
                 hidden_dim: int = 768, freeze_backbone: bool = False):
        super().__init__()
        
        # Load pre-trained ViT-Large
        self.vit = ViTModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Get ViT output dimension
        vit_dim = self.vit.config.hidden_size  # 1024 for ViT-Large
        
        # Projection layer to match hidden_dim
        self.projection = nn.Linear(vit_dim, hidden_dim)
        
        # Spatial attention for ROI focus
        self.spatial_attention = SpatialAttention(hidden_dim)
        
        # Feature enhancement layers
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Position embedding for patch features
        self.patch_size = 16  # ViT patch size
        self.num_patches = (224 // self.patch_size) ** 2  # 196 patches for 224x224
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT backbone.
        
        Args:
            x: [B, C, H, W] input images
            
        Returns:
            [B, hidden_dim] feature vectors
        """
        batch_size = x.size(0)
        
        # Pass through ViT
        outputs = self.vit(pixel_values=x, output_hidden_states=True)
        
        # Get the last hidden state [B, num_patches + 1, vit_dim]
        # +1 for CLS token
        hidden_states = outputs.last_hidden_state
        
        # Extract CLS token (first token)
        cls_token = hidden_states[:, 0]  # [B, vit_dim]
        
        # Project to target dimension
        features = self.projection(cls_token)  # [B, hidden_dim]
        
        # Enhance features
        enhanced_features = self.feature_enhancer(features)
        
        # Add residual connection
        output_features = features + enhanced_features
        
        return output_features
    
    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features for spatial attention.
        
        Args:
            x: [B, C, H, W] input images
            
        Returns:
            [B, num_patches, hidden_dim] patch features
        """
        batch_size = x.size(0)
        
        # Pass through ViT
        outputs = self.vit(pixel_values=x, output_hidden_states=True)
        
        # Get patch features (excluding CLS token)
        patch_features = outputs.last_hidden_state[:, 1:]  # [B, num_patches, vit_dim]
        
        # Project to target dimension
        projected_patches = self.projection(patch_features)  # [B, num_patches, hidden_dim]
        
        return projected_patches
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get 2D feature maps for spatial processing.
        
        Args:
            x: [B, C, H, W] input images
            
        Returns:
            [B, hidden_dim, H', W'] feature maps
        """
        batch_size = x.size(0)
        
        # Extract patch features
        patch_features = self.extract_patch_features(x)  # [B, num_patches, hidden_dim]
        
        # Reshape to 2D feature maps
        h = w = int(self.num_patches ** 0.5)  # 14x14 for 224x224 input
        feature_maps = patch_features.transpose(1, 2).view(batch_size, self.hidden_dim, h, w)
        
        # Apply spatial attention
        attended_features = self.spatial_attention(feature_maps)
        
        return attended_features

class MultiScaleViTBackbone(nn.Module):
    """
    Multi-scale ViT backbone for better feature extraction.
    """
    
    def __init__(self, model_name: str = "google/vit-large-patch16-224", 
                 hidden_dim: int = 768):
        super().__init__()
        
        self.backbone = ViTBackbone(model_name, hidden_dim)
        
        # Multi-scale feature fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale feature extraction.
        
        Args:
            x: [B, C, H, W] input images
            
        Returns:
            [B, hidden_dim] fused features
        """
        # Original scale features
        original_features = self.backbone(x)
        
        # Downsampled features (simulate different scale)
        downsampled = F.interpolate(x, scale_factor=0.75, mode='bilinear', align_corners=False)
        downsampled_features = self.backbone(downsampled)
        
        # Fuse features
        fused_features = torch.cat([original_features, downsampled_features], dim=1)
        output = self.scale_fusion(fused_features)
        
        return output 