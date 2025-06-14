"""
Bidirectional GRU Temporal Encoder with Multi-Head Attention.
Optimized for cutting behavior detection across 5-frame sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for temporal modeling."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, hidden_dim] input sequences
            mask: [B, T] attention mask (optional)
            
        Returns:
            [B, T, hidden_dim] attended features
        """
        batch_size, seq_len, _ = x.size()
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output_proj(attended)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""
    
    def __init__(self, hidden_dim: int, max_len: int = 10):
        super().__init__()
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, hidden_dim] input sequences
            
        Returns:
            [B, T, hidden_dim] position-encoded sequences
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class BidirectionalGRUEncoder(nn.Module):
    """
    Bidirectional GRU encoder with multi-head attention.
    Optimized for 5-frame temporal sequences.
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 2, num_heads: int = 8, 
                 dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Divide by 2 for bidirectional
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Multi-head attention
        if use_attention:
            self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through bidirectional GRU encoder.
        
        Args:
            x: [B, T, hidden_dim] input sequences
            mask: [B, T] sequence mask (optional)
            
        Returns:
            Tuple of:
            - [B, T, hidden_dim] sequence features
            - [B, hidden_dim] aggregated features
        """
        batch_size, seq_len, _ = x.size()
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Bidirectional GRU
        gru_out, hidden = self.gru(x)  # [B, T, hidden_dim], [2*num_layers, B, hidden_dim//2]
        
        # Apply dropout
        gru_out = self.dropout(gru_out)
        
        # Multi-head attention (if enabled)
        if self.use_attention:
            # Self-attention with residual connection
            attn_out = self.attention(gru_out, mask)
            gru_out = self.attention_norm(gru_out + attn_out)
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(gru_out)
        sequence_features = self.ffn_norm(gru_out + ffn_out)
        
        # Aggregate sequence features
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(sequence_features)
            masked_features = sequence_features * mask_expanded
            aggregated = masked_features.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            # Simple average pooling
            aggregated = sequence_features.mean(dim=1)
        
        # Final projection
        aggregated = self.output_proj(aggregated)
        
        return sequence_features, aggregated

class TemporalTransformer(nn.Module):
    """
    Transformer-based temporal encoder as an alternative to GRU.
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 3, num_heads: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer encoder.
        
        Args:
            x: [B, T, hidden_dim] input sequences
            mask: [B, T] sequence mask (optional)
            
        Returns:
            Tuple of:
            - [B, T, hidden_dim] sequence features
            - [B, hidden_dim] aggregated features
        """
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask for transformer
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # Transformer uses True for masked positions
        
        # Pass through transformer
        sequence_features = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Aggregate sequence features
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(sequence_features)
            masked_features = sequence_features * mask_expanded
            aggregated = masked_features.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            # Simple average pooling
            aggregated = sequence_features.mean(dim=1)
        
        # Final projection
        aggregated = self.output_proj(aggregated)
        
        return sequence_features, aggregated

class HybridTemporalEncoder(nn.Module):
    """
    Hybrid encoder combining GRU and Transformer for best performance.
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 2, num_heads: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        
        # GRU encoder
        self.gru_encoder = BidirectionalGRUEncoder(
            hidden_dim, num_layers, num_heads, dropout, use_attention=False
        )
        
        # Transformer encoder
        self.transformer_encoder = TemporalTransformer(
            hidden_dim, num_layers=1, num_heads=num_heads, dropout=dropout
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through hybrid encoder.
        
        Args:
            x: [B, T, hidden_dim] input sequences
            mask: [B, T] sequence mask (optional)
            
        Returns:
            Tuple of:
            - [B, T, hidden_dim] sequence features
            - [B, hidden_dim] aggregated features
        """
        # GRU features
        gru_seq, gru_agg = self.gru_encoder(x, mask)
        
        # Transformer features
        trans_seq, trans_agg = self.transformer_encoder(x, mask)
        
        # Fuse aggregated features
        fused_agg = torch.cat([gru_agg, trans_agg], dim=1)
        final_agg = self.fusion(fused_agg)
        
        # Use GRU sequence features (could also fuse these)
        return gru_seq, final_agg 