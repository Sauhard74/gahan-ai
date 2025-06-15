"""
DETR-style decoder heads for cutting behavior detection.
Includes classification and bounding box regression heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class MLP(nn.Module):
    """Simple multi-layer perceptron."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DETRDecoderLayer(nn.Module):
    """Single DETR decoder layer with self-attention and cross-attention."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries: torch.Tensor, memory: torch.Tensor, 
                query_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            queries: [B, num_queries, hidden_dim] object queries
            memory: [B, T, hidden_dim] encoded sequence features
            query_mask: [B, num_queries] query mask (optional)
            memory_mask: [B, T] memory mask (optional)
            
        Returns:
            [B, num_queries, hidden_dim] updated queries
        """
        # Self-attention
        q_self = self.self_attn(queries, queries, queries, key_padding_mask=query_mask)[0]
        queries = self.self_attn_norm(queries + self.dropout(q_self))
        
        # Cross-attention
        q_cross = self.cross_attn(queries, memory, memory, key_padding_mask=memory_mask)[0]
        queries = self.cross_attn_norm(queries + self.dropout(q_cross))
        
        # Feed-forward
        q_ffn = self.ffn(queries)
        queries = self.ffn_norm(queries + self.dropout(q_ffn))
        
        return queries

class DETRDecoder(nn.Module):
    """DETR decoder with multiple layers."""
    
    def __init__(self, hidden_dim: int, num_layers: int = 6, num_heads: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DETRDecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, queries: torch.Tensor, memory: torch.Tensor,
                query_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            queries: [B, num_queries, hidden_dim] object queries
            memory: [B, T, hidden_dim] encoded sequence features
            query_mask: [B, num_queries] query mask (optional)
            memory_mask: [B, T] memory mask (optional)
            
        Returns:
            [B, num_queries, hidden_dim] decoded features
        """
        for layer in self.layers:
            queries = layer(queries, memory, query_mask, memory_mask)
        
        return self.norm(queries)

class CuttingDetectionHeads(nn.Module):
    """
    Detection heads for cutting behavior detection.
    Includes classification, bounding box regression, and cutting behavior prediction.
    """
    
    def __init__(self, hidden_dim: int, num_classes: int = 4, num_queries: int = 100):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Bounding box regression head
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Cutting behavior head
        self.cutting_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Confidence/objectness head
        self.objectness_embed = nn.Linear(hidden_dim, 1)
        
    def forward(self, decoder_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            decoder_output: [B, num_queries, hidden_dim] decoder features
            
        Returns:
            Dictionary with predictions
        """
        # Classification logits
        class_logits = self.class_head(decoder_output)  # [B, num_queries, num_classes]
        
        # Bounding box coordinates - use center-width-height parameterization
        bbox_raw = self.bbox_embed(decoder_output)  # [B, num_queries, 4]
        
        # Split into center coordinates and dimensions
        center_x = torch.sigmoid(bbox_raw[..., 0])  # [0, 1]
        center_y = torch.sigmoid(bbox_raw[..., 1])  # [0, 1]
        width = torch.sigmoid(bbox_raw[..., 2])     # [0, 1]
        height = torch.sigmoid(bbox_raw[..., 3])    # [0, 1]
        
        # Convert to x1, y1, x2, y2 format ensuring valid boxes
        half_width = width / 2
        half_height = height / 2
        
        x1 = torch.clamp(center_x - half_width, min=0.0, max=1.0)
        y1 = torch.clamp(center_y - half_height, min=0.0, max=1.0)
        x2 = torch.clamp(center_x + half_width, min=0.0, max=1.0)
        y2 = torch.clamp(center_y + half_height, min=0.0, max=1.0)
        
        # Ensure x2 > x1 and y2 > y1 with minimum box size
        min_size = 0.01  # 1% of image size
        x2 = torch.maximum(x2, x1 + min_size)
        y2 = torch.maximum(y2, y1 + min_size)
        
        # Stack to create final bbox coordinates
        bbox_coords = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, num_queries, 4]
        
        # Cutting behavior probability
        cutting_logits = self.cutting_embed(decoder_output)  # [B, num_queries, 1]
        
        # Objectness score
        objectness_logits = self.objectness_embed(decoder_output)  # [B, num_queries, 1]
        
        return {
            'pred_logits': class_logits,
            'pred_boxes': bbox_coords,
            'pred_cutting': cutting_logits,
            'pred_objectness': objectness_logits
        }

class SequenceCuttingHead(nn.Module):
    """
    Sequence-level cutting behavior prediction head.
    Predicts whether any cutting behavior occurs in the entire sequence.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.sequence_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, aggregated_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            aggregated_features: [B, hidden_dim] sequence-level features
            
        Returns:
            [B, 1] sequence cutting probability
        """
        return self.sequence_classifier(aggregated_features)

class CombinedDetectionHeads(nn.Module):
    """
    Combined detection heads for both object-level and sequence-level predictions.
    """
    
    def __init__(self, hidden_dim: int, num_classes: int = 4, num_queries: int = 100):
        super().__init__()
        
        # Object-level detection heads
        self.object_heads = CuttingDetectionHeads(hidden_dim, num_classes, num_queries)
        
        # Sequence-level cutting head
        self.sequence_head = SequenceCuttingHead(hidden_dim)
        
    def forward(self, decoder_output: torch.Tensor, 
                aggregated_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            decoder_output: [B, num_queries, hidden_dim] decoder features
            aggregated_features: [B, hidden_dim] sequence-level features
            
        Returns:
            Dictionary with all predictions
        """
        # Object-level predictions
        object_preds = self.object_heads(decoder_output)
        
        # Sequence-level prediction
        sequence_cutting = self.sequence_head(aggregated_features)
        
        # Combine predictions
        predictions = {
            **object_preds,
            'sequence_cutting': sequence_cutting
        }
        
        return predictions

class AdaptiveQueryGenerator(nn.Module):
    """
    Adaptive query generator that creates object queries based on input features.
    """
    
    def __init__(self, hidden_dim: int, num_queries: int = 100):
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Query adaptation network
        self.query_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, aggregated_features: torch.Tensor) -> torch.Tensor:
        """
        Generate adaptive object queries.
        
        Args:
            aggregated_features: [B, hidden_dim] sequence-level features
            
        Returns:
            [B, num_queries, hidden_dim] object queries
        """
        batch_size = aggregated_features.size(0)
        
        # Get base queries
        query_indices = torch.arange(self.num_queries, device=aggregated_features.device)
        base_queries = self.query_embed(query_indices)  # [num_queries, hidden_dim]
        
        # Expand for batch
        base_queries = base_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Adapt queries based on input features
        adaptation = self.query_adapter(aggregated_features)  # [B, hidden_dim]
        adaptation = adaptation.unsqueeze(1).expand(-1, self.num_queries, -1)
        
        # Combine base queries with adaptation
        adaptive_queries = base_queries + adaptation
        
        return adaptive_queries 