"""
Main Cutting Detector Model.
Combines ViT backbone, GRU temporal encoder, and DETR decoder for cutting behavior detection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .vit_backbone import ViTBackbone, MultiScaleViTBackbone
from .temporal_encoder import BidirectionalGRUEncoder, HybridTemporalEncoder
from .detr_decoder_heads import (
    DETRDecoder, CombinedDetectionHeads, AdaptiveQueryGenerator
)

class CuttingDetector(nn.Module):
    """
    Main model for cutting behavior detection.
    
    Architecture:
    1. ViT-Large backbone for frame-level feature extraction
    2. Bidirectional GRU + Attention for temporal modeling
    3. DETR-style decoder for object detection and cutting classification
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract config parameters
        self.hidden_dim = config.get('hidden_dim', 768)
        self.num_classes = config.get('num_classes', 4)
        self.sequence_length = config.get('sequence_length', 5)
        self.num_queries = config.get('num_queries', 100)
        
        # Backbone configuration
        backbone_name = config.get('backbone', 'google/vit-large-patch16-224')
        use_multiscale = config.get('use_multiscale_backbone', False)
        
        # Temporal encoder configuration
        temporal_config = config.get('temporal_encoder', {})
        self.temporal_type = temporal_config.get('type', 'bidirectional_gru')
        
        # Decoder configuration
        decoder_config = config.get('decoder', {})
        
        # Initialize backbone
        if use_multiscale:
            self.backbone = MultiScaleViTBackbone(backbone_name, self.hidden_dim)
        else:
            self.backbone = ViTBackbone(backbone_name, self.hidden_dim)
        
        # Initialize temporal encoder
        if self.temporal_type == 'bidirectional_gru':
            self.temporal_encoder = BidirectionalGRUEncoder(
                hidden_dim=self.hidden_dim,
                num_layers=temporal_config.get('num_layers', 2),
                num_heads=temporal_config.get('attention_heads', 8),
                dropout=temporal_config.get('dropout', 0.1),
                use_attention=temporal_config.get('use_attention', True)
            )
        elif self.temporal_type == 'hybrid':
            self.temporal_encoder = HybridTemporalEncoder(
                hidden_dim=self.hidden_dim,
                num_layers=temporal_config.get('num_layers', 2),
                num_heads=temporal_config.get('attention_heads', 8),
                dropout=temporal_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown temporal encoder type: {self.temporal_type}")
        
        # Initialize DETR decoder
        self.decoder = DETRDecoder(
            hidden_dim=self.hidden_dim,
            num_layers=decoder_config.get('num_layers', 6),
            num_heads=decoder_config.get('num_heads', 8),
            dropout=decoder_config.get('dropout', 0.1)
        )
        
        # Initialize query generator
        self.query_generator = AdaptiveQueryGenerator(
            hidden_dim=self.hidden_dim,
            num_queries=self.num_queries
        )
        
        # Initialize detection heads
        self.detection_heads = CombinedDetectionHeads(
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_queries=self.num_queries
        )
        
        # Class mapping
        self.class_names = ['Background', 'Car', 'MotorBike', 'EgoVehicle']
        
    def forward(self, images: torch.Tensor, 
                targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the cutting detector.
        
        Args:
            images: [B, T, C, H, W] sequence of images
            targets: List of target dictionaries (for training)
            
        Returns:
            Dictionary with predictions
        """
        batch_size, seq_len, channels, height, width = images.shape
        
        # Reshape for backbone processing
        images_flat = images.view(batch_size * seq_len, channels, height, width)
        
        # Extract frame-level features
        frame_features = self.backbone(images_flat)  # [B*T, hidden_dim]
        
        # Reshape back to sequence format
        sequence_features = frame_features.view(batch_size, seq_len, self.hidden_dim)
        
        # Temporal encoding
        temporal_features, aggregated_features = self.temporal_encoder(sequence_features)
        
        # Generate adaptive object queries
        object_queries = self.query_generator(aggregated_features)
        
        # DETR decoding
        decoded_features = self.decoder(object_queries, temporal_features)
        
        # Generate predictions
        predictions = self.detection_heads(decoded_features, aggregated_features)
        
        return predictions
    
    def predict(self, images: torch.Tensor, 
                confidence_threshold: float = 0.3,
                iou_threshold: float = 0.5) -> List[Dict]:
        """
        Generate predictions for inference.
        
        Args:
            images: [B, T, C, H, W] sequence of images
            confidence_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of prediction dictionaries
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(images)
            
            # Post-process predictions
            predictions = self._post_process_predictions(
                outputs, confidence_threshold, iou_threshold
            )
        
        return predictions
    
    def _post_process_predictions(self, outputs: Dict[str, torch.Tensor],
                                confidence_threshold: float,
                                iou_threshold: float) -> List[Dict]:
        """
        Post-process model outputs to generate final predictions.
        
        Args:
            outputs: Model outputs dictionary
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of processed predictions
        """
        batch_size = outputs['pred_logits'].size(0)
        predictions = []
        
        for i in range(batch_size):
            # Get predictions for this sample
            logits = outputs['pred_logits'][i]  # [num_queries, num_classes]
            boxes = outputs['pred_boxes'][i]    # [num_queries, 4]
            cutting_logits = outputs['pred_cutting'][i]  # [num_queries, 1]
            objectness = outputs['pred_objectness'][i]   # [num_queries, 1]
            
            # Convert logits to probabilities
            class_probs = torch.softmax(logits, dim=-1)
            cutting_probs = torch.sigmoid(cutting_logits)
            objectness_probs = torch.sigmoid(objectness)
            
            # Get class predictions (excluding background)
            max_probs, class_ids = class_probs[:, 1:].max(dim=-1)
            class_ids += 1  # Adjust for background class
            
            # Combine with objectness for final confidence
            confidence_scores = max_probs * objectness_probs.squeeze(-1)
            
            # Filter by confidence threshold
            confident_mask = confidence_scores >= confidence_threshold
            
            if confident_mask.sum() == 0:
                # No confident predictions
                predictions.append({
                    'boxes': [],
                    'labels': [],
                    'scores': [],
                    'cutting_probs': [],
                    'has_cutting': False
                })
                continue
            
            # Filter predictions
            filtered_boxes = boxes[confident_mask]
            filtered_labels = class_ids[confident_mask]
            filtered_scores = confidence_scores[confident_mask]
            filtered_cutting = cutting_probs[confident_mask].squeeze(-1)
            
            # Apply NMS
            keep_indices = self._apply_nms(filtered_boxes, filtered_scores, iou_threshold)
            
            final_boxes = filtered_boxes[keep_indices]
            final_labels = filtered_labels[keep_indices]
            final_scores = filtered_scores[keep_indices]
            final_cutting = filtered_cutting[keep_indices]
            
            # Determine if sequence has cutting behavior
            sequence_cutting_prob = torch.sigmoid(outputs['sequence_cutting'][i]).item()
            has_cutting = sequence_cutting_prob > 0.5 or final_cutting.max().item() > 0.5
            
            predictions.append({
                'boxes': final_boxes.cpu().numpy().tolist(),
                'labels': final_labels.cpu().numpy().tolist(),
                'scores': final_scores.cpu().numpy().tolist(),
                'cutting_probs': final_cutting.cpu().numpy().tolist(),
                'has_cutting': has_cutting,
                'sequence_cutting_prob': sequence_cutting_prob
            })
        
        return predictions
    
    def _apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor, 
                   iou_threshold: float) -> torch.Tensor:
        """
        Apply Non-Maximum Suppression.
        
        Args:
            boxes: [N, 4] bounding boxes
            scores: [N] confidence scores
            iou_threshold: IoU threshold
            
        Returns:
            Indices of boxes to keep
        """
        from torchvision.ops import nms
        
        # Convert normalized coordinates to absolute coordinates for NMS
        # Assuming input image size of 224x224 (ViT input size)
        abs_boxes = boxes * 224
        
        # Apply NMS
        keep_indices = nms(abs_boxes, scores, iou_threshold)
        
        return keep_indices
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'CuttingDetector',
            'backbone': type(self.backbone).__name__,
            'temporal_encoder': type(self.temporal_encoder).__name__,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'num_queries': self.num_queries,
            'sequence_length': self.sequence_length,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'class_names': self.class_names
        }

def create_cutting_detector(config: Dict) -> CuttingDetector:
    """
    Factory function to create a cutting detector model.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        CuttingDetector model
    """
    model = CuttingDetector(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    
    # Apply weight initialization (excluding pre-trained ViT)
    model.temporal_encoder.apply(init_weights)
    model.decoder.apply(init_weights)
    model.query_generator.apply(init_weights)
    model.detection_heads.apply(init_weights)
    
    return model

# Backward compatibility alias
CuttingBehaviorModel = CuttingDetector 