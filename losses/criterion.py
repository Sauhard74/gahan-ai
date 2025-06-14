"""
Comprehensive loss function for cutting behavior detection.
Includes focal loss, GIoU loss, and 10x false negative penalty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from utils.hungarian_matcher import HungarianMatcher

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for bounding box regression.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: [N, 4] predicted boxes in format [x1, y1, x2, y2]
            target_boxes: [N, 4] target boxes in format [x1, y1, x2, y2]
            
        Returns:
            GIoU loss
        """
        # Calculate IoU
        iou, union = self._box_iou(pred_boxes, target_boxes)
        
        # Calculate enclosing box
        lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        enclose_area = wh[:, 0] * wh[:, 1]
        
        # Calculate GIoU
        giou = iou - (enclose_area - union) / enclose_area
        
        # GIoU loss
        loss = 1 - giou
        
        return loss.mean()
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate IoU between boxes."""
        area1 = self._box_area(boxes1)
        area2 = self._box_area(boxes2)
        
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        union = area1 + area2 - inter
        iou = inter / union
        
        return iou, union
    
    def _box_area(self, boxes: torch.Tensor) -> torch.Tensor:
        """Calculate box areas."""
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy with 10x penalty for false negatives.
    """
    
    def __init__(self, false_negative_penalty: float = 10.0):
        super().__init__()
        self.false_negative_penalty = false_negative_penalty
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N] predicted probabilities (after sigmoid)
            targets: [N] binary targets (0 or 1)
            
        Returns:
            Weighted BCE loss
        """
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Apply penalty for false negatives
        # False negative: target=1, prediction<0.5
        false_negative_mask = (targets == 1) & (inputs < 0.5)
        
        # Apply penalty
        weighted_loss = bce_loss.clone()
        weighted_loss[false_negative_mask] *= self.false_negative_penalty
        
        return weighted_loss.mean()

class CuttingDetectionLoss(nn.Module):
    """
    Combined loss function for cutting behavior detection.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Loss weights
        self.weight_class = config.get('classification', 1.0)
        self.weight_bbox = config.get('bbox_regression', 5.0)
        self.weight_giou = config.get('giou', 2.0)
        self.weight_cutting = config.get('cutting', 3.0)
        self.weight_sequence = config.get('sequence', 2.0)
        
        # False negative penalty
        self.false_negative_penalty = config.get('false_negative_penalty', 10.0)
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.giou_loss = GIoULoss()
        self.weighted_bce = WeightedBCELoss(self.false_negative_penalty)
        
        # Hungarian matcher for training
        self.matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=5.0,
            cost_giou=2.0
        )
        
        # Class weights for imbalanced dataset
        self.register_buffer('class_weights', torch.tensor([0.1, 1.0, 1.0, 1.0]))  # Lower weight for background
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute the losses.
        
        Args:
            outputs: Model outputs dictionary
            targets: List of target dictionaries
            
        Returns:
            Dictionary of losses
        """
        # Check if we have any valid targets
        has_valid_targets = any(len(t['labels']) > 0 for t in targets)
        
        if not has_valid_targets:
            # Return zero losses if no valid targets
            device = outputs['pred_logits'].device
            return {
                'loss': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_class': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_bbox': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_giou': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_cutting': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_sequence': torch.tensor(0.0, device=device, requires_grad=True)
            }
        
        # Hungarian matching for training
        try:
            indices = self.matcher(outputs, targets)
        except Exception as e:
            print(f"Warning: Hungarian matcher failed: {e}")
            # Return zero losses if matching fails
            device = outputs['pred_logits'].device
            return {
                'loss': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_class': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_bbox': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_giou': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_cutting': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_sequence': torch.tensor(0.0, device=device, requires_grad=True)
            }
        
        # Compute individual losses
        loss_class = self._classification_loss(outputs, targets, indices)
        loss_bbox = self._bbox_regression_loss(outputs, targets, indices)
        loss_giou = self._giou_loss(outputs, targets, indices)
        loss_cutting = self._cutting_loss(outputs, targets, indices)
        loss_sequence = self._sequence_cutting_loss(outputs, targets)
        
        # Ensure all losses are positive
        loss_class = torch.clamp(loss_class, min=0.0)
        loss_bbox = torch.clamp(loss_bbox, min=0.0)
        loss_giou = torch.clamp(loss_giou, min=0.0)
        loss_cutting = torch.clamp(loss_cutting, min=0.0)
        loss_sequence = torch.clamp(loss_sequence, min=0.0)
        
        # Combine losses
        total_loss = (
            self.weight_class * loss_class +
            self.weight_bbox * loss_bbox +
            self.weight_giou * loss_giou +
            self.weight_cutting * loss_cutting +
            self.weight_sequence * loss_sequence
        )
        
        return {
            'loss': total_loss,
            'loss_class': loss_class,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
            'loss_cutting': loss_cutting,
            'loss_sequence': loss_sequence
        }
    
    def _classification_loss(self, outputs: Dict, targets: List[Dict], 
                           indices: List[Tuple]) -> torch.Tensor:
        """Compute classification loss."""
        pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes]
        
        # Get target classes
        target_classes = []
        for i, (src_idx, tgt_idx) in enumerate(indices):
            target_classes_i = torch.full(
                (pred_logits.shape[1],), 0, dtype=torch.int64, device=pred_logits.device
            )
            if len(tgt_idx) > 0:
                target_classes_i[src_idx] = targets[i]['labels'][tgt_idx]
            target_classes.append(target_classes_i)
        
        target_classes = torch.stack(target_classes)  # [B, num_queries]
        
        # Flatten for loss computation
        pred_logits_flat = pred_logits.view(-1, pred_logits.shape[-1])
        target_classes_flat = target_classes.view(-1)
        
        # Apply class weights
        class_weights = self.class_weights.to(pred_logits.device)
        
        # Focal loss with class weights
        loss = F.cross_entropy(
            pred_logits_flat, target_classes_flat, 
            weight=class_weights, reduction='mean'
        )
        
        return loss
    
    def _bbox_regression_loss(self, outputs: Dict, targets: List[Dict], 
                            indices: List[Tuple]) -> torch.Tensor:
        """Compute bounding box regression loss."""
        pred_boxes = outputs['pred_boxes']  # [B, num_queries, 4]
        
        # Get matched predictions and targets
        src_boxes = []
        target_boxes = []
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_boxes.append(pred_boxes[i][src_idx])
                target_boxes.append(targets[i]['boxes'][tgt_idx])
        
        if len(src_boxes) == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        
        src_boxes = torch.cat(src_boxes, dim=0)
        target_boxes = torch.cat(target_boxes, dim=0)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='mean')
        
        return loss_bbox
    
    def _giou_loss(self, outputs: Dict, targets: List[Dict], 
                   indices: List[Tuple]) -> torch.Tensor:
        """Compute GIoU loss."""
        pred_boxes = outputs['pred_boxes']  # [B, num_queries, 4]
        
        # Get matched predictions and targets
        src_boxes = []
        target_boxes = []
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_boxes.append(pred_boxes[i][src_idx])
                target_boxes.append(targets[i]['boxes'][tgt_idx])
        
        if len(src_boxes) == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        
        src_boxes = torch.cat(src_boxes, dim=0)
        target_boxes = torch.cat(target_boxes, dim=0)
        
        # GIoU loss
        loss_giou = self.giou_loss(src_boxes, target_boxes)
        
        return loss_giou
    
    def _cutting_loss(self, outputs: Dict, targets: List[Dict], 
                     indices: List[Tuple]) -> torch.Tensor:
        """Compute cutting behavior loss."""
        pred_cutting = outputs['pred_cutting']  # [B, num_queries, 1]
        
        # Get matched predictions and targets
        src_cutting = []
        target_cutting = []
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_cutting.append(pred_cutting[i][src_idx].squeeze(-1))
                # Extract cutting labels from targets
                cutting_labels = targets[i].get('cutting', torch.zeros(len(tgt_idx)))
                target_cutting.append(cutting_labels)
        
        if len(src_cutting) == 0:
            return torch.tensor(0.0, device=pred_cutting.device, requires_grad=True)
        
        src_cutting = torch.cat(src_cutting, dim=0)
        target_cutting = torch.cat(target_cutting, dim=0).float()
        
        # Apply sigmoid to predictions
        src_cutting_prob = torch.sigmoid(src_cutting)
        
        # Weighted BCE loss with false negative penalty
        loss_cutting = self.weighted_bce(src_cutting_prob, target_cutting)
        
        return loss_cutting
    
    def _sequence_cutting_loss(self, outputs: Dict, targets: List[Dict]) -> torch.Tensor:
        """Compute sequence-level cutting loss."""
        pred_sequence = outputs['sequence_cutting']  # [B, 1]
        
        # Get sequence-level cutting targets
        target_sequence = []
        for target in targets:
            has_cutting = target.get('has_cutting', False)
            target_sequence.append(float(has_cutting))
        
        target_sequence = torch.tensor(target_sequence, device=pred_sequence.device, dtype=torch.float32)
        
        # Apply sigmoid to predictions
        pred_sequence_prob = torch.sigmoid(pred_sequence.squeeze(-1))
        
        # Use standard BCE loss instead of weighted BCE for sequence level
        loss_sequence = F.binary_cross_entropy(pred_sequence_prob, target_sequence)
        
        return loss_sequence

def create_criterion(config: Dict) -> CuttingDetectionLoss:
    """
    Factory function to create the loss criterion.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        CuttingDetectionLoss criterion
    """
    return CuttingDetectionLoss(config) 