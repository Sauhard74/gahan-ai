"""
Comprehensive loss function for cutting behavior detection.
Includes focal loss, GIoU loss, and 10x false negative penalty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from utils.hungarian_matcher import HungarianMatcher

# Import the GIoU function directly
def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU (GIoU) between two sets of boxes.
    
    Args:
        boxes1: [N, 4] boxes in format [x1, y1, x2, y2]
        boxes2: [M, 4] boxes in format [x1, y1, x2, y2]
        
    Returns:
        [N, M] GIoU matrix
    """
    # Handle empty inputs
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.empty((len(boxes1), len(boxes2)), device=boxes1.device if len(boxes1) > 0 else boxes2.device)
    
    # Ensure both tensors are on the same device
    if boxes1.device != boxes2.device:
        boxes2 = boxes2.to(boxes1.device)
    
    # Fix invalid boxes
    boxes1 = _fix_invalid_boxes(boxes1)
    boxes2 = _fix_invalid_boxes(boxes2)
    
    # Compute regular IoU and union
    iou, union = _box_iou(boxes1, boxes2)
    
    # Compute enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area_enclosing = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Compute GIoU
    giou = iou - (area_enclosing - union) / (area_enclosing + 1e-7)
    
    return giou

def _fix_invalid_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Fix invalid bounding boxes where x2 < x1 or y2 < y1."""
    if len(boxes) == 0:
        return boxes
    
    # Work on a copy to avoid in-place operations
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # Ensure x2 >= x1 and y2 >= y1 WITHOUT in-place operations
    fixed_x2 = torch.maximum(x2, x1 + 1e-6)
    fixed_y2 = torch.maximum(y2, y1 + 1e-6)
    
    # Create new tensor with fixed coordinates
    fixed = torch.stack([x1, y1, fixed_x2, fixed_y2], dim=1)
    
    return fixed

def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple:
    """Compute IoU and union between two sets of boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-7)
    
    return iou, union

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
        # Handle empty predictions or targets gracefully
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            print(f"⚠️ Empty boxes detected: pred={len(pred_boxes)}, target={len(target_boxes)}")
            return torch.zeros(1, device=pred_boxes.device if len(pred_boxes) > 0 else target_boxes.device)
        
        # Validate box format and fix if needed - return fixed boxes
        pred_boxes_fixed, invalid_pred = self._validate_and_fix_boxes(pred_boxes, "predictions")
        target_boxes_fixed, invalid_target = self._validate_and_fix_boxes(target_boxes, "targets")
        
        if invalid_pred > 0 or invalid_target > 0:
            print(f"Warning: Found {invalid_pred + invalid_target} invalid boxes, fixing...")
        
        try:
            # Compute GIoU loss using fixed boxes
            giou_loss = 1 - generalized_box_iou(pred_boxes_fixed, target_boxes_fixed).diag()
            
            # Clamp to ensure non-negative values
            giou_loss = torch.clamp(giou_loss, min=0.0)
            
            return giou_loss.mean()
            
        except Exception as e:
            print(f"❌ Error computing GIoU loss: {e}")
            return torch.zeros(1, device=pred_boxes.device, requires_grad=True)
    
    def _validate_and_fix_boxes(self, boxes: torch.Tensor, box_type: str) -> tuple:
        """Validate and fix box format WITHOUT in-place operations."""
        invalid_boxes = 0
        fixed_boxes = boxes.clone()  # Work on a copy
        
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            if x1 > x2 or y1 > y2:
                # Only print warning for first few invalid boxes to avoid spam
                if invalid_boxes < 3:
                    print(f"Warning: {box_type} box {i} has incorrect format: {boxes[i]}")
                invalid_boxes += 1
                # Fix without in-place operation
                clamped_box = torch.clamp(boxes[i], min=0.0)
                fixed_boxes[i] = clamped_box
        
        # Only print summary if there were many invalid boxes
        if invalid_boxes > 3:
            print(f"Warning: Found {invalid_boxes} total invalid {box_type} boxes (showing first 3)")
        
        return fixed_boxes, invalid_boxes

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy with 10x penalty for false negatives.
    Uses BCEWithLogitsLoss for autocast compatibility.
    """
    
    def __init__(self, false_negative_penalty: float = 10.0):
        super().__init__()
        self.false_negative_penalty = false_negative_penalty
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N] predicted logits (before sigmoid)
            targets: [N] binary targets (0 or 1)
            
        Returns:
            Weighted BCE loss
        """
        # Use BCEWithLogitsLoss for autocast compatibility
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply penalty for false negatives
        # Convert logits to probabilities for masking
        probs = torch.sigmoid(inputs)
        false_negative_mask = (targets == 1) & (probs < 0.5)
        
        # Apply penalty WITHOUT in-place operation
        penalty_weights = torch.ones_like(bce_loss)
        penalty_weights = torch.where(false_negative_mask, 
                                    penalty_weights * self.false_negative_penalty, 
                                    penalty_weights)
        weighted_loss = bce_loss * penalty_weights
        
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
                # Ensure target boxes are on the same device
                target_box = targets[i]['boxes'][tgt_idx].to(pred_boxes.device)
                target_boxes.append(target_box)
        
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
                # Ensure target boxes are on the same device
                target_box = targets[i]['boxes'][tgt_idx].to(pred_boxes.device)
                target_boxes.append(target_box)
        
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
                # Extract cutting labels from targets and ensure correct device
                cutting_labels = targets[i].get('cutting', torch.zeros(len(tgt_idx), device=pred_cutting.device))
                if cutting_labels.device != pred_cutting.device:
                    cutting_labels = cutting_labels.to(pred_cutting.device)
                target_cutting.append(cutting_labels)
        
        if len(src_cutting) == 0:
            return torch.tensor(0.0, device=pred_cutting.device, requires_grad=True)
        
        src_cutting = torch.cat(src_cutting, dim=0)
        target_cutting = torch.cat(target_cutting, dim=0).float()
        
        # Use logits directly (don't apply sigmoid here)
        loss_cutting = self.weighted_bce(src_cutting, target_cutting)
        
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
        
        # Use BCEWithLogitsLoss for autocast compatibility
        loss_sequence = F.binary_cross_entropy_with_logits(pred_sequence.squeeze(-1), target_sequence)
        
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