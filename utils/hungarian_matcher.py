"""
Hungarian Matcher for DETR-style training.
IMPORTANT: This is ONLY for training loss calculation, NOT for evaluation F1 scores!
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict

class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for DETR-style object detection training.
    
    This module computes an assignment between the targets and the predictions of the network.
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as no-objects).
    """
    
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        """
        Args:
            cost_class: Relative weight of the classification error in the matching cost
            cost_bbox: Relative weight of the L1 error of the bounding box coordinates
            cost_giou: Relative weight of the GIoU loss of the bounding boxes
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs can't be 0"
    
    @torch.no_grad()
    def forward(self, outputs: Dict, targets: List[Dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs the matching.
        
        Args:
            outputs: Dictionary with:
                - "pred_logits": [batch_size, num_queries, num_classes]
                - "pred_boxes": [batch_size, num_queries, 4]
            targets: List of targets (len(targets) = batch_size), where each target is a dict with:
                - "labels": [num_target_boxes] (where num_target_boxes is the number of ground-truth objects)
                - "boxes": [num_target_boxes, 4]
                
        Returns:
            List of tuples (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        
        # Handle empty targets case
        valid_targets = []
        for target in targets:
            if len(target["labels"]) > 0 and len(target["boxes"]) > 0:
                valid_targets.append(target)
        
        if not valid_targets:
            # No valid targets - return empty matches for all samples
            return [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)) 
                    for _ in range(batch_size)]
        
        # Flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Also concat the target labels and boxes
        try:
            tgt_ids = torch.cat([v["labels"] for v in targets if len(v["labels"]) > 0])
            tgt_bbox = torch.cat([v["boxes"] for v in targets if len(v["boxes"]) > 0])
            
            # Ensure target tensors are on the same device as predictions
            if len(tgt_ids) > 0:
                tgt_ids = tgt_ids.to(out_prob.device)
            if len(tgt_bbox) > 0:
                tgt_bbox = tgt_bbox.to(out_bbox.device)
                
        except RuntimeError as e:
            # Handle case where all targets are empty
            return [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)) 
                    for _ in range(batch_size)]
        
        if len(tgt_ids) == 0 or len(tgt_bbox) == 0:
            # No valid targets
            return [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)) 
                    for _ in range(batch_size)]
        
        # Compute the classification cost
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the GIoU cost between boxes
        cost_giou = -self._generalized_box_iou(out_bbox, tgt_bbox)
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]
    
    def _generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Generalized IoU from https://giou.stanford.edu/
        
        The boxes should be in [x1, y1, x2, y2] format
        
        Returns:
            GIoU matrix of shape [N, M]
        """
        # Fix invalid boxes before computing IoU
        boxes1 = self._fix_invalid_boxes(boxes1)
        boxes2 = self._fix_invalid_boxes(boxes2)
        
        # Degenerate boxes gives inf / nan results, so do an early check
        if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
            print(f"Warning: Invalid boxes1 detected after fixing")
            # Force fix any remaining invalid boxes
            boxes1[:, 2:] = torch.maximum(boxes1[:, 2:], boxes1[:, :2] + 1e-6)
            
        if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
            print(f"Warning: Invalid boxes2 detected after fixing")
            # Force fix any remaining invalid boxes
            boxes2[:, 2:] = torch.maximum(boxes2[:, 2:], boxes2[:, :2] + 1e-6)
        
        iou, union = self._box_iou(boxes1, boxes2)
        
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        area = wh[:, :, 0] * wh[:, :, 1]
        
        return iou - (area - union) / area
    
    def _fix_invalid_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Fix invalid bounding boxes where x2 < x1 or y2 < y1.
        
        Args:
            boxes: [N, 4] boxes in format [x1, y1, x2, y2]
            
        Returns:
            Fixed boxes
        """
        if len(boxes) == 0:
            return boxes
        
        fixed_boxes = boxes.clone()
        
        # Find invalid boxes
        invalid_x = fixed_boxes[:, 2] < fixed_boxes[:, 0]  # x2 < x1
        invalid_y = fixed_boxes[:, 3] < fixed_boxes[:, 1]  # y2 < y1
        
        if invalid_x.any() or invalid_y.any():
            # Only print warning if there are many invalid boxes to avoid spam
            total_invalid = invalid_x.sum() + invalid_y.sum()
            if total_invalid <= 5:
                print(f"Warning: Found {total_invalid} invalid boxes, fixing...")
            elif total_invalid <= 20:
                print(f"Warning: Found {total_invalid} invalid boxes, fixing... (reduced logging)")
            # Don't print for very large numbers to avoid spam
            
            # Fix x coordinates: swap if x2 < x1 WITHOUT in-place operations
            if invalid_x.any():
                x1_vals = fixed_boxes[:, 0]
                x2_vals = fixed_boxes[:, 2]
                # Create new tensor with swapped values where needed
                new_x1 = torch.where(invalid_x, x2_vals, x1_vals)
                new_x2 = torch.where(invalid_x, x1_vals, x2_vals)
                fixed_boxes = torch.cat([
                    new_x1.unsqueeze(1),
                    fixed_boxes[:, 1:2],
                    new_x2.unsqueeze(1),
                    fixed_boxes[:, 3:4]
                ], dim=1)
            
            # Fix y coordinates: swap if y2 < y1 WITHOUT in-place operations
            if invalid_y.any():
                y1_vals = fixed_boxes[:, 1]
                y2_vals = fixed_boxes[:, 3]
                # Create new tensor with swapped values where needed
                new_y1 = torch.where(invalid_y, y2_vals, y1_vals)
                new_y2 = torch.where(invalid_y, y1_vals, y2_vals)
                fixed_boxes = torch.cat([
                    fixed_boxes[:, 0:1],
                    new_y1.unsqueeze(1),
                    fixed_boxes[:, 2:3],
                    new_y2.unsqueeze(1)
                ], dim=1)
        
        # Ensure minimum box size WITHOUT in-place operations
        min_size = 1e-6
        x1, y1, x2, y2 = fixed_boxes[:, 0], fixed_boxes[:, 1], fixed_boxes[:, 2], fixed_boxes[:, 3]
        new_x2 = torch.maximum(x2, x1 + min_size)
        new_y2 = torch.maximum(y2, y1 + min_size)
        
        fixed_boxes = torch.stack([x1, y1, new_x2, new_y2], dim=1)
        
        return fixed_boxes
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute IoU between two sets of boxes.
        
        Args:
            boxes1: [N, 4]
            boxes2: [M, 4]
            
        Returns:
            iou: [N, M]
            union: [N, M]
        """
        area1 = self._box_area(boxes1)
        area2 = self._box_area(boxes2)
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
        
        union = area1[:, None] + area2 - inter
        
        iou = inter / union
        return iou, union
    
    def _box_area(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Computes the area of a set of bounding boxes.
        
        Args:
            boxes: [N, 4] boxes in format [x1, y1, x2, y2]
            
        Returns:
            area: [N] area of each box
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) 