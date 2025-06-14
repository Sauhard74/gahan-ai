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
        
        # Flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
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
        # Degenerate boxes gives inf / nan results, so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        
        iou, union = self._box_iou(boxes1, boxes2)
        
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        area = wh[:, :, 0] * wh[:, :, 1]
        
        return iou - (area - union) / area
    
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