"""
Evaluation utilities for cutting behavior detection.
Implements proper IoU-based evaluation, not Hungarian matching for F1 scores.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from .roi_ops import calculate_iou

def match_predictions_to_targets(pred_boxes: torch.Tensor, pred_classes: torch.Tensor, 
                                pred_scores: torch.Tensor, target_boxes: torch.Tensor, 
                                target_classes: torch.Tensor, iou_threshold: float = 0.5,
                                confidence_threshold: float = 0.3) -> Dict[str, Any]:
    """
    Match predictions to ground truth targets using IoU threshold.
    This is the CORRECT way to evaluate detection, not Hungarian matching.
    
    Args:
        pred_boxes: [N, 4] predicted bounding boxes
        pred_classes: [N] predicted class labels
        pred_scores: [N] prediction confidence scores
        target_boxes: [M, 4] ground truth bounding boxes
        target_classes: [M] ground truth class labels
        iou_threshold: IoU threshold for positive match
        confidence_threshold: Confidence threshold for predictions
        
    Returns:
        Dictionary with TP, FP, FN counts per class
    """
    device = pred_boxes.device
    num_classes = max(pred_classes.max().item(), target_classes.max().item()) + 1
    
    # Filter predictions by confidence
    confident_mask = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[confident_mask]
    pred_classes = pred_classes[confident_mask]
    pred_scores = pred_scores[confident_mask]
    
    # Initialize counters
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)
    
    if len(pred_boxes) == 0 and len(target_boxes) == 0:
        return {'tp': tp, 'fp': fp, 'fn': fn}
    
    if len(pred_boxes) == 0:
        # All targets are false negatives
        for cls in target_classes:
            fn[cls] += 1
        return {'tp': tp, 'fp': fp, 'fn': fn}
    
    if len(target_boxes) == 0:
        # All predictions are false positives
        for cls in pred_classes:
            fp[cls] += 1
        return {'tp': tp, 'fp': fp, 'fn': fn}
    
    # Calculate IoU matrix
    iou_matrix = calculate_iou(pred_boxes, target_boxes)  # [N, M]
    
    # Track which targets have been matched
    target_matched = torch.zeros(len(target_boxes), dtype=torch.bool, device=device)
    
    # Sort predictions by confidence (highest first)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    
    for pred_idx in sorted_indices:
        pred_cls = pred_classes[pred_idx]
        
        # Find best matching target
        ious_for_pred = iou_matrix[pred_idx]
        best_target_idx = torch.argmax(ious_for_pred)
        best_iou = ious_for_pred[best_target_idx]
        
        # Check if it's a valid match
        if (best_iou >= iou_threshold and 
            not target_matched[best_target_idx] and
            target_classes[best_target_idx] == pred_cls):
            # True positive
            tp[pred_cls] += 1
            target_matched[best_target_idx] = True
        else:
            # False positive
            fp[pred_cls] += 1
    
    # Count false negatives (unmatched targets)
    for target_idx, matched in enumerate(target_matched):
        if not matched:
            target_cls = target_classes[target_idx]
            fn[target_cls] += 1
    
    return {'tp': tp, 'fp': fp, 'fn': fn}

def calculate_f1_scores(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> Dict[str, float]:
    """
    Calculate F1 scores from TP, FP, FN counts.
    
    Args:
        tp: True positives per class
        fp: False positives per class  
        fn: False negatives per class
        
    Returns:
        Dictionary with precision, recall, f1 per class and overall
    """
    # Avoid division by zero
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Calculate macro averages (excluding background class 0)
    valid_classes = tp[1:] + fp[1:] + fn[1:] > 0
    if valid_classes.sum() > 0:
        macro_precision = precision[1:][valid_classes].mean().item()
        macro_recall = recall[1:][valid_classes].mean().item()
        macro_f1 = f1[1:][valid_classes].mean().item()
    else:
        macro_precision = macro_recall = macro_f1 = 0.0
    
    # Calculate micro averages (excluding background)
    total_tp = tp[1:].sum()
    total_fp = fp[1:].sum()
    total_fn = fn[1:].sum()
    
    micro_precision = (total_tp / (total_tp + total_fp + 1e-8)).item()
    micro_recall = (total_tp / (total_tp + total_fn + 1e-8)).item()
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8))
    
    return {
        'precision_per_class': precision.cpu().numpy(),
        'recall_per_class': recall.cpu().numpy(),
        'f1_per_class': f1.cpu().numpy(),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    }

def evaluate_cutting_detection(predictions: List[Dict], targets: List[Dict], 
                             iou_threshold: float = 0.5, 
                             confidence_threshold: float = 0.3) -> Dict[str, float]:
    """
    Evaluate cutting detection performance across a dataset.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for matching
        confidence_threshold: Confidence threshold for predictions
        
    Returns:
        Evaluation metrics dictionary
    """
    total_tp = torch.zeros(4)  # 4 classes: Background, Car, MotorBike, EgoVehicle
    total_fp = torch.zeros(4)
    total_fn = torch.zeros(4)
    
    for pred, target in zip(predictions, targets):
        # Convert to tensors
        pred_boxes = torch.tensor(pred.get('boxes', []), dtype=torch.float32)
        pred_classes = torch.tensor(pred.get('labels', []), dtype=torch.long)
        pred_scores = torch.tensor(pred.get('scores', []), dtype=torch.float32)
        
        target_boxes = torch.tensor(target.get('boxes', []), dtype=torch.float32)
        target_classes = torch.tensor(target.get('labels', []), dtype=torch.long)
        
        # Match predictions to targets
        matches = match_predictions_to_targets(
            pred_boxes, pred_classes, pred_scores,
            target_boxes, target_classes,
            iou_threshold, confidence_threshold
        )
        
        total_tp += matches['tp'].cpu()
        total_fp += matches['fp'].cpu()
        total_fn += matches['fn'].cpu()
    
    # Calculate final metrics
    metrics = calculate_f1_scores(total_tp, total_fp, total_fn)
    
    return metrics

def calculate_cutting_behavior_f1(predictions: List[Dict], targets: List[Dict]) -> float:
    """
    Calculate F1 score specifically for cutting behavior detection.
    This focuses on the cutting attribute rather than just object detection.
    
    Args:
        predictions: List of predictions with cutting behavior
        targets: List of targets with cutting behavior
        
    Returns:
        F1 score for cutting behavior
    """
    tp = fp = fn = 0
    
    for pred, target in zip(predictions, targets):
        pred_cutting = pred.get('has_cutting', False)
        target_cutting = target.get('has_cutting', False)
        
        if pred_cutting and target_cutting:
            tp += 1
        elif pred_cutting and not target_cutting:
            fp += 1
        elif not pred_cutting and target_cutting:
            fn += 1
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1 