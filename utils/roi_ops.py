"""
ROI Operations for Cutting Behavior Detection
Focus on designated region where lane-cutting is relevant
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any

def filter_objects_by_roi(objects: List[Dict], roi_bounds: List[int], 
                         image_size: Tuple[int, int]) -> List[Dict]:
    """
    Filter objects to only include those in the ROI.
    
    Args:
        objects: List of object dictionaries with bounding boxes
        roi_bounds: [x1, y1, x2, y2] ROI coordinates
        image_size: (width, height) of the image
        
    Returns:
        Filtered list of objects in ROI
    """
    if not roi_bounds:
        return objects
    
    x1_roi, y1_roi, x2_roi, y2_roi = roi_bounds
    filtered_objects = []
    
    for obj in objects:
        bbox = obj.get('bbox', [])
        if len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = bbox
        
        # Check if object center is in ROI
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if (x1_roi <= center_x <= x2_roi and 
            y1_roi <= center_y <= y2_roi):
            filtered_objects.append(obj)
    
    return filtered_objects

def normalize_bbox(bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
    """Normalize bounding box coordinates to [0, 1]."""
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return [x1/width, y1/height, x2/width, y2/height]

def denormalize_bbox(bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
    """Denormalize bounding box coordinates from [0, 1] to pixel coordinates."""
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return [x1*width, y1*height, x2*width, y2*height]

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of bounding boxes.
    
    Args:
        box1: [N, 4] tensor of boxes in format [x1, y1, x2, y2]
        box2: [M, 4] tensor of boxes in format [x1, y1, x2, y2]
        
    Returns:
        [N, M] tensor of IoU values
    """
    # Expand dimensions for broadcasting
    box1 = box1.unsqueeze(1)  # [N, 1, 4]
    box2 = box2.unsqueeze(0)  # [1, M, 4]
    
    # Calculate intersection
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = area1 + area2 - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)
    return iou

def crop_roi_from_image(image: np.ndarray, roi_bounds: List[int]) -> np.ndarray:
    """
    Crop ROI from image.
    
    Args:
        image: Input image array [H, W, C]
        roi_bounds: [x1, y1, x2, y2] ROI coordinates
        
    Returns:
        Cropped image
    """
    if not roi_bounds:
        return image
    
    x1, y1, x2, y2 = roi_bounds
    return image[y1:y2, x1:x2]

def adjust_bbox_for_roi(bbox: List[float], roi_bounds: List[int]) -> List[float]:
    """
    Adjust bounding box coordinates relative to ROI crop.
    
    Args:
        bbox: [x1, y1, x2, y2] in original image coordinates
        roi_bounds: [x1, y1, x2, y2] ROI coordinates
        
    Returns:
        Adjusted bounding box coordinates
    """
    if not roi_bounds:
        return bbox
    
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    x1, y1, x2, y2 = bbox
    
    # Adjust coordinates relative to ROI
    adj_x1 = max(0, x1 - roi_x1)
    adj_y1 = max(0, y1 - roi_y1)
    adj_x2 = min(roi_x2 - roi_x1, x2 - roi_x1)
    adj_y2 = min(roi_y2 - roi_y1, y2 - roi_y1)
    
    return [adj_x1, adj_y1, adj_x2, adj_y2] 