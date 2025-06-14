"""
Custom collate function for batching sequences with variable number of objects.
"""

import torch
from typing import List, Dict, Any

def collate_sequences(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for sequence data with variable number of objects.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    # Stack image sequences
    images = torch.stack([item['images'] for item in batch])  # [B, T, C, H, W]
    
    # Collect targets for each sample
    targets = []
    for item in batch:
        # Flatten sequence data for Hungarian matcher
        all_labels = []
        all_boxes = []
        
        # Combine all frames' annotations into single tensors
        for frame_labels, frame_boxes in zip(item['labels'], item['boxes']):
            if len(frame_labels) > 0:  # Only add if there are objects in this frame
                all_labels.append(frame_labels)
                all_boxes.append(frame_boxes)
        
        # Concatenate all frames' data
        if all_labels:
            combined_labels = torch.cat(all_labels, dim=0)
            combined_boxes = torch.cat(all_boxes, dim=0)
        else:
            # Handle empty case
            combined_labels = torch.tensor([], dtype=torch.long)
            combined_boxes = torch.tensor([], dtype=torch.float32).reshape(0, 4)
        
        target = {
            'labels': combined_labels,  # Flattened tensor for Hungarian matcher
            'boxes': combined_boxes,    # Flattened tensor for Hungarian matcher
            'has_cutting': item['has_cutting'],
            # Keep original sequence structure for other uses
            'sequence_labels': item['labels'],  # Original list of tensors
            'sequence_boxes': item['boxes']     # Original list of tensors
        }
        targets.append(target)
    
    return {
        'images': images,
        'targets': targets
    }

def collate_detection_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for detection data (single frame).
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    images = torch.stack([item['image'] for item in batch])
    
    targets = []
    for item in batch:
        target = {
            'labels': item['labels'],
            'boxes': item['boxes'],
            'has_cutting': item.get('has_cutting', False)
        }
        targets.append(target)
    
    return {
        'images': images,
        'targets': targets
    }

def pad_sequences(sequences: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of tensors with different lengths
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor [batch_size, max_length, ...]
    """
    if not sequences:
        return torch.empty(0)
    
    max_len = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)
    
    # Get the shape of the rest of the dimensions
    rest_shape = sequences[0].shape[1:]
    
    # Create padded tensor
    padded = torch.full((batch_size, max_len) + rest_shape, pad_value, 
                       dtype=sequences[0].dtype, device=sequences[0].device)
    
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        padded[i, :length] = seq
    
    return padded

def create_padding_mask(lengths: List[int], max_length: int = None) -> torch.Tensor:
    """
    Create padding mask for sequences.
    
    Args:
        lengths: List of sequence lengths
        max_length: Maximum length (if None, use max of lengths)
        
    Returns:
        Boolean mask [batch_size, max_length] where True indicates valid positions
    """
    if max_length is None:
        max_length = max(lengths)
    
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask 