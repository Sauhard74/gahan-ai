"""
Optimized Cut-in Dataset with ROI filtering, class balancing, and caching.
"""

import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Any
import pickle
import hashlib
from pathlib import Path
import logging

from utils.roi_ops import filter_objects_by_roi, normalize_bbox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CutInSequenceDataset(Dataset):
    """
    Dataset for cutting behavior detection with ROI filtering and class balancing.
    """
    
    def __init__(self, 
                 images_dir: str,
                 annotations_dir: str,
                 sequence_length: int = 5,
                 image_size: Tuple[int, int] = (224, 224),
                 roi_bounds: Optional[List[int]] = None,
                 oversample_positive: int = 10,
                 augment: bool = True):
        
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.roi_bounds = roi_bounds or [480, 540, 1440, 1080]
        self.oversample_positive = oversample_positive
        self.augment = augment
        
        # Class mapping
        self.class_to_idx = {
            'Background': 0, 'Car': 1, 'MotorBike': 2, 'EgoVehicle': 3
        }
        
        # Initialize transforms
        self._init_transforms()
        
        # Load sequences
        self.sequences = self._load_sequences()
        self.balanced_sequences = self._balance_classes()
        
        logger.info(f"Dataset: {len(self.balanced_sequences)} sequences")
    
    def _init_transforms(self):
        """Initialize augmentation transforms."""
        if self.augment:
            self.transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.1),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _load_sequences(self) -> List[Dict]:
        """Load all sequences."""
        image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                           list(self.images_dir.glob('*.png')))
        
        sequences = []
        for i in range(0, len(image_files) - self.sequence_length + 1, self.sequence_length):
            sequence_files = image_files[i:i + self.sequence_length]
            sequence_data = self._process_sequence(sequence_files)
            if sequence_data:
                sequences.append(sequence_data)
        
        return sequences
    
    def _process_sequence(self, sequence_files: List[Path]) -> Optional[Dict]:
        """Process a single sequence."""
        try:
            sequence_data = {
                'image_paths': [str(f) for f in sequence_files],
                'annotations': [],
                'has_cutting': False
            }
            
            for img_path in sequence_files:
                ann_name = img_path.stem + '.xml'
                ann_path = self.annotations_dir / ann_name
                
                if ann_path.exists():
                    annotation = self._parse_annotation(ann_path, img_path)
                    sequence_data['annotations'].append(annotation)
                    
                    if annotation and any(obj.get('cutting', False) for obj in annotation['objects']):
                        sequence_data['has_cutting'] = True
                else:
                    sequence_data['annotations'].append({
                        'image_path': str(img_path),
                        'image_size': (1920, 1080),
                        'objects': []
                    })
            
            return sequence_data
        except Exception as e:
            logger.warning(f"Failed to process sequence: {e}")
            return None
    
    def _parse_annotation(self, ann_path: Path, img_path: Path) -> Optional[Dict]:
        """Parse XML annotation file."""
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            # Get image size
            size_elem = root.find('size')
            if size_elem is not None:
                width = int(size_elem.find('width').text)
                height = int(size_elem.find('height').text)
                image_size = (width, height)
            else:
                image_size = (1920, 1080)  # Default
            
            # Parse objects
            objects = []
            for obj in root.findall('object'):
                try:
                    name = obj.find('name').text
                    if name not in self.class_to_idx:
                        continue
                    
                    bbox = obj.find('bndbox')
                    x1 = float(bbox.find('xmin').text)
                    y1 = float(bbox.find('ymin').text)
                    x2 = float(bbox.find('xmax').text)
                    y2 = float(bbox.find('ymax').text)
                    
                    # Get cutting attribute
                    cutting = False
                    cutting_elem = obj.find('attributes/attribute[name="Cutting"]')
                    if cutting_elem is not None:
                        cutting_value = cutting_elem.find('value')
                        if cutting_value is not None:
                            cutting = cutting_value.text.lower() == 'true'
                    
                    obj_data = {
                        'class': name,
                        'class_id': self.class_to_idx[name],
                        'bbox': [x1, y1, x2, y2],
                        'cutting': cutting
                    }
                    
                    objects.append(obj_data)
                except Exception as e:
                    continue
            
            # Filter by ROI
            if self.roi_bounds:
                objects = filter_objects_by_roi(objects, self.roi_bounds, image_size)
            
            return {
                'image_path': str(img_path),
                'image_size': image_size,
                'objects': objects
            }
        except Exception as e:
            return None
    
    def _balance_classes(self) -> List[Dict]:
        """Balance classes by oversampling positive sequences."""
        positive_sequences = [seq for seq in self.sequences if seq['has_cutting']]
        negative_sequences = [seq for seq in self.sequences if not seq['has_cutting']]
        
        balanced_sequences = negative_sequences.copy()
        
        # Add oversampled positive sequences
        for _ in range(self.oversample_positive):
            balanced_sequences.extend(positive_sequences)
        
        np.random.shuffle(balanced_sequences)
        return balanced_sequences
    
    def __len__(self) -> int:
        return len(self.balanced_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sequence sample."""
        try:
            sequence = self.balanced_sequences[idx]
            
            # Load images
            images = []
            for img_path in sequence['image_paths']:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                else:
                    images.append(np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8))
            
            # Ensure sequence length
            while len(images) < self.sequence_length:
                images.append(images[-1] if images else np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8))
            images = images[:self.sequence_length]
            
            # Process annotations
            labels_list = []
            boxes_list = []
            
            for annotation in sequence['annotations'][:self.sequence_length]:
                if annotation and annotation['objects']:
                    frame_labels = [obj['class_id'] for obj in annotation['objects']]
                    frame_boxes = [normalize_bbox(obj['bbox'], annotation['image_size']) 
                                 for obj in annotation['objects']]
                    
                    labels_list.append(torch.tensor(frame_labels, dtype=torch.long))
                    boxes_list.append(torch.tensor(frame_boxes, dtype=torch.float32))
                else:
                    labels_list.append(torch.tensor([], dtype=torch.long))
                    boxes_list.append(torch.tensor([], dtype=torch.float32).reshape(0, 4))
            
            # Pad to sequence length
            while len(labels_list) < self.sequence_length:
                labels_list.append(torch.tensor([], dtype=torch.long))
                boxes_list.append(torch.tensor([], dtype=torch.float32).reshape(0, 4))
            
            # Apply transforms
            transformed_images = []
            for img in images:
                transformed = self.transform(image=img)
                transformed_images.append(transformed['image'])
            
            images_tensor = torch.stack(transformed_images)
            
            return {
                'images': images_tensor,
                'labels': labels_list,
                'boxes': boxes_list,
                'has_cutting': sequence['has_cutting']
            }
        except Exception as e:
            logger.error(f"Failed to load sample {idx}: {e}")
            return self._get_dummy_sample()
    
    def _get_dummy_sample(self) -> Dict[str, Any]:
        """Get dummy sample for error cases."""
        dummy_images = torch.zeros(self.sequence_length, 3, self.image_size[1], self.image_size[0])
        dummy_labels = [torch.tensor([], dtype=torch.long) for _ in range(self.sequence_length)]
        dummy_boxes = [torch.tensor([], dtype=torch.float32).reshape(0, 4) for _ in range(self.sequence_length)]
        
        return {
            'images': dummy_images,
            'labels': dummy_labels,
            'boxes': dummy_boxes,
            'has_cutting': False
        }

def create_datasets(config: Dict) -> Tuple[Dataset, Dataset]:
    """Create train and validation datasets."""
    train_dataset = CutInSequenceDataset(
        images_dir=config['train_images'],
        annotations_dir=config['train_annotations'],
        sequence_length=config.get('sequence_length', 5),
        roi_bounds=config.get('roi_bounds'),
        oversample_positive=config.get('oversample_positive', 10),
        augment=True
    )
    
    val_dataset = CutInSequenceDataset(
        images_dir=config['val_images'],
        annotations_dir=config['val_annotations'],
        sequence_length=config.get('sequence_length', 5),
        roi_bounds=config.get('roi_bounds'),
        oversample_positive=1,
        augment=False
    )
    
    return train_dataset, val_dataset 