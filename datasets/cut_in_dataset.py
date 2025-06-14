"""
Optimized Cut-in Dataset for Gahan AI structure.
Each REC folder contains images and Annotations subfolder.
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
import random

from utils.roi_ops import filter_objects_by_roi, normalize_bbox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CutInSequenceDataset(Dataset):
    """
    Dataset for cutting behavior detection with Gahan AI structure.
    Structure: dataset_root/Train|Test/REC_*/[images + Annotations/]
    """
    
    def __init__(self, 
                 dataset_root: str,
                 split: str = "Train",  # "Train" or "Test"
                 sequence_length: int = 5,
                 image_size: Tuple[int, int] = (224, 224),
                 roi_bounds: Optional[List[int]] = None,
                 oversample_positive: int = 10,
                 augment: bool = True,
                 val_split_ratio: float = 0.2,
                 is_validation: bool = False):
        
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.roi_bounds = roi_bounds or [480, 540, 1440, 1080]
        self.oversample_positive = oversample_positive
        self.augment = augment
        self.val_split_ratio = val_split_ratio
        self.is_validation = is_validation
        
        # Class mapping
        self.class_to_idx = {
            'Background': 0, 'Car': 1, 'MotorBike': 2, 'EgoVehicle': 3
        }
        
        # Initialize transforms
        self._init_transforms()
        
        # Load sequences from REC folders
        self.sequences = self._load_sequences_from_rec_folders()
        
        # Split train/val if using Train split
        if self.split == "Train":
            self.sequences = self._split_train_val()
        
        # Balance classes
        self.balanced_sequences = self._balance_classes()
        
        logger.info(f"Dataset ({split}{'_val' if is_validation else ''}): {len(self.balanced_sequences)} sequences")
    
    def _init_transforms(self):
        """Initialize image transforms."""
        if self.augment and not self.is_validation:
            self.transform = A.Compose([
                A.Resize(self.image_size[1], self.image_size[0]),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.1),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size[1], self.image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _load_sequences_from_rec_folders(self) -> List[Dict]:
        """Load sequences from REC folders."""
        split_path = self.dataset_root / self.split
        
        if not split_path.exists():
            raise ValueError(f"Split path not found: {split_path}")
        
        sequences = []
        rec_folders = [f for f in split_path.iterdir() if f.is_dir() and f.name.startswith('REC_')]
        
        logger.info(f"Found {len(rec_folders)} REC folders in {split_path}")
        
        for rec_folder in rec_folders:
            annotations_folder = rec_folder / "Annotations"
            
            if not annotations_folder.exists():
                logger.warning(f"No Annotations folder in {rec_folder}")
                continue
            
            # Get all image files in Annotations folder (not REC folder)
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(annotations_folder.glob(ext))
            
            # Get all annotation files in Annotations folder
            annotation_files = list(annotations_folder.glob('*.xml'))
            
            if len(image_files) == 0 or len(annotation_files) == 0:
                logger.warning(f"No images or annotations in {annotations_folder}")
                continue
            
            # Sort files by frame number
            image_files.sort(key=lambda x: self._extract_frame_number(x.name))
            annotation_files.sort(key=lambda x: self._extract_frame_number(x.name))
            
            # Create sequences of specified length
            rec_sequences = self._create_sequences_from_folder(rec_folder, image_files, annotation_files)
            sequences.extend(rec_sequences)
        
        logger.info(f"Created {len(sequences)} sequences from {len(rec_folders)} REC folders")
        return sequences
    
    def _extract_frame_number(self, filename: str) -> int:
        """Extract frame number from filename like frame_000001.jpg"""
        try:
            # Extract number from frame_XXXXXX.ext
            parts = filename.split('_')
            if len(parts) >= 2:
                number_part = parts[1].split('.')[0]
                return int(number_part)
            return 0
        except:
            return 0
    
    def _create_sequences_from_folder(self, rec_folder: Path, image_files: List[Path], 
                                    annotation_files: List[Path]) -> List[Dict]:
        """Create sequences from a single REC folder."""
        sequences = []
        
        # Match images with annotations (both are in Annotations folder)
        matched_pairs = []
        for img_file in image_files:
            # Find corresponding annotation in same folder
            ann_name = img_file.stem + '.xml'
            ann_file = img_file.parent / ann_name  # Same folder as image
            
            if ann_file.exists():
                matched_pairs.append((img_file, ann_file))
        
        if len(matched_pairs) < self.sequence_length:
            logger.warning(f"Not enough matched pairs in {rec_folder}: {len(matched_pairs)}")
            return sequences
        
        # Create overlapping sequences
        for i in range(len(matched_pairs) - self.sequence_length + 1):
            sequence_pairs = matched_pairs[i:i + self.sequence_length]
            
            sequence_data = {
                'rec_folder': str(rec_folder),
                'image_paths': [str(pair[0]) for pair in sequence_pairs],
                'annotation_paths': [str(pair[1]) for pair in sequence_pairs],
                'has_cutting': False
            }
            
            # Check if any frame in sequence has cutting behavior
            for _, ann_path in sequence_pairs:
                if self._has_cutting_behavior(ann_path):
                    sequence_data['has_cutting'] = True
                    break
            
            sequences.append(sequence_data)
        
        return sequences
    
    def _has_cutting_behavior(self, ann_path: Path) -> bool:
        """Check if annotation file contains cutting behavior."""
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                cutting_elem = obj.find('attributes/attribute[name="Cutting"]')
                if cutting_elem is not None:
                    cutting_value = cutting_elem.find('value')
                    if cutting_value is not None and cutting_value.text.lower() == 'true':
                        return True
            return False
        except:
            return False
    
    def _split_train_val(self) -> List[Dict]:
        """Split training data into train/val."""
        if self.val_split_ratio == 0:
            return self.sequences
        
        # Group sequences by REC folder to avoid data leakage
        rec_groups = {}
        for seq in self.sequences:
            rec_folder = seq['rec_folder']
            if rec_folder not in rec_groups:
                rec_groups[rec_folder] = []
            rec_groups[rec_folder].append(seq)
        
        # Split REC folders
        rec_folders = list(rec_groups.keys())
        random.shuffle(rec_folders)
        
        val_count = int(len(rec_folders) * self.val_split_ratio)
        
        if self.is_validation:
            selected_folders = rec_folders[:val_count]
        else:
            selected_folders = rec_folders[val_count:]
        
        # Collect sequences from selected folders
        selected_sequences = []
        for folder in selected_folders:
            selected_sequences.extend(rec_groups[folder])
        
        return selected_sequences
    
    def _parse_annotation(self, ann_path: Path) -> Optional[Dict]:
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
        
        # Add oversampled positive sequences (only if not validation)
        if not self.is_validation:
            for _ in range(self.oversample_positive):
                balanced_sequences.extend(positive_sequences)
        else:
            balanced_sequences.extend(positive_sequences)
        
        random.shuffle(balanced_sequences)
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
            
            for ann_path in sequence['annotation_paths'][:self.sequence_length]:
                annotation = self._parse_annotation(Path(ann_path))
                
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
    dataset_root = config['paths']['dataset_root']
    train_split = config['paths']['train_split']
    val_split_ratio = config['paths']['val_split_ratio']
    
    train_dataset = CutInSequenceDataset(
        dataset_root=dataset_root,
        split=train_split,
        sequence_length=config['model']['sequence_length'],
        roi_bounds=config['dataset']['roi_bounds'],
        oversample_positive=config['dataset']['oversample_positive'],
        val_split_ratio=val_split_ratio,
        is_validation=False,
        augment=True
    )
    
    val_dataset = CutInSequenceDataset(
        dataset_root=dataset_root,
        split=train_split,
        sequence_length=config['model']['sequence_length'],
        roi_bounds=config['dataset']['roi_bounds'],
        oversample_positive=1,  # No oversampling for validation
        val_split_ratio=val_split_ratio,
        is_validation=True,
        augment=False
    )
    
    return train_dataset, val_dataset 