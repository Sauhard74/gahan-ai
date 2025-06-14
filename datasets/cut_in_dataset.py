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
        
        logger.info(f"🔍 Found {len(rec_folders)} REC folders in {split_path}")
        logger.info(f"📁 Processing REC folders for {self.split} split...")
        
        successful_folders = 0
        total_images = 0
        total_annotations = 0
        
        for i, rec_folder in enumerate(rec_folders):
            logger.info(f"📂 [{i+1}/{len(rec_folders)}] Processing: {rec_folder.name}")
            
            annotations_folder = rec_folder / "Annotations"
            
            if not annotations_folder.exists():
                logger.warning(f"❌ No Annotations folder in {rec_folder.name}")
                continue
            
            # Get all image files in Annotations folder (not REC folder)
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                found_images = list(annotations_folder.glob(ext))
                image_files.extend(found_images)
                if found_images:
                    logger.debug(f"   📸 Found {len(found_images)} {ext} files")
            
            # Get all annotation files in Annotations folder
            annotation_files = list(annotations_folder.glob('*.xml'))
            
            logger.info(f"   📊 Found {len(image_files)} images, {len(annotation_files)} annotations")
            
            if len(image_files) == 0:
                logger.warning(f"   ⚠️  No images found in {rec_folder.name}/Annotations")
                continue
                
            if len(annotation_files) == 0:
                logger.warning(f"   ⚠️  No XML annotations found in {rec_folder.name}/Annotations")
                continue
            
            # Sort files by frame number
            image_files.sort(key=lambda x: self._extract_frame_number(x.name))
            annotation_files.sort(key=lambda x: self._extract_frame_number(x.name))
            
            logger.debug(f"   🔢 Frame range: {self._extract_frame_number(image_files[0].name)} to {self._extract_frame_number(image_files[-1].name)}")
            
            # Create sequences of specified length
            rec_sequences = self._create_sequences_from_folder(rec_folder, image_files, annotation_files)
            
            if rec_sequences:
                sequences.extend(rec_sequences)
                successful_folders += 1
                total_images += len(image_files)
                total_annotations += len(annotation_files)
                
                # Count cutting sequences
                cutting_sequences = sum(1 for seq in rec_sequences if seq['has_cutting'])
                logger.info(f"   ✅ Created {len(rec_sequences)} sequences ({cutting_sequences} with cutting behavior)")
            else:
                logger.warning(f"   ❌ No sequences created from {rec_folder.name}")
        
        logger.info(f"🎯 Dataset loading summary:")
        logger.info(f"   📁 Successful folders: {successful_folders}/{len(rec_folders)}")
        logger.info(f"   📸 Total images: {total_images}")
        logger.info(f"   📄 Total annotations: {total_annotations}")
        logger.info(f"   🎬 Total sequences: {len(sequences)}")
        
        # Count cutting behavior
        cutting_sequences = sum(1 for seq in sequences if seq['has_cutting'])
        logger.info(f"   ✂️  Sequences with cutting: {cutting_sequences}/{len(sequences)} ({cutting_sequences/len(sequences)*100:.1f}%)")
        
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
        unmatched_images = []
        
        logger.debug(f"   🔗 Matching {len(image_files)} images with annotations...")
        
        for img_file in image_files:
            # Find corresponding annotation in same folder
            ann_name = img_file.stem + '.xml'
            ann_file = img_file.parent / ann_name  # Same folder as image
            
            if ann_file.exists():
                matched_pairs.append((img_file, ann_file))
            else:
                unmatched_images.append(img_file.name)
        
        if unmatched_images:
            logger.debug(f"   ⚠️  {len(unmatched_images)} images without matching XML files")
            if len(unmatched_images) <= 5:  # Show first few examples
                logger.debug(f"      Examples: {', '.join(unmatched_images)}")
        
        logger.debug(f"   ✅ Matched {len(matched_pairs)} image-annotation pairs")
        
        if len(matched_pairs) < self.sequence_length:
            logger.warning(f"   ❌ Not enough matched pairs in {rec_folder.name}: {len(matched_pairs)} < {self.sequence_length}")
            return sequences
        
        # Create overlapping sequences
        num_possible_sequences = len(matched_pairs) - self.sequence_length + 1
        logger.debug(f"   🎬 Creating {num_possible_sequences} overlapping sequences...")
        
        cutting_count = 0
        
        for i in range(num_possible_sequences):
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
                    cutting_count += 1
                    break
            
            sequences.append(sequence_data)
        
        logger.debug(f"   📊 Sequence stats: {len(sequences)} total, {cutting_count} with cutting behavior")
        
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
            logger.info("📊 No validation split requested (val_split_ratio = 0)")
            return self.sequences
        
        logger.info(f"🔄 Splitting data for {'validation' if self.is_validation else 'training'} set...")
        
        # Group sequences by REC folder to avoid data leakage
        rec_groups = {}
        for seq in self.sequences:
            rec_folder = seq['rec_folder']
            if rec_folder not in rec_groups:
                rec_groups[rec_folder] = []
            rec_groups[rec_folder].append(seq)
        
        logger.info(f"   📁 Grouping by REC folders: {len(rec_groups)} unique folders")
        
        # Show folder distribution
        for folder, seqs in list(rec_groups.items())[:3]:  # Show first 3 as examples
            cutting_seqs = sum(1 for s in seqs if s['has_cutting'])
            folder_name = Path(folder).name
            logger.debug(f"      {folder_name}: {len(seqs)} sequences ({cutting_seqs} cutting)")
        
        # Split REC folders
        rec_folders = list(rec_groups.keys())
        random.shuffle(rec_folders)
        
        val_count = int(len(rec_folders) * self.val_split_ratio)
        
        if self.is_validation:
            selected_folders = rec_folders[:val_count]
            logger.info(f"   📋 Selected {len(selected_folders)} folders for VALIDATION ({self.val_split_ratio:.1%})")
        else:
            selected_folders = rec_folders[val_count:]
            logger.info(f"   📋 Selected {len(selected_folders)} folders for TRAINING ({1-self.val_split_ratio:.1%})")
        
        # Collect sequences from selected folders
        selected_sequences = []
        total_cutting = 0
        
        for folder in selected_folders:
            folder_sequences = rec_groups[folder]
            selected_sequences.extend(folder_sequences)
            cutting_in_folder = sum(1 for s in folder_sequences if s['has_cutting'])
            total_cutting += cutting_in_folder
        
        logger.info(f"   ✅ Final split: {len(selected_sequences)} sequences ({total_cutting} with cutting)")
        
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
                    
                    # Validate and fix bounding box
                    x1, y1, x2, y2 = self._validate_bbox(x1, y1, x2, y2, image_size, ann_path)
                    
                    # Skip if bbox is invalid after fixing
                    if x1 >= x2 or y1 >= y2:
                        logger.debug(f"   ⚠️  Skipping invalid bbox in {ann_path.name}: [{x1}, {y1}, {x2}, {y2}]")
                        continue
                    
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
                    logger.debug(f"   ⚠️  Error parsing object in {ann_path.name}: {e}")
                    continue
            
            # Filter by ROI
            if self.roi_bounds:
                objects = filter_objects_by_roi(objects, self.roi_bounds, image_size)
            
            return {
                'image_size': image_size,
                'objects': objects
            }
        except Exception as e:
            logger.debug(f"   ❌ Error parsing annotation {ann_path.name}: {e}")
            return None
    
    def _validate_bbox(self, x1: float, y1: float, x2: float, y2: float, 
                      image_size: tuple, ann_path: Path) -> tuple:
        """Validate and fix bounding box coordinates."""
        width, height = image_size
        
        # Clamp to image boundaries
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # Fix swapped coordinates
        if x1 > x2:
            logger.debug(f"   🔧 Fixing swapped x coordinates in {ann_path.name}: {x1} > {x2}")
            x1, x2 = x2, x1
        
        if y1 > y2:
            logger.debug(f"   🔧 Fixing swapped y coordinates in {ann_path.name}: {y1} > {y2}")
            y1, y2 = y2, y1
        
        # Ensure minimum box size
        min_size = 1.0
        if x2 - x1 < min_size:
            x2 = x1 + min_size
        if y2 - y1 < min_size:
            y2 = y1 + min_size
        
        # Final clamp after fixing
        x2 = min(x2, width)
        y2 = min(y2, height)
        
        return x1, y1, x2, y2
    
    def _balance_classes(self) -> List[Dict]:
        """Balance classes by oversampling positive sequences."""
        positive_sequences = [seq for seq in self.sequences if seq['has_cutting']]
        negative_sequences = [seq for seq in self.sequences if not seq['has_cutting']]
        
        logger.info(f"⚖️  Class balancing for {'validation' if self.is_validation else 'training'} set:")
        logger.info(f"   📊 Original distribution:")
        logger.info(f"      Positive (cutting): {len(positive_sequences)}")
        logger.info(f"      Negative (no cutting): {len(negative_sequences)}")
        
        if len(positive_sequences) == 0:
            logger.warning(f"   ⚠️  No positive sequences found! All sequences are negative.")
        
        balanced_sequences = negative_sequences.copy()
        
        # Add oversampled positive sequences (only if not validation)
        if not self.is_validation and self.oversample_positive > 1:
            logger.info(f"   🔄 Applying {self.oversample_positive}x oversampling to positive sequences...")
            for i in range(self.oversample_positive):
                balanced_sequences.extend(positive_sequences)
                logger.debug(f"      Added copy {i+1}/{self.oversample_positive}: +{len(positive_sequences)} sequences")
        else:
            balanced_sequences.extend(positive_sequences)
            if self.is_validation:
                logger.info(f"   📋 No oversampling for validation set")
            else:
                logger.info(f"   📋 No oversampling applied (oversample_positive = {self.oversample_positive})")
        
        # Final counts
        final_positive = sum(1 for seq in balanced_sequences if seq['has_cutting'])
        final_negative = sum(1 for seq in balanced_sequences if not seq['has_cutting'])
        
        logger.info(f"   ✅ Final balanced distribution:")
        logger.info(f"      Positive (cutting): {final_positive}")
        logger.info(f"      Negative (no cutting): {final_negative}")
        logger.info(f"      Total sequences: {len(balanced_sequences)}")
        
        if final_positive > 0:
            ratio = final_negative / final_positive
            logger.info(f"      Negative:Positive ratio: {ratio:.1f}:1")
        
        random.shuffle(balanced_sequences)
        logger.debug(f"   🔀 Shuffled {len(balanced_sequences)} sequences")
        
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