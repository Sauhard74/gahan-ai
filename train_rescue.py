#!/usr/bin/env python3
"""
RESCUE GPU Training Script for Cutting Behavior Detection
Enhanced stability and debugging to fix F1 0.000 issue
"""

import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import argparse
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.cutting_detector import CuttingDetector
from datasets.cut_in_dataset import CutInSequenceDataset
from losses.criterion import CuttingDetectionLoss
from utils.collate_fn import collate_sequences
from utils.evaluation import evaluate_cutting_detection, calculate_cutting_behavior_f1

class RescueTrainer:
    """Enhanced trainer with stability features and debugging for F1 0.000 issue."""
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = self._setup_device()
        
        # Create model, datasets, criterion, optimizer
        self.model = self._create_model()
        self.train_loader, self.val_loader = self._create_dataloaders()
        self.criterion = self._create_criterion()
        self.optimizer, self.scheduler = self._create_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.checkpoint_dir = Path(self.config['training']['save_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Enhanced tracking for debugging
        self.loss_history = []
        self.f1_history = []
        self.learning_rates = []
        self.gradient_norms = []
        
        # Stability features
        self.patience_counter = 0
        self.max_patience = 6
        self.min_lr = 1e-6
        
        print(f"🚀 RESCUE Trainer initialized")
        print(f"📊 Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"📈 Training: {len(self.train_loader)} batches, Validation: {len(self.val_loader)} batches")
        print(f"🎯 Target F1: {self.config['training']['target_f1']}")
        
        # Debug model architecture
        self._debug_model_architecture()
    
    def _setup_device(self):
        """Enhanced device setup with memory optimization."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            
            # Enhanced CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Memory optimization
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            return device
        else:
            print("🔧 Using CPU (CUDA not available)")
            return torch.device('cpu')
    
    def _debug_model_architecture(self):
        """Debug model architecture to understand complexity."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n🔍 MODEL ARCHITECTURE DEBUG:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Hidden dim: {self.config['model']['hidden_dim']}")
        print(f"   Num queries: {self.config['model']['num_queries']}")
        print(f"   Sequence length: {self.config['model']['sequence_length']}")
        
        # Check for parameter initialization
        zero_params = sum(1 for p in self.model.parameters() if torch.all(p == 0))
        print(f"   Zero-initialized parameters: {zero_params}")
        
        # Check gradient flow
        grad_params = sum(1 for p in self.model.parameters() if p.requires_grad)
        print(f"   Parameters requiring gradients: {grad_params}")
    
    def _create_model(self):
        """Create model with enhanced initialization."""
        model = CuttingDetector(self.config['model']).to(self.device)
        
        # Enhanced model initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        
        # Apply initialization to new layers (not pretrained backbone)
        for name, module in model.named_modules():
            if 'backbone' not in name:  # Skip pretrained backbone
                module.apply(init_weights)
        
        return model
    
    def _create_dataloaders(self):
        """Create dataloaders with enhanced debugging."""
        train_dataset = CutInSequenceDataset(
            dataset_root=self.config['data']['data_dir'],
            split=self.config['data']['train_split'],
            sequence_length=self.config['model']['sequence_length'],
            image_size=tuple(self.config['data']['image_size']),
            oversample_positive=self.config['data']['oversample_positive'],
            val_split_ratio=self.config['data']['val_split_ratio'],
            is_validation=False,
            augment=True
        )
        
        val_dataset = CutInSequenceDataset(
            dataset_root=self.config['data']['data_dir'],
            split=self.config['data']['train_split'],
            sequence_length=self.config['model']['sequence_length'],
            image_size=tuple(self.config['data']['image_size']),
            oversample_positive=1,
            val_split_ratio=self.config['data']['val_split_ratio'],
            is_validation=True,
            augment=False
        )
        
        print(f"\n📊 DATASET DEBUG:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        # Check dataset balance
        if hasattr(train_dataset, 'positive_samples'):
            pos_ratio = train_dataset.positive_samples / len(train_dataset)
            print(f"   Positive sample ratio: {pos_ratio:.3f}")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True, 
            num_workers=self.config['data']['num_workers'],
            collate_fn=collate_sequences, 
            pin_memory=self.config['data']['pin_memory'],
            drop_last=True,  # Ensure consistent batch sizes
            persistent_workers=True if self.config['data']['num_workers'] > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=False, 
            num_workers=self.config['data']['num_workers'],
            collate_fn=collate_sequences, 
            pin_memory=self.config['data']['pin_memory'],
            drop_last=False,
            persistent_workers=True if self.config['data']['num_workers'] > 0 else False
        )
        
        return train_loader, val_loader
    
    def _create_criterion(self):
        """Create criterion with enhanced debugging."""
        criterion = CuttingDetectionLoss(self.config['training']['loss_weights']).to(self.device)
        
        print(f"\n🎯 LOSS CONFIGURATION:")
        for key, weight in self.config['training']['loss_weights'].items():
            print(f"   {key}: {weight}")
        
        return criterion
    
    def _create_optimizer(self):
        """Create optimizer with enhanced settings."""
        # Separate learning rates for backbone and head
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.config['training']['learning_rate'] * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': self.config['training']['learning_rate']}
        ], weight_decay=self.config['training']['weight_decay'])
        
        # Enhanced scheduler with warmup
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize F1
            factor=self.config['training']['scheduler_gamma'],
            patience=3,
            verbose=True,
            min_lr=self.min_lr
        )
        
        print(f"\n⚙️ OPTIMIZER CONFIGURATION:")
        print(f"   Backbone LR: {self.config['training']['learning_rate'] * 0.1:.2e}")
        print(f"   Head LR: {self.config['training']['learning_rate']:.2e}")
        print(f"   Weight decay: {self.config['training']['weight_decay']:.2e}")
        
        return optimizer, scheduler
    
    def _debug_batch(self, batch, batch_idx):
        """Debug batch content for troubleshooting."""
        if batch_idx == 0:  # Debug first batch only
            print(f"\n🔍 BATCH DEBUG (Batch {batch_idx}):")
            print(f"   Images shape: {batch['images'].shape}")
            print(f"   Batch size: {len(batch['targets'])}")
            
            # Check targets
            total_objects = 0
            cutting_samples = 0
            for i, target in enumerate(batch['targets']):
                num_objects = len(target['labels']) if isinstance(target['labels'], torch.Tensor) else sum(len(labels) for labels in target['labels'])
                total_objects += num_objects
                if target['has_cutting']:
                    cutting_samples += 1
                
                if i == 0:  # Debug first target
                    print(f"   Target 0 - Objects: {num_objects}, Has cutting: {target['has_cutting']}")
                    if isinstance(target['labels'], torch.Tensor):
                        print(f"   Target 0 - Labels: {target['labels'].tolist()}")
                    else:
                        print(f"   Target 0 - Labels (sequence): {[labels.tolist() for labels in target['labels'][:2]]}")
            
            print(f"   Total objects in batch: {total_objects}")
            print(f"   Cutting samples: {cutting_samples}/{len(batch['targets'])}")
    
    def train_epoch(self):
        """Enhanced training loop with debugging."""
        self.model.train()
        total_loss = 0.0
        loss_components = {'classification': 0.0, 'bbox_regression': 0.0, 'giou': 0.0, 'cutting': 0.0, 'sequence': 0.0}
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        print(f"\n🔄 Training Epoch {self.current_epoch + 1}")
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            try:
                # Debug first batch
                if batch_idx == 0:
                    self._debug_batch(batch, batch_idx)
                
                # Move data to device
                images = batch['images'].to(self.device, non_blocking=True)
                targets = []
                for target in batch['targets']:
                    target_dict = {
                        'labels': target['labels'].to(self.device, non_blocking=True),
                        'boxes': target['boxes'].to(self.device, non_blocking=True),
                        'has_cutting': target['has_cutting']
                    }
                    if 'cutting' in target:
                        target_dict['cutting'] = target['cutting'].to(self.device, non_blocking=True)
                    targets.append(target_dict)
                
                # Forward pass with AMP
                if self.config['training']['use_amp']:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss_dict = self.criterion(outputs, targets)
                        loss = loss_dict['loss'] / gradient_accumulation_steps
                else:
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict['loss'] / gradient_accumulation_steps
                
                # Backward pass
                if self.config['training']['use_amp']:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config['training'].get('max_grad_norm', 0) > 0:
                        if self.config['training']['use_amp']:
                            self.scaler.unscale_(self.optimizer)
                        
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['training']['max_grad_norm']
                        )
                        self.gradient_norms.append(grad_norm.item())
                    
                    # Optimizer step
                    if self.config['training']['use_amp']:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                # Update metrics
                actual_loss = loss.item() * gradient_accumulation_steps
                total_loss += actual_loss
                
                # Track loss components
                for key in loss_components:
                    if key in loss_dict:
                        loss_components[key] += loss_dict[key].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{actual_loss:.4f}',
                    'Avg': f'{total_loss/(batch_idx+1):.4f}'
                })
                
                # Memory cleanup
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
        
        # Final gradient update
        if len(self.train_loader) % gradient_accumulation_steps != 0:
            if self.config['training'].get('max_grad_norm', 0) > 0:
                if self.config['training']['use_amp']:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['max_grad_norm']
                )
            
            if self.config['training']['use_amp']:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        
        # Log loss components
        print(f"\n📊 LOSS BREAKDOWN:")
        for key, value in loss_components.items():
            avg_component = value / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
            print(f"   {key}: {avg_component:.4f}")
        
        # Log gradient norms
        if self.gradient_norms:
            avg_grad_norm = np.mean(self.gradient_norms[-len(self.train_loader):])
            print(f"   Avg gradient norm: {avg_grad_norm:.4f}")
        
        return avg_loss
    
    def validate(self):
        """Enhanced validation with detailed debugging."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        print(f"\n🔍 Validating...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                if batch is None:
                    continue
                
                try:
                    # Move data to device
                    images = batch['images'].to(self.device, non_blocking=True)
                    targets = []
                    for target in batch['targets']:
                        target_dict = {
                            'labels': target['labels'].to(self.device, non_blocking=True),
                            'boxes': target['boxes'].to(self.device, non_blocking=True),
                            'has_cutting': target['has_cutting']
                        }
                        if 'cutting' in target:
                            target_dict['cutting'] = target['cutting'].to(self.device, non_blocking=True)
                        targets.append(target_dict)
                    
                    # Forward pass
                    if self.config['training']['use_amp']:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss_dict = self.criterion(outputs, targets)
                    else:
                        outputs = self.model(images)
                        loss_dict = self.criterion(outputs, targets)
                    
                    total_loss += loss_dict['loss'].item()
                    
                    # Convert outputs for evaluation
                    batch_predictions = self._convert_outputs_to_predictions(outputs)
                    batch_targets = self._convert_targets_for_evaluation(targets)
                    
                    all_predictions.extend(batch_predictions)
                    all_targets.extend(batch_targets)
                    
                    # Debug first validation batch
                    if batch_idx == 0:
                        self._debug_predictions(batch_predictions[:2], batch_targets[:2])
                    
                except Exception as e:
                    print(f"❌ Error in validation batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        
        # Calculate metrics
        if all_predictions and all_targets:
            metrics = evaluate_cutting_detection(
                all_predictions, all_targets,
                iou_threshold=self.config['evaluation']['iou_threshold'],
                confidence_threshold=self.config['evaluation']['confidence_threshold']
            )
            
            cutting_f1 = calculate_cutting_behavior_f1(all_predictions, all_targets)
            metrics['cutting_f1'] = cutting_f1
            
            # Debug metrics calculation
            self._debug_metrics(all_predictions, all_targets, metrics)
        else:
            metrics = {'micro_f1': 0.0, 'cutting_f1': 0.0}
        
        return avg_loss, metrics
    
    def _debug_predictions(self, predictions, targets):
        """Debug predictions vs targets."""
        print(f"\n🔍 PREDICTION DEBUG:")
        for i, (pred, target) in enumerate(zip(predictions[:2], targets[:2])):
            print(f"   Sample {i}:")
            print(f"     Predicted boxes: {len(pred['boxes'])}")
            print(f"     Target boxes: {len(target['boxes'])}")
            print(f"     Predicted cutting: {pred['has_cutting']}")
            print(f"     Target cutting: {target['has_cutting']}")
            
            if len(pred['scores']) > 0:
                max_score = np.max(pred['scores'])
                print(f"     Max confidence: {max_score:.4f}")
            else:
                print(f"     No predictions above threshold")
    
    def _debug_metrics(self, predictions, targets, metrics):
        """Debug metrics calculation."""
        print(f"\n📊 METRICS DEBUG:")
        
        # Count predictions and targets
        total_pred_boxes = sum(len(p['boxes']) for p in predictions)
        total_target_boxes = sum(len(t['boxes']) for t in targets)
        cutting_predictions = sum(1 for p in predictions if p['has_cutting'])
        cutting_targets = sum(1 for t in targets if t['has_cutting'])
        
        print(f"   Total predicted boxes: {total_pred_boxes}")
        print(f"   Total target boxes: {total_target_boxes}")
        print(f"   Cutting predictions: {cutting_predictions}/{len(predictions)}")
        print(f"   Cutting targets: {cutting_targets}/{len(targets)}")
        print(f"   Micro F1: {metrics['micro_f1']:.4f}")
        print(f"   Cutting F1: {metrics['cutting_f1']:.4f}")
        
        # Check confidence distribution
        all_scores = np.concatenate([p['scores'] for p in predictions if len(p['scores']) > 0])
        if len(all_scores) > 0:
            print(f"   Score range: {np.min(all_scores):.4f} - {np.max(all_scores):.4f}")
            print(f"   Mean score: {np.mean(all_scores):.4f}")
        else:
            print(f"   No predictions with scores!")
    
    def _convert_outputs_to_predictions(self, outputs):
        """Convert model outputs to prediction format."""
        batch_size = outputs['pred_logits'].size(0)
        predictions = []
        
        for i in range(batch_size):
            # Get predictions for this sample
            logits = outputs['pred_logits'][i]
            boxes = outputs['pred_boxes'][i]
            objectness = outputs.get('pred_objectness', torch.ones_like(logits[:, 0:1]))[i]
            cutting_logits = outputs.get('pred_cutting', torch.zeros_like(logits[:, 0:1]))[i]
            
            # Convert to probabilities
            class_probs = torch.softmax(logits, dim=-1)
            objectness_probs = torch.sigmoid(objectness)
            cutting_probs = torch.sigmoid(cutting_logits)
            
            # Get class predictions (excluding background)
            max_probs, class_ids = class_probs[:, 1:].max(dim=-1)
            class_ids += 1  # Adjust for background class
            
            # Combine with objectness for confidence
            confidence_scores = max_probs * objectness_probs.squeeze(-1)
            
            # Apply confidence threshold
            conf_threshold = self.config['evaluation']['confidence_threshold']
            valid_mask = confidence_scores > conf_threshold
            
            # Filter predictions
            valid_boxes = boxes[valid_mask].cpu().numpy()
            valid_labels = class_ids[valid_mask].cpu().numpy()
            valid_scores = confidence_scores[valid_mask].cpu().numpy()
            
            # Determine cutting behavior
            has_cutting = False
            if 'sequence_cutting' in outputs:
                sequence_cutting_prob = torch.sigmoid(outputs['sequence_cutting'][i]).item()
                has_cutting = sequence_cutting_prob > 0.5
            elif len(cutting_probs) > 0:
                has_cutting = cutting_probs.max().item() > 0.5
            
            predictions.append({
                'boxes': valid_boxes,
                'labels': valid_labels,
                'scores': valid_scores,
                'has_cutting': has_cutting
            })
        
        return predictions
    
    def _convert_targets_for_evaluation(self, targets):
        """Convert targets to evaluation format."""
        eval_targets = []
        
        for target in targets:
            # Handle both sequence format and flattened format
            if isinstance(target['labels'], list):
                # Sequence format - combine all frames
                all_boxes = []
                all_labels = []
                for frame_boxes, frame_labels in zip(target['boxes'], target['labels']):
                    if len(frame_boxes) > 0:
                        all_boxes.append(frame_boxes.cpu().numpy())
                        all_labels.append(frame_labels.cpu().numpy())
                
                if all_boxes:
                    combined_boxes = np.concatenate(all_boxes, axis=0)
                    combined_labels = np.concatenate(all_labels, axis=0)
                else:
                    combined_boxes = np.array([]).reshape(0, 4)
                    combined_labels = np.array([])
            else:
                # Flattened format
                combined_boxes = target['boxes'].cpu().numpy()
                combined_labels = target['labels'].cpu().numpy()
            
            eval_targets.append({
                'boxes': combined_boxes,
                'labels': combined_labels,
                'has_cutting': target['has_cutting']
            })
        
        return eval_targets
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'config': self.config,
            'loss_history': self.loss_history,
            'f1_history': self.f1_history,
            'learning_rates': self.learning_rates
        }
        
        if self.config['training']['use_amp']:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved with F1: {self.best_f1:.4f}")
    
    def train(self):
        """Main training loop with enhanced monitoring."""
        print("🚀 Starting RESCUE training...")
        print(f"📊 Training for {self.config['training']['num_epochs']} epochs")
        print(f"🎯 Target F1 score: {self.config['training']['target_f1']}")
        
        # Initialize AMP scaler if using mixed precision
        if self.config['training']['use_amp']:
            self.scaler = torch.cuda.amp.GradScaler()
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            print(f"\n{'='*60}")
            print(f"🔄 EPOCH {epoch+1}/{self.config['training']['num_epochs']}")
            print(f"{'='*60}")
            
            # Train epoch
            epoch_start = time.time()
            train_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start
            
            # Validate
            val_loss, metrics = self.validate()
            
            # Update scheduler
            current_f1 = metrics['micro_f1']
            self.scheduler.step(current_f1)
            
            # Track history
            self.loss_history.append({'train': train_loss, 'val': val_loss})
            self.f1_history.append(current_f1)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Log results
            print(f"\n📊 EPOCH {epoch+1} RESULTS:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val F1: {current_f1:.4f}")
            print(f"   Cutting F1: {metrics.get('cutting_f1', 0.0):.4f}")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"   Epoch Time: {epoch_time/60:.1f}min")
            
            # Check for improvement
            is_best = current_f1 > self.best_f1
            if is_best:
                self.best_f1 = current_f1
                self.patience_counter = 0
                print(f"🎉 New best F1: {self.best_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"⏳ No improvement for {self.patience_counter} epochs")
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best)
            
            # Early stopping checks
            if current_f1 >= self.config['training']['target_f1']:
                print(f"🎉 Target F1 {self.config['training']['target_f1']} reached! Stopping training.")
                break
            
            if self.patience_counter >= self.max_patience:
                print(f"⏹️ Early stopping: No improvement for {self.max_patience} epochs")
                break
            
            if self.optimizer.param_groups[0]['lr'] < self.min_lr:
                print(f"⏹️ Learning rate too low: {self.optimizer.param_groups[0]['lr']:.2e}")
                break
        
        total_time = time.time() - start_time
        print(f"\n🏁 Training completed in {total_time/3600:.1f} hours")
        print(f"🏆 Best F1 score: {self.best_f1:.4f}")
        
        # Final analysis
        self._final_analysis()
    
    def _final_analysis(self):
        """Perform final analysis of training."""
        print(f"\n📈 TRAINING ANALYSIS:")
        
        if len(self.f1_history) > 0:
            max_f1_epoch = np.argmax(self.f1_history) + 1
            print(f"   Best F1 achieved at epoch: {max_f1_epoch}")
            print(f"   F1 progression: {self.f1_history}")
        
        if len(self.learning_rates) > 0:
            print(f"   Final learning rate: {self.learning_rates[-1]:.2e}")
        
        if len(self.gradient_norms) > 0:
            avg_grad_norm = np.mean(self.gradient_norms)
            print(f"   Average gradient norm: {avg_grad_norm:.4f}")

def main():
    parser = argparse.ArgumentParser(description='RESCUE GPU Training for Cutting Behavior Detection')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RescueTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_f1 = checkpoint.get('best_f1', 0.0)
        
        if trainer.config['training']['use_amp'] and 'scaler_state_dict' in checkpoint:
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ Resumed from epoch {trainer.current_epoch}")
    
    try:
        # Start training
        trainer.train()
        print("🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        print("💾 Saving current state...")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 