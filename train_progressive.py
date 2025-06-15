#!/usr/bin/env python3
"""
PROGRESSIVE GPU Training Script for Cutting Behavior Detection
High-performance training with enhanced parallelism for 6-7 epochs to F1 0.85
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

class ProgressiveTrainer:
    """High-performance trainer with progressive features for 6-7 epochs to F1 0.85."""
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device with enhanced optimizations
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
        
        # Progressive features
        self.progressive_config = self.config.get('progressive_features', {})
        self.warmup_epochs = self.progressive_config.get('warmup_epochs', 1)
        self.unfreeze_backbone_epoch = self.progressive_config.get('unfreeze_backbone_epoch', 2)
        self.adaptive_batch_size = self.progressive_config.get('adaptive_batch_size', False)
        self.current_batch_size = self.config['training']['batch_size']
        
        # Performance tracking
        self.epoch_times = []
        self.f1_history = []
        self.loss_history = []
        self.gpu_memory_usage = []
        
        # Early convergence detection
        self.convergence_patience = self.progressive_config.get('convergence_patience', 3)
        self.min_improvement = self.progressive_config.get('min_improvement', 0.02)
        self.no_improvement_count = 0
        
        print(f"🚀 PROGRESSIVE Trainer initialized")
        print(f"📊 Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"📈 Training: {len(self.train_loader)} batches, Validation: {len(self.val_loader)} batches")
        print(f"🎯 Target F1: {self.config['training']['target_f1']} in {self.config['training']['num_epochs']} epochs")
        
        # Initialize progressive features
        self._initialize_progressive_features()
    
    def _setup_device(self):
        """Enhanced device setup with maximum optimizations."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            
            # Maximum CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Enhanced memory optimization
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            
            # Set memory fraction
            max_memory_fraction = self.config.get('gpu_optimizations', {}).get('max_memory_fraction', 0.92)
            torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            return device
        else:
            print("🔧 Using CPU (CUDA not available)")
            return torch.device('cpu')
    
    def _initialize_progressive_features(self):
        """Initialize progressive training features."""
        print(f"\n🔧 PROGRESSIVE FEATURES:")
        print(f"   Warmup epochs: {self.warmup_epochs}")
        print(f"   Backbone unfreeze epoch: {self.unfreeze_backbone_epoch}")
        print(f"   Adaptive batch sizing: {self.adaptive_batch_size}")
        print(f"   Convergence patience: {self.convergence_patience}")
        
        # Freeze backbone initially if progressive unfreezing is enabled
        if self.unfreeze_backbone_epoch > 0:
            self._freeze_backbone()
            print(f"   🔒 Backbone frozen until epoch {self.unfreeze_backbone_epoch}")
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = True
        print(f"🔓 Backbone unfrozen at epoch {self.current_epoch + 1}")
    
    def _create_model(self):
        """Create model with enhanced optimizations."""
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
            if 'backbone' not in name:
                module.apply(init_weights)
        
        # Compile model for maximum speed if enabled
        if self.config.get('gpu_optimizations', {}).get('compile_model', False):
            try:
                print("🔥 Compiling model with torch.compile...")
                model = torch.compile(model, mode='max-autotune')
                print("✅ Model compilation successful")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")
        
        return model
    
    def _create_dataloaders(self):
        """Create high-performance dataloaders."""
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
        
        print(f"\n📊 DATASET INFO:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        # High-performance dataloader settings
        dataloader_kwargs = {
            'num_workers': self.config['data']['num_workers'],
            'pin_memory': self.config['data']['pin_memory'],
            'prefetch_factor': self.config['data'].get('prefetch_factor', 4),
            'persistent_workers': self.config.get('gpu_optimizations', {}).get('dataloader_persistent_workers', True),
            'collate_fn': collate_sequences,
        }
        
        # Set multiprocessing context if specified
        mp_context = self.config.get('gpu_optimizations', {}).get('multiprocessing_context')
        if mp_context and dataloader_kwargs['num_workers'] > 0:
            import multiprocessing as mp
            dataloader_kwargs['multiprocessing_context'] = mp.get_context(mp_context)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.current_batch_size,
            shuffle=True,
            drop_last=True,
            **dataloader_kwargs
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.current_batch_size,
            shuffle=False,
            drop_last=False,
            **dataloader_kwargs
        )
        
        return train_loader, val_loader
    
    def _create_criterion(self):
        """Create criterion with enhanced settings."""
        return CuttingDetectionLoss(self.config['training']['loss_weights']).to(self.device)
    
    def _create_optimizer(self):
        """Create optimizer with progressive learning rates."""
        # Separate parameter groups for progressive learning
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Different learning rates for backbone and head
        base_lr = self.config['training']['learning_rate']
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': base_lr * 0.1, 'name': 'backbone'},
            {'params': head_params, 'lr': base_lr, 'name': 'head'}
        ], weight_decay=self.config['training']['weight_decay'])
        
        # Progressive scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config['training']['scheduler_step_size'],
            gamma=self.config['training']['scheduler_gamma']
        )
        
        print(f"\n⚙️ OPTIMIZER SETUP:")
        print(f"   Backbone LR: {base_lr * 0.1:.2e}")
        print(f"   Head LR: {base_lr:.2e}")
        print(f"   Weight decay: {self.config['training']['weight_decay']:.2e}")
        
        return optimizer, scheduler
    
    def _apply_warmup(self, epoch):
        """Apply learning rate warmup."""
        if epoch < self.warmup_epochs:
            warmup_factor = self.progressive_config.get('warmup_factor', 0.1)
            warmup_lr_scale = warmup_factor + (1.0 - warmup_factor) * epoch / self.warmup_epochs
            
            for param_group in self.optimizer.param_groups:
                if param_group['name'] == 'backbone':
                    param_group['lr'] = self.config['training']['learning_rate'] * 0.1 * warmup_lr_scale
                else:
                    param_group['lr'] = self.config['training']['learning_rate'] * warmup_lr_scale
            
            print(f"🔥 Warmup LR scale: {warmup_lr_scale:.3f}")
    
    def _adaptive_batch_sizing(self):
        """Implement adaptive batch sizing based on GPU memory."""
        if not self.adaptive_batch_size:
            return
        
        try:
            # Get current GPU memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                memory_utilization = memory_used / memory_total
                
                max_batch_size = self.progressive_config.get('max_batch_size', 32)
                min_batch_size = self.progressive_config.get('min_batch_size', 16)
                
                # Adjust batch size based on memory utilization
                if memory_utilization < 0.7 and self.current_batch_size < max_batch_size:
                    new_batch_size = min(self.current_batch_size + 2, max_batch_size)
                    print(f"📈 Increasing batch size: {self.current_batch_size} → {new_batch_size}")
                    self.current_batch_size = new_batch_size
                elif memory_utilization > 0.9 and self.current_batch_size > min_batch_size:
                    new_batch_size = max(self.current_batch_size - 2, min_batch_size)
                    print(f"📉 Decreasing batch size: {self.current_batch_size} → {new_batch_size}")
                    self.current_batch_size = new_batch_size
                
        except Exception as e:
            print(f"⚠️ Adaptive batch sizing error: {e}")
    
    def train_epoch(self):
        """High-performance training loop."""
        self.model.train()
        total_loss = 0.0
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        # Apply progressive features
        if self.current_epoch < self.warmup_epochs:
            self._apply_warmup(self.current_epoch)
        
        if self.current_epoch == self.unfreeze_backbone_epoch:
            self._unfreeze_backbone()
        
        print(f"\n🔄 Training Epoch {self.current_epoch + 1}")
        
        # High-performance progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}", 
                   ncols=100, leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            try:
                # Move data to device with non-blocking transfer
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
                with torch.cuda.amp.autocast(enabled=self.config['training']['use_amp']):
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict['loss'] / gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Update weights
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config['training'].get('max_grad_norm', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['training']['max_grad_norm']
                        )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                # Update metrics
                actual_loss = loss.item() * gradient_accumulation_steps
                total_loss += actual_loss
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{actual_loss:.4f}',
                    'Avg': f'{total_loss/(batch_idx+1):.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Memory cleanup
                if batch_idx % self.config.get('gpu_optimizations', {}).get('empty_cache_frequency', 30) == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        return avg_loss
    
    def validate(self):
        """High-performance validation loop."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        print(f"\n🔍 Validating...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", 
                                                 ncols=100, leave=False)):
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
                    with torch.cuda.amp.autocast(enabled=self.config['training']['use_amp']):
                        outputs = self.model(images)
                        loss_dict = self.criterion(outputs, targets)
                    
                    total_loss += loss_dict['loss'].item()
                    
                    # Convert outputs for evaluation
                    batch_predictions = self._convert_outputs_to_predictions(outputs)
                    batch_targets = self._convert_targets_for_evaluation(targets)
                    
                    all_predictions.extend(batch_predictions)
                    all_targets.extend(batch_targets)
                    
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
        else:
            metrics = {'micro_f1': 0.0, 'cutting_f1': 0.0}
        
        return avg_loss, metrics
    
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
    
    def _check_early_convergence(self, current_f1):
        """Check for early convergence."""
        if len(self.f1_history) == 0:
            self.no_improvement_count = 0
            return False
        
        best_f1 = max(self.f1_history)
        improvement = current_f1 - best_f1
        
        if improvement < self.min_improvement:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        
        return self.no_improvement_count >= self.convergence_patience
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_f1': self.best_f1,
            'config': self.config,
            'f1_history': self.f1_history,
            'loss_history': self.loss_history,
            'epoch_times': self.epoch_times
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved with F1: {self.best_f1:.4f}")
    
    def train(self):
        """Main progressive training loop."""
        print("🚀 Starting PROGRESSIVE training...")
        print(f"📊 Training for {self.config['training']['num_epochs']} epochs")
        print(f"🎯 Target F1 score: {self.config['training']['target_f1']}")
        
        # Initialize AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config['training']['use_amp'])
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            print(f"\n{'='*60}")
            print(f"🔄 EPOCH {epoch+1}/{self.config['training']['num_epochs']}")
            print(f"{'='*60}")
            
            # Adaptive batch sizing
            self._adaptive_batch_sizing()
            
            # Train epoch
            epoch_start = time.time()
            train_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # Validate
            val_loss, metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track history
            current_f1 = metrics['micro_f1']
            self.f1_history.append(current_f1)
            self.loss_history.append({'train': train_loss, 'val': val_loss})
            
            # GPU memory tracking
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1024**3
                self.gpu_memory_usage.append(memory_gb)
            
            # Log results
            print(f"\n📊 EPOCH {epoch+1} RESULTS:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val F1: {current_f1:.4f}")
            print(f"   Cutting F1: {metrics.get('cutting_f1', 0.0):.4f}")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"   Epoch Time: {epoch_time/60:.1f}min")
            if torch.cuda.is_available():
                print(f"   GPU Memory: {memory_gb:.1f}GB")
            
            # Check for improvement
            is_best = current_f1 > self.best_f1
            if is_best:
                self.best_f1 = current_f1
                print(f"🎉 New best F1: {self.best_f1:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best)
            
            # Early stopping checks
            if current_f1 >= self.config['training']['target_f1']:
                print(f"🎉 Target F1 {self.config['training']['target_f1']} reached! Stopping training.")
                break
            
            # Early convergence check
            if self._check_early_convergence(current_f1):
                print(f"🔄 Early convergence detected. Stopping training.")
                break
        
        total_time = time.time() - start_time
        print(f"\n🏁 Training completed in {total_time/3600:.1f} hours")
        print(f"🏆 Best F1 score: {self.best_f1:.4f}")
        
        # Performance summary
        self._performance_summary()
    
    def _performance_summary(self):
        """Print performance summary."""
        print(f"\n📈 PERFORMANCE SUMMARY:")
        if self.epoch_times:
            avg_epoch_time = np.mean(self.epoch_times)
            print(f"   Average epoch time: {avg_epoch_time/60:.1f}min")
        
        if self.gpu_memory_usage:
            avg_memory = np.mean(self.gpu_memory_usage)
            max_memory = np.max(self.gpu_memory_usage)
            print(f"   Average GPU memory: {avg_memory:.1f}GB")
            print(f"   Peak GPU memory: {max_memory:.1f}GB")
        
        if self.f1_history:
            max_f1_epoch = np.argmax(self.f1_history) + 1
            print(f"   Best F1 achieved at epoch: {max_f1_epoch}")
            print(f"   F1 progression: {[f'{f1:.3f}' for f1 in self.f1_history]}")

def main():
    parser = argparse.ArgumentParser(description='Progressive GPU Training for Cutting Behavior Detection')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ProgressiveTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_f1 = checkpoint.get('best_f1', 0.0)
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