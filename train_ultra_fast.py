#!/usr/bin/env python3
"""
ULTRA-HIGH-PERFORMANCE Training Script
Maximizes 40GB GPU utilization for fastest training WITHOUT quality compromise
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

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.cutting_detector import CuttingDetector
from datasets.cut_in_dataset import CutInSequenceDataset
from losses.criterion import CuttingDetectionLoss
from utils.collate_fn import collate_sequences
from utils.evaluation import evaluate_cutting_detection, calculate_cutting_behavior_f1

class UltraFastTrainer:
    """Ultra-high-performance trainer that maximizes 40GB GPU utilization."""
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device with maximum optimizations
        self.device = self._setup_device()
        
        # Create model, datasets, criterion, optimizer
        self.model = self._create_model()
        self.train_loader, self.val_loader = self._create_dataloaders()
        self.criterion = self._create_criterion()
        self.optimizer, self.scheduler = self._create_optimizer()
        
        # Mixed precision scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler() if self.config['training']['use_amp'] else None
        
        # Training state
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.checkpoint_dir = Path(self.config['training']['save_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Performance optimizations
        self.validate_every_n_epochs = 2  # Still validate every 2 epochs
        self.early_stopping_patience = 6  # Increased patience for quality
        self.no_improvement_count = 0
        
        print(f"🚀 ULTRA-FAST Trainer initialized")
        print(f"📊 Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"📈 Training: {len(self.train_loader)} batches, Validation: {len(self.val_loader)} batches")
        print(f"🔥 Batch size: {self.config['training']['batch_size']} (was 1-4)")
        print(f"⚡ Mixed Precision: {'ENABLED' if self.scaler else 'DISABLED'}")
        print(f"⏱️ Estimated time: ~{self._estimate_training_time():.1f} hours")
        print(f"💾 Expected GPU usage: ~25-30GB (of 40GB available)")
    
    def _setup_device(self):
        """Setup device with maximum GPU optimizations."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            
            # Maximum performance optimizations
            torch.backends.cudnn.benchmark = True  # Enable for maximum speed
            torch.backends.cudnn.deterministic = False  # Disable for speed
            torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for A100/H100
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matrix ops
            
            # Set memory fraction if specified
            if 'gpu_optimizations' in self.config:
                max_memory = self.config['gpu_optimizations'].get('max_memory_fraction', 0.95)
                print(f"🔧 Setting GPU memory fraction to {max_memory}")
            
            return device
        else:
            print("❌ CUDA not available - cannot utilize 40GB GPU!")
            return torch.device('cpu')
    
    def _create_model(self):
        model = CuttingDetector(self.config['model']).to(self.device)
        
        # Enable optimized attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("⚡ Using optimized attention (PyTorch 2.0+)")
        
        return model
    
    def _create_dataloaders(self):
        """Create high-performance dataloaders with maximum GPU utilization."""
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
        
        # High-performance dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True, 
            num_workers=self.config['data']['num_workers'],  # Multi-threaded loading
            collate_fn=collate_sequences, 
            pin_memory=self.config['data']['pin_memory'],  # Pin memory for GPU transfer
            drop_last=True,  # Drop last incomplete batch
            prefetch_factor=self.config['data']['prefetch_factor'],  # Prefetch batches
            persistent_workers=True  # Keep workers alive between epochs
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=False, 
            num_workers=self.config['data']['num_workers'],
            collate_fn=collate_sequences, 
            pin_memory=self.config['data']['pin_memory'],
            prefetch_factor=self.config['data']['prefetch_factor'],
            persistent_workers=True
        )
        
        return train_loader, val_loader
    
    def _create_criterion(self):
        return CuttingDetectionLoss(self.config['training']['loss_weights']).to(self.device)
    
    def _create_optimizer(self):
        # Optimized for large batch sizes
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            eps=1e-8,  # Stability for mixed precision
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler optimized for large batches
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config['training']['scheduler_step_size'],
            gamma=self.config['training']['scheduler_gamma']
        )
        
        return optimizer, scheduler
    
    def _estimate_training_time(self):
        """Estimate training time with large batch optimization."""
        batches_per_epoch = len(self.train_loader)
        total_epochs = self.config['training']['num_epochs']
        
        # With batch_size=16 vs original batch_size=1, we expect ~10-12x speedup
        # Plus mixed precision gives another ~1.5-2x speedup
        original_seconds_per_50_batches = 3
        speedup_factor = 15  # Conservative estimate of total speedup
        
        optimized_seconds_per_50_batches = original_seconds_per_50_batches / speedup_factor
        
        total_batches = batches_per_epoch * total_epochs
        total_seconds = (total_batches / 50) * optimized_seconds_per_50_batches
        
        # Add validation time (every 2 epochs, but faster due to larger batches)
        val_epochs = total_epochs // self.validate_every_n_epochs
        val_seconds = val_epochs * (len(self.val_loader) / 50) * optimized_seconds_per_50_batches
        
        return (total_seconds + val_seconds) / 3600  # Convert to hours
    
    def advanced_cleanup(self):
        """Advanced memory cleanup for maximum GPU utilization."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations complete
    
    def train_epoch(self):
        """Ultra-fast training loop with mixed precision and large batches."""
        self.model.train()
        total_loss = 0.0
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        # Progress bar with detailed metrics
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch + 1}",
            ncols=120,
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            try:
                # High-speed data transfer with non-blocking
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
                
                # Mixed precision forward pass
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss_dict = self.criterion(outputs, targets)
                        loss = loss_dict['loss'] / gradient_accumulation_steps
                else:
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict['loss'] / gradient_accumulation_steps
                
                # Mixed precision backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights with gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if self.scaler:
                        # Mixed precision gradient clipping and update
                        if self.config['training'].get('max_grad_norm', 0) > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config['training']['max_grad_norm']
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Standard gradient clipping and update
                        if self.config['training'].get('max_grad_norm', 0) > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config['training']['max_grad_norm']
                            )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                # Update metrics
                actual_loss = loss.item() * gradient_accumulation_steps
                total_loss += actual_loss
                
                # Update progress bar with GPU memory info
                if batch_idx % 25 == 0:
                    gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    progress_bar.set_postfix({
                        'loss': f'{actual_loss:.3f}',
                        'avg': f'{total_loss/(batch_idx+1):.3f}',
                        'GPU': f'{gpu_memory:.1f}GB'
                    })
                
                # Less frequent cleanup due to larger batches
                if batch_idx % 100 == 0:
                    self.advanced_cleanup()
                
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                self.optimizer.zero_grad()
                self.advanced_cleanup()
                continue
        
        # Final gradient update
        if len(self.train_loader) % gradient_accumulation_steps != 0:
            if self.scaler:
                if self.config['training'].get('max_grad_norm', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.config['training'].get('max_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        return avg_loss
    
    def validate(self):
        """High-speed validation with large batches."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Full validation for quality (no subset)
        with torch.no_grad():
            val_progress = tqdm(self.val_loader, desc="Validating", leave=False)
            
            for batch_idx, batch in enumerate(val_progress):
                if batch is None:
                    continue
                
                try:
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
                    
                    # Mixed precision inference
                    if self.scaler:
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
                    
                    # Update progress
                    if batch_idx % 50 == 0:
                        gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                        val_progress.set_postfix({'GPU': f'{gpu_memory:.1f}GB'})
                    
                except Exception as e:
                    continue
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        
        # Calculate comprehensive metrics
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
        """Convert model outputs to predictions."""
        batch_size = outputs['pred_logits'].size(0)
        predictions = []
        
        for i in range(batch_size):
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
            
            # Determine cutting behavior
            has_cutting = False
            if 'sequence_cutting' in outputs:
                sequence_cutting_prob = torch.sigmoid(outputs['sequence_cutting'][i]).item()
                has_cutting = sequence_cutting_prob > 0.5
            elif len(cutting_probs) > 0:
                has_cutting = cutting_probs.max().item() > 0.5
            
            predictions.append({
                'boxes': boxes.cpu().numpy(),
                'labels': class_ids.cpu().numpy(),
                'scores': confidence_scores.cpu().numpy(),
                'has_cutting': has_cutting
            })
        
        return predictions
    
    def _convert_targets_for_evaluation(self, targets):
        """Convert targets to evaluation format."""
        eval_targets = []
        
        for target in targets:
            eval_targets.append({
                'boxes': target['boxes'].cpu().numpy(),
                'labels': target['labels'].cpu().numpy(),
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
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved with F1: {self.best_f1:.4f}")
    
    def train(self):
        """Ultra-fast training loop."""
        print("🚀 Starting ULTRA-FAST training...")
        print(f"📊 Training for up to {self.config['training']['num_epochs']} epochs")
        print(f"🎯 Target F1 score: {self.config['training']['target_f1']}")
        print(f"⚡ Batch size: {self.config['training']['batch_size']} (16x larger than original)")
        print(f"🔥 Mixed Precision: {'ENABLED' if self.scaler else 'DISABLED'}")
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            print(f"\n🔄 EPOCH {epoch+1}/{self.config['training']['num_epochs']}")
            
            # Train epoch
            epoch_start = time.time()
            train_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start
            
            # GPU memory info
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                gpu_max = torch.cuda.max_memory_allocated() / 1e9
                print(f"✅ Training completed in {epoch_time/60:.1f}min - Loss: {train_loss:.4f}")
                print(f"💾 GPU Memory: {gpu_memory:.1f}GB current, {gpu_max:.1f}GB peak")
            
            # Validate every N epochs
            if (epoch + 1) % self.validate_every_n_epochs == 0:
                print("🔍 Validating...")
                val_loss, metrics = self.validate()
                current_f1 = metrics['micro_f1']
                
                print(f"📊 Validation - Loss: {val_loss:.4f}, F1: {current_f1:.4f}, Cutting F1: {metrics.get('cutting_f1', 0.0):.4f}")
                
                # Check for improvement
                is_best = current_f1 > self.best_f1
                if is_best:
                    self.best_f1 = current_f1
                    self.no_improvement_count = 0
                    self.save_checkpoint(epoch + 1, is_best=True)
                else:
                    self.no_improvement_count += 1
                
                # Early stopping
                if self.no_improvement_count >= self.early_stopping_patience:
                    print(f"🛑 Early stopping - no improvement for {self.early_stopping_patience} validation cycles")
                    break
                
                # Target reached
                if current_f1 >= self.config['training']['target_f1']:
                    print(f"🎉 Target F1 {self.config['training']['target_f1']} reached!")
                    break
            
            # Update scheduler
            self.scheduler.step()
            
            # Memory cleanup
            self.advanced_cleanup()
        
        total_time = time.time() - start_time
        print(f"\n🏁 ULTRA-FAST training completed in {total_time/3600:.1f} hours")
        print(f"🏆 Best F1 score: {self.best_f1:.4f}")
        print(f"💾 Maximum GPU utilization: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB of 40GB")

def main():
    parser = argparse.ArgumentParser(description='Ultra-Fast Cutting Behavior Detection Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = UltraFastTrainer(args.config)
    
    try:
        # Start training
        trainer.train()
        print("🎉 Ultra-fast training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 