#!/usr/bin/env python3
"""
FAST Training Script for Cutting Behavior Detection
Optimized to complete training in under 8 hours
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

class FastTrainer:
    """Speed-optimized trainer for 8-hour training window."""
    
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
        
        # Speed optimizations
        self.validate_every_n_epochs = 2  # Validate every 2 epochs instead of every epoch
        self.early_stopping_patience = 5
        self.no_improvement_count = 0
        
        print(f"🚀 Fast Trainer initialized")
        print(f"📊 Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"📈 Training: {len(self.train_loader)} batches, Validation: {len(self.val_loader)} batches")
        print(f"⏱️ Estimated time: ~{self._estimate_training_time():.1f} hours")
    
    def _setup_device(self):
        """Simple device setup."""
        if torch.cuda.is_available():
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            torch.backends.cudnn.benchmark = True  # Enable for speed
            torch.backends.cudnn.deterministic = False  # Disable for speed
            return torch.device('cuda')
        else:
            print("🔧 Using CPU")
            return torch.device('cpu')
    
    def _create_model(self):
        model = CuttingDetector(self.config['model']).to(self.device)
        return model
    
    def _create_dataloaders(self):
        """Create optimized dataloaders."""
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
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True, 
            num_workers=0,
            collate_fn=collate_sequences, 
            pin_memory=False,
            drop_last=True  # Drop last incomplete batch for speed
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=False, 
            num_workers=0,
            collate_fn=collate_sequences, 
            pin_memory=False,
            drop_last=False
        )
        
        return train_loader, val_loader
    
    def _create_criterion(self):
        return CuttingDetectionLoss(self.config['training']['loss_weights']).to(self.device)
    
    def _create_optimizer(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config['training']['scheduler_step_size'],
            gamma=self.config['training']['scheduler_gamma']
        )
        
        return optimizer, scheduler
    
    def _estimate_training_time(self):
        """Estimate total training time."""
        batches_per_epoch = len(self.train_loader)
        total_epochs = self.config['training']['num_epochs']
        seconds_per_50_batches = 3  # Current speed
        
        total_batches = batches_per_epoch * total_epochs
        total_seconds = (total_batches / 50) * seconds_per_50_batches
        
        # Add validation time (every 2 epochs)
        val_epochs = total_epochs // self.validate_every_n_epochs
        val_seconds = val_epochs * len(self.val_loader) * 0.5  # Assume 0.5 sec per val batch
        
        return (total_seconds + val_seconds) / 3600  # Convert to hours
    
    def simple_cleanup(self):
        """Simple memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def train_epoch(self):
        """Fast training loop with minimal logging."""
        self.model.train()
        total_loss = 0.0
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
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
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['loss'] / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
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
                
                # Update progress bar less frequently
                if batch_idx % 100 == 0:
                    progress_bar.set_postfix({
                        'loss': f'{actual_loss:.3f}',
                        'avg': f'{total_loss/(batch_idx+1):.3f}'
                    })
                
                # Memory cleanup less frequently
                if batch_idx % 50 == 0:
                    self.simple_cleanup()
                
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                self.optimizer.zero_grad()
                self.simple_cleanup()
                continue
        
        # Final gradient update
        if len(self.train_loader) % gradient_accumulation_steps != 0:
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
        """Fast validation with reduced dataset."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Validate on subset for speed
        max_val_batches = min(len(self.val_loader), 500)  # Limit validation batches
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= max_val_batches:
                    break
                    
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
                    
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    total_loss += loss_dict['loss'].item()
                    
                    # Convert outputs for evaluation (simplified)
                    batch_predictions = self._convert_outputs_to_predictions(outputs)
                    batch_targets = self._convert_targets_for_evaluation(targets)
                    
                    all_predictions.extend(batch_predictions)
                    all_targets.extend(batch_targets)
                    
                except Exception as e:
                    continue
        
        avg_loss = total_loss / max_val_batches if max_val_batches > 0 else 0.0
        
        # Calculate metrics on subset
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
        """Simplified output conversion."""
        batch_size = outputs['pred_logits'].size(0)
        predictions = []
        
        for i in range(batch_size):
            logits = outputs['pred_logits'][i]
            boxes = outputs['pred_boxes'][i]
            
            # Simple conversion
            class_probs = torch.softmax(logits, dim=-1)
            max_probs, class_ids = class_probs[:, 1:].max(dim=-1)
            class_ids += 1
            
            # Simple cutting detection
            has_cutting = False
            if 'sequence_cutting' in outputs:
                has_cutting = torch.sigmoid(outputs['sequence_cutting'][i]).item() > 0.5
            
            predictions.append({
                'boxes': boxes.cpu().numpy(),
                'labels': class_ids.cpu().numpy(),
                'scores': max_probs.cpu().numpy(),
                'has_cutting': has_cutting
            })
        
        return predictions
    
    def _convert_targets_for_evaluation(self, targets):
        """Simplified target conversion."""
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
        
        # Save only best model to save space
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved with F1: {self.best_f1:.4f}")
    
    def train(self):
        """Fast training loop with early stopping."""
        print("🚀 Starting FAST training...")
        print(f"📊 Training for up to {self.config['training']['num_epochs']} epochs")
        print(f"🎯 Target F1 score: {self.config['training']['target_f1']}")
        print(f"⚡ Validating every {self.validate_every_n_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            print(f"\n🔄 EPOCH {epoch+1}/{self.config['training']['num_epochs']}")
            
            # Train epoch
            epoch_start = time.time()
            train_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start
            
            print(f"✅ Training completed in {epoch_time/60:.1f}min - Loss: {train_loss:.4f}")
            
            # Validate only every N epochs
            if (epoch + 1) % self.validate_every_n_epochs == 0:
                print("🔍 Validating...")
                val_loss, metrics = self.validate()
                current_f1 = metrics['micro_f1']
                
                print(f"📊 Validation - Loss: {val_loss:.4f}, F1: {current_f1:.4f}")
                
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
            self.simple_cleanup()
        
        total_time = time.time() - start_time
        print(f"\n🏁 Training completed in {total_time/3600:.1f} hours")
        print(f"🏆 Best F1 score: {self.best_f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Fast Cutting Behavior Detection Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = FastTrainer(args.config)
    
    try:
        # Start training
        trainer.train()
        print("🎉 Fast training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 