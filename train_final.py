#!/usr/bin/env python3
"""
Simple Single-Process GPU Training Script for Cutting Behavior Detection
Removes all parallel processing and complex CUDA optimizations for stability
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

class SimpleTrainer:
    """Simple, single-threaded trainer with minimal CUDA complexity."""
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device - simple approach
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
        
        print(f"✅ Simple Trainer initialized")
        print(f"📊 Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"📈 Training: {len(self.train_loader)} batches, Validation: {len(self.val_loader)} batches")
    
    def _setup_device(self):
        """Simple device setup without complex memory management."""
        force_cpu = self.config.get('force_cpu', False)
        
        if force_cpu:
            print("🔧 Using CPU (forced)")
            return torch.device('cpu')
        elif torch.cuda.is_available():
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            # Simple CUDA setup - no complex memory management
            torch.backends.cudnn.benchmark = False  # Disable for stability
            torch.backends.cudnn.deterministic = True  # Enable for reproducibility
            return torch.device('cuda')
        else:
            print("🔧 Using CPU (CUDA not available)")
            return torch.device('cpu')
    
    def _create_model(self):
        model = CuttingDetector(self.config['model']).to(self.device)
        return model
    
    def _create_dataloaders(self):
        """Create simple, single-threaded dataloaders."""
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
        
        # Simple dataloaders - single threaded, no parallel processing
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True, 
            num_workers=0,  # Single-threaded
            collate_fn=collate_sequences, 
            pin_memory=False,  # Disable for simplicity
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=False, 
            num_workers=0,  # Single-threaded
            collate_fn=collate_sequences, 
            pin_memory=False,  # Disable for simplicity
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
    
    def simple_cleanup(self):
        """Simple memory cleanup without complex logic."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def train_epoch(self):
        """Simple, synchronous training loop."""
        self.model.train()
        total_loss = 0.0
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        print(f"🔄 Training Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            if batch is None:
                print(f"⚠️ Skipping None batch {batch_idx}")
                continue
            
            try:
                # Simple, synchronous data transfer
                images = batch['images'].to(self.device)
                targets = []
                for target in batch['targets']:
                    target_dict = {
                        'labels': target['labels'].to(self.device),
                        'boxes': target['boxes'].to(self.device),
                        'has_cutting': target['has_cutting']
                    }
                    if 'cutting' in target:
                        target_dict['cutting'] = target['cutting'].to(self.device)
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
                
                # Simple progress logging
                if batch_idx % 50 == 0:
                    print(f"   Batch {batch_idx}/{len(self.train_loader)}: Loss = {actual_loss:.4f}")
                
                # Simple cleanup every 20 batches
                if batch_idx % 20 == 0:
                    self.simple_cleanup()
                
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                # Simple error handling - just skip and continue
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
        print(f"✅ Training completed. Average Loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self):
        """Simple validation loop."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        print(f"🔍 Validating...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch is None:
                    continue
                
                try:
                    # Simple, synchronous data transfer
                    images = batch['images'].to(self.device)
                    targets = []
                    for target in batch['targets']:
                        target_dict = {
                            'labels': target['labels'].to(self.device),
                            'boxes': target['boxes'].to(self.device),
                            'has_cutting': target['has_cutting']
                        }
                        if 'cutting' in target:
                            target_dict['cutting'] = target['cutting'].to(self.device)
                        targets.append(target_dict)
                    
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    total_loss += loss_dict['loss'].item()
                    
                    # Convert outputs for evaluation
                    batch_predictions = self._convert_outputs_to_predictions(outputs)
                    batch_targets = self._convert_targets_for_evaluation(targets)
                    
                    all_predictions.extend(batch_predictions)
                    all_targets.extend(batch_targets)
                    
                    # Simple progress
                    if batch_idx % 20 == 0:
                        print(f"   Validation batch {batch_idx}/{len(self.val_loader)}")
                    
                    # Simple cleanup
                    if batch_idx % 10 == 0:
                        self.simple_cleanup()
                        
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
        
        print(f"✅ Validation completed. Average Loss: {avg_loss:.4f}")
        return avg_loss, metrics
    
    def _convert_outputs_to_predictions(self, outputs):
        """Convert model outputs to prediction format for evaluation."""
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
            'config': self.config
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
        """Main training loop."""
        print("🚀 Starting GPU-optimized training...")
        print(f"📊 Training for {self.config['training']['num_epochs']} epochs")
        print(f"🎯 Target F1 score: {self.config['training']['target_f1']}")
        
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
            
            print(f"✅ Epoch {epoch+1} completed in {epoch_time/60:.1f}min")
            
            # Validate
            val_loss, metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log results
            current_f1 = metrics['micro_f1']
            print(f"\n📊 EPOCH {epoch+1} RESULTS:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val F1: {current_f1:.4f}")
            print(f"   Cutting F1: {metrics.get('cutting_f1', 0.0):.4f}")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            is_best = current_f1 > self.best_f1
            if is_best:
                self.best_f1 = current_f1
            
            self.save_checkpoint(epoch + 1, is_best)
            
            # Early stopping check
            if current_f1 >= self.config['training']['target_f1']:
                print(f"🎉 Target F1 {self.config['training']['target_f1']} reached! Stopping training.")
                break
        
        total_time = time.time() - start_time
        print(f"\n🏁 Training completed in {total_time/3600:.1f} hours")
        print(f"🏆 Best F1 score: {self.best_f1:.4f}")

    def cleanup_and_exit(self):
        """Cleanup resources before exit."""
        print("🧹 Cleaning up resources...")
        self.simple_cleanup()
        
        if torch.cuda.is_available():
            # Final memory report
            memory_gb = self.memory_manager.get_memory_gb()
            print(f"📊 Final Memory Status:")
            print(f"   Allocated: {memory_gb:.2f}GB")
        
        print("✅ Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description='GPU-Optimized Cutting Behavior Detection Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SimpleTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
        
    finally:
        trainer.cleanup_and_exit()

if __name__ == "__main__":
    main() 