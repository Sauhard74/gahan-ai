"""
Optimized Training Script for Cutting Behavior Detection.
Incorporates all insights: mixed precision, proper F1 evaluation, CUDA optimizations.
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import wandb
from typing import Dict, List, Any

# Import our modules
from models.cutting_detector import create_cutting_detector
from datasets.cut_in_dataset import create_datasets
from losses.criterion import create_criterion
from utils.collate_fn import collate_sequences
from utils.evaluation import evaluate_cutting_detection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedTrainer:
    """
    Optimized trainer with all performance enhancements.
    """
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config['device']['use_cuda'] else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize mixed precision scaler (fix deprecated warning)
        if self.config['training']['use_amp'] and self.device.type == 'cuda':
            try:
                # Try new API first
                from torch.amp import GradScaler
                self.scaler = GradScaler('cuda')
            except (ImportError, TypeError):
                # Fall back to old API
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Create model, datasets, and training components
        self.model = self._create_model()
        self.train_loader, self.val_loader = self._create_dataloaders()
        self.criterion = self._create_criterion()
        self.optimizer, self.scheduler = self._create_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # Initialize wandb if configured
        if self.config.get('use_wandb', False):
            wandb.init(project="cutting-detection", config=self.config)
    
    def _create_model(self) -> nn.Module:
        """Create and initialize the model."""
        model = create_cutting_detector(self.config['model'])
        model = model.to(self.device)
        
        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        # Log model info
        model_info = model.module.get_model_info() if hasattr(model, 'module') else model.get_model_info()
        logger.info(f"Model: {model_info}")
        
        return model
    
    def _create_dataloaders(self) -> tuple:
        """Create train and validation dataloaders."""
        # Create datasets - pass full config, not just dataset section
        train_dataset, val_dataset = create_datasets(self.config)
        
        # Log dataset info
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")
        
        # Create dataloaders with optimizations
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['device']['num_workers'],
            pin_memory=self.config['device']['pin_memory'],
            collate_fn=collate_sequences,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['device']['num_workers'],
            pin_memory=self.config['device']['pin_memory'],
            collate_fn=collate_sequences,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        return train_loader, val_loader
    
    def _create_criterion(self):
        """Create loss criterion."""
        return create_criterion(self.config['training']['loss_weights'])
    
    def _create_optimizer(self):
        """Create optimizer and scheduler."""
        # Optimizer
        if self.config['training']['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        
        # Scheduler
        if self.config['training']['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config['training']['num_epochs']
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.1
            )
        
        return optimizer, scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device with non-blocking transfer
            images = batch['images'].to(self.device, non_blocking=True)
            targets = []
            
            for i in range(len(batch['targets'])):
                target = {
                    'labels': [labels.to(self.device, non_blocking=True) for labels in batch['targets'][i]['labels']],
                    'boxes': [boxes.to(self.device, non_blocking=True) for boxes in batch['targets'][i]['boxes']],
                    'has_cutting': batch['targets'][i]['has_cutting']
                }
                targets.append(target)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)  # More efficient
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict['loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['loss']
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        return {'total_loss': avg_loss, **loss_components}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model with proper F1 evaluation."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                images = batch['images'].to(self.device, non_blocking=True)
                targets = []
                
                for i in range(len(batch['targets'])):
                    target = {
                        'labels': [labels.to(self.device, non_blocking=True) for labels in batch['targets'][i]['labels']],
                        'boxes': [boxes.to(self.device, non_blocking=True) for boxes in batch['targets'][i]['boxes']],
                        'has_cutting': batch['targets'][i]['has_cutting']
                    }
                    targets.append(target)
                
                # Forward pass
                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss_dict = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                
                val_loss += loss_dict['loss'].item()
                
                # Convert outputs to predictions for evaluation
                batch_predictions = self._convert_outputs_to_predictions(outputs)
                batch_targets = self._convert_targets_for_evaluation(targets)
                
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)
        
        # Calculate proper F1 scores using IoU-based evaluation
        metrics = evaluate_cutting_detection(
            all_predictions, all_targets,
            iou_threshold=self.config['evaluation']['iou_threshold'],
            confidence_threshold=self.config['evaluation']['confidence_threshold']
        )
        
        metrics['val_loss'] = val_loss / len(self.val_loader)
        
        return metrics
    
    def _convert_outputs_to_predictions(self, outputs: Dict) -> List[Dict]:
        """Convert model outputs to prediction format for evaluation."""
        batch_size = outputs['pred_logits'].size(0)
        predictions = []
        
        for i in range(batch_size):
            # Get predictions for this sample
            logits = outputs['pred_logits'][i]
            boxes = outputs['pred_boxes'][i]
            objectness = outputs['pred_objectness'][i]
            
            # Convert to probabilities
            class_probs = torch.softmax(logits, dim=-1)
            objectness_probs = torch.sigmoid(objectness)
            
            # Get class predictions (excluding background)
            max_probs, class_ids = class_probs[:, 1:].max(dim=-1)
            class_ids += 1  # Adjust for background class
            
            # Combine with objectness for confidence
            confidence_scores = max_probs * objectness_probs.squeeze(-1)
            
            predictions.append({
                'boxes': boxes.cpu().numpy(),
                'labels': class_ids.cpu().numpy(),
                'scores': confidence_scores.cpu().numpy()
            })
        
        return predictions
    
    def _convert_targets_for_evaluation(self, targets: List[Dict]) -> List[Dict]:
        """Convert targets to evaluation format."""
        eval_targets = []
        
        for target in targets:
            # Combine all frames' annotations
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
            
            eval_targets.append({
                'boxes': combined_boxes,
                'labels': combined_labels
            })
        
        return eval_targets
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with F1: {metrics['micro_f1']:.4f}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            current_f1 = val_metrics['micro_f1']
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f}, "
                       f"Val F1: {current_f1:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_f1': current_f1,
                    'val_loss': val_metrics['val_loss'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = current_f1 > self.best_f1
            if is_best:
                self.best_f1 = current_f1
            
            self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping check
            if current_f1 >= self.config['evaluation']['target_f1']:
                logger.info(f"Target F1 {self.config['evaluation']['target_f1']} reached! Stopping training.")
                break
        
        logger.info(f"Training completed. Best F1: {self.best_f1:.4f}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Cutting Detection Model")
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = OptimizedTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_f1 = checkpoint['metrics'].get('micro_f1', 0.0)
        logger.info(f"Resumed from epoch {trainer.current_epoch}")
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 