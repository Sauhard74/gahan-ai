"""
Debug Training Script - Test with minimal data to identify issues.
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path

# Import our modules
from models.cutting_detector import create_cutting_detector
from datasets.cut_in_dataset import create_datasets
from losses.criterion import create_criterion
from utils.collate_fn import collate_sequences

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_training():
    """Debug the training pipeline with minimal data."""
    
    # Load config
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config for debugging
    config['dataset']['oversample_positive'] = 2  # Reduce oversampling
    config['training']['batch_size'] = 2  # Small batch size
    
    logger.info("🔧 Starting debug training with minimal data...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("📊 Creating datasets...")
    try:
        train_dataset, val_dataset = create_datasets(config)
        logger.info(f"✅ Train dataset: {len(train_dataset)} samples")
        logger.info(f"✅ Val dataset: {len(val_dataset)} samples")
    except Exception as e:
        logger.error(f"❌ Dataset creation failed: {e}")
        return
    
    # Create dataloaders
    logger.info("🔄 Creating dataloaders...")
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,  # No multiprocessing for debugging
            collate_fn=collate_sequences
        )
        logger.info(f"✅ Train loader: {len(train_loader)} batches")
    except Exception as e:
        logger.error(f"❌ DataLoader creation failed: {e}")
        return
    
    # Create model
    logger.info("🤖 Creating model...")
    try:
        model = create_cutting_detector(config['model'])
        model = model.to(device)
        logger.info(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        return
    
    # Create criterion
    logger.info("📏 Creating loss criterion...")
    try:
        criterion = create_criterion(config['training']['loss_weights'])
        logger.info("✅ Criterion created")
    except Exception as e:
        logger.error(f"❌ Criterion creation failed: {e}")
        return
    
    # Test single batch
    logger.info("🧪 Testing single batch...")
    try:
        model.train()
        
        # Get first batch
        batch = next(iter(train_loader))
        logger.info(f"📦 Batch loaded:")
        logger.info(f"   Images shape: {batch['images'].shape}")
        logger.info(f"   Targets: {len(batch['targets'])} samples")
        
        # Move to device
        images = batch['images'].to(device)
        targets = []
        for i, target in enumerate(batch['targets']):
            target_dict = {
                'labels': target['labels'].to(device),
                'boxes': target['boxes'].to(device),
                'has_cutting': target['has_cutting']
            }
            targets.append(target_dict)
            logger.info(f"   Target {i}: {len(target['labels'])} objects, cutting={target['has_cutting']}")
        
        # Forward pass
        logger.info("⏩ Forward pass...")
        outputs = model(images)
        logger.info(f"✅ Forward pass successful")
        logger.info(f"   Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"   {key}: {value.shape}")
        
        # Loss computation
        logger.info("📊 Computing loss...")
        loss_dict = criterion(outputs, targets)
        logger.info(f"✅ Loss computation successful")
        logger.info(f"   Loss components:")
        for key, value in loss_dict.items():
            logger.info(f"   {key}: {value.item():.4f}")
        
        # Backward pass
        logger.info("⏪ Backward pass...")
        loss = loss_dict['loss']
        loss.backward()
        logger.info(f"✅ Backward pass successful")
        
        logger.info("🎉 Debug test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test multiple batches
    logger.info("🔄 Testing multiple batches...")
    try:
        total_loss = 0.0
        num_batches = min(5, len(train_loader))  # Test first 5 batches
        
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
                
            logger.info(f"📦 Batch {i+1}/{num_batches}")
            
            # Move to device
            images = batch['images'].to(device)
            targets = []
            for target in batch['targets']:
                target_dict = {
                    'labels': target['labels'].to(device),
                    'boxes': target['boxes'].to(device),
                    'has_cutting': target['has_cutting']
                }
                targets.append(target_dict)
            
            # Forward pass
            outputs = model(images)
            
            # Loss computation
            loss_dict = criterion(outputs, targets)
            total_loss += loss_dict['loss'].item()
            
            logger.info(f"   Loss: {loss_dict['loss'].item():.4f}")
        
        avg_loss = total_loss / num_batches
        logger.info(f"✅ Average loss over {num_batches} batches: {avg_loss:.4f}")
        
        if avg_loss < 0:
            logger.error("❌ NEGATIVE LOSS DETECTED! This indicates a bug in the loss function.")
        elif avg_loss > 100:
            logger.warning("⚠️  Very high loss detected. Check data normalization and loss scaling.")
        else:
            logger.info("✅ Loss values look reasonable.")
            
    except Exception as e:
        logger.error(f"❌ Multi-batch test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training() 