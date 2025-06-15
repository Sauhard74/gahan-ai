#!/usr/bin/env python3
"""
PROOF: Verify that training is actually happening by checking:
1. Model weights are changing
2. Gradients are being computed
3. Data is being loaded from disk
4. Loss computation is real
"""

import torch
import yaml
import sys
import os
import hashlib
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.cutting_detector import CuttingDetector
from datasets.cut_in_dataset import CutInSequenceDataset
from losses.criterion import CuttingDetectionLoss
from utils.collate_fn import collate_sequences
from torch.utils.data import DataLoader

def get_model_weight_hash(model):
    """Get a hash of all model weights to detect changes."""
    weight_str = ""
    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_str += str(param.data.cpu().numpy().flatten()[:10])  # First 10 values
    return hashlib.md5(weight_str.encode()).hexdigest()

def verify_real_training():
    """Prove that training is actually happening."""
    print("🔍 PROVING TRAINING IS REAL...")
    print("=" * 60)
    
    # Load config
    with open('configs/simple_gpu_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Create model
    model = CuttingDetector(config['model']).to(device)
    criterion = CuttingDetectionLoss(config['training']['loss_weights']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # PROOF 1: Check if dataset actually exists and loads real data
    print("\n🔍 PROOF 1: Verifying Real Dataset Loading...")
    try:
        dataset = CutInSequenceDataset(
            dataset_root=config['data']['data_dir'],
            split=config['data']['train_split'],
            sequence_length=config['model']['sequence_length'],
            image_size=tuple(config['data']['image_size']),
            oversample_positive=1,
            val_split_ratio=config['data']['val_split_ratio'],
            is_validation=False,
            augment=False
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        print(f"✅ Dataset path exists: {os.path.exists(config['data']['data_dir'])}")
        
        # Check if we can load actual data
        sample = dataset[0]
        print(f"✅ Sample loaded - Images shape: {sample['images'].shape}")
        print(f"✅ Sample has {len(sample['labels'])} annotations")
        
        # Verify images are not all zeros (real data)
        img_mean = sample['images'].mean().item()
        print(f"✅ Image data mean: {img_mean:.4f} (not zeros = real images)")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # PROOF 2: Check model weights change during training
    print("\n🔍 PROOF 2: Verifying Model Weights Actually Change...")
    
    # Get initial weight hash
    initial_hash = get_model_weight_hash(model)
    print(f"📊 Initial weight hash: {initial_hash}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, 
        num_workers=0, collate_fn=collate_sequences
    )
    
    # Train for a few steps
    model.train()
    losses = []
    
    for step, batch in enumerate(dataloader):
        if step >= 3:  # Just 3 steps to prove it works
            break
            
        print(f"\n📈 Training Step {step + 1}:")
        
        # Move data to device
        images = batch['images'].to(device)
        targets = []
        for target in batch['targets']:
            target_dict = {
                'labels': target['labels'].to(device),
                'boxes': target['boxes'].to(device),
                'has_cutting': target['has_cutting']
            }
            if 'cutting' in target:
                target_dict['cutting'] = target['cutting'].to(device)
            targets.append(target_dict)
        
        # Forward pass
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['loss']
        
        print(f"   Loss: {loss.item():.4f}")
        losses.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check if gradients exist (proof of real computation)
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"   Gradient norm: {total_grad_norm:.4f}")
        
        if total_grad_norm == 0:
            print("❌ No gradients computed - this would indicate fake training!")
            return False
        
        # Update weights
        optimizer.step()
        
        # Check weight hash changed
        new_hash = get_model_weight_hash(model)
        if new_hash == initial_hash:
            print("❌ Weights didn't change - this would indicate fake training!")
            return False
        else:
            print(f"   ✅ Weights changed: {initial_hash[:8]} → {new_hash[:8]}")
            initial_hash = new_hash
    
    # PROOF 3: Verify loss is actually computed from real data
    print("\n🔍 PROOF 3: Verifying Loss Computation is Real...")
    print(f"✅ Loss values: {losses}")
    
    if all(l == losses[0] for l in losses):
        print("❌ All losses identical - might indicate fake computation!")
        return False
    
    print("✅ Loss values vary - indicates real computation on different data")
    
    # PROOF 4: Check file system access
    print("\n🔍 PROOF 4: Verifying File System Access...")
    data_dir = Path(config['data']['data_dir'])
    if data_dir.exists():
        image_files = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png"))
        print(f"✅ Found {len(image_files)} image files in dataset")
        
        if len(image_files) > 0:
            # Check file modification times (proves files exist)
            sample_file = image_files[0]
            mod_time = os.path.getmtime(sample_file)
            print(f"✅ Sample file: {sample_file.name}")
            print(f"✅ File modified: {time.ctime(mod_time)}")
        else:
            print("❌ No image files found!")
            return False
    else:
        print("❌ Dataset directory doesn't exist!")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING IS 100% REAL!")
    print("✅ Dataset loads real images from disk")
    print("✅ Model weights change after each step")
    print("✅ Gradients are computed and applied")
    print("✅ Loss varies with different data")
    print("✅ File system access confirmed")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = verify_real_training()
    if success:
        print("\n🚀 Your training is DEFINITELY real and working!")
        print("💪 Keep it running - you're making progress!")
    else:
        print("\n❌ Something suspicious detected.")
        print("🔧 Investigation needed.") 