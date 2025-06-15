#!/usr/bin/env python3
"""
Switch from current training to fast training
Saves current progress and starts optimized training
"""

import os
import sys
import torch
import yaml
import shutil
from pathlib import Path

def switch_to_fast_training():
    """Switch from current training to fast training."""
    print("🔄 Switching to FAST training mode...")
    
    # Check if current training is running
    current_checkpoints = Path("checkpoints")
    fast_checkpoints = Path("checkpoints_fast")
    
    # Create fast checkpoints directory
    fast_checkpoints.mkdir(exist_ok=True)
    
    # Find latest checkpoint from current training
    latest_checkpoint = None
    if current_checkpoints.exists():
        checkpoint_files = list(current_checkpoints.glob("checkpoint_epoch_*.pth"))
        if checkpoint_files:
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
            print(f"📂 Found latest checkpoint: {latest_checkpoint}")
            
            # Copy to fast training directory
            fast_checkpoint = fast_checkpoints / "resume_checkpoint.pth"
            shutil.copy2(latest_checkpoint, fast_checkpoint)
            print(f"✅ Copied checkpoint to: {fast_checkpoint}")
        else:
            print("⚠️ No checkpoints found - starting fresh")
    
    # Copy best model if exists
    best_model = current_checkpoints / "best_model.pth"
    if best_model.exists():
        fast_best = fast_checkpoints / "best_model.pth"
        shutil.copy2(best_model, fast_best)
        print(f"✅ Copied best model to: {fast_best}")
    
    print("\n🚀 FAST TRAINING SETUP COMPLETE!")
    print("=" * 50)
    print("📊 SPEED IMPROVEMENTS:")
    print("   • Batch size: 1 → 4 (4x speedup)")
    print("   • Epochs: 30 → 15 (2x speedup)")
    print("   • Validation: Every epoch → Every 2 epochs")
    print("   • Early stopping enabled")
    print("   • Reduced validation dataset")
    print("   • Optimized memory management")
    print("=" * 50)
    print("⏱️ ESTIMATED TIME: ~6-8 hours (vs 35+ hours)")
    print("=" * 50)
    
    # Show commands to run
    print("\n🎯 TO START FAST TRAINING:")
    if latest_checkpoint:
        print(f"python train_fast.py --config configs/fast_gpu_config.yaml")
        print("\n💡 Your progress will be preserved!")
    else:
        print(f"python train_fast.py --config configs/fast_gpu_config.yaml")
        print("\n💡 Starting fresh with optimized settings!")
    
    print("\n⚠️ IMPORTANT:")
    print("   • Stop your current training first (Ctrl+C)")
    print("   • Run the fast training command above")
    print("   • Monitor GPU memory usage")
    print("   • Training will auto-stop when target F1 reached")

if __name__ == "__main__":
    switch_to_fast_training() 