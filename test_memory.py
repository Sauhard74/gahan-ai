#!/usr/bin/env python3
"""
Quick memory test for 40GB A100 GPU
"""

import torch
import yaml
from models.cutting_detector import CuttingDetector

def test_memory():
    print("🧪 Testing memory usage on 40GB A100...")
    
    # Load config
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    print(f"🚀 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Create model
        print("🤖 Creating model...")
        model = CuttingDetector(config['model']).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {total_params:,}")
        
        # Test forward pass
        batch_size = config['training']['batch_size']
        seq_len = config['model']['sequence_length']
        
        print(f"🔄 Testing forward pass (batch_size={batch_size}, seq_len={seq_len})...")
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len, 3, 224, 224).to(device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_percent = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
        
        print(f"✅ Forward pass successful!")
        print(f"💾 Memory used: {memory_used:.2f} GB ({memory_percent:.1f}%)")
        
        if memory_percent < 80:
            print("🎉 Memory usage is SAFE for training!")
        elif memory_percent < 90:
            print("⚠️ Memory usage is HIGH but should work")
        else:
            print("❌ Memory usage is TOO HIGH - reduce batch size further")
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        print("💡 Try reducing batch_size or model size further")
    
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    test_memory() 