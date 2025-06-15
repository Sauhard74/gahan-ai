#!/usr/bin/env python3
"""
Test script to verify in-place operation fixes
"""

import torch
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.cutting_detector import CuttingDetector
from losses.criterion import CuttingDetectionLoss

def test_inplace_fixes():
    """Test that in-place operations are fixed."""
    print("🧪 Testing in-place operation fixes...")
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Create simple config
    config = {
        'model': {
            'backbone': 'google/vit-base-patch16-224',
            'hidden_dim': 256,
            'num_classes': 4,
            'sequence_length': 2,
            'num_queries': 20,
            'temporal_encoder': {
                'type': 'bidirectional_gru',
                'hidden_size': 128,
                'num_layers': 1,
                'dropout': 0.1,
                'use_attention': False
            },
            'decoder': {
                'num_queries': 20,
                'num_layers': 2,
                'dropout': 0.1
            }
        },
        'training': {
            'loss_weights': {
                'classification': 1.0,
                'bbox_regression': 5.0,
                'giou': 2.0,
                'cutting': 3.0,
                'sequence': 2.0
            }
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Create model and criterion
    model = CuttingDetector(config['model']).to(device)
    criterion = CuttingDetectionLoss(config['training']['loss_weights']).to(device)
    
    # Create dummy data
    batch_size = 2
    seq_len = 2
    images = torch.randn(batch_size, seq_len, 3, 224, 224).to(device)
    
    # Create dummy targets
    targets = []
    for i in range(batch_size):
        targets.append({
            'labels': torch.tensor([1, 2], device=device),
            'boxes': torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.7, 0.7]], device=device),
            'has_cutting': True,
            'cutting': torch.tensor([1.0, 0.0], device=device)
        })
    
    print("✅ Created model and dummy data")
    
    # Test forward pass with gradient computation
    try:
        model.train()
        
        # Forward pass
        outputs = model(images)
        print("✅ Forward pass successful")
        
        # Loss computation
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['loss']
        print(f"✅ Loss computation successful: {loss.item():.4f}")
        
        # Backward pass (this is where in-place operations would fail)
        loss.backward()
        print("✅ Backward pass successful - no in-place operation errors!")
        
        return True
        
    except RuntimeError as e:
        if "inplace operation" in str(e).lower():
            print(f"❌ In-place operation error still exists: {e}")
            return False
        else:
            print(f"❌ Other error: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_inplace_fixes()
    if success:
        print("\n🎉 All in-place operation fixes are working correctly!")
        print("✅ Training should now work without gradient computation errors")
    else:
        print("\n❌ In-place operation issues still exist")
        print("🔧 Additional fixes may be needed") 