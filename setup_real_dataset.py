"""
Setup script for real dataset configuration.
Helps configure the correct dataset path and validates the structure.
"""

import os
import yaml
from pathlib import Path
import argparse

def detect_environment():
    """Detect the current environment (Colab, Kaggle, or local)."""
    if 'COLAB_GPU' in os.environ:
        print("🔍 Detected Google Colab environment")
        return 'colab'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        print("🔍 Detected Kaggle environment")
        return 'kaggle'
    else:
        print("🔍 Detected local environment")
        return 'local'

def find_dataset_paths(environment):
    """Find potential dataset paths based on environment."""
    print(f"🔎 Searching for dataset in {environment} environment...")
    
    if environment == 'colab':
        potential_paths = [
            '/content/drive/MyDrive/dataset',
            '/content/drive/MyDrive/cutting_behavior_dataset',
            '/content/dataset',
            '/content/data',
            '/content/distribution'
        ]
    elif environment == 'kaggle':
        potential_paths = [
            '/kaggle/input',
            '/kaggle/working/dataset',
            '/kaggle/working/data'
        ]
    else:  # local
        potential_paths = [
            './dataset',
            './data',
            '../dataset',
            '../data',
            '~/dataset',
            '~/data'
        ]
    
    found_paths = []
    for path in potential_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            print(f"✅ Found potential dataset path: {expanded_path}")
            found_paths.append(expanded_path)
        else:
            print(f"❌ Path not found: {expanded_path}")
    
    return found_paths

def validate_dataset_structure(dataset_path):
    """Validate that the dataset has the expected structure."""
    print(f"🔍 Validating dataset structure at: {dataset_path}")
    
    required_splits = ['Train', 'Test']  # Common splits
    found_splits = []
    
    for split in required_splits:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            print(f"✅ Found {split} split")
            found_splits.append(split)
            
            # Check for REC folders
            rec_folders = [f for f in os.listdir(split_path) 
                          if os.path.isdir(os.path.join(split_path, f)) and f.startswith('REC_')]
            print(f"   📁 Found {len(rec_folders)} REC folders")
            
            if len(rec_folders) > 0:
                # Check first REC folder structure
                first_rec = os.path.join(split_path, rec_folders[0])
                annotations_path = os.path.join(first_rec, 'Annotations')
                if os.path.exists(annotations_path):
                    print(f"   ✅ Annotations folder found in {rec_folders[0]}")
                    
                    # Count files
                    image_files = []
                    xml_files = []
                    for file in os.listdir(annotations_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_files.append(file)
                        elif file.lower().endswith('.xml'):
                            xml_files.append(file)
                    
                    print(f"   📸 Sample counts - Images: {len(image_files)}, Annotations: {len(xml_files)}")
                else:
                    print(f"   ❌ No Annotations folder in {rec_folders[0]}")
        else:
            print(f"❌ {split} split not found")
    
    return len(found_splits) > 0, found_splits

def update_config_file(dataset_path, config_path='configs/experiment_config.yaml'):
    """Update the configuration file with the correct dataset path."""
    print(f"📝 Updating configuration file: {config_path}")
    
    try:
        # Read current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update dataset path
        old_path = config['data']['data_dir']
        config['data']['data_dir'] = dataset_path
        
        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Updated dataset path:")
        print(f"   Old: {old_path}")
        print(f"   New: {dataset_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating config file: {e}")
        return False

def main():
    print("🚀 Setting up real dataset configuration...")
    print("=" * 60)
    
    # Detect environment
    env = detect_environment()
    
    # Find potential dataset paths
    potential_paths = find_dataset_paths(env)
    
    if not potential_paths:
        print("❌ No potential dataset paths found!")
        print("💡 Please ensure your dataset is uploaded and accessible.")
        if env == 'colab':
            print("   For Colab: Mount Google Drive and place dataset in /content/drive/MyDrive/")
        elif env == 'kaggle':
            print("   For Kaggle: Add dataset as input or upload to /kaggle/working/")
        else:
            print("   For local: Place dataset in ./dataset/ or ./data/")
        return False
    
    # Validate each potential path
    valid_paths = []
    for path in potential_paths:
        is_valid, splits = validate_dataset_structure(path)
        if is_valid:
            valid_paths.append((path, splits))
            print(f"✅ Valid dataset found at: {path}")
        else:
            print(f"❌ Invalid dataset structure at: {path}")
    
    if not valid_paths:
        print("❌ No valid datasets found!")
        print("💡 Expected structure:")
        print("   dataset/")
        print("   ├── Train/")
        print("   │   ├── REC_YYYY_MM_DD_HH_MM_SS_F/")
        print("   │   │   └── Annotations/")
        print("   │   │       ├── *.jpg")
        print("   │   │       └── *.xml")
        print("   └── Test/")
        print("       └── ...")
        return False
    
    # Use the first valid path
    selected_path, splits = valid_paths[0]
    print(f"🎯 Selected dataset: {selected_path}")
    print(f"📊 Available splits: {', '.join(splits)}")
    
    # Update configuration
    success = update_config_file(selected_path)
    
    if success:
        print("\n🎉 Dataset setup completed successfully!")
        print("✅ Configuration file updated")
        print("🚀 You can now run training with:")
        print("   python train_final.py --config configs/experiment_config.yaml")
    else:
        print("\n❌ Dataset setup failed!")
        print("💡 Please manually update the 'data_dir' in configs/experiment_config.yaml")
    
    return success

if __name__ == "__main__":
    main() 