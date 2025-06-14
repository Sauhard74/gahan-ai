"""
Setup script for real dataset configuration.
Helps configure the correct dataset path and validates the structure.
"""

import os
import yaml
from pathlib import Path
import argparse

def find_dataset_paths():
    """Find potential dataset paths."""
    potential_paths = [
        "/content/distribution",  # Colab
        "dataset/dataset/distribution",  # Local nested
        "dataset/distribution",  # Local simple
        "../dataset/distribution",  # Parent directory
        "../../dataset/distribution",  # Grandparent directory
        "/kaggle/input/distribution",  # Kaggle
        "/data/distribution",  # Common data directory
    ]
    
    found_paths = []
    for path in potential_paths:
        if Path(path).exists():
            train_path = Path(path) / "Train"
            if train_path.exists():
                rec_folders = [f for f in train_path.iterdir() if f.is_dir() and f.name.startswith('REC_')]
                if len(rec_folders) > 0:
                    found_paths.append((path, len(rec_folders)))
    
    return found_paths

def validate_dataset_structure(dataset_root: str):
    """Validate the dataset structure."""
    root_path = Path(dataset_root)
    
    print(f"🔍 Validating dataset structure at: {dataset_root}")
    
    if not root_path.exists():
        print(f"❌ Dataset root not found: {dataset_root}")
        return False
    
    # Check Train folder
    train_path = root_path / "Train"
    if not train_path.exists():
        print(f"❌ Train folder not found: {train_path}")
        return False
    
    # Check REC folders
    rec_folders = [f for f in train_path.iterdir() if f.is_dir() and f.name.startswith('REC_')]
    if len(rec_folders) == 0:
        print(f"❌ No REC folders found in: {train_path}")
        return False
    
    print(f"✅ Found {len(rec_folders)} REC folders")
    
    # Check first few REC folders for structure
    valid_folders = 0
    total_images = 0
    total_annotations = 0
    
    for i, rec_folder in enumerate(rec_folders[:3]):  # Check first 3
        print(f"📂 Checking {rec_folder.name}...")
        
        annotations_folder = rec_folder / "Annotations"
        if not annotations_folder.exists():
            print(f"   ⚠️  No Annotations folder in {rec_folder.name}")
            continue
        
        # Count images and annotations
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(annotations_folder.glob(ext)))
        
        annotation_files = list(annotations_folder.glob('*.xml'))
        
        print(f"   📊 Found {len(image_files)} images, {len(annotation_files)} annotations")
        
        if len(image_files) > 0 and len(annotation_files) > 0:
            valid_folders += 1
            total_images += len(image_files)
            total_annotations += len(annotation_files)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total REC folders: {len(rec_folders)}")
    print(f"   Valid folders checked: {valid_folders}/3")
    print(f"   Estimated total images: ~{total_images * len(rec_folders) // 3:,}")
    print(f"   Estimated total annotations: ~{total_annotations * len(rec_folders) // 3:,}")
    
    if valid_folders > 0:
        print(f"✅ Dataset structure is valid!")
        return True
    else:
        print(f"❌ Dataset structure is invalid!")
        return False

def update_config(dataset_root: str, config_path: str = "configs/experiment_config.yaml"):
    """Update the configuration file with the correct dataset path."""
    
    print(f"🔧 Updating configuration file: {config_path}")
    
    # Load current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update dataset root
    config['paths']['dataset_root'] = dataset_root
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Configuration updated with dataset_root: {dataset_root}")

def main():
    parser = argparse.ArgumentParser(description="Setup real dataset configuration")
    parser.add_argument('--dataset-path', type=str, help='Path to dataset root')
    parser.add_argument('--auto-find', action='store_true', help='Automatically find dataset')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml', 
                       help='Path to config file')
    
    args = parser.parse_args()
    
    print("🚀 Setting up real dataset configuration...")
    
    if args.dataset_path:
        # Use provided path
        dataset_path = args.dataset_path
        print(f"📁 Using provided dataset path: {dataset_path}")
        
    elif args.auto_find:
        # Auto-find dataset
        print("🔍 Searching for dataset...")
        found_paths = find_dataset_paths()
        
        if not found_paths:
            print("❌ No valid dataset found in common locations.")
            print("Please provide the dataset path manually using --dataset-path")
            return
        
        # Use the path with most REC folders
        dataset_path, num_folders = max(found_paths, key=lambda x: x[1])
        print(f"✅ Found dataset with {num_folders} REC folders at: {dataset_path}")
        
    else:
        # Interactive mode
        print("🔍 Searching for potential dataset locations...")
        found_paths = find_dataset_paths()
        
        if found_paths:
            print("Found potential dataset locations:")
            for i, (path, num_folders) in enumerate(found_paths):
                print(f"  {i+1}. {path} ({num_folders} REC folders)")
            
            while True:
                try:
                    choice = input(f"\nSelect dataset location (1-{len(found_paths)}) or enter custom path: ")
                    if choice.isdigit() and 1 <= int(choice) <= len(found_paths):
                        dataset_path = found_paths[int(choice)-1][0]
                        break
                    else:
                        dataset_path = choice
                        break
                except (ValueError, KeyboardInterrupt):
                    print("Invalid choice. Please try again.")
        else:
            dataset_path = input("Enter dataset root path: ")
    
    # Validate dataset structure
    if validate_dataset_structure(dataset_path):
        # Update configuration
        update_config(dataset_path, args.config)
        
        print(f"\n🎉 Setup complete!")
        print(f"📊 Dataset: {dataset_path}")
        print(f"⚙️  Config: {args.config}")
        print(f"\n🚀 You can now run training with:")
        print(f"   python train_final.py --config {args.config}")
        
    else:
        print(f"\n❌ Setup failed. Please check your dataset path and structure.")

if __name__ == "__main__":
    main() 