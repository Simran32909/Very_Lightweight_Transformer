#!/usr/bin/env python3
"""
Comprehensive setup script for 1MSharada training.
This script will:
1. Generate train/val/test splits (8000/1000/1000 samples)
2. Test the data loading functionality
3. Verify the training configuration
"""

import os
import json
import random
import sys
from pathlib import Path

def generate_splits():
    """Generate train/val/test splits for 1MSharada dataset."""
    print("Step 1: Generating data splits...")
    
    data_dir = "/scratch/tathagata.ghosh/datasets/1MSharada"
    output_dir = "/scratch/tathagata.ghosh/Very_Lightweight_Transformer/data/1MSharada_splits"
    
    # Find all JSON files
    json_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        print("❌ No JSON files found!")
        return False
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(json_files)
    
    # Create splits (8000/1000/1000 = 10000 total)
    train_files = json_files[:80000]
    val_files = json_files[80000:90000]
    test_files = json_files[90000:100000]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    splits = {
        'train.json': train_files,
        'val.json': val_files,
        'test.json': test_files
    }
    
    for split_name, files in splits.items():
        output_file = os.path.join(output_dir, split_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(files, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {split_name}: {len(files)} files")
    
    return True

def test_data_loading():
    """Test the data loading functionality."""
    print("\nStep 2: Testing data loading...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from src.data.data_utils import read_data_1msharada
        
        splits_dir = "/scratch/tathagata.ghosh/Very_Lightweight_Transformer/data/1MSharada_splits"
        train_split = os.path.join(splits_dir, 'train.json')
        
        if not os.path.exists(train_split):
            print("❌ Train split not found!")
            return False
        
        image_paths, texts = read_data_1msharada(train_split)
        print(f"✅ Data loading test: {len(image_paths)} images, {len(texts)} texts")
        
        if len(image_paths) > 0:
            print(f"✅ Sample text: {texts[0][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def verify_configuration():
    """Verify the training configuration files exist."""
    print("\nStep 3: Verifying configuration...")
    
    config_files = [
        "configs/data/1msharada.yaml",
        "configs/train_1msharada.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} not found!")
            return False
    
    return True

def main():
    print("Setting up 1MSharada training...")
    print("=" * 50)
    
    # Step 1: Generate splits
    if not generate_splits():
        print("❌ Failed to generate splits!")
        return
    
    # Step 2: Test data loading
    if not test_data_loading():
        print("❌ Data loading test failed!")
        return
    
    # Step 3: Verify configuration
    if not verify_configuration():
        print("❌ Configuration verification failed!")
        return
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nTo start training, run:")
    print("cd /scratch/tathagata.ghosh/Very_Lightweight_Transformer")
    print("python src/train_hybrid.py --config-name train_1msharada.yaml")
    print("\nOr to test only (no training):")
    print("python src/train_hybrid.py --config-name train_1msharada.yaml train=false")

if __name__ == "__main__":
    main()
