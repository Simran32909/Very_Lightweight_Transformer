#!/usr/bin/env python3
"""
Test script to verify the training setup with 1MSharada dataset.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_data_splits():
    """Test if data splits exist and are valid."""
    splits_dir = "/scratch/tathagata.ghosh/Very_Lightweight_Transformer/data/1MSharada_splits"
    
    for split_name in ['train', 'val', 'test']:
        split_file = os.path.join(splits_dir, f'{split_name}.json')
        if not os.path.exists(split_file):
            print(f"❌ Split file not found: {split_file}")
            return False
        
        with open(split_file, 'r', encoding='utf-8') as f:
            files = json.load(f)
        
        print(f"✅ {split_name}: {len(files)} files")
        
        # Test a few files
        for i, json_file in enumerate(files[:3]):
            if not os.path.exists(json_file):
                print(f"❌ JSON file not found: {json_file}")
                return False
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                image_path = data.get('image_path', '')
                if image_path.startswith('./'):
                    image_path = image_path.replace('./data/1MSharada/', '/scratch/tathagata.ghosh/datasets/1MSharada/')
                
                if not os.path.exists(image_path):
                    print(f"❌ Image file not found: {image_path}")
                    return False
                
                if not data.get('original_text', '').strip():
                    print(f"❌ Empty text in: {json_file}")
                    return False
                    
            except Exception as e:
                print(f"❌ Error reading {json_file}: {e}")
                return False
    
    return True

def test_data_loading():
    """Test if the data loading function works."""
    try:
        from src.data.data_utils import read_data_1msharada
        
        splits_dir = "/scratch/tathagata.ghosh/Very_Lightweight_Transformer/data/1MSharada_splits"
        train_split = os.path.join(splits_dir, 'train.json')
        
        if not os.path.exists(train_split):
            print("❌ Train split not found, skipping data loading test")
            return False
        
        image_paths, texts = read_data_1msharada(train_split)
        print(f"✅ Data loading test: {len(image_paths)} images, {len(texts)} texts")
        
        if len(image_paths) > 0:
            print(f"✅ Sample text: {texts[0][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    print("Testing 1MSharada training setup...")
    print("=" * 50)
    
    # Test 1: Data splits
    print("\n1. Testing data splits...")
    if not test_data_splits():
        print("❌ Data splits test failed!")
        return
    
    # Test 2: Data loading
    print("\n2. Testing data loading...")
    if not test_data_loading():
        print("❌ Data loading test failed!")
        return
    
    print("\n✅ All tests passed! Training setup is ready.")
    print("\nTo start training, run:")
    print("cd /scratch/tathagata.ghosh/Very_Lightweight_Transformer")
    print("python src/train_hybrid.py --config-name train_1msharada.yaml")

if __name__ == "__main__":
    main()
