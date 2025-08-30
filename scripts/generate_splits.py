#!/usr/bin/env python3
"""
Generate train/val/test splits for 1MSharada dataset.
"""

import os
import json
import random

def main():
    # Configuration
    data_dir = "/scratch/tathagata.ghosh/datasets/1MSharada"
    output_dir = "/scratch/tathagata.ghosh/Very_Lightweight_Transformer/data/1MSharada_splits"
    
    print(f"Searching for JSON files in {data_dir}...")
    
    # Find all JSON files
    json_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        print("No JSON files found!")
        return
    
    # Shuffle files
    random.seed(42)
    random.shuffle(json_files)
    
    # Create splits (8000/1000/1000 = 10000 total)
    train_files = json_files[:8000]
    val_files = json_files[8000:9000]
    test_files = json_files[9000:10000]
    
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
        print(f"Saved {split_name}: {len(files)} files")
    
    print(f"\nAll splits saved to: {output_dir}")
    
    # Test a sample file
    if len(train_files) > 0:
        sample_file = train_files[0]
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"\nSample data from {sample_file}:")
            print(f"Original text: {data.get('original_text', 'N/A')[:100]}...")
            print(f"Image path: {data.get('image_path', 'N/A')}")
        except Exception as e:
            print(f"Error reading sample file: {e}")

if __name__ == "__main__":
    main()
