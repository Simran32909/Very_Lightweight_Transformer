#!/usr/bin/env python3
"""
Utility script to create train/val/test splits for the 1MSharada dataset.
This script will:
1. Find all JSON files in the dataset
2. Create train/val/test splits with specified sample counts
3. Save the splits to JSON files
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

def find_all_json_files(data_dir: str) -> List[str]:
    """Find all JSON files in the dataset directory."""
    json_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def read_json_file(json_path: str) -> Dict:
    """Read a JSON file and return its content."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_splits(json_files: List[str], 
                  train_samples: int = 8000, 
                  val_samples: int = 1000, 
                  test_samples: int = 1000,
                  seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits from the list of JSON files.
    
    Args:
        json_files: List of paths to JSON files
        train_samples: Number of training samples
        val_samples: Number of validation samples  
        test_samples: Number of test samples
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    random.seed(seed)
    
    # Shuffle the files
    shuffled_files = json_files.copy()
    random.shuffle(shuffled_files)
    
    # Split the files
    train_files = shuffled_files[:train_samples]
    val_files = shuffled_files[train_samples:train_samples + val_samples]
    test_files = shuffled_files[train_samples + val_samples:train_samples + val_samples + test_samples]
    
    return train_files, val_files, test_files

def save_splits(train_files: List[str], 
                val_files: List[str], 
                test_files: List[str], 
                output_dir: str):
    """Save the splits to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train split
    with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_files, f, indent=2, ensure_ascii=False)
    
    # Save validation split
    with open(os.path.join(output_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_files, f, indent=2, ensure_ascii=False)
    
    # Save test split
    with open(os.path.join(output_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_files, f, indent=2, ensure_ascii=False)
    
    print(f"Saved splits to {output_dir}")
    print(f"Train samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")

def main():
    # Configuration
    data_dir = "/scratch/tathagata.ghosh/datasets/1MSharada"
    output_dir = "/scratch/tathagata.ghosh/Very_Lightweight_Transformer/data/1MSharada_splits"
    
    # Sample counts (total 10000)
    train_samples = 8000
    val_samples = 1000
    test_samples = 1000
    
    print(f"Finding JSON files in {data_dir}...")
    json_files = find_all_json_files(data_dir)
    print(f"Found {len(json_files)} JSON files")
    
    if len(json_files) < (train_samples + val_samples + test_samples):
        print(f"Warning: Only {len(json_files)} files available, but {train_samples + val_samples + test_samples} requested")
        # Adjust sample counts
        total_available = len(json_files)
        train_samples = int(total_available * 0.8)
        val_samples = int(total_available * 0.1)
        test_samples = total_available - train_samples - val_samples
        print(f"Adjusted to: train={train_samples}, val={val_samples}, test={test_samples}")
    
    print("Creating splits...")
    train_files, val_files, test_files = create_splits(
        json_files, train_samples, val_samples, test_samples
    )
    
    print("Saving splits...")
    save_splits(train_files, val_files, test_files, output_dir)
    
    # Print some sample data for verification
    print("\nSample data from train split:")
    sample_file = train_files[0]
    sample_data = read_json_file(sample_file)
    print(f"File: {sample_file}")
    print(f"Original text: {sample_data.get('original_text', 'N/A')[:100]}...")
    print(f"Image path: {sample_data.get('image_path', 'N/A')}")

if __name__ == "__main__":
    main()
