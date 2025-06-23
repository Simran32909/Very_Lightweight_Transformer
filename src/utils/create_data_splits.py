import os
import argparse
import random
from sklearn.model_selection import train_test_split

def create_data_splits(data_dir: str, output_dir: str, test_size: float = 0.2, val_size: float = 0.1):
    """
    Scans a directory for samples, splits them into train, val, and test sets,
    and saves the file lists.

    The script assumes a directory structure like:
    .../data/<id_part1>/<id_part2>/<sample_id>/

    Args:
        data_dir (str): The root directory where the data samples are stored.
        output_dir (str): The directory where train.txt, val.txt, and test.txt will be saved.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the *training* dataset to include in the validation split.
    """
    print(f"Scanning for samples in: {data_dir}")
    sample_paths = []

    # Walk through the directory structure to find sample folders
    # Assumes a structure of data_dir/xx/yy/sample_id/
    for root, dirs, files in os.walk(data_dir):
        # A sample directory is one that contains a .json file
        if any(f.endswith('.json') for f in files):
            relative_path = os.path.relpath(root, data_dir)
            sample_paths.append(relative_path)

    if not sample_paths:
        print("Error: No samples found. Check your data directory and its structure.")
        return

    print(f"Found {len(sample_paths)} total samples.")
    random.shuffle(sample_paths)

    # Split into train and test sets first
    train_paths, test_paths = train_test_split(sample_paths, test_size=test_size, random_state=42)

    # Split the initial training set further into a final training set and a validation set
    train_paths, val_paths = train_test_split(train_paths, test_size=val_size, random_state=42)

    print(f"Train set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")
    print(f"Test set size: {len(test_paths)}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write the files
    for split_name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
        output_path = os.path.join(output_dir, f"{split_name}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            for path in paths:
                # Use forward slashes for cross-platform compatibility
                f.write(path.replace('\\', '/') + '\n')
        print(f"Saved {split_name} split to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create train/val/test splits for the dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root of the dataset where sample folders are located."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the train.txt, val.txt, and test.txt files."
    )
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data for the test set.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of the remaining data for the validation set.")
    
    args = parser.parse_args()
    
    # We calculate val_size based on the remaining part after test split
    # sklearn's train_test_split val_size is relative to the input set.
    # So we'll pass val_size / (1 - test_size) to get the desired overall proportion.
    val_proportion = args.val_size / (1 - args.test_size)

    create_data_splits(args.data_dir, args.output_dir, args.test_size, val_proportion) 