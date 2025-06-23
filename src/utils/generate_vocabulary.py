import os
import json
import argparse
from collections import Counter

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"


def generate_vocabulary(data_dir: str, split_file_path: str, output_path: str):
    """
    Generates a vocabulary file from the training data split file.
    It reads the list of training samples, collects all unique characters
    from the 'original_text' field in the corresponding JSON files,
    and saves them to a vocabulary file.
    """
    print(f"Generating vocabulary from split file: {split_file_path}")
    
    char_counter = Counter()

    if not os.path.exists(split_file_path):
        print(f"Error: Split file not found at {split_file_path}")
        return

    with open(split_file_path, 'r', encoding='utf-8') as f:
        sample_rel_paths = [line.strip() for line in f]

    for rel_path in sample_rel_paths:
        sample_id = os.path.basename(rel_path)
        json_path = os.path.join(data_dir, rel_path, f"{sample_id}.json")

        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = data.get("original_text", "")
            char_counter.update(text)
        except Exception as e:
            print(f"Warning: Could not process {json_path}. Skipping. Error: {e}")

    print(f"Found {len(char_counter)} unique characters in the training set.")

    # Create vocabulary list with special tokens first
    vocabulary = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    
    # Add all unique characters from the dataset, sorted for consistency
    sorted_chars = sorted(char_counter.keys())
    vocabulary.extend(sorted_chars)

    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for char in vocabulary:
            f.write(char + '\n')

    print(f"Vocabulary of size {len(vocabulary)} saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a vocabulary file for the Sharada dataset from a split file.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root of the Sharada dataset."
    )
    parser.add_argument(
        "--split_file_path",
        type=str,
        required=True,
        help="Path to the training split file (e.g., train.txt)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated vocabulary file (e.g., vocab.txt)."
    )
    args = parser.parse_args()
    
    generate_vocabulary(args.data_dir, args.split_file_path, args.output_path) 