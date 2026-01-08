#!/usr/bin/env python3
"""
Convert training data from instruction/input/output format to MLX-compatible format
"""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_jsonl_format(input_file: str, output_file: str):
    """
    Convert from:
    {"instruction": "...", "input": "...", "output": "[...]"}
    To:
    {"text": "instruction + input", "completion": "output"}
    """
    converted_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            try:
                data = json.loads(line.strip())

                # Combine instruction and input as the prompt
                text = f"{data['instruction']}\n\n{data['input']}"

                # Output is already the completion
                completion = data['output']

                # Create new format
                new_data = {
                    "text": text,
                    "completion": completion
                }

                f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                converted_count += 1

            except json.JSONDecodeError as e:
                logging.warning(f"Skipping malformed line: {e}")
                continue

    logging.info(f"Converted {converted_count} examples")
    return converted_count

def main():
    # Convert training data - use the simplified version
    logging.info("Converting training data format...")
    train_count = convert_jsonl_format("train_simplified.jsonl", "data/train_mlx.jsonl")

    # Split the simplified data for validation
    import random
    with open("train_simplified.jsonl", 'r', encoding='utf-8') as f:
        all_data = f.readlines()

    random.seed(42)
    random.shuffle(all_data)

    # 90% train, 10% validation
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # Write validation data
    with open("data/valid.jsonl", 'w', encoding='utf-8') as f:
        f.writelines(val_data)

    val_count = convert_jsonl_format("data/valid.jsonl", "data/valid_mlx.jsonl")

    # Create a copy for test data (same format as validation)
    import shutil
    shutil.copy("data/valid_mlx.jsonl", "data/test_mlx.jsonl")

    logging.info("Data conversion complete!")
    logging.info(f"Training examples: {train_count}")
    logging.info(f"Validation examples: {val_count}")
    logging.info(f"Test examples: {val_count}")

    # Show sample
    with open("data/train_mlx.jsonl", 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline().strip())
        logging.info("\nSample converted format:")
        logging.info(f"Text (first 100 chars): {sample['text'][:100]}...")
        logging.info(f"Completion (first 100 chars): {sample['completion'][:100]}...")

if __name__ == "__main__":
    main()
