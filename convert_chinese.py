#!/usr/bin/env python3
"""
Convert Chinese text between Traditional and Simplified characters
"""

import os
import json
import opencc
from pathlib import Path

def convert_file_traditional_to_simplified(input_file: str, output_file: str):
    """Convert Traditional Chinese to Simplified Chinese"""
    converter = opencc.OpenCC('t2s')  # Traditional to Simplified

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            try:
                data = json.loads(line.strip())
                # Convert the text content
                if 'input' in data:
                    data['input'] = converter.convert(data['input'])
                if 'output' in data:
                    data['output'] = converter.convert(data['output'])

                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                continue

def convert_file_simplified_to_traditional(input_file: str, output_file: str):
    """Convert Simplified Chinese to Traditional Chinese"""
    converter = opencc.OpenCC('s2t')  # Simplified to Traditional

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            try:
                data = json.loads(line.strip())
                # Convert the text content
                if 'input' in data:
                    data['input'] = converter.convert(data['input'])
                if 'output' in data:
                    data['output'] = converter.convert(data['output'])

                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                continue

def convert_pdf_text_to_traditional(text: str) -> str:
    """Convert PDF extracted text to Traditional Chinese"""
    converter = opencc.OpenCC('s2t')  # Simplified to Traditional
    return converter.convert(text)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert Chinese character sets")
    parser.add_argument("--train-to-simplified", action="store_true",
                        help="Convert training data from Traditional to Simplified")
    parser.add_argument("--train-to-traditional", action="store_true",
                        help="Convert training data from Simplified to Traditional")
    parser.add_argument("--pdf-text", type=str,
                        help="Convert a string of PDF text to Traditional Chinese")

    args = parser.parse_args()

    if args.train_to_simplified:
        print("Converting training data to Simplified Chinese...")
        convert_file_traditional_to_simplified("train.jsonl", "train_simplified.jsonl")
        print("Done! Created train_simplified.jsonl")

    elif args.train_to_traditional:
        print("Converting training data to Traditional Chinese...")
        convert_file_simplified_to_traditional("train.jsonl", "train_traditional.jsonl")
        print("Done! Created train_traditional.jsonl")

    elif args.pdf_text:
        result = convert_pdf_text_to_traditional(args.pdf_text)
        print("Converted text:")
        print(result)

    else:
        print("Usage examples:")
        print("  Convert training data to Simplified: python convert_chinese.py --train-to-simplified")
        print("  Convert training data to Traditional: python convert_chinese.py --train-to-traditional")
        print("  Convert PDF text: python convert_chinese.py --pdf-text '实录经录'")

if __name__ == "__main__":
    main()
