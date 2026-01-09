#!/usr/bin/env python3
"""
Fine-tune Qwen2.5 model for Ming Shilu NER using MLX framework
Optimized for Apple Silicon (M3 Pro)
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# --- Configuration ---
TRAIN_DATA = "train.jsonl"
MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"
CHECKPOINT_DIR = "checkpoints"
ADAPTER_DIR = "adapters"

# Training hyperparameters
DEFAULT_CONFIG = {
    "batch_size": 1,
    "lora_layers": 16,
    "iters": 1000,
    "steps_per_eval": 100,
    "val_batches": 10,
    "learning_rate": 1e-5,
    "save_every": 200,
    "test_split": 0.1
}


def prepare_data_split(input_file: str, output_dir: str, test_split: float = 0.1):
    """
    Split training data into train/validation sets and convert to MLX format
    """
    import random
    import subprocess

    logging.info(f"Reading training data from {input_file}")

    # First convert the data format
    logging.info("Converting data format for MLX...")
    result = subprocess.run([
        "python", "convert_data_format.py"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        logging.error(f"Data conversion failed: {result.stderr}")
        raise RuntimeError("Failed to convert data format")

    logging.info("Data conversion successful")

    # Use the converted files
    train_file = os.path.join(output_dir, "train_mlx.jsonl")
    val_file = os.path.join(output_dir, "valid_mlx.jsonl")

    # Rename to MLX expected names
    import shutil
    mlx_train = os.path.join(output_dir, "train.jsonl")
    mlx_valid = os.path.join(output_dir, "valid.jsonl")
    mlx_test = os.path.join(output_dir, "test.jsonl")

    shutil.copy(train_file, mlx_train)
    shutil.copy(val_file, mlx_valid)
    shutil.copy(val_file, mlx_test)  # Use validation as test

    # Count examples
    with open(mlx_train, 'r', encoding='utf-8') as f:
        train_count = sum(1 for _ in f)

    with open(mlx_valid, 'r', encoding='utf-8') as f:
        val_count = sum(1 for _ in f)

    logging.info(f"Data preparation complete:")
    logging.info(f"  Training examples: {train_count}")
    logging.info(f"  Validation examples: {val_count}")
    logging.info(f"  Test examples: {val_count}")

    return mlx_train, mlx_valid


def train_model(config: dict):
    """
    Train the model using MLX LoRA
    """
    import subprocess
    
    # Prepare data directory
    data_dir = "data"
    train_file, val_file = prepare_data_split(
        TRAIN_DATA, 
        data_dir, 
        config['test_split']
    )
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    
    # Build training command
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", MODEL_NAME,
        "--train",
        "--data", data_dir,
        "--batch-size", str(config['batch_size']),
        "--num-layers", str(config['lora_layers']),
        "--iters", str(config['iters']),
        "--steps-per-eval", str(config['steps_per_eval']),
        "--val-batches", str(config['val_batches']),
        "--learning-rate", str(config['learning_rate']),
        "--save-every", str(config['save_every']),
        "--adapter-path", ADAPTER_DIR,
    ]
    
    logging.info("=" * 60)
    logging.info("Starting MLX LoRA Fine-Tuning")
    logging.info("=" * 60)
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Training examples: {len(open(train_file).readlines())}")
    logging.info(f"Validation examples: {len(open(val_file).readlines())}")
    logging.info(f"Configuration:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
    logging.info("=" * 60)
    
    # Run training
    try:
        logging.info("Executing training command...")
        logging.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        
        logging.info("=" * 60)
        logging.info("Training completed successfully!")
        logging.info(f"Adapter saved to: {ADAPTER_DIR}")
        logging.info("=" * 60)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed with error: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during training: {e}")
        return False


def fuse_model(adapter_path: str, output_path: str):
    """
    Fuse the LoRA adapter with the base model for standalone deployment
    """
    import subprocess
    
    logging.info("=" * 60)
    logging.info("Fusing LoRA adapter with base model...")
    logging.info("=" * 60)
    
    cmd = [
        "python", "-m", "mlx_lm.fuse",
        "--model", MODEL_NAME,
        "--adapter-path", adapter_path,
        "--save-path", output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"Fused model saved to: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Model fusion failed: {e}")
        return False


def test_model(adapter_path: str, test_cases: list):
    """
    Test the fine-tuned model with sample inputs
    """
    try:
        from mlx_lm import load, generate
        
        logging.info("=" * 60)
        logging.info("Testing fine-tuned model...")
        logging.info("=" * 60)
        
        # Load model with adapter
        model, tokenizer = load(MODEL_NAME, adapter_path=adapter_path)
        
        for i, test_case in enumerate(test_cases, 1):
            logging.info(f"\nTest Case {i}:")
            logging.info(f"Input: {test_case}")
            
            prompt = f"你是明朝歷史資料專家。請從以下文本中提取官員資訊，輸出為 JSON List。\n\n{test_case}"
            
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=512,
                temp=0.0
            )
            
            logging.info(f"Output: {response}")
            logging.info("-" * 60)
        
        return True
        
    except ImportError:
        logging.warning("MLX-LM not installed. Skipping model testing.")
        logging.warning("Install with: pip install mlx mlx-lm")
        return False
    except Exception as e:
        logging.error(f"Model testing failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 for Ming Shilu NER")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG['batch_size'],
                        help="Training batch size")
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG['lora_layers'],
                        help="Number of LoRA layers")
    parser.add_argument("--iters", type=int, default=DEFAULT_CONFIG['iters'],
                        help="Training iterations")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help="Learning rate")
    parser.add_argument("--test-split", type=float, default=DEFAULT_CONFIG['test_split'],
                        help="Validation split ratio")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: Train for only 100 iterations with faster evaluation")
    parser.add_argument("--fuse", action="store_true",
                        help="Fuse adapter with base model after training")
    parser.add_argument("--test", action="store_true",
                        help="Test model after training")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (for testing existing model)")

    args = parser.parse_args()

    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config.update({
        'batch_size': args.batch_size,
        'num_layers': args.num_layers,
        'iters': args.iters,
        'learning_rate': args.learning_rate,
        'test_split': args.test_split
    })

    # Debug mode overrides
    if args.debug:
        config.update({
            'iters': 100,
            'steps_per_eval': 10,  # Evaluate every 10 iterations in debug mode
            'save_every': 50,      # Save every 50 iterations in debug mode
        })
        logging.info("🔧 DEBUG MODE ENABLED: Training for 100 iterations only")
    
    # Check if training data exists
    if not os.path.exists(TRAIN_DATA):
        logging.error(f"Training data not found: {TRAIN_DATA}")
        logging.error("Please run prepare_training_data.py first to generate training data.")
        return
    
    # Train model
    if not args.skip_train:
        success = train_model(config)
        if not success:
            logging.error("Training failed. Exiting.")
            return
    
    # Fuse model (optional)
    if args.fuse:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fused_path = os.path.join(CHECKPOINT_DIR, f"fused_model_{timestamp}")
        fuse_model(ADAPTER_DIR, fused_path)
    
    # Test model (optional)
    if args.test:
        test_cases = [
            "宣德九年，少师蹇义卒。",
            "命行在工部尚书李友直提督供应柴炭。",
            "太监沐敬、丰城侯李贤率军出征。"
        ]
        test_model(ADAPTER_DIR, test_cases)
    
    logging.info("=" * 60)
    logging.info("All tasks completed!")
    logging.info(f"Adapter location: {ADAPTER_DIR}")
    logging.info(f"Checkpoints location: {CHECKPOINT_DIR}")
    logging.info("=" * 60)
    logging.info("\nNext steps:")
    logging.info("1. Test the model: python train_model.py --skip-train --test")
    logging.info("2. Fuse the model: python train_model.py --skip-train --fuse")
    logging.info("3. Use in pipeline: Update runNER.py to use the adapter")


if __name__ == "__main__":
    main()
