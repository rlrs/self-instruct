#!/usr/bin/env python3
"""Convert training data to standard SFT formats and upload to HuggingFace."""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def convert_to_chat_format(data: List[Dict]) -> List[Dict]:
    """Convert to standard chat/conversation format."""
    chat_data = []
    for item in data:
        # Standard chat format with messages array
        chat_item = {
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"].strip()}
            ]
        }
        chat_data.append(chat_item)
    return chat_data


def convert_to_alpaca_format(data: List[Dict]) -> List[Dict]:
    """Convert to Alpaca-style format."""
    alpaca_data = []
    for item in data:
        prompt = item["prompt"]
        
        # Try to parse instruction and input from prompt
        if "\n\n" in prompt:
            parts = prompt.split("\n\n", 1)
            instruction = parts[0]
            input_text = parts[1] if len(parts) > 1 else ""
        else:
            instruction = prompt
            input_text = ""
        
        # Remove common prefixes
        for prefix in ["Instruktion: ", "Opgave: ", "Spørgsmål: "]:
            if instruction.startswith(prefix):
                instruction = instruction[len(prefix):]
                break
        
        # Remove "Svar:" suffix if present
        if instruction.endswith("\nSvar:"):
            instruction = instruction[:-6]
        if input_text.endswith("\nSvar:"):
            input_text = input_text[:-6]
        
        alpaca_item = {
            "instruction": instruction.strip(),
            "input": input_text.strip(),
            "output": item["completion"].strip()
        }
        alpaca_data.append(alpaca_item)
    return alpaca_data


def convert_to_sharegpt_format(data: List[Dict]) -> List[Dict]:
    """Convert to ShareGPT format."""
    sharegpt_data = []
    for i, item in enumerate(data):
        sharegpt_item = {
            "id": f"conv_{i}",
            "conversations": [
                {"from": "human", "value": item["prompt"]},
                {"from": "gpt", "value": item["completion"].strip()}
            ]
        }
        sharegpt_data.append(sharegpt_item)
    return sharegpt_data


def convert_to_openai_format(data: List[Dict]) -> List[Dict]:
    """Convert to OpenAI fine-tuning format."""
    openai_data = []
    for item in data:
        openai_item = {
            "prompt": item["prompt"] + "\n\n###\n\n",
            "completion": " " + item["completion"].strip() + " ###"
        }
        openai_data.append(openai_item)
    return openai_data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data as JSONL."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data: List[Dict], file_path: str):
    """Save data as JSON."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_huggingface_dataset(train_data: List[Dict], val_data: List[Dict], format_name: str) -> DatasetDict:
    """Create HuggingFace dataset from data."""
    if format_name == "chat":
        # For chat format, we need to convert messages to string for HF
        train_df = pd.DataFrame([
            {"text": json.dumps(item, ensure_ascii=False)} for item in train_data
        ])
        val_df = pd.DataFrame([
            {"text": json.dumps(item, ensure_ascii=False)} for item in val_data
        ])
    else:
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
    
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df)
    })
    
    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Convert and upload training data")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/generated/training_data",
        help="Input directory with train/validation splits"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/formatted",
        help="Output directory for formatted data"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["chat", "alpaca"],
        choices=["chat", "alpaca", "sharegpt", "openai"],
        help="Output formats to generate"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        help="HuggingFace repository name (e.g., username/dataset-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace repository private"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace API token (can also use HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input_dir)
    train_data = load_jsonl(input_path / "train_split.jsonl")
    val_data = load_jsonl(input_path / "validation.jsonl")
    
    logger.info(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to different formats
    format_converters = {
        "chat": convert_to_chat_format,
        "alpaca": convert_to_alpaca_format,
        "sharegpt": convert_to_sharegpt_format,
        "openai": convert_to_openai_format
    }
    
    converted_data = {}
    
    for format_name in args.formats:
        logger.info(f"Converting to {format_name} format...")
        converter = format_converters[format_name]
        
        train_converted = converter(train_data)
        val_converted = converter(val_data)
        
        # Save formatted data
        format_dir = output_path / format_name
        format_dir.mkdir(exist_ok=True)
        
        if format_name == "openai":
            # OpenAI uses JSONL
            save_jsonl(train_converted, format_dir / "train.jsonl")
            save_jsonl(val_converted, format_dir / "validation.jsonl")
        else:
            # Others typically use JSON
            save_json(train_converted, format_dir / "train.json")
            save_json(val_converted, format_dir / "validation.json")
            
            # Also save JSONL versions
            save_jsonl(train_converted, format_dir / "train.jsonl")
            save_jsonl(val_converted, format_dir / "validation.jsonl")
        
        converted_data[format_name] = (train_converted, val_converted)
        logger.info(f"Saved {format_name} format to {format_dir}")
    
    # Upload to HuggingFace if requested
    if args.upload:
        if not args.repo_name:
            logger.error("--repo-name is required when uploading to HuggingFace")
            return
            
        # Login to HuggingFace
        if args.hf_token:
            login(token=args.hf_token)
        else:
            logger.info("No HF token provided, using environment or cached credentials")
        
        # Upload each format as a separate dataset config
        for format_name in args.formats:
            logger.info(f"Uploading {format_name} format to HuggingFace...")
            train_converted, val_converted = converted_data[format_name]
            
            # Create dataset
            dataset = create_huggingface_dataset(train_converted, val_converted, format_name)
            
            # Upload to hub
            dataset.push_to_hub(
                args.repo_name,
                config_name=format_name,
                private=args.private
            )
            
        logger.info(f"Successfully uploaded to https://huggingface.co/datasets/{args.repo_name}")
    
    # Print summary
    logger.info("\nConversion Summary:")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Formats generated: {', '.join(args.formats)}")
    logger.info(f"Training examples: {len(train_data)}")
    logger.info(f"Validation examples: {len(val_data)}")
    
    if args.upload:
        logger.info(f"\nDataset uploaded to: https://huggingface.co/datasets/{args.repo_name}")
        logger.info("You can load it with:")
        for format_name in args.formats:
            logger.info(f"  dataset = load_dataset('{args.repo_name}', '{format_name}')")


if __name__ == "__main__":
    main()