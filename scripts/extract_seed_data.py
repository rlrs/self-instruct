#!/usr/bin/env python3
"""Extract Danish seed tasks from alpaca-seed-eval.xlsx and convert to JSONL format."""

import json
import pandas as pd
from pathlib import Path


def extract_alpaca_sheet(df: pd.DataFrame) -> list:
    """Extract tasks from the alpaca-da sheet."""
    tasks = []
    
    for idx, row in df.iterrows():
        # Skip if not corrected or marked as 'no'
        if row.get('corrected') != 'yes':
            continue
            
        # Skip rows without Danish instruction
        if pd.isna(row.get('instruction.1')):
            continue
        
        # Use corrected versions if available, otherwise fall back to regular versions
        instruction = str(row['corrected_instruction']).strip() if pd.notna(row.get('corrected_instruction')) else str(row['instruction.1']).strip()
        input_text = str(row['corrected_input']).strip() if pd.notna(row.get('corrected_input')) else str(row['input.1']).strip() if pd.notna(row.get('input.1')) else ""
        output = str(row['corrected_output']).strip() if pd.notna(row.get('corrected_output')) else str(row['output.1']).strip() if pd.notna(row.get('output.1')) else ""
        
        task = {
            "id": f"alpaca_task_{idx}",
            "name": row.get('name', ''),
            "instruction": instruction,
            "instances": [{
                "input": input_text,
                "output": output
            }],
            "is_classification": None  # Will be determined later
        }
        
        # Add metadata
        task["metadata"] = {
            "source": "alpaca-da",
            "has_input": bool(input_text),
            "original_source": row.get('source_dataset', '')
        }
        
        tasks.append(task)
    
    return tasks


def extract_trustllm_sheet(df: pd.DataFrame) -> list:
    """Extract tasks from the trustllm sheet."""
    tasks = []
    
    for idx, row in df.iterrows():
        # Skip if not useful
        if row.get('useful') != 'yes':
            continue
            
        # Skip if not corrected when corrected version exists
        if row.get('corrected') == 'no' and pd.isna(row.get('corrected_instruction')):
            continue
        
        # Use corrected versions if available
        instruction = str(row['corrected_instruction']).strip() if pd.notna(row.get('corrected_instruction')) else str(row['instruction']).strip()
        output = str(row['corrected_completion']).strip() if pd.notna(row.get('corrected_completion')) else str(row['completion']).strip() if pd.notna(row.get('completion')) else ""
        
        # Skip if no valid instruction or output
        if not instruction or not output:
            continue
        
        task = {
            "id": f"trustllm_task_{row['ID']}",
            "name": f"trustllm_{idx}",
            "instruction": instruction,
            "instances": [{
                "input": "",  # TrustLLM tasks don't have separate inputs
                "output": output
            }],
            "is_classification": None  # Will be determined later
        }
        
        # Add metadata
        task["metadata"] = {
            "source": "trustllm",
            "has_input": False,
            "original_source": row.get('source_dataset', '')
        }
        
        tasks.append(task)
    
    return tasks


def extract_seed_data(excel_path: str, output_path: str) -> None:
    """Extract Danish tasks from Excel and save as JSONL."""
    # Read both sheets
    excel_file = pd.ExcelFile(excel_path)
    
    all_tasks = []
    
    # Process alpaca-da sheet
    if 'alpaca-da' in excel_file.sheet_names:
        df_alpaca = pd.read_excel(excel_file, sheet_name='alpaca-da')
        alpaca_tasks = extract_alpaca_sheet(df_alpaca)
        all_tasks.extend(alpaca_tasks)
        print(f"Extracted {len(alpaca_tasks)} tasks from alpaca-da sheet")
    
    # Process trustllm sheet
    if 'trustllm' in excel_file.sheet_names:
        df_trustllm = pd.read_excel(excel_file, sheet_name='trustllm')
        trustllm_tasks = extract_trustllm_sheet(df_trustllm)
        all_tasks.extend(trustllm_tasks)
        print(f"Extracted {len(trustllm_tasks)} tasks from trustllm sheet")
    
    # Write to JSONL file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for task in all_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
    
    print(f"\nTotal extracted tasks: {len(all_tasks)}")
    print(f"Tasks with input: {sum(1 for t in all_tasks if t['metadata']['has_input'])}")
    print(f"Tasks without input: {sum(1 for t in all_tasks if not t['metadata']['has_input'])}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract Danish seed tasks from Excel")
    parser.add_argument(
        "--input", 
        default="alpaca-seed-eval.xlsx",
        help="Path to the Excel file"
    )
    parser.add_argument(
        "--output",
        default="data/seed_tasks.jsonl",
        help="Output JSONL file path"
    )
    
    args = parser.parse_args()
    extract_seed_data(args.input, args.output)