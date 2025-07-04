"""Filter and prepare training data for fine-tuning."""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
import yaml
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class TrainingDataProcessor:
    """Process and filter training data."""
    
    def __init__(self, config_path: str):
        """Initialize the processor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.seen_instructions = set()
        self.seen_instances = set()
        
    def _hash_text(self, text: str) -> str:
        """Create hash of text for deduplication."""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
        
    def filter_instance(self, instance: Dict[str, str]) -> bool:
        """Check if an instance passes quality filters."""
        input_text = instance.get('input', '')
        output_text = instance.get('output', '')
        
        # Must have output
        if not output_text or len(output_text.strip()) < 5:
            return False
            
        # Check for repetitive output (output shouldn't just repeat input)
        if input_text and output_text.strip() == input_text.strip():
            return False
            
        # Check for placeholder text
        placeholder_patterns = ['[', ']', '<', '>', '...', 'TODO', 'FIXME', 'XXX']
        for pattern in placeholder_patterns:
            if pattern in output_text:
                return False
                
        # Length constraints
        if len(output_text) > 5000:  # Too long
            return False
            
        return True
        
    def deduplicate_instances(self, instances: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate instances."""
        unique_instances = []
        
        for instance in instances:
            # Create hash of input+output
            instance_text = f"{instance.get('input', '')}|||{instance.get('output', '')}"
            instance_hash = self._hash_text(instance_text)
            
            if instance_hash not in self.seen_instances:
                self.seen_instances.add(instance_hash)
                unique_instances.append(instance)
                
        return unique_instances
        
    def format_for_training(self, task: Dict, instance: Dict[str, str]) -> List[Dict[str, str]]:
        """Format a task instance for training with random template selection."""
        instruction = task['instruction']
        input_text = instance.get('input', '')
        output_text = instance.get('output', '')
        
        # Define templates matching the original paper's approach
        if input_text:
            # Templates for tasks WITH input
            templates_with_input = [
                # Explicit Input/Output labels
                ("{instruction}\nInput: {input}\nOutput:", " {output}"),
                ("{instruction}\n\nInput: {input}\n\nOutput:", " {output}"),
                ("Opgave: {instruction}\nInput: {input}\nOutput:", " {output}"),
                # Danish variations with different labels
                ("Instruktion: {instruction}\n\nInput: {input}\n\nSvar:", " {output}"),
                ("{instruction}\n\nTekst: {input}\n\nSvar:", " {output}"),
                # No Input label
                ("{instruction}\n\n{input}\n\nOutput:", " {output}"),
                ("{instruction}\n\n{input}\n\n", "{output}"),
                ("{instruction}\n{input}\n\n", "{output}"),
                ("Opgave: {instruction}\n\n{input}\n\n", "{output}"),
            ]
            prompt_template, completion_template = random.choice(templates_with_input)
            prompt = prompt_template.format(instruction=instruction.strip(), input=input_text.strip())
            completion = completion_template.format(output=output_text.strip())
        else:
            # Templates for tasks WITHOUT input
            templates_without_input = [
                # With Output/Svar label
                ("{instruction} Output:", " {output}"),
                ("{instruction}\nOutput:", " {output}"),
                ("{instruction}\n\nOutput:", " {output}"),
                ("{instruction}\nSvar:", " {output}"),
                ("{instruction}\n\nSvar:", " {output}"),
                # With prefix
                ("Opgave: {instruction}\n\nSvar:", " {output}"),
                ("Instruktion: {instruction}\n\nOutput:", " {output}"),
                ("Spørgsmål: {instruction}\nSvar:", " {output}"),
                # No label
                ("{instruction}\n", "{output}"),
                ("{instruction}\n\n", "{output}"),
                ("Opgave: {instruction}\n\n", "{output}"),
            ]
            prompt_template, completion_template = random.choice(templates_without_input)
            prompt = prompt_template.format(instruction=instruction.strip())
            completion = completion_template.format(output=output_text.strip())
        
        # Return single formatted example (not a list)
        return [{
            "prompt": prompt,
            "completion": completion
        }]
        
    def process_tasks(self, tasks: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Process tasks and generate training examples."""
        all_examples = []
        statistics = defaultdict(int)
        
        # Initialize counters to ensure they exist
        statistics['classification_tasks'] = 0
        statistics['generation_tasks'] = 0
        
        # Debug: log initial state
        logger.debug(f"Initial statistics keys: {list(statistics.keys())}")
        
        for task in tasks:
            instruction = task['instruction']
            instances = task.get('instances', [])
            
            # Skip if no instances
            if not instances:
                statistics['tasks_without_instances'] += 1
                continue
                
            # Check if instruction is duplicate
            instruction_hash = self._hash_text(instruction)
            if instruction_hash in self.seen_instructions:
                statistics['duplicate_instructions'] += 1
                continue
                
            self.seen_instructions.add(instruction_hash)
            
            # Filter and deduplicate instances
            valid_instances = [inst for inst in instances if self.filter_instance(inst)]
            unique_instances = self.deduplicate_instances(valid_instances)
            
            statistics['total_tasks'] += 1
            statistics['total_instances'] += len(instances)
            statistics['valid_instances'] += len(valid_instances)
            statistics['unique_instances'] += len(unique_instances)
            
            # Generate training examples
            for instance in unique_instances:
                examples = self.format_for_training(task, instance)
                all_examples.extend(examples)
                
            # Track task types
            if task.get('is_classification'):
                statistics['classification_tasks'] += 1
            else:
                statistics['generation_tasks'] += 1
        
        # Debug: log final state before conversion
        logger.debug(f"Final statistics before dict conversion: {dict(statistics)}")
        final_stats = dict(statistics)
        logger.debug(f"Final statistics after dict conversion: {final_stats}")
        logger.debug(f"'classification_tasks' in final_stats: {'classification_tasks' in final_stats}")
                
        return all_examples, final_stats
        
    def run(self, input_files: List[str], output_dir: str, include_seed: bool = True):
        """Run the data processing pipeline."""
        all_tasks = []
        
        # Load all input files
        for input_file in input_files:
            logger.info(f"Loading {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    task = json.loads(line)
                    all_tasks.append(task)
                    
        logger.info(f"Loaded {len(all_tasks)} total tasks")
        
        # Process tasks
        training_examples, statistics = self.process_tasks(all_tasks)
        
        # Shuffle examples
        random.shuffle(training_examples)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training data in different formats
        
        # 1. JSONL format (for fine-tuning)
        train_file = output_path / "train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                
        # 2. Split into train/validation
        split_idx = int(len(training_examples) * 0.95)
        train_examples = training_examples[:split_idx]
        val_examples = training_examples[split_idx:]
        
        train_split_file = output_path / "train_split.jsonl"
        with open(train_split_file, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                
        val_file = output_path / "validation.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for example in val_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                
        # 3. Save statistics
        stats_file = output_path / "statistics.json"
        statistics['total_training_examples'] = len(training_examples)
        statistics['train_examples'] = len(train_examples)
        statistics['validation_examples'] = len(val_examples)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
            
        # Log summary
        logger.info("Processing complete!")
        logger.info(f"Total tasks processed: {statistics.get('total_tasks', 0)}")
        logger.info(f"Classification tasks: {statistics.get('classification_tasks', 0)}")
        logger.info(f"Generation tasks: {statistics.get('generation_tasks', 0)}")
        logger.info(f"Total instances: {statistics.get('total_instances', 0)}")
        logger.info(f"Valid instances: {statistics.get('valid_instances', 0)}")
        logger.info(f"Unique instances: {statistics.get('unique_instances', 0)}")
        logger.info(f"Total training examples: {len(training_examples)}")
        logger.info(f"Train/Val split: {len(train_examples)}/{len(val_examples)}")
        logger.info(f"Output directory: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        nargs='+',
        required=True,
        help="Input JSONL files"
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated/training",
        help="Output directory"
    )
    parser.add_argument(
        "--include-seed",
        action="store_true",
        help="Include seed tasks in training data"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = TrainingDataProcessor(args.config)
    processor.run(args.input, args.output_dir, args.include_seed)


if __name__ == "__main__":
    main()