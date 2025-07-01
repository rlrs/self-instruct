"""Generate instances (input-output pairs) for instructions."""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
from tqdm import tqdm
import random
import asyncio

from src.simple_vllm_client import GenerationConfig, generate_many
from src.utils.prompts import (
    INSTANCE_GENERATION_INPUT_FIRST_PROMPT,
    INSTANCE_GENERATION_NO_INPUT_PROMPT,
    INSTANCE_GENERATION_OUTPUT_FIRST_PROMPT
)

logger = logging.getLogger(__name__)


class InstanceGenerator:
    """Generate instances for instructions."""
    
    def __init__(self, config_path: str):
        """Initialize the instance generator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.base_url = self.config['vllm']['base_url']
        self.model_name = self.config['vllm'].get('model_name')
        self.generation_config = GenerationConfig(**self.config['generation']['instance'])
        
    def detect_if_input_needed(self, instruction: str) -> bool:
        """Detect if an instruction needs input based on keywords and patterns."""
        instruction_lower = instruction.lower()
        
        # Keywords that suggest input is needed
        input_keywords = [
            'følgende', 'given', 'baseret på', 'ud fra', 'denne', 'dette',
            'nedenstående', 'ovenstående', 'teksten', 'sætningen', 'beskrivelsen',
            'informationen', 'dataen', 'input', 'eksempel', 'tilfælde'
        ]
        
        # Keywords that suggest no input is needed
        no_input_keywords = [
            'generer', 'skriv', 'opret', 'lav', 'formuler', 'design',
            'brainstorm', 'foreslå', 'opfind', 'skab'
        ]
        
        # Check for input keywords
        has_input_keyword = any(keyword in instruction_lower for keyword in input_keywords)
        
        # Check for no-input keywords at the beginning
        starts_with_no_input = any(instruction_lower.startswith(keyword) for keyword in no_input_keywords)
        
        # If instruction asks to process "following" or similar, it needs input
        if has_input_keyword and not starts_with_no_input:
            return True
            
        # If instruction is purely generative, it doesn't need input
        if starts_with_no_input and not has_input_keyword:
            return False
            
        # Default: check if instruction references something that should be provided
        return has_input_keyword
        
    def parse_instances(self, text: str, has_input: bool) -> List[Dict[str, str]]:
        """Parse instances from base model generated text."""
        instances = []
        
        # Base model continues from our prompt
        # Split by "Eksempel" to get individual examples
        examples = re.split(r'Eksempel\s*\d+:', text)
        
        for example in examples:
            example = example.strip()
            if not example:
                continue
                
            if has_input:
                # Look for Input: ... Output: ... pattern
                match = re.search(r'Input:\s*(.+?)\s*Output:\s*(.+?)(?=$)', example, re.DOTALL)
                if match:
                    instances.append({
                        'input': match.group(1).strip(),
                        'output': match.group(2).strip()
                    })
            else:
                # For no-input tasks, the whole example is the output
                output = example.strip()
                # Remove "Output:" prefix if present
                output = re.sub(r'^Output:\s*', '', output)
                if output and len(output.split()) > 2:  # Basic quality check
                    instances.append({
                        'input': "",
                        'output': output
                    })
                    
        # Strategy 2: Look for Input:/Output: pairs without example numbers
        if not instances and has_input:
            pattern = r'Input:\s*(.+?)\s*Output:\s*(.+?)(?=Input:|$)'
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for input_text, output_text in matches:
                instances.append({
                    'input': input_text.strip(),
                    'output': output_text.strip()
                })
                
        # Strategy 3: For no-input tasks, look for standalone outputs
        if not instances and not has_input:
            pattern = r'Output:\s*(.+?)(?=Output:|Eksempel|$)'
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for output_text in matches:
                output_text = output_text.strip()
                if output_text:
                    instances.append({
                        'input': "",
                        'output': output_text
                    })
                    
        return instances
        
    def parse_classification_instances(self, text: str) -> Tuple[List[str], List[Dict[str, str]]]:
        """Parse instances from classification task generation - base model style."""
        categories = []
        instances = []
        
        # For base model, we use output-first approach
        # First line after "Klasse:" contains the categories
        lines = text.strip().split('\n')
        if lines:
            # Extract unique categories from the generated text
            for line in lines:
                if ':' in line:
                    # This might be a category definition
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        cat = parts[1].strip()
                        if cat and cat not in categories:
                            categories.append(cat)
                        
        # Parse instances - for classification, we generate output-first
        # So we need to reconstruct the instances
        instances = self.parse_instances(text, has_input=True)
        
        return categories, instances
        
    def generate_instances_input_first(self, instruction: str, num_instances: int) -> List[Dict[str, str]]:
        """Generate instances using input-first approach."""
        prompt = INSTANCE_GENERATION_INPUT_FIRST_PROMPT.format(
            instruction=instruction,
            num_instances=num_instances
        )
        
        try:
            response = self.client.generate(prompt, self.generation_config)
            instances = self.parse_instances(response, has_input=True)
            return instances[:num_instances]
        except Exception as e:
            logger.error(f"Error generating instances: {e}")
            return []
            
    def generate_instances_no_input(self, instruction: str, num_instances: int) -> List[Dict[str, str]]:
        """Generate instances for tasks without input."""
        prompt = INSTANCE_GENERATION_NO_INPUT_PROMPT.format(
            instruction=instruction,
            num_instances=num_instances
        )
        
        try:
            response = self.client.generate(prompt, self.generation_config)
            instances = self.parse_instances(response, has_input=False)
            return instances[:num_instances]
        except Exception as e:
            logger.error(f"Error generating instances: {e}")
            return []
            
    def generate_instances_output_first(self, instruction: str, num_instances: int) -> List[Dict[str, str]]:
        """Generate instances using output-first approach for classification tasks."""
        prompt = INSTANCE_GENERATION_OUTPUT_FIRST_PROMPT.format(
            instruction=instruction,
            num_instances=num_instances
        )
        
        try:
            response = self.client.generate(prompt, self.generation_config)
            categories, instances = self.parse_classification_instances(response)
            
            # Log categories for debugging
            if categories:
                logger.debug(f"Found categories: {categories}")
                
            return instances[:num_instances]
        except Exception as e:
            logger.error(f"Error generating instances: {e}")
            return []
            
    def generate_for_task(self, task: Dict) -> Dict:
        """Generate instances for a single task."""
        instruction = task['instruction']
        is_classification = task.get('is_classification', False)
        num_instances = self.config['pipeline']['num_instances_per_instruction']
        
        # Check if task already has instances from seed data
        if task.get('instances') and len(task['instances']) > 0:
            logger.debug(f"Task {task['id']} already has instances, skipping generation")
            return task
            
        # Determine generation strategy
        if is_classification:
            # Use output-first approach for classification
            instances = self.generate_instances_output_first(instruction, num_instances)
        else:
            # Detect if input is needed
            needs_input = self.detect_if_input_needed(instruction)
            
            if needs_input:
                instances = self.generate_instances_input_first(instruction, num_instances)
            else:
                instances = self.generate_instances_no_input(instruction, num_instances)
                
        # Update task with instances
        task['instances'] = instances
        task['metadata'] = task.get('metadata', {})
        task['metadata']['num_instances_generated'] = len(instances)
        task['metadata']['generation_method'] = 'output_first' if is_classification else ('input_first' if self.detect_if_input_needed(instruction) else 'no_input')
        
        return task
        
    async def generate_instances_batch_async(self, tasks: List[Dict]) -> List[Dict]:
        """Generate instances for multiple tasks concurrently."""
        num_instances = self.config['pipeline']['num_instances_per_instruction']
        
        # Prepare prompts and metadata
        prompts = []
        prompt_metadata = []
        tasks_to_generate = []
        
        for i, task in enumerate(tasks):
            # Skip if task already has instances
            if task.get('instances') and len(task['instances']) > 0:
                continue
                
            instruction = task['instruction']
            is_classification = task.get('is_classification', False)
            
            # Determine prompt based on task type
            if is_classification:
                prompt = INSTANCE_GENERATION_OUTPUT_FIRST_PROMPT.format(
                    instruction=instruction,
                    num_instances=num_instances
                )
                generation_method = 'output_first'
            else:
                needs_input = self.detect_if_input_needed(instruction)
                if needs_input:
                    prompt = INSTANCE_GENERATION_INPUT_FIRST_PROMPT.format(
                        instruction=instruction,
                        num_instances=num_instances
                    )
                    generation_method = 'input_first'
                else:
                    prompt = INSTANCE_GENERATION_NO_INPUT_PROMPT.format(
                        instruction=instruction,
                        num_instances=num_instances
                    )
                    generation_method = 'no_input'
            
            prompts.append(prompt)
            prompt_metadata.append({
                'task_index': i,
                'is_classification': is_classification,
                'generation_method': generation_method,
                'needs_input': generation_method != 'no_input'
            })
            tasks_to_generate.append(task)
        
        if not prompts:
            logger.info("All tasks already have instances, skipping generation")
            return tasks
        
        # Generate all instances at once
        logger.info(f"Generating instances for {len(prompts)} tasks...")
        responses = await generate_many(
            prompts,
            self.generation_config,
            self.base_url,
            self.model_name
        )
        
        # Process responses
        for task, response, metadata in zip(tasks_to_generate, responses, prompt_metadata):
            if not response:
                logger.error(f"Instance generation failed for task {task['id']}")
                continue
                
            # Parse instances based on generation method
            if metadata['is_classification']:
                _, instances = self.parse_classification_instances(response)
            else:
                instances = self.parse_instances(
                    response, 
                    has_input=metadata['needs_input']
                )
            
            # Update task
            task['instances'] = instances[:num_instances]
            task['metadata'] = task.get('metadata', {})
            task['metadata']['num_instances_generated'] = len(instances)
            task['metadata']['generation_method'] = metadata['generation_method']
        
        return tasks
    
    def run(self, input_file: str, output_file: str):
        """Run the instance generation pipeline."""
        # Load tasks
        tasks = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line)
                tasks.append(task)
                
        logger.info(f"Loaded {len(tasks)} tasks")
        
        # Generate instances using concurrent requests
        updated_tasks = asyncio.run(self.generate_instances_batch_async(tasks))
            
        # Statistics
        total_instances = sum(len(t.get('instances', [])) for t in updated_tasks)
        tasks_with_instances = sum(1 for t in updated_tasks if t.get('instances'))
        
        logger.info(f"Generated {total_instances} instances for {tasks_with_instances} tasks")
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for task in updated_tasks:
                f.write(json.dumps(task, ensure_ascii=False) + '\n')
                
        logger.info(f"Saved to {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate instances for tasks")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with classified tasks"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    generator = InstanceGenerator(args.config)
    generator.run(args.input, args.output)


if __name__ == "__main__":
    main()