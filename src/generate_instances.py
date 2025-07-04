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

from src.openai_client import GenerationConfig, generate_many
from src.utils.prompts import (
    INSTANCE_GENERATION_INPUT_FIRST_PROMPT,
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
            
        self.base_url = self.config['server']['base_url']
        self.model_name = self.config['server'].get('model_name')
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
        
    def parse_input_output(self, text: str) -> Tuple[str, str]:
        """Parse input and output from text using the original paper's approach."""
        # Look for Output marker to split input/output
        if re.findall(r"Output\s*\d*\s*:", text):
            parts = re.split(r"Output\s*\d*\s*:", text, 1)
            inst_input = parts[0].strip()
            inst_output = parts[1].strip() if len(parts) > 1 else ""
        else:
            # No Output marker found, treat everything as output with no input
            inst_input = ""
            inst_output = text.strip()
            
        # To avoid the case where multiple input/output pairs are generated
        if re.findall(r"Input\s*\d*\s*:", inst_output):
            inst_output = re.split(r"Input\s*\d*\s*:", inst_output)[0].strip()
            
        # Remove the prefix "Input:" from the input string if present
        inst_input = re.sub(r"^Input\s*\d*\s*:", "", inst_input).strip()
        
        return inst_input, inst_output
    
    def parse_instances(self, text: str, generation_method: str = 'input_first') -> List[Dict[str, str]]:
        """Parse instances from base model generated text matching the original paper's approach."""
        instances = []
        text = text.strip()
        
        # Check if we have Example markers
        if re.findall(r"Eksempel\s*\d*\.?", text):
            # Split by Example markers
            instance_texts = re.split(r"Eksempel\s*\d*\.?", text)
            instance_texts = [it.strip() for it in instance_texts if it.strip()]
            
            for instance_text in instance_texts:
                inst_input, inst_output = self.parse_input_output(instance_text)
                if inst_output:  # Only add if we have output
                    instances.append({
                        'input': inst_input,
                        'output': inst_output
                    })
        elif re.findall(r"Output\s*\d*\s*:", text):
            # No Example markers but has Output marker - parse as single instance
            inst_input, inst_output = self.parse_input_output(text)
            if inst_output:
                instances.append({
                    'input': inst_input,
                    'output': inst_output
                })
        else:
            # No markers at all - treat entire text as output
            # This handles cases where model continues directly after "Output:" in prompt
            if text:
                instances.append({
                    'input': "",
                    'output': text
                })
        
        return instances
        
    def parse_classification_instances(self, text: str) -> Tuple[List[str], List[Dict[str, str]]]:
        """Parse instances from classification task generation matching the original paper's approach."""
        categories = []
        instances = []
        
        # Check if we have the expected "Klasseetiket:" marker
        if "Klasseetiket:" not in text:
            # Fallback to regular parsing if format is unexpected
            return [], self.parse_instances(text, generation_method='output_first')
        
        # Split by "Klasseetiket:" to get each instance
        instance_texts = text.split("Klasseetiket:")[1:]  # Skip the part before first marker
        
        for instance_text in instance_texts:
            instance_text = instance_text.strip()
            if not instance_text:
                continue
                
            # Split by newline - first line is the class label, rest is the input
            lines = instance_text.split('\n', 1)
            
            if len(lines) >= 1:
                # First line is the class label (output)
                class_label = lines[0].strip()
                if class_label and class_label not in categories:
                    categories.append(class_label)
                
                # Rest is the input (if any)
                if len(lines) > 1:
                    input_text = lines[1].strip()
                else:
                    input_text = ""
                
                # For classification, output is the class label, input is the text to classify
                if class_label:
                    instances.append({
                        'input': input_text,
                        'output': class_label
                    })
        
        return categories, instances
        
    def generate_instances_input_first(self, instruction: str, num_instances: int) -> List[Dict[str, str]]:
        """Generate instances using input-first approach for all non-classification tasks."""
        prompt = INSTANCE_GENERATION_INPUT_FIRST_PROMPT.format(
            instruction=instruction
        )
        
        try:
            # Always log the first few for debugging
            logger.info(f"Generating instances for instruction: {instruction[:100]}...")
            logger.info(f"Prompt ending: ...{prompt[-200:]}")
            
            response = self.client.generate(prompt, self.generation_config)
            
            logger.info(f"Raw response (first 300 chars): {response[:300]}")
            logger.info(f"Response length: {len(response)} characters")
            
            instances = self.parse_instances(response, generation_method='input_first')
            logger.info(f"Parsed {len(instances)} instances")
            if instances:
                logger.info(f"First instance: {instances[0]}")
            
            return instances[:num_instances]
        except Exception as e:
            logger.error(f"Error generating instances: {e}")
            return []
            
    def generate_instances_output_first(self, instruction: str, num_instances: int) -> List[Dict[str, str]]:
        """Generate instances using output-first approach for classification tasks."""
        prompt = INSTANCE_GENERATION_OUTPUT_FIRST_PROMPT.format(
            instruction=instruction
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
            # Use input-first approach for all non-classification tasks
            instances = self.generate_instances_input_first(instruction, num_instances)
                
        # Update task with instances
        task['instances'] = instances
        task['metadata'] = task.get('metadata', {})
        task['metadata']['num_instances_generated'] = len(instances)
        task['metadata']['generation_method'] = 'output_first' if is_classification else 'input_first'
        
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
                    instruction=instruction
                )
                generation_method = 'output_first'
            else:
                prompt = INSTANCE_GENERATION_INPUT_FIRST_PROMPT.format(
                    instruction=instruction
                )
                generation_method = 'input_first'
            
            prompts.append(prompt)
            prompt_metadata.append({
                'task_index': i,
                'is_classification': is_classification,
                'generation_method': generation_method
            })
            tasks_to_generate.append(task)
        
        if not prompts:
            logger.info("All tasks already have instances, skipping generation")
            return tasks
        
        # Save first few prompts for debugging
        debug_dir = Path("debug_prompts")
        debug_dir.mkdir(exist_ok=True)
        for i, (prompt, task) in enumerate(zip(prompts[:5], tasks_to_generate[:5])):
            debug_file = debug_dir / f"instance_prompt_{i}_{task['id']}.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"Task ID: {task['id']}\n")
                f.write(f"Instruction: {task['instruction']}\n")
                f.write(f"Is Classification: {task.get('is_classification', False)}\n")
                f.write(f"Generation Method: {prompt_metadata[i]['generation_method']}\n")
                f.write(f"\n{'='*80}\nFULL PROMPT:\n{'='*80}\n\n")
                f.write(prompt)
                f.write(f"\n\n{'='*80}\nEND OF PROMPT\n{'='*80}\n")
            logger.info(f"Saved debug prompt to {debug_file}")
        
        # Generate all instances at once
        logger.info(f"Generating instances for {len(prompts)} tasks...")
        responses = await generate_many(
            prompts,
            self.generation_config,
            self.base_url,
            self.model_name
        )
        
        # Process responses
        for i, (task, response, metadata) in enumerate(zip(tasks_to_generate, responses, prompt_metadata)):
            try:
                if not response:
                    logger.error(f"Instance generation failed for task {task['id']}: Empty response from model")
                    logger.error(f"  Instruction: {task['instruction'][:100]}...")
                    logger.error(f"  Is classification: {metadata['is_classification']}")
                    logger.error(f"  Generation method: {metadata['generation_method']}")
                    task['instances'] = []
                    task['metadata'] = task.get('metadata', {})
                    task['metadata']['num_instances_generated'] = 0
                    task['metadata']['generation_failed'] = True
                    task['metadata']['failure_reason'] = "Empty response from model"
                    continue
                
                # Save responses for debugging
                if i < 5:
                    debug_file = debug_dir / f"instance_response_{i}_{task['id']}.txt"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(f"Task ID: {task['id']}\n")
                        f.write(f"Instruction: {task['instruction']}\n")
                        f.write(f"Is Classification: {metadata['is_classification']}\n")
                        f.write(f"Generation Method: {metadata['generation_method']}\n")
                        f.write(f"\n{'='*80}\nRESPONSE:\n{'='*80}\n\n")
                        f.write(response if response else "[EMPTY RESPONSE]")
                        f.write(f"\n\n{'='*80}\nEND OF RESPONSE\n{'='*80}\n")
                    logger.info(f"Saved debug response to {debug_file}")
                    
                # Parse instances based on generation method
                if metadata['is_classification']:
                    _, instances = self.parse_classification_instances(response)
                else:
                    # For input_first template, parse both input and no-input cases
                    instances = self.parse_instances(response, generation_method='input_first')
                
                # Update task
                task['instances'] = instances[:num_instances]
                task['metadata'] = task.get('metadata', {})
                task['metadata']['num_instances_generated'] = len(instances)
                task['metadata']['generation_method'] = metadata['generation_method']
                
                # Debug logging for first few tasks
                if i < 3 and instances:
                    logger.info(f"Parsed {len(instances)} instances, first: {instances[0]}")
                
                if not instances:
                    logger.warning(f"No instances parsed for task {task['id']} from response: {response[:200]}...")
                    
            except Exception as e:
                logger.error(f"Error processing response for task {task['id']}: {str(e)}", exc_info=True)
                task['instances'] = []
                task['metadata'] = task.get('metadata', {})
                task['metadata']['num_instances_generated'] = 0
                task['metadata']['generation_failed'] = True
                task['metadata']['failure_reason'] = str(e)
        
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