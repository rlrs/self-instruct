"""Generate new instructions using Self-Instruct methodology."""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Set
import logging
from rouge_score import rouge_scorer
import yaml
from tqdm import tqdm
import asyncio

from src.openai_client import GenerationConfig, generate_many
from src.utils.prompts import INSTRUCTION_GENERATION_PROMPT, format_instruction_examples

logger = logging.getLogger(__name__)


class InstructionGenerator:
    """Generate new instructions based on seed tasks."""
    
    def __init__(self, config_path: str, previous_instructions_file: str = None):
        """Initialize the instruction generator.
        
        Args:
            config_path: Path to configuration file
            previous_instructions_file: Optional path to previous instructions for bootstrapping
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.base_url = self.config['server']['base_url']
        self.model_name = self.config['server'].get('model_name')
        self.generation_config = GenerationConfig(**self.config['generation']['instruction'])
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        
        # Load existing instructions
        self.all_instructions = []
        self.instruction_set = set()
        self.machine_instructions = []  # Track machine-generated separately
        
        # Load previous instructions if provided (for bootstrapping)
        if previous_instructions_file and Path(previous_instructions_file).exists():
            self.load_previous_instructions(previous_instructions_file)
        
    def load_previous_instructions(self, file_path: str):
        """Load previously generated instructions for bootstrapping."""
        logger.info(f"Loading previous instructions from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line)
                instruction = task['instruction']
                self.all_instructions.append(instruction)
                self.instruction_set.add(instruction.lower().strip())
                self.machine_instructions.append(instruction)
        logger.info(f"Loaded {len(self.machine_instructions)} previous machine-generated instructions")
        
    def load_seed_tasks(self, seed_file: str) -> List[Dict]:
        """Load seed tasks from JSONL file."""
        tasks = []
        with open(seed_file, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line)
                tasks.append(task)
                self.all_instructions.append(task['instruction'])
                self.instruction_set.add(task['instruction'].lower().strip())
        return tasks
        
    def is_similar(self, instruction: str, threshold: float = None) -> bool:
        """Check if instruction is too similar to existing ones using ROUGE-L."""
        if threshold is None:
            threshold = self.config['filtering']['rouge_threshold']
            
        instruction_lower = instruction.lower().strip()
        
        # Quick exact match check
        if instruction_lower in self.instruction_set:
            return True
            
        # ROUGE-L similarity check
        for existing in self.all_instructions:
            scores = self.rouge_scorer.score(existing, instruction)
            if scores['rougeL'].fmeasure > threshold:
                return True
                
        return False
        
    def filter_instruction(self, instruction: str) -> bool:
        """Check if instruction passes quality filters."""
        # Length check
        min_len = self.config['filtering']['min_instruction_length']
        max_len = self.config['filtering']['max_instruction_length']
        
        if len(instruction) < min_len:
            logger.debug(f"Filtered (too short): {instruction[:50]}...")
            return False
        if len(instruction) > max_len:
            logger.debug(f"Filtered (too long): {instruction[:50]}...")
            return False
            
        # Blacklist keywords
        blacklist = self.config['filtering']['blacklist_keywords']
        instruction_lower = instruction.lower()
        
        for keyword in blacklist:
            if keyword.lower() in instruction_lower:
                logger.debug(f"Filtered (blacklist '{keyword}'): {instruction[:50]}...")
                return False
                
        # Basic quality checks
        if instruction.count(' ') < 2:  # Too short
            logger.debug(f"Filtered (not enough words): {instruction[:50]}...")
            return False
            
        return True
        
    def parse_generated_instructions(self, text: str) -> List[str]:
        """Parse instructions from generated text - base model format."""
        instructions = []
        
        # Split by newline and number pattern like original
        # The model continues from where we left off (e.g., "9. instruction")
        raw_instructions = re.split(r'\n\d+\s?\.\s?', text)
        
        for inst in raw_instructions:
            inst = re.sub(r"\s+", " ", inst).strip()
            inst = inst.strip()
            if inst == "":
                continue
            # Basic quality check - at least 3 words
            if len(inst.split()) > 3:
                instructions.append(inst)
                    
        return instructions
        
    def create_prompts(self, num_prompts: int, num_instructions_per_prompt: int) -> List[str]:
        """Create multiple instruction generation prompts with bootstrapping."""
        prompts = []
        
        for _ in range(num_prompts):
            # Sample examples for each prompt
            num_examples = self.config['pipeline']['num_prompt_instructions']
            
            # Separate human seed instructions from all instructions
            seed_instructions = self.all_instructions[:len(self.seed_tasks)]
            
            # Determine how many machine vs human examples to use
            # Follow original paper: use 0-2 machine examples based on availability
            num_machine_available = len(self.machine_instructions)
            num_machine_to_use = min(2, num_machine_available)  # Use up to 2 machine examples
            num_human_to_use = num_examples - num_machine_to_use
            
            examples = []
            
            # Sample machine-generated instructions if available
            if num_machine_to_use > 0:
                machine_examples = random.sample(self.machine_instructions, num_machine_to_use)
                examples.extend(machine_examples)
                logger.debug(f"Sampled {num_machine_to_use} machine-generated examples")
            
            # Sample human seed instructions
            human_examples = random.sample(seed_instructions, min(num_human_to_use, len(seed_instructions)))
            examples.extend(human_examples)
            logger.debug(f"Sampled {len(human_examples)} human seed examples")
            
            # Shuffle to mix machine and human examples
            random.shuffle(examples)
            
            # Create prompt with base model style
            examples_text = format_instruction_examples(examples[:num_examples], num_examples)
            # Next number after examples
            next_num = num_examples + 1
            prompt = INSTRUCTION_GENERATION_PROMPT.format(
                examples=examples_text,
                next_number=next_num
            )
            prompts.append(prompt)
            
        return prompts
        
    async def generate_single_prompt_async(self, semaphore, generated_instructions, pbar, stats, target_count):
        """Generate instructions from a single prompt."""
        async with semaphore:
            # Check if we've already reached our target
            if stats['kept'] >= target_count:
                return
                
            # Create a single prompt
            prompts = self.create_prompts(1, 10)  # Generate ~10 instructions per prompt
            prompt = prompts[0]
            
            try:
                # Use generate_many with a single prompt for consistency
                responses = await generate_many(
                    [prompt],
                    self.generation_config,
                    self.base_url,
                    self.model_name
                )
                
                response = responses[0] if responses else None
                if not response:
                    return
                    
                new_instructions = self.parse_generated_instructions(response)
                logger.debug(f"Generated {len(new_instructions)} instructions")
                
                # Filter and add valid instructions
                for instruction in new_instructions:
                    # Check if we've reached target before processing each instruction
                    if stats['kept'] >= target_count:
                        break
                        
                    stats['total_generated'] += 1
                    
                    if not self.filter_instruction(instruction):
                        stats['filtered_by_quality'] += 1
                        continue
                        
                    if not self.is_similar(instruction):
                        generated_instructions.append(instruction)
                        self.all_instructions.append(instruction)
                        self.instruction_set.add(instruction.lower().strip())
                        pbar.update(1)
                        stats['kept'] += 1
                    else:
                        stats['filtered_by_similarity'] += 1
                        
                # Update progress bar description
                pbar.set_description(
                    f"Generating (kept: {stats['kept']}, "
                    f"filtered: {stats['filtered_by_quality'] + stats['filtered_by_similarity']}/{stats['total_generated']})"
                )
                        
            except Exception as e:
                logger.error(f"Generation error: {e}", exc_info=True)
    
    async def generate_batch_async(self, num_instructions: int = 100) -> List[str]:
        """Generate a batch of new instructions with high concurrency."""
        generated_instructions = []
        
        pbar = tqdm(total=num_instructions, desc="Generating instructions")
        
        # Track statistics
        stats = {
            'total_generated': 0,
            'filtered_by_quality': 0,
            'filtered_by_similarity': 0,
            'kept': 0
        }
        
        # Use a semaphore to control concurrency
        # vLLM can handle high concurrency, so we can be aggressive
        max_concurrent = self.config.get('pipeline', {}).get('max_concurrent_requests', 100)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Generate more than needed to account for filtering
        # Estimate based on typical filtering rate
        overgenerate_factor = 2.0  # Generate 2x what we need
        prompts_needed = int(num_instructions * overgenerate_factor / 10)  # ~10 instructions per prompt
        
        # Create all tasks upfront
        tasks = []
        for _ in range(prompts_needed):
            task = self.generate_single_prompt_async(semaphore, generated_instructions, pbar, stats, num_instructions)
            tasks.append(task)
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
        
        # If we still need more, generate additional ones
        while (num_missing := num_instructions - len(generated_instructions)) > 0:
            additional_tasks = []
            for _ in range(2 * num_missing):
                task = self.generate_single_prompt_async(semaphore, generated_instructions, pbar, stats, num_instructions)
                additional_tasks.append(task)
            
            await asyncio.gather(*additional_tasks)
            
            # Break if we're not making progress
            if stats['kept'] == len(generated_instructions):
                logger.warning("Not making progress, stopping generation")
                break
                
        pbar.close()
        
        # Log final statistics
        logger.info(f"Generation statistics:")
        logger.info(f"  Total generated: {stats['total_generated']}")
        logger.info(f"  Filtered by quality: {stats['filtered_by_quality']}")
        logger.info(f"  Filtered by similarity: {stats['filtered_by_similarity']}")
        logger.info(f"  Kept: {len(generated_instructions)}")
        
        return generated_instructions[:num_instructions]
        
    def generate_batch(self, num_instructions: int = 100) -> List[str]:
        """Synchronous wrapper for batch generation."""
        return asyncio.run(self.generate_batch_async(num_instructions))
        
    def run(self, output_dir: str, start_id: int = 0):
        """Run the instruction generation pipeline.
        
        Args:
            output_dir: Directory to save output
            start_id: Starting ID for generated tasks (for continuous numbering across iterations)
        """
        # Load seed tasks
        seed_file = self.config['data']['seed_file']
        self.seed_tasks = self.load_seed_tasks(seed_file)
        logger.info(f"Loaded {len(self.seed_tasks)} seed tasks")
        
        # Log bootstrapping info
        if self.machine_instructions:
            logger.info(f"Bootstrapping with {len(self.machine_instructions)} machine-generated instructions")
            logger.info(f"Total instruction pool size: {len(self.all_instructions)}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate instructions
        num_to_generate = self.config['pipeline']['num_instructions_per_iteration']
        new_instructions = self.generate_batch(num_to_generate)
        
        # Save generated instructions
        output_file = output_path / "generated_instructions.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, instruction in enumerate(new_instructions):
                task = {
                    "id": f"generated_task_{start_id + i}",
                    "instruction": instruction,
                    "instances": [],  # Will be filled later
                    "is_classification": None  # Will be determined later
                }
                f.write(json.dumps(task, ensure_ascii=False) + '\n')
                
        logger.info(f"Generated {len(new_instructions)} new instructions")
        logger.info(f"Saved to {output_file}")
        
        # Also save all instructions for reference
        all_instructions_file = output_path / "all_instructions.json"
        with open(all_instructions_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_instructions, f, ensure_ascii=False, indent=2)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate new instructions")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    generator = InstructionGenerator(args.config)
    generator.run(args.output_dir)


if __name__ == "__main__":
    main()