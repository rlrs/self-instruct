#!/usr/bin/env python3
"""Run the complete Self-Instruct pipeline."""

import argparse
import logging
import sys
from pathlib import Path
import time
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generate_instructions import InstructionGenerator
from src.classify_tasks import TaskClassifier
from src.generate_instances import InstanceGenerator
from src.prepare_training_data import TrainingDataProcessor

logger = logging.getLogger(__name__)


def setup_logging(level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )


def run_pipeline(config_path: str, output_dir: str, iterations: int = 1, continue_from: int = None):
    """Run the complete Self-Instruct pipeline with bootstrapping.
    
    Args:
        config_path: Path to configuration file
        output_dir: Base output directory
        iterations: Number of iterations to run
        continue_from: If specified, continue from this iteration number
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track all generated files and cumulative instructions
    all_generated_files = []
    cumulative_instructions_file = output_path / "cumulative_instructions.jsonl"
    
    # Initialize cumulative file with seed tasks if it doesn't exist
    if not cumulative_instructions_file.exists():
        seed_file = Path("data/seed_tasks.jsonl")
        if seed_file.exists():
            import shutil
            shutil.copy(seed_file, cumulative_instructions_file)
            logger.info(f"Initialized cumulative instructions with seed tasks")
    
    # Track total generated count for continuous ID numbering
    total_generated_count = 0
    
    # If continuing from a specific iteration, load existing state
    start_iteration = 0
    if continue_from is not None:
        start_iteration = continue_from
        logger.info(f"Continuing from iteration {continue_from}")
        
        # Count existing instructions to maintain ID continuity
        existing_count = 0
        with open(cumulative_instructions_file, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line)
                if task['id'].startswith('generated_task_'):
                    existing_count += 1
        
        total_generated_count = existing_count
        logger.info(f"Found {existing_count} existing generated instructions")
        
        # Collect existing generated files
        for i in range(1, continue_from + 1):
            iter_dir = output_path / f"iteration_{i}"
            instances_file = iter_dir / "tasks_with_instances.jsonl"
            if instances_file.exists():
                all_generated_files.append(str(instances_file))
    
    for iteration in range(start_iteration, iterations):
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting iteration {iteration + 1}/{iterations}")
        logger.info(f"{'='*50}\n")
        
        iter_dir = output_path / f"iteration_{iteration + 1}"
        iter_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        
        # Step 1: Generate instructions with bootstrapping
        logger.info("Step 1: Generating new instructions...")
        
        # For iterations after the first, use cumulative instructions for bootstrapping
        if iteration > 0:
            instruction_gen = InstructionGenerator(config_path, str(cumulative_instructions_file))
        else:
            instruction_gen = InstructionGenerator(config_path)
            
        instruction_gen.run(str(iter_dir), start_id=total_generated_count)
        
        generated_instructions_file = iter_dir / "generated_instructions.jsonl"
        
        # Step 2: Classify tasks
        logger.info("\nStep 2: Classifying tasks...")
        classifier = TaskClassifier(config_path)
        classified_file = iter_dir / "classified_tasks.jsonl"
        classifier.run(
            input_file=str(generated_instructions_file),
            output_file=str(classified_file)
        )
        
        # Step 3: Generate instances
        logger.info("\nStep 3: Generating instances...")
        instance_gen = InstanceGenerator(config_path)
        instances_file = iter_dir / "tasks_with_instances.jsonl"
        instance_gen.run(
            input_file=str(classified_file),
            output_file=str(instances_file)
        )
        
        all_generated_files.append(str(instances_file))
        
        # Step 3.5: Append new instructions to cumulative file for bootstrapping
        logger.info("\nUpdating cumulative instructions for bootstrapping...")
        new_count = 0
        with open(instances_file, 'r', encoding='utf-8') as f_in:
            with open(cumulative_instructions_file, 'a', encoding='utf-8') as f_out:
                for line in f_in:
                    f_out.write(line)
                    new_count += 1
        
        total_generated_count += new_count
        logger.info(f"Added {new_count} new instructions to cumulative pool")
        logger.info(f"Total cumulative instructions: {total_generated_count + 114}")  # 114 seed tasks
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nIteration {iteration + 1} completed in {elapsed_time:.2f} seconds")
    
    # Step 4: Prepare training data from all iterations
    logger.info(f"\n{'='*50}")
    logger.info("Step 4: Preparing final training data...")
    logger.info(f"{'='*50}\n")
    
    # Use the cumulative file which includes both seed and all generated instructions
    processor = TrainingDataProcessor(config_path)
    training_dir = output_path / "training_data"
    processor.run(
        input_files=[str(cumulative_instructions_file)],
        output_dir=str(training_dir),
        include_seed=False  # Seeds are already in cumulative file
    )
    
    # Save pipeline metadata
    metadata = {
        "iterations": iterations,
        "config_file": config_path,
        "output_directory": str(output_path),
        "cumulative_instructions_file": str(cumulative_instructions_file),
        "generated_files": all_generated_files,
        "total_instructions": total_generated_count + 114,  # Including seed tasks
        "training_data_directory": str(training_dir),
        "bootstrapping": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_file = output_path / "pipeline_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("Pipeline completed successfully!")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Training data: {training_dir}")
    logger.info(f"{'='*50}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the Self-Instruct pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single iteration
  python scripts/run_pipeline.py
  
  # Run multiple iterations
  python scripts/run_pipeline.py --iterations 3
  
  # Continue from iteration 5 and run to iteration 10
  python scripts/run_pipeline.py --continue-from 5 --iterations 10
  
  # Just generate training data from existing iterations
  python scripts/run_pipeline.py --continue-from 5 --iterations 5
  
  # Use custom config
  python scripts/run_pipeline.py --config configs/custom.yaml
        """
    )
    
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated",
        help="Output directory for all generated data"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run"
    )
    parser.add_argument(
        "--continue-from",
        type=int,
        default=None,
        help="Continue from a specific iteration (e.g., --continue-from 3 to skip iterations 1-3)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Check if model server is running
    logger.info("Checking model server connection...")
    try:
        import yaml
        import aiohttp
        import asyncio
        
        # Load config to get server URL
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        base_url = config['server']['base_url']
        
        # Simple health check
        async def check_health():
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/health") as response:
                    return response.status == 200
        
        is_healthy = asyncio.run(check_health())
        
        if is_healthy:
            logger.info("Model server is running and accessible")
        else:
            raise Exception("Model server health check failed")
            
    except Exception as e:
        logger.error(f"Failed to connect to model server: {e}")
        logger.error("Please ensure model server is running at the configured URL")
        sys.exit(1)
    
    # Run pipeline
    try:
        run_pipeline(
            config_path=args.config,
            output_dir=args.output_dir,
            iterations=args.iterations,
            continue_from=args.continue_from
        )
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()