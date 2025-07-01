"""Classify tasks as classification or non-classification tasks."""

import json
import logging
from pathlib import Path
from typing import List, Dict
import yaml
from tqdm import tqdm
import asyncio

from src.simple_vllm_client import GenerationConfig, generate_many
from src.utils.prompts import CLASSIFICATION_PROMPT, format_classification_examples

logger = logging.getLogger(__name__)


# Danish classification task examples
CLASSIFICATION_EXAMPLES = [
    "Klassificer følgende tekst som positiv, negativ eller neutral.",
    "Afgør om følgende sætning er grammatisk korrekt eller ukorrekt.",
    "Identificer hvilket sprog følgende tekst er skrevet på.",
    "Kategoriser følgende nyhedsartikel som sport, politik, teknologi eller underholdning.",
    "Afgør om følgende e-mail er spam eller ikke-spam.",
    "Klassificer følgende anmeldelse på en skala fra 1 til 5 stjerner.",
    "Identificer om følgende udsagn er sandt eller falsk.",
    "Kategoriser følgende produkt som elektronik, tøj, mad eller møbler.",
    "Afgør om følgende tekst indeholder personlige oplysninger eller ej.",
    "Klassificer følgende billede som portræt, landskab eller abstrakt.",
    "Identificer tonen i følgende besked som formel eller uformel.",
    "Afgør om følgende påstand er faktuel eller meningsbaseret."
]

# Danish non-classification task examples
NON_CLASSIFICATION_EXAMPLES = [
    "Skriv en kort historie om en rejse til månen.",
    "Forklar hvordan fotosyntese fungerer.",
    "Oversæt følgende sætning fra engelsk til dansk.",
    "Opsummer hovedpunkterne i følgende artikel.",
    "Generer fem kreative navne til en ny restaurant.",
    "Løs følgende matematiske ligning.",
    "Skriv en e-mail til din chef om at bede om ferie.",
    "Beskriv fremgangsmåden for at bage en kage.",
    "List fem fordele ved at dyrke motion regelmæssigt.",
    "Omskriv følgende sætning så den bliver mere formel.",
    "Udvid følgende ide til en fuld forretningsplan.",
    "Generer spørgsmål til et job interview.",
    "Skriv en produktbeskrivelse for en ny smartphone.",
    "Forklar forskellen mellem to begreber.",
    "Lav en daglig rutine for en studerende.",
    "Skriv dialoger mellem to karakterer i en film.",
    "Beregn den samlede pris inklusive moms.",
    "Formuler argumenter for og imod et emne.",
    "Opret en opskrift med de givne ingredienser."
]


class TaskClassifier:
    """Classify tasks as classification or non-classification."""
    
    def __init__(self, config_path: str):
        """Initialize the task classifier.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.base_url = self.config['vllm']['base_url']
        self.model_name = self.config['vllm'].get('model_name')
        self.generation_config = GenerationConfig(**self.config['generation']['classification'])
        
        # We don't need the Danish examples anymore with the simplified English prompt
        
    async def classify_batch_async(self, tasks: List[Dict]) -> List[Dict]:
        """Classify a batch of tasks using async requests.
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            Updated tasks with is_classification field
        """
        # Create prompts for all tasks
        prompts = []
        for task in tasks:
            prompt = CLASSIFICATION_PROMPT.format(
                instruction=task['instruction']
            )
            prompts.append(prompt)
        
        # Generate all classifications at once
        logger.info(f"Classifying {len(prompts)} tasks...")
        responses = await generate_many(
            prompts,
            self.generation_config,
            self.base_url,
            self.model_name
        )
        
        # Process responses
        classified_tasks = []
        for task, response in zip(tasks, responses):
            if not response:
                logger.error(f"Classification failed for task {task['id']}")
                task['is_classification'] = False
            else:
                # Parse response - clean it first
                response_clean = response.strip().lower()
                
                # Remove any trailing punctuation or whitespace
                response_clean = response_clean.rstrip('.,!? \n\r\t')
                
                # Check for Danish "ja" (yes) or "nej" (no)
                if 'ja' in response_clean or response_clean == 'j':
                    task['is_classification'] = True
                elif 'nej' in response_clean or response_clean == 'n':
                    task['is_classification'] = False
                else:
                    # Log the full response for debugging
                    logger.warning(f"Unexpected classification response for task '{task['instruction'][:50]}...': '{response}' (cleaned: '{response_clean}')")
                    task['is_classification'] = False
            
            classified_tasks.append(task)
        
        return classified_tasks
            
    def classify_batch(self, tasks: List[Dict]) -> List[Dict]:
        """Synchronous wrapper for batch classification."""
        return asyncio.run(self.classify_batch_async(tasks))
        
    def run(self, input_file: str, output_file: str):
        """Run the classification pipeline.
        
        Args:
            input_file: Path to input JSONL file with tasks
            output_file: Path to output JSONL file
        """
        # Load tasks
        tasks = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line)
                tasks.append(task)
                
        logger.info(f"Loaded {len(tasks)} tasks to classify")
        
        # Classify tasks
        classified_tasks = self.classify_batch(tasks)
        
        # Count results
        num_classification = sum(1 for t in classified_tasks if t['is_classification'])
        num_non_classification = len(classified_tasks) - num_classification
        
        logger.info(f"Classification tasks: {num_classification}")
        logger.info(f"Non-classification tasks: {num_non_classification}")
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for task in classified_tasks:
                f.write(json.dumps(task, ensure_ascii=False) + '\n')
                
        logger.info(f"Saved classified tasks to {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify tasks")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with tasks"
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
    
    classifier = TaskClassifier(args.config)
    classifier.run(args.input, args.output)


if __name__ == "__main__":
    main()