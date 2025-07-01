#!/usr/bin/env python3
"""
Score instruction-tuning data quality using an instruction-tuned model.
Assumes a local server running with OpenAI-compatible API.
"""

import json
import argparse
from typing import Dict, List, Tuple
import requests
from dataclasses import dataclass
import time
from tqdm import tqdm
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor


@dataclass
class QualityScores:
    """Scores for a single instruction-completion pair."""
    instruction_alignment_completeness: int  # 1-5
    response_coherence: int                  # 1-5
    danish_language_appropriateness: int     # 1-5
    assistant_persona: int                   # 1-5
    information_quality: int                 # 1-5
    
    @property
    def total_score(self) -> float:
        """Simple average of all scores."""
        return (self.instruction_alignment_completeness + 
                self.response_coherence + 
                self.danish_language_appropriateness + 
                self.assistant_persona + 
                self.information_quality) / 5.0
    
    def to_dict(self) -> Dict:
        return {
            "instruction_alignment_completeness": self.instruction_alignment_completeness,
            "response_coherence": self.response_coherence,
            "danish_language_appropriateness": self.danish_language_appropriateness,
            "assistant_persona": self.assistant_persona,
            "information_quality": self.information_quality,
            "total_score": self.total_score
        }


class QualityScorer:
    """Score instruction-tuning data quality using an LLM."""
    
    def __init__(self, api_url: str = "http://localhost:8000/v1/chat/completions"):
        self.api_url = api_url
        self.system_prompt = """You are evaluating the quality of instruction-completion pairs for training a Danish language assistant. 

IMPORTANT: A good assistant should:
- Point out when instructions are unclear, incomplete, or missing information
- Ask for clarification rather than making assumptions
- Never invent content when the instruction refers to something not provided
- Acknowledge when a task cannot be completed due to missing information

Please analyze each instruction-completion pair across these 5 dimensions and provide scores from 1-5:

1. **Instruction Alignment & Completeness**: How well and completely does the response follow the instruction?
   - 1: Ignores instruction, just repeats it, invents content when instruction refers to missing information, or gives an extremely brief/incomplete response
   - 2: Attempts to follow but makes assumptions about missing information, misses key points, or is overly compressed
   - 3: Partially addresses the instruction with moderate completeness, may have minor assumption issues
   - 4: Mostly complete response with minor omissions, properly handles any ambiguities
   - 5: Perfectly follows instruction, asks for clarification when needed, never invents missing content

2. **Response Coherence**: How logical and well-structured is the response?
   - 1: Incoherent, contradictory, or nonsensical
   - 2: Some logical gaps or confusion
   - 3: Generally makes sense with minor issues
   - 4: Clear logical flow throughout
   - 5: Exceptionally well-structured and coherent

3. **Danish Language Appropriateness**: Is the response appropriate for Danish context and language?
   - 1: Wrong language or completely culturally inappropriate
   - 2: Uses English when Danish expected, or major cultural errors
   - 3: Mostly Danish but some unnecessary English or cultural confusion
   - 4: Good Danish usage with minor issues
   - 5: Perfect Danish language and cultural context (or correctly uses English when appropriate)

4. **Assistant Persona**: Does the response maintain proper assistant behavior?
   - 1: Roleplays as human with fake personal experiences
   - 2: Makes up personal anecdotes or claims impossible abilities
   - 3: Mostly appropriate but occasional persona slips
   - 4: Good assistant behavior with minor issues
   - 5: Perfect assistant persona - helpful, honest about limitations, no false experiences

5. **Information Quality**: Is the information accurate, relevant, and helpful?
   - 1: Inaccurate, irrelevant, or potentially harmful information
   - 2: Some accuracy issues or largely unhelpful
   - 3: Generally accurate but could be more helpful
   - 4: Good quality information with minor issues
   - 5: Excellent, accurate, and genuinely helpful information

Please analyze each dimension and provide your reasoning, then end with your scores as exactly 5 comma-separated numbers."""
        
        self.user_prompt_template = """INSTRUCTION:
{instruction}

COMPLETION:
{completion}

Please analyze this instruction-completion pair and provide your scores."""

    def score_single(self, instruction: str, completion: str) -> QualityScores:
        """Score a single instruction-completion pair."""
        user_prompt = self.user_prompt_template.format(
            instruction=instruction,
            completion=completion
        )
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": "google/gemma-3-27b-it",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result['choices'][0]['message']['content'].strip()
            
            # Parse the scores with multiple fallback methods
            scores = self._extract_scores(response_text)
            if len(scores) != 5:
                raise ValueError(f"Expected 5 scores, got {len(scores)}: {scores}")
            
            return QualityScores(
                instruction_alignment=scores[0],
                response_completeness=scores[1],
                information_density=scores[2],
                response_coherence=scores[3],
                format_appropriateness=scores[4]
            )
            
        except Exception as e:
            print(f"Error scoring: {e}")
            # Return neutral scores on error
            return QualityScores(3, 3, 3, 3, 3)
    
    def _extract_scores(self, response_text: str) -> List[int]:
        """Extract scores from response text using multiple fallback methods."""
        import re
        
        # Method 1: Look for explicit score line at the end
        lines = response_text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if re.match(r'^[\d\s,]+$', line) and ',' in line:
                try:
                    scores = [int(s.strip()) for s in line.split(',')]
                    if len(scores) == 5 and all(1 <= s <= 5 for s in scores):
                        return scores
                except ValueError:
                    continue
        
        # Method 2: Look for "Scores:" or similar patterns
        score_patterns = [
            r'scores?:\s*([1-5]\s*,\s*[1-5]\s*,\s*[1-5]\s*,\s*[1-5]\s*,\s*[1-5])',
            r'final scores?:\s*([1-5]\s*,\s*[1-5]\s*,\s*[1-5]\s*,\s*[1-5]\s*,\s*[1-5])',
            r'rating[s]?:\s*([1-5]\s*,\s*[1-5]\s*,\s*[1-5]\s*,\s*[1-5]\s*,\s*[1-5])',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    scores = [int(s.strip()) for s in match.group(1).split(',')]
                    if len(scores) == 5 and all(1 <= s <= 5 for s in scores):
                        return scores
                except ValueError:
                    continue
        
        # Method 3: Look for any sequence of 5 comma-separated numbers in valid range
        all_numbers = re.findall(r'([1-5]\s*,\s*[1-5]\s*,\s*[1-5]\s*,\s*[1-5]\s*,\s*[1-5])', response_text)
        if all_numbers:
            try:
                scores = [int(s.strip()) for s in all_numbers[-1].split(',')]
                if len(scores) == 5 and all(1 <= s <= 5 for s in scores):
                    return scores
            except ValueError:
                pass
        
        # Method 4: Extract individual dimension scores from text
        dimension_scores = []
        dimension_patterns = [
            r'instruction alignment.*?([1-5])',
            r'response coherence.*?([1-5])',
            r'danish language.*?([1-5])',
            r'assistant persona.*?([1-5])',
            r'information quality.*?([1-5])',
        ]
        
        for pattern in dimension_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 5:
                    dimension_scores.append(score)
        
        if len(dimension_scores) == 5:
            return dimension_scores
        
        raise ValueError(f"Could not extract valid scores from response: {response_text[:200]}...")
    
    async def score_single_async(self, session: aiohttp.ClientSession, instruction: str, completion: str, index: int) -> Dict:
        """Score a single instruction-completion pair asynchronously."""
        user_prompt = self.user_prompt_template.format(
            instruction=instruction,
            completion=completion
        )
        
        payload = {
            "model": "google/gemma-3-27b-it",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            async with session.post(self.api_url, json=payload, timeout=60) as response:
                response.raise_for_status()
                result = await response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                
                scores_list = self._extract_scores(response_text)
                scores = QualityScores(
                    instruction_alignment_completeness=scores_list[0],
                    response_coherence=scores_list[1],
                    danish_language_appropriateness=scores_list[2],
                    assistant_persona=scores_list[3],
                    information_quality=scores_list[4]
                )
                
                return {
                    'index': index,
                    'instruction': instruction,
                    'completion': completion,
                    'scores': scores.to_dict(),
                    'reasoning': response_text
                }
                
        except Exception as e:
            print(f"Error scoring example {index}: {e}")
            scores = QualityScores(
                instruction_alignment_completeness=3,
                response_coherence=3,
                danish_language_appropriateness=3,
                assistant_persona=3,
                information_quality=3
            )
            return {
                'index': index,
                'instruction': instruction,
                'completion': completion,
                'scores': scores.to_dict(),
                'reasoning': f"Error during scoring: {str(e)}"
            }
    
    async def score_dataset_async(self, data_path: str, output_path: str, limit: int = None, max_concurrent: int = 20, filter_config: Dict = None):
        """Score dataset asynchronously with limited concurrency."""
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if limit:
                lines = lines[:limit]
        
        # Parse examples
        examples = []
        for i, line in enumerate(lines):
            try:
                data = json.loads(line.strip())
                
                if 'messages' in data:
                    messages = data['messages']
                    instruction = ''
                    completion = ''
                    
                    for msg in messages:
                        if msg.get('role') == 'user':
                            instruction = msg.get('content', '')
                        elif msg.get('role') == 'assistant':
                            completion = msg.get('content', '')
                else:
                    instruction = data.get('prompt', '')
                    completion = data.get('completion', '')
                
                examples.append((i, instruction, completion))
            except json.JSONDecodeError:
                print(f"Error parsing line {i}")
                continue
        
        # Score asynchronously with limited concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def score_with_limit(session, index, instruction, completion):
            async with semaphore:
                return await self.score_single_async(session, instruction, completion, index)
        
        # Open output file for incremental writing
        results = []
        
        async with aiohttp.ClientSession() as session:
            tasks = [score_with_limit(session, idx, inst, comp) for idx, inst, comp in examples]
            
            # Process with progress bar and save incrementally
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('[\n')  # Start JSON array
                first_result = True
                
                for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scoring examples"):
                    result = await task
                    results.append(result)
                    
                    # Write result immediately
                    if not first_result:
                        f.write(',\n')
                    json.dump(result, f, ensure_ascii=False, indent=2)
                    f.flush()  # Ensure it's written to disk
                    first_result = False
                
                f.write('\n]')  # Close JSON array
        
        # Sort results by index for summary (file already saved)
        results.sort(key=lambda x: x['index'])
        
        # Apply filtering if configured
        if filter_config:
            filtered_results, filtered_out = self._apply_filters(results, filter_config)
            
            # Save filtered dataset
            if 'filtered_output_path' in filter_config:
                self._save_filtered_dataset(filtered_results, filter_config['filtered_output_path'])
            
            self._print_filtering_summary(results, filtered_results, filtered_out, filter_config)
        else:
            self._print_summary(results)
        
        return results
    
    def score_dataset(self, data_path: str, output_path: str, limit: int = None, max_concurrent: int = 20, filter_config: Dict = None):
        """Score an entire dataset and save results (wrapper for async method)."""
        return asyncio.run(self.score_dataset_async(data_path, output_path, limit, max_concurrent, filter_config))
    
    def _print_summary(self, results: List[Dict]):
        """Print summary statistics of the scoring."""
        if not results:
            print("No results to summarize")
            return
        
        # Calculate averages for each dimension
        dimensions = ['instruction_alignment_completeness', 'response_coherence', 
                     'danish_language_appropriateness', 'assistant_persona', 
                     'information_quality', 'total_score']
        
        print("\n=== Scoring Summary ===")
        print(f"Total examples scored: {len(results)}")
        print("\nAverage scores by dimension:")
        
        for dim in dimensions:
            scores = [r['scores'][dim] for r in results]
            avg = sum(scores) / len(scores)
            print(f"  {dim}: {avg:.2f}")
        
        # Find problematic examples
        print("\nExamples with low total scores (<2.5):")
        low_scoring = [r for r in results if r['scores']['total_score'] < 2.5]
        for r in low_scoring[:5]:  # Show first 5
            print(f"  Index {r['index']}: {r['scores']['total_score']:.1f}")
            print(f"    Instruction: {r['instruction'][:100]}...")
            print(f"    Scores: {[r['scores'][d] for d in dimensions[:-1]]}")
    
    def _apply_filters(self, results: List[Dict], filter_config: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Apply quality filters and return (passed, filtered_out) results."""
        passed = []
        filtered_out = []
        
        for result in results:
            scores = result['scores']
            should_keep = True
            filter_reasons = []
            
            # Check minimum total score
            if 'min_total_score' in filter_config:
                if scores['total_score'] < filter_config['min_total_score']:
                    should_keep = False
                    filter_reasons.append(f"total_score={scores['total_score']:.2f} < {filter_config['min_total_score']}")
            
            # Check individual dimension thresholds
            if 'dimension_thresholds' in filter_config:
                for dim, min_score in filter_config['dimension_thresholds'].items():
                    if dim in scores and scores[dim] < min_score:
                        should_keep = False
                        filter_reasons.append(f"{dim}={scores[dim]} < {min_score}")
            
            # Check response length if configured
            if 'min_response_length' in filter_config:
                if len(result['completion']) < filter_config['min_response_length']:
                    should_keep = False
                    filter_reasons.append(f"response_length={len(result['completion'])} < {filter_config['min_response_length']}")
            
            if should_keep:
                passed.append(result)
            else:
                result['filter_reasons'] = filter_reasons
                filtered_out.append(result)
        
        return passed, filtered_out
    
    def _save_filtered_dataset(self, filtered_results: List[Dict], output_path: str):
        """Save filtered dataset in training format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in filtered_results:
                # Convert back to training format
                training_example = {
                    "messages": [
                        {"role": "user", "content": result['instruction']},
                        {"role": "assistant", "content": result['completion']}
                    ]
                }
                f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
    
    def _print_filtering_summary(self, original_results: List[Dict], filtered_results: List[Dict], 
                                filtered_out: List[Dict], filter_config: Dict):
        """Print summary of filtering results."""
        print("\n=== Filtering Summary ===")
        print(f"Original examples: {len(original_results)}")
        print(f"Passed filters: {len(filtered_results)} ({len(filtered_results)/len(original_results)*100:.1f}%)")
        print(f"Filtered out: {len(filtered_out)} ({len(filtered_out)/len(original_results)*100:.1f}%)")
        
        # Print filter configuration
        print("\nFilter Configuration:")
        if 'min_total_score' in filter_config:
            print(f"  Minimum total score: {filter_config['min_total_score']}")
        if 'dimension_thresholds' in filter_config:
            print("  Dimension thresholds:")
            for dim, threshold in filter_config['dimension_thresholds'].items():
                print(f"    {dim}: >= {threshold}")
        if 'min_response_length' in filter_config:
            print(f"  Minimum response length: {filter_config['min_response_length']} characters")
        
        # Show reasons for filtering
        if filtered_out:
            print("\nTop filtering reasons:")
            reason_counts = {}
            for result in filtered_out:
                for reason in result.get('filter_reasons', []):
                    reason_type = reason.split('=')[0]
                    reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
            
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count} examples ({count/len(filtered_out)*100:.1f}% of filtered)")
        
        # Print quality stats for filtered dataset
        print("\nQuality stats for filtered dataset:")
        dimensions = ['instruction_alignment_completeness', 'response_coherence', 
                     'danish_language_appropriateness', 'assistant_persona', 
                     'information_quality', 'total_score']
        
        for dim in dimensions:
            scores = [r['scores'][dim] for r in filtered_results]
            if scores:
                avg = sum(scores) / len(scores)
                print(f"  {dim}: {avg:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Score instruction-tuning data quality")
    parser.add_argument("input_file", help="Path to JSONL file with training data")
    parser.add_argument("output_file", help="Path to save scoring results")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions",
                       help="API URL for the scoring model")
    parser.add_argument("--limit", type=int, help="Limit number of examples to score")
    parser.add_argument("--max-concurrent", type=int, default=1000,
                       help="Maximum number of concurrent requests (default: 1000)")
    
    # Filtering options
    parser.add_argument("--filter", action="store_true", help="Apply quality filters")
    parser.add_argument("--min-total-score", type=float, default=3.5,
                       help="Minimum total score to pass filter (default: 3.5)")
    parser.add_argument("--min-instruction-alignment", type=int, default=3,
                       help="Minimum instruction alignment score (default: 3)")
    parser.add_argument("--min-information-quality", type=int, default=3,
                       help="Minimum information quality score (default: 3)")
    parser.add_argument("--min-assistant-persona", type=int, default=3,
                       help="Minimum assistant persona score (default: 3)")
    parser.add_argument("--min-response-length", type=int, default=20,
                       help="Minimum response length in characters (default: 20)")
    parser.add_argument("--filtered-output", type=str,
                       help="Path to save filtered training data (JSONL format)")
    
    args = parser.parse_args()
    
    # Build filter configuration
    filter_config = None
    if args.filter:
        filter_config = {
            'min_total_score': args.min_total_score,
            'dimension_thresholds': {
                'instruction_alignment_completeness': args.min_instruction_alignment,
                'information_quality': args.min_information_quality,
                'assistant_persona': args.min_assistant_persona
            },
            'min_response_length': args.min_response_length
        }
        
        if args.filtered_output:
            filter_config['filtered_output_path'] = args.filtered_output
    
    scorer = QualityScorer(api_url=args.api_url)
    scorer.score_dataset(args.input_file, args.output_file, limit=args.limit, 
                        max_concurrent=args.max_concurrent, filter_config=filter_config)


if __name__ == "__main__":
    main()