#!/usr/bin/env python3
"""
Apply quality filters to already-scored data.
"""

import json
import argparse
from typing import Dict, List, Tuple


def apply_filters(results: List[Dict], filter_config: Dict) -> Tuple[List[Dict], List[Dict]]:
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


def save_filtered_dataset(filtered_results: List[Dict], output_path: str, format_type: str = "torchtune"):
    """Save filtered dataset in specified format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in filtered_results:
            # Strip leading/trailing whitespace from completion
            completion = result['completion'].strip()
            
            if format_type == "torchtune":
                # Torchtune format with input/output
                training_example = {
                    "input": result['instruction'],
                    "output": completion
                }
            elif format_type == "messages":
                # Messages format
                training_example = {
                    "messages": [
                        {"role": "user", "content": result['instruction']},
                        {"role": "assistant", "content": completion}
                    ]
                }
            else:  # original format
                training_example = {
                    "prompt": result['instruction'],
                    "completion": completion
                }
            
            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')


def print_filtering_summary(original_results: List[Dict], filtered_results: List[Dict], 
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
    parser = argparse.ArgumentParser(description="Apply filters to already-scored data")
    parser.add_argument("scores_file", help="Path to JSON file with scoring results")
    parser.add_argument("output_file", help="Path to save filtered training data")
    parser.add_argument("--format", choices=["original", "messages", "torchtune"], 
                       default="original",
                       help="Output format (default: original)")
    
    # Filtering options
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
    
    args = parser.parse_args()
    
    # Load scored data
    print(f"Loading scores from {args.scores_file}...")
    with open(args.scores_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Build filter configuration
    filter_config = {
        'min_total_score': args.min_total_score,
        'dimension_thresholds': {
            'instruction_alignment_completeness': args.min_instruction_alignment,
            'information_quality': args.min_information_quality,
            'assistant_persona': args.min_assistant_persona
        },
        'min_response_length': args.min_response_length
    }
    
    # Apply filters
    filtered_results, filtered_out = apply_filters(results, filter_config)
    
    # Save filtered dataset
    save_filtered_dataset(filtered_results, args.output_file, args.format)
    print(f"\nFiltered dataset saved to: {args.output_file}")
    
    # Print summary
    print_filtering_summary(results, filtered_results, filtered_out, filter_config)


if __name__ == "__main__":
    main()