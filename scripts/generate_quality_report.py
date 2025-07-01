#!/usr/bin/env python3
"""Generate comprehensive quality report from scoring results."""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import argparse


def load_scored_data(file_path):
    """Load scored data and return as list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_score_statistics(scored_data):
    """Calculate statistics for each scoring dimension and overall."""
    dimensions = [
        'instruction_alignment_completeness',
        'response_coherence', 
        'danish_language_appropriateness',
        'assistant_persona',
        'information_quality'
    ]
    
    stats = {}
    
    # Calculate total scores
    total_scores = []
    for item in scored_data:
        total = sum(item['scores'][dim] for dim in dimensions)
        total_scores.append(total)
    
    stats['total'] = {
        'mean': statistics.mean(total_scores),
        'median': statistics.median(total_scores),
        'stdev': statistics.stdev(total_scores) if len(total_scores) > 1 else 0,
        'min': min(total_scores),
        'max': max(total_scores)
    }
    
    # Calculate per-dimension statistics
    for dim in dimensions:
        scores = [item['scores'][dim] for item in scored_data]
        stats[dim] = {
            'mean': statistics.mean(scores),
            'median': statistics.median(scores),
            'stdev': statistics.stdev(scores) if len(scores) > 1 else 0,
            'min': min(scores),
            'max': max(scores),
            'distribution': dict(Counter(scores))
        }
    
    return stats


def identify_quality_issues(scored_data):
    """Identify common quality issues based on scores."""
    issues = defaultdict(list)
    
    for i, item in enumerate(scored_data):
        scores = item['scores']
        total_score = sum(scores.values())
        
        # Low overall quality
        if total_score < 15:  # Less than 3 average
            issues['low_overall_quality'].append({
                'index': i,
                'total_score': total_score,
                'instruction': item['instruction'][:100] + '...' if len(item['instruction']) > 100 else item['instruction']
            })
        
        # Specific dimension issues
        if scores['instruction_alignment_completeness'] <= 2:
            issues['poor_instruction_alignment'].append({
                'index': i,
                'score': scores['instruction_alignment_completeness'],
                'instruction': item['instruction'][:100] + '...' if len(item['instruction']) > 100 else item['instruction']
            })
        
        if scores['response_coherence'] <= 2:
            issues['poor_coherence'].append({
                'index': i,
                'score': scores['response_coherence'],
                'completion': item['completion'][:100] + '...' if len(item['completion']) > 100 else item['completion']
            })
        
        if scores['danish_language_appropriateness'] <= 2:
            issues['poor_danish'].append({
                'index': i,
                'score': scores['danish_language_appropriateness'],
                'instruction': item['instruction'][:100] + '...' if len(item['instruction']) > 100 else item['instruction']
            })
        
        if scores['assistant_persona'] <= 2:
            issues['persona_issues'].append({
                'index': i,
                'score': scores['assistant_persona'],
                'completion': item['completion'][:100] + '...' if len(item['completion']) > 100 else item['completion']
            })
        
        if scores['information_quality'] <= 2:
            issues['poor_information_quality'].append({
                'index': i,
                'score': scores['information_quality'],
                'completion': item['completion'][:100] + '...' if len(item['completion']) > 100 else item['completion']
            })
    
    return issues


def analyze_response_lengths(scored_data):
    """Analyze response length patterns."""
    lengths = [len(item['completion']) for item in scored_data]
    
    length_stats = {
        'mean': statistics.mean(lengths),
        'median': statistics.median(lengths),
        'stdev': statistics.stdev(lengths) if len(lengths) > 1 else 0,
        'min': min(lengths),
        'max': max(lengths)
    }
    
    # Length distribution
    length_buckets = {
        'very_short': 0,  # < 50 chars
        'short': 0,       # 50-100 chars
        'medium': 0,      # 100-200 chars
        'long': 0,        # 200-500 chars
        'very_long': 0    # > 500 chars
    }
    
    for length in lengths:
        if length < 50:
            length_buckets['very_short'] += 1
        elif length < 100:
            length_buckets['short'] += 1
        elif length < 200:
            length_buckets['medium'] += 1
        elif length < 500:
            length_buckets['long'] += 1
        else:
            length_buckets['very_long'] += 1
    
    return length_stats, length_buckets


def find_best_and_worst_examples(scored_data, n=5):
    """Find best and worst examples based on total scores."""
    # Sort by total score
    sorted_data = sorted(scored_data, key=lambda x: sum(x['scores'].values()))
    
    worst_examples = sorted_data[:n]
    best_examples = sorted_data[-n:]
    
    return best_examples, worst_examples


def generate_report(scored_data_path, output_path=None):
    """Generate comprehensive quality report."""
    print(f"Loading scored data from {scored_data_path}...")
    scored_data = load_scored_data(scored_data_path)
    
    print(f"Analyzing {len(scored_data)} scored examples...")
    
    # Calculate statistics
    stats = calculate_score_statistics(scored_data)
    
    # Identify issues
    issues = identify_quality_issues(scored_data)
    
    # Analyze lengths
    length_stats, length_buckets = analyze_response_lengths(scored_data)
    
    # Find examples
    best_examples, worst_examples = find_best_and_worst_examples(scored_data)
    
    # Generate report
    report = {
        'summary': {
            'total_examples': len(scored_data),
            'overall_quality': {
                'mean_total_score': stats['total']['mean'],
                'median_total_score': stats['total']['median'],
                'score_range': f"{stats['total']['min']}-{stats['total']['max']}"
            }
        },
        'dimension_statistics': {
            dim: {
                'mean': stats[dim]['mean'],
                'distribution': stats[dim]['distribution']
            }
            for dim in ['instruction_alignment_completeness', 'response_coherence', 
                       'danish_language_appropriateness', 'assistant_persona', 'information_quality']
        },
        'quality_issues': {
            issue_type: {
                'count': len(examples),
                'percentage': len(examples) / len(scored_data) * 100,
                'examples': examples[:3]  # Show first 3 examples
            }
            for issue_type, examples in issues.items()
        },
        'response_length_analysis': {
            'statistics': length_stats,
            'distribution': length_buckets,
            'distribution_percentages': {
                k: v / len(scored_data) * 100 for k, v in length_buckets.items()
            }
        },
        'best_examples': [
            {
                'total_score': sum(ex['scores'].values()),
                'instruction': ex['instruction'],
                'completion': ex['completion'],
                'scores': ex['scores']
            }
            for ex in best_examples
        ],
        'worst_examples': [
            {
                'total_score': sum(ex['scores'].values()),
                'instruction': ex['instruction'],
                'completion': ex['completion'],
                'scores': ex['scores']
            }
            for ex in worst_examples
        ]
    }
    
    # Print summary to console
    print("\n" + "="*60)
    print("QUALITY REPORT SUMMARY")
    print("="*60)
    
    print(f"\nTotal Examples: {report['summary']['total_examples']:,}")
    print(f"Mean Total Score: {report['summary']['overall_quality']['mean_total_score']:.2f}/25")
    print(f"Median Total Score: {report['summary']['overall_quality']['median_total_score']:.2f}/25")
    
    print("\nDimension Averages:")
    for dim, data in report['dimension_statistics'].items():
        print(f"  {dim}: {data['mean']:.2f}/5")
    
    print("\nResponse Length Distribution:")
    for category, percentage in report['response_length_analysis']['distribution_percentages'].items():
        print(f"  {category}: {percentage:.1f}%")
    
    print("\nQuality Issues:")
    for issue_type, data in report['quality_issues'].items():
        if data['count'] > 0:
            print(f"  {issue_type}: {data['count']} examples ({data['percentage']:.1f}%)")
    
    # Save full report
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nFull report saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Generate quality report from scored data')
    parser.add_argument('--input', '-i', required=True, help='Path to scored data JSON file')
    parser.add_argument('--output', '-o', help='Path to save report JSON file')
    
    args = parser.parse_args()
    
    generate_report(args.input, args.output)


if __name__ == '__main__':
    main()