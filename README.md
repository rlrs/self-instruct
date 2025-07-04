# Self-Instruct for Danish

This is an implementation of the Self-Instruct methodology for generating Danish instruction-following data. The implementation is based on the paper "SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions" but adapted for Danish language and local LLM inference.

## Overview

Self-Instruct is a framework for improving language models' instruction-following capabilities by bootstrapping off their own generations. This implementation:

- Generates Danish instruction data from a seed set
- Uses local model inference servers (vLLM, SGLang, etc.) instead of OpenAI API
- Leverages server-side continuous batching - sends concurrent requests and lets the server optimize throughput
- Implements all core Self-Instruct components: instruction generation, task classification, instance generation, and data filtering
- Produces training data ready for fine-tuning

## Installation

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```

## Quick Start

1. **Extract seed data from Excel:**
```bash
python scripts/extract_seed_data.py
```

2. **Start model serving server** (in a separate terminal):

For data generation (requires a **base pre-trained model**):
```bash
python -m sglang.launch_server --model google/gemma-3-27b-pt --context-length 4096 --port 8000
```

Example base models:
- `google/gemma-3-27b-pt` (base pre-trained)
- `meta-llama/Llama-3.1-70B` (base)
- `mistralai/Mistral-7B-v0.1` (base)

**Note:** The main pipeline requires a base (pre-trained) model, NOT an instruction-tuned model. Instruction-tuned models may produce lower quality synthetic data due to their existing instruction-following biases.

3. **Run the pipeline:**
```bash
python scripts/run_pipeline.py
```

## Pipeline Components

### 1. Instruction Generation (`src/generate_instructions.py`)
- Generates new task instructions based on seed examples
- Uses ROUGE-L similarity to ensure diversity
- Filters out low-quality instructions

### 2. Task Classification (`src/classify_tasks.py`)
- Classifies tasks as classification or non-classification
- Important for determining instance generation strategy

### 3. Instance Generation (`src/generate_instances.py`)
- Generates input-output examples for each instruction
- Uses different strategies:
  - **Input-first**: For non-classification tasks
  - **Output-first**: For classification tasks (prevents label bias)

### 4. Data Processing (`src/prepare_training_data.py`)
- Filters and deduplicates instances
- Formats data for fine-tuning with multiple prompt templates

## Quality Scoring and Filtering

After generating training data, you can optionally score and filter it based on quality metrics:

### 1. Score the generated data:

First, start a server with an **instruction-tuned model** for quality evaluation:
```bash
python -m sglang.launch_server --model google/gemma-3-27b-it --context-length 4096 --port 8000
```

Then run the scoring:
```bash
python scripts/score_quality.py data/generated/training_data/train.jsonl data/quality_scores.json
```

This evaluates each example on 5 dimensions:
- **Instruction Alignment & Completeness**: How well the response follows the instruction
- **Response Coherence**: Logical structure and clarity
- **Danish Language Appropriateness**: Proper Danish usage and cultural context
- **Assistant Persona**: Maintains appropriate assistant behavior (no fake personal experiences)
- **Information Quality**: Accuracy and helpfulness of the response

### 2. Filter based on quality scores:

```bash
python scripts/filter_scored_data.py data/quality_scores.json data/filtered_train.jsonl \
  --format torchtune \
  --min-total-score 3.5 \
  --min-instruction-alignment 3 \
  --min-information-quality 3 \
  --min-assistant-persona 3
```

Available output formats:
- `--format original`: Keep prompt/completion format
- `--format messages`: Convert to chat messages format
- `--format torchtune`: Convert to torchtune's input/output format

## Configuration

Edit `configs/default.yaml` to customize:

- Model server URL and settings
- Generation parameters (temperature, max_tokens, etc.)
- **Batch size for concurrent requests** (default: 200, adjust based on your server capacity)
- Number of instructions/instances to generate
- Filtering thresholds
- Output paths

### Performance Tuning

The pipeline leverages server-side continuous batching:

- Modern inference servers (vLLM, SGLang) automatically batch concurrent requests for optimal GPU utilization
- The pipeline sends multiple requests concurrently and lets the server handle scheduling
- No manual batching needed - the server optimizes based on available GPU memory

For best performance:
- Ensure your inference server has sufficient GPU memory
- Monitor server logs to check utilization
- Adjust the number of concurrent requests in the config if needed

## Running Multiple Iterations

To generate more data through multiple iterations:

```bash
python scripts/run_pipeline.py --iterations 3
```

## Output Structure

```
data/generated/
  iteration_1/
    generated_instructions.jsonl
    classified_tasks.jsonl
    tasks_with_instances.jsonl
  iteration_2/
    ...
  training_data/
    train.jsonl          # All training examples  
    statistics.json      # Generation statistics
  pipeline_metadata.json
```

## Example Usage

### Generate instructions only:
```bash
python -m src.generate_instructions --output-dir custom_output
```

### Classify tasks:
```bash
python -m src.classify_tasks --input tasks.jsonl --output classified.jsonl
```

### Generate instances:
```bash
python -m src.generate_instances --input classified.jsonl --output instances.jsonl
```

### Prepare training data:
```bash
python -m src.prepare_training_data --input file1.jsonl file2.jsonl --output-dir training
```

## Notes

- The pipeline expects an OpenAI-compatible model server running at `http://localhost:8000` (configurable)
- All prompts and examples are in Danish
- The seed data comes from `alpaca-seed-eval.xlsx` with Danish translations
- Generation quality depends on the underlying model's Danish capabilities
- Use base (pre-trained) models for data generation, instruction-tuned models for quality scoring

## Citation

If you use this implementation, please cite the original Self-Instruct paper:

```bibtex
@article{wang2022self,
  title={Self-Instruct: Aligning Language Models with Self-Generated Instructions},
  author={Wang, Yizhong and Kordi, Yeganeh and Mishra, Swaroop and Liu, Alisa and Smith, Noah A. and Khashabi, Daniel and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2212.10560},
  year={2022}
}
```