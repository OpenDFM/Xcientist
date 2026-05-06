# Code Usage Guide

## Quick Start

### Activate Environment
```bash
source project/venv/bin/activate
```

### Run Full Method Experiment
```bash
cd project/
python run_experiment.py \
  --data data/locomo10.json \
  --embedding-model ../model_candidate/model_share/all-MiniLM-L6-v2/ \
  --condition full \
  --output ../results/full_results.json
```

### Run Baseline Experiment
```bash
cd project/
python run_experiment.py \
  --data data/locomo10.json \
  --embedding-model ../model_candidate/model_share/all-MiniLM-L6-v2/ \
  --condition baseline \
  --output ../results/baseline_results.json
```

### Run Ablation Experiments
```bash
cd project/

# Disable note enricher
python run_experiment.py --data data/locomo10.json \
  --embedding-model ../model_candidate/model_share/all-MiniLM-L6-v2/ \
  --condition ablation:modular_atomic_note_enricher \
  --output ../results/ablation_enricher.json

# Disable handle index
python run_experiment.py --data data/locomo10.json \
  --embedding-model ../model_candidate/model_share/all-MiniLM-L6-v2/ \
  --condition ablation:ultra_sparse_facet_handle_index \
  --output ../results/ablation_handles.json

# Disable slotted reranker
python run_experiment.py --data data/locomo10.json \
  --embedding-model ../model_candidate/model_share/all-MiniLM-L6-v2/ \
  --condition ablation:sloted_evidence_reranker \
  --output ../results/ablation_reranker.json

# Disable adjudicator
python run_experiment.py --data data/locomo10.json \
  --embedding-model ../model_candidate/model_share/all-MiniLM-L6-v2/ \
  --condition ablation:dedup_minority_aware_provenance_adjudicator \
  --output ../results/ablation_adjudicator.json
```

### Limit Samples for Testing
```bash
cd project/
python run_experiment.py \
  --data data/locomo10.json \
  --embedding-model ../model_candidate/model_share/all-MiniLM-L6-v2/ \
  --condition full \
  --num-samples 3 \
  --max-questions 10 \
  --output ../results/test_run.json
```

## Required Environment Variables
- `OPENAI_API_KEY` — OpenAI API key for gpt-4o-mini
- `OPENAI_BASE_URL` — OpenAI API base URL

## CLI Arguments
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data` | Yes | — | Path to locomo10.json |
| `--embedding-model` | Yes | — | Path to sentence-transformers model |
| `--condition` | Yes | — | `baseline`, `full`, or `ablation:<component>` |
| `--num-samples` | No | all | Number of LoCoMo samples to process |
| `--max-questions` | No | all | Max QA questions per sample |
| `--output` | Yes | — | Output JSON file path |
| `--model` | No | gpt-4o-mini | LLM model name |

## Output Format
The output JSON contains:
- `condition`: the experiment condition name
- `predictions`: per-sample QA predictions with generated answers and F1 scores
- `metrics`: aggregate metrics including overall F1 and per-category breakdown
- `component_config`: which components were enabled/disabled
- `component_status`: component enable/disable status

## Python API Usage
```python
from memory.system import SlottedMemorySystem
from llm_utils import generate_answer
from eval.evaluation import eval_question_answering
import json

# Create memory system
system = SlottedMemorySystem(
    embedding_model_path='model_candidate/model_share/all-MiniLM-L6-v2/',
    component_config={
        'modular_atomic_note_enricher': True,
        'ultra_sparse_facet_handle_index': True,
        'sloted_evidence_reranker': True,
        'dedup_minority_aware_provenance_adjudicator': True
    }
)

# Load and ingest data
data = json.load(open('project/data/locomo10.json'))
system.ingest([data[0]])

# Query for evidence
result = system.query('What did the speaker talk about?')
evidence_text = system.get_evidence_text(result)

# Generate answer
answer = generate_answer('What did the speaker talk about?', evidence_text)
print(answer)
```