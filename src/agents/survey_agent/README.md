# Deep Survey

Deep Survey is an automated academic survey generation pipeline. Given a research topic, it collects related papers, expands the paper set through references and citations, reads and analyzes papers with LLMs, clusters related work, optionally builds relation/code analysis artifacts, generates a survey draft, revises it, refines citations, and optionally evaluates the final survey.

## Repository layout

```text
.
├── Dockerfile                    # Container environment for Deep Survey and MinerU
├── config/                       # Hydra/YAML runtime configurations
│   ├── evaluation/               # Evaluation experiment configs
│   └── *.yaml                    # Default and task-specific configs
├── scripts/                      # Main runnable entry points
├── modules/                      # Core pipeline modules
├── utils/                        # API, logging, file, PDF, markdown and helper utilities
├── topics/                       # Topic lists and benchmark topic definitions
├── baselines/                    # Baseline systems and baseline outputs
├── tests/                        # Development tests and scratch files
├── logs/                         # Runtime logs, ignored by git
├── outputs/                      # Generated surveys and artifacts, ignored by git
└── database/                     # Local paper/cache/database artifacts, ignored by git
```

`logs/`, `outputs/`, and `database/` are generated/runtime directories and are ignored by git. The `logs/` directory can usually be ignored when reading the codebase.

## Pipeline overview

The main Deep Survey pipeline follows these stages:

1. Related work collection
   - Searches seed papers for the target topic.
   - Expands papers through references and citations.
   - Can use local graph/cache data when configured.
2. Paper database construction
   - Builds an embedding/database view over collected papers.
3. Paper comprehension
   - Reads papers or abstracts.
   - Writes reusable paper keynotes into cache.
4. Clustering and analysis
   - Clusters related papers.
   - Optionally performs intra-cluster and inter-cluster analysis.
   - Optionally generates relation graphs and relation tables.
5. Optional code/environment report generation
   - Collects and analyzes code repositories linked to papers.
   - Generates code and environment reports when enabled.
6. Survey generation
   - Generates an outline.
   - Drafts sections/subsections.
   - Reviews and revises the draft.
   - Refines final paper text and references.
7. Optional evaluation
   - Evaluates generated survey quality and citation quality when enabled.

## Core modules

- `modules/work_collector.py`: collects seed papers and expands them via references/citations.
- `modules/database.py`: builds retrieval databases over collected papers.
- `modules/work_analyzer.py`: reads papers, extracts keynotes, clusters papers, and performs relation/cluster analysis.
- `modules/survey_generator.py`: generates outlines, drafts survey sections, reviews/revises drafts, and saves final survey outputs.
- `modules/judge.py`: evaluates survey outputs when evaluation is enabled.
- `modules/code_collector.py`: collects code repository information.
- `modules/code_report_generator.py`: generates code/environment reports from collected repositories.
- `modules/paper_graph_retriever.py`: retrieves papers from local graph/database resources.

## Important scripts

### Single-topic Hydra entry point

```bash
python3 scripts/run_deep_survey.py
```

This uses Hydra with the default config declared in the script:

```text
config/deep_survey_fast.yaml
```

You can override Hydra config values from the command line, for example:

```bash
python3 scripts/run_deep_survey.py BasicInfo.topic="Graph Neural Networks"
```

### Batch entry point with explicit YAML config

```bash
python3 scripts/run_deep_survey_batch_arg.py --config ./config/personal/deep_survey_batch_0514.yaml
```

This is the most explicit batch runner. It reads a YAML file, loads topics according to `BasicInfo`, and runs the pipeline for each topic. For example, `config/personal/deep_survey_batch_0514.yaml` uses:

- `BasicInfo.user_defiend_benchmarks: True`
- `BasicInfo.topic_path: ./topics/human_compare_topics.txt`
- `BasicInfo.output_base_dir: ./outputs/0514_mimo_pro_human_test`
- `BasicInfo.skip_exist: True`

### Adapter entry point

```bash
python3 scripts/run_deep_survey_adapter.py --workspace /workspace --config /workspace/config/runtime.yaml
```

The adapter is designed for external orchestration. It writes standardized artifacts such as:

```text
/workspace/survey/output/survey.md
/workspace/survey/output/survey.json
/workspace/survey/output/evaluation.txt
/workspace/logs/relation_graph.json
/workspace/logs/relation_table.json
/workspace/logs/clustering_result.json
/workspace/logs/draft.json
/workspace/logs/deep_survey.log
```

### Baseline evaluation

```bash
python3 scripts/baseline_evaluation.py
```

This script evaluates outputs from Deep Survey and baseline systems such as AutoSurvey, SurveyForge, LiRA, and human-written surveys according to the configured benchmark/evaluation settings.

### Other utility scripts

- `scripts/run_keynotes_gen.py`: pre-generate paper keynotes.
- `scripts/run_work_clusters.py`: run work clustering utilities.
- `scripts/memory_monitor.sh`: monitor memory usage during long-running jobs.
- `scripts/run_model_ablation.sh`: run model ablation experiments.

## Configuration

Configurations live under `config/`. The common top-level sections are:

```yaml
BasicInfo:
  topic: ""
  topic_path: ./topics/human_compare_topics.txt
  output_base_dir: ./outputs/example
  cache_path: ./database
  topic_max_retry: 5
  skip_exist: True

APIInfo:
  llm_api_key: "..."
  llm_api_base_url: "..."
  llm_model_name: "..."
  batch_chat_agent_worker: 2
  chat_timeout: 3600
  batch_chat_timeout: 3600

ModuleInfo:
  WorkCollector:
    max_seed_paper_num: 15
    reference_graph_depth: 1
  WorkAnalyzer:
    abstract_only_mode: False
    paper_reading_max_retry: 6
  SurveyGenerator:
    include_initial_analysis: True
    include_relation_graph: False
    include_relation_table: False
    include_code_report: False
    enable_review_and_revise: True
  Judge:
    skip_evaluation: True
```

Before running experiments, check at least these fields:

- `BasicInfo.topic` or `BasicInfo.topic_path`
- `BasicInfo.output_base_dir`
- `BasicInfo.cache_path`
- `APIInfo.llm_api_key`
- `APIInfo.llm_api_base_url`
- `APIInfo.llm_model_name`
- `APIInfo.batch_chat_agent_worker`
- `ModuleInfo.Judge.skip_evaluation`

Avoid committing real API keys. Prefer using environment variables or local-only config files for secrets.

## Topics

Topic lists are stored in `topics/`. Batch runs can use:

```yaml
BasicInfo:
  user_defiend_benchmarks: True
  topic_path: ./topics/human_compare_topics.txt
```

Benchmark topic dictionaries are defined in:

```text
topics/Benchmark_topics.py
```

Relevant switches include:

- `BasicInfo.AutoSurvey_benchmark`
- `BasicInfo.SurveyGen_benchmark`
- `BasicInfo.sub_benchmark_test`
- `BasicInfo.user_defiend_benchmarks`

## Outputs

Generated files are typically written under `BasicInfo.output_base_dir`. A batch run stores per-domain/per-topic results and may produce:

- Markdown survey files
- JSON survey files with paper text and references
- Evaluation result files
- Analysis artifacts such as clustering results, relation graphs, relation tables, draft files, code reports, and environment reports

The exact outputs depend on `ModuleInfo.SurveyGenerator` and `ModuleInfo.Judge` switches.

## Docker environment

The repository includes a Dockerfile at `Dockerfile`. It is the intended environment configuration for Deep Survey and PDF parsing utilities.

The image is based on:

```dockerfile
FROM vllm/vllm-openai:v0.10.1.1
```

It installs:

- Noto fonts and fontconfig for Chinese text rendering
- `libgl1` for OpenCV support
- `mineru[core]` for PDF parsing
- `hydra-core`, `asciinet`, `tenacity`, `trafilatura`, `readability-lxml`, `html2text`
- OpenJDK 25
- `rich`, `loguru`, `sentence-transformers`, `hdbscan`

It also sets:

```dockerfile
ENV HF_ENDPOINT=https://hf-mirror.com
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]
```

and downloads MinerU models during image build:

```bash
mineru-models-download -s huggingface -m all
```

Build example:

```bash
docker build -t deep-survey:latest .
```

Run example:

```bash
docker run --gpus all --rm -it \
  -v /hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey:/workspace/deep-survey \
  -w /workspace/deep-survey \
  deep-survey:latest \
  python3 scripts/run_deep_survey_batch_arg.py --config ./config/personal/deep_survey_batch_0514.yaml
```

If building fails around the final Python package installation block, check the trailing shell continuation after `pip install hdbscan`. The last command in a chained `RUN` block should not leave a dangling `&& \` before the next Docker instruction.

## PDF and MinerU utilities

PDF/markdown related utilities are under `utils/`, especially:

```text
utils/mineru_utils.py
utils/convert_to_md.py
utils/html_utils.py
```

`utils/mineru_utils.py` contains MinerU-based parsing helpers for converting PDFs into markdown-like outputs. The Dockerfile installs MinerU and pre-downloads its models so PDF parsing can run with local MinerU model sources.

## Recommended run patterns

### Foreground run

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_deep_survey_batch_arg.py \
  --config ./config/personal/deep_survey_batch_0514.yaml
```

### Background run with log capture

```bash
mkdir -p logs
nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python3 scripts/run_deep_survey_batch_arg.py --config ./config/personal/deep_survey_batch_0514.yaml' \
  > ./logs/deep_survey_batch.log 2>&1 &
```

### Memory monitoring

```bash
bash scripts/memory_monitor.sh
```

For long experiments on remote/HPC machines, prefer using `tmux`, `screen`, or a job scheduler so the process is not interrupted by SSH session disconnection.

## Troubleshooting

### `Killed`

`Killed` can be caused by OOM, scheduler limits, or external signals. Check:

- system/job scheduler logs
- shell/session disconnection
- memory logs
- GPU/CPU memory usage
- custom signal traps if enabled

### `ProxyError` or API timeout

In this project, repeated `ProxyError` can also indicate upstream API timeout, connection reset, gateway failure, or too much concurrent traffic. Consider reducing:

```yaml
APIInfo:
  batch_chat_agent_worker: 1
  low_flow_mode: True
  low_flow_latency: 2
  exponential_backoff: True
```

### Hugging Face network issues

The Dockerfile sets:

```bash
HF_ENDPOINT=https://hf-mirror.com
```

If the environment cannot access Hugging Face, use a mirror, pre-download models, or configure offline cache paths.

### Logs directory

`logs/` is runtime-only and ignored by git. It is useful for debugging local runs but should not be treated as source code.

## Development notes

- The codebase is Python-based and uses Hydra/OmegaConf-style YAML configs.
- Most modules are initialized from the selected config, so behavior is usually controlled by YAML rather than command-line flags.
- Long-running batch jobs depend heavily on API stability, retry settings, and concurrency settings.
- Generated caches under `database/` can speed up later runs but are not source files.
