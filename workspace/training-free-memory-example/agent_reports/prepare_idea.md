# Training-Free Slotted Evidence Retrieval for Scalable LLM Agent Memory

## 1. Idea Summary
This work refines the span-grounded atomic-note memory paradigm for LLM agents by shifting the main contribution away from heavier corroboration or canonical-claim construction and toward lightweight note enrichment plus staged, capacity-capped retrieval. Each immutable atomic note remains the sole authoritative memory unit, but is augmented at write time with cached embeddings, normalized entity and time tags, QA-oriented keywords, a short context descriptor, provenance-family labels, and a sparse set of deterministic local links. At read time, retrieval follows a two-stage design: a bounded ANN-plus-lexical screener proposes candidates, then a deterministic slotted evidence reranker assembles a fixed-size bundle containing support, provenance, temporal, and conflict anchors. Exact normalized handles are kept intentionally sparse and are used only as optional soft boosts rather than hard filters, while a lightweight deduplication and minority-conflict pass prevents duplicate flooding without erasing useful contradictions. The result is a training-free, implementation-ready memory system designed for low latency, graceful degradation under noisy metadata, and improved QA-oriented retrieval as memory grows.

## 2. Idea JSON Components (exact order from idea.json.idea_contract.components)
1. **sloted_evidence_reranker** - Deterministic, training‑free read‑path reranker that builds a bounded candidate pool (ANN + lexical), applies fixed feature scoring (including optional handle boosts and facet‑confidence weighting), and assembles a fixed‑size, sloted evidence bundle (support/provenance/temporal/conflict) with tightly capped local expansion.
2. **modular_atomic_note_enricher** - Deterministic write-path support module that converts raw spans into bounded enriched notes containing only the retrieval facets needed for scalable QA, preventing silent metadata bloat and enabling clean ablation of each facet.
3. **ultra_sparse_facet_handle_index** - High-precision support index of note-attached exact handles for trivial entity, relation, and time normalizations. It improves screening without creating a separate canonical claim layer.
4. **dedup_minority_aware_provenance_adjudicator** - Guardrail module that downweights duplicate-source flooding, preserves bounded minority contradictions, and keeps the final evidence bundle compact and provenance-diverse.

## 3. Code Implementation Guidance
All code must be implemented in the `project/` directory. Do not import from `repos/` (which is reference only). The system must be runtime self-contained in `project/`. Use the provided virtual environment at `project/venv` (Python 3.12.13). Required packages are already installed (see Environment Variable Usage Guidance). The implementation should be training-free: embeddings are cached at write time, indices are compact, local links are sparse and deterministic, scoring uses fixed rules rather than learned weights, and all candidate and expansion sizes are explicitly capped.

### Key Implementation Points:
- **Atomic Notes**: Immutable span-grounded notes are the only memory authority.
- **Write-time Enrichment**: Deterministic, rule-based enrichment of notes with bounded retrieval schema (cached embedding, normalized entity/time tags with confidence, QA-oriented keywords, short context descriptor, provenance-family label, sparse local links).
- **Micro-handles**: Optional exact normalizations (canonical date forms, reliable entity aliases, exact relation phrases) attached to notes as soft retrieval seeds/boosts.
- **Read-time Retrieval**: Two-stage: 
  1. Bounded screener (ANN search over cached embeddings + lexical overlap + optional entity/time filters) → compact candidate pool.
  2. Deterministic slotted reranker (fixed feature scoring: semantic similarity, lexical overlap, optional handle boost, temporal compatibility weighted by facet confidence, provenance diversity preference, duplicate-family penalties) → fixed-capacity evidence bundle with explicit slots (support, provenance, temporal, conflict).
- **Degraded Mode**: When entity/time facets are missing or low-confidence, expand candidate pool slightly and shift weight to semantic/lexical evidence.
- **Deduplication**: Lightweight adjudication step to suppress repeated evidence from same provenance family while preserving bounded minority contradictions.

## 4. Component Correspondence
Each canonical idea component maps to a specific module in the implementation:
- `sloted_evidence_reranker` → The deterministic slotted evidence reranker module (read-path).
- `modular_atomic_note_enricher` → The deterministic note enrichment module (write-path).
- `ultra_sparse_facet_handle_index` → The index of exact handles (entity, relation, time) attached to notes.
- `dedup_minority_aware_provenance_adjudicator` → The lightweight deduplication and minority-conflict guardrail.

## 5. Dataset Usage Guidance
- **Primary Dataset**: Use only the local LoCoMo dataset at `repos/locomo-main/locomo-main/data/locomo10.json`.
- **No External Data**: Do not download or generate any other benchmark data (e.g., NaturalQuestions, HotpotQA, TimeQA, synthetic stress tests).
- **Evaluation**: Run evaluation directly inside the LoCoMo repository (`repos/locomo-main/locomo-main`) using the provided metric modules.
- **Metric Reuse**: Use `task_eval/evaluation.py::eval_question_answering` and `task_eval/evaluation_stats.py::analyze_aggr_acc` for scoring.

## 6. Environment Variable Usage Guidance
- **OPENAI_API_KEY**: Required for accessing the OpenAI API (used via litellm for the LLM generator gpt-4o-mini).
- **OPENAI_API_BASE**: Required for specifying the OpenAI API base URL (paired with OPENAI_API_KEY).
- **Virtual Environment**: Use the provided virtual environment at `project/venv` (Python 3.12.13). Activate with `source project/venv/bin/activate`.
- **Package Installation**: All required packages are already installed in the virtual environment (see `prepare_target_inventory.json` for the list).

## 7. Resource Acquisition Log
No external resource acquisition is required for the synthesis phase. All resources (repositories, datasets, models, environment) have been validated and are available locally.

## 8. Repository-to-Dataset Mapping
- **LoCoMo Repository** (`repos/locomo-main/locomo-main/`) → **LoCoMo Dataset** (`repos/locomo-main/locomo-main/data/locomo10.json`).
- **A-mem-main Repository** (`project/A-mem-main/A-mem-main/`) → Reference implementation only, not used for data.

## 9. Real Experiment Targets

### Verified Model Paths
- **Embedding Model**: `model_candidate/model_share/all-MiniLM-L6-v2/` (sentence-transformers, 384-dim, verified loadable via `SentenceTransformer('model_candidate/model_share/all-MiniLM-L6-v2')`)
- **LLM Generator**: `gpt-4o-mini` via litellm (API-only, requires `OPENAI_API_KEY` and `OPENAI_API_BASE`)

### Verified Dataset Paths
- **LoCoMo Dataset**: `repos/locomo-main/locomo-main/data/locomo10.json` (2.8MB, verified JSON)

### Benchmark Entrypoints
- **Evaluation QA**: `repos/locomo-main/locomo-main/task_eval/evaluation.py::eval_question_answering`
- **Evaluation Stats**: `repos/locomo-main/locomo-main/task_eval/evaluation_stats.py::analyze_aggr_acc`

### Run Commands
```bash
# Activate virtual environment
source project/venv/bin/activate

# Run the experiment (example - actual entry point to be implemented in project/)
cd project/
python run_experiment.py --data ../repos/locomo-main/locomo-main/data/locomo10.json \
  --embedding-model ../model_candidate/model_share/all-MiniLM-L6-v2/ \
  --output ../results/

# Run evaluation (from LoCoMo repo)
cd repos/locomo-main/locomo-main/
python task_eval/evaluate_qa.py --predictions ../../results/predictions.json
```

### Expected Output Locations
- Experiment results: `results/standard/` (standard runs), `results/ablation/` (ablation studies)
- Predictions: `results/predictions.json`
- Metrics: `results/metrics.json`

### Environment Requirements
- Python 3.12.13 (via `project/venv`)
- All packages pre-installed (see Environment Variable Usage Guidance)
- `OPENAI_API_KEY` and `OPENAI_API_BASE` must be set in environment

## 10. Canonical Idea Components
The canonical idea components are (in exact order):
1. sloted_evidence_reranker
2. modular_atomic_note_enricher
3. ultra_sparse_facet_handle_index
4. dedup_minority_aware_provenance_adjudicator

Each component must be implemented as a separate, ablatable module to enable clean attribution and ablation studies.