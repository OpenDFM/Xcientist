## Local Package Imports

Some modules are designed to be imported as standalone packages. Run the editable installs once per environment so Python can resolve `agents` and `memory` without tweaking `PYTHONPATH`.

```bash
python -m pip install -e .
```

Example usage after installation:

```python
# memory package (driven by src/memory/setup.py)
from memory.memory_system.vectorstore import VectorStore
```

These setups keep the repository as the single source of truth while letting interactive sessions or downstream tools import the project modules directly.

# 🧪 Test Progress

At 2026.1.1, we finish the whole test and debug for MemPrism!

# Slot (Short-term) API
| Module / Test Item | Status |
|--------------------|---------|
| add_slot    | ✅ Debugged |
| clear_container    | ✅ Debugged |
| get_container_size   | ✅ Debugged |
| filter_and_route_slots  | ✅ Debugged |
| compress_slots      | ✅ Debugged |
| query      | ✅ Debugged |
| transfer_slot_to_text | ✅ Debugged |
| transfer_experiment_agent_context_to_working_slots | ✅ Debugged |
| generate_long_term_memory | ✅ Debugged |
| transfer_slot_to_semantic_record | ✅ Debugged |
| transfer_slot_to_episodic_record | ✅ Debugged |
| transfer_slot_to_procedural_record | ✅ Debugged |
| multi_thread_process | ✅ Debugged |

# Long-term Memory API
| Module / Test Item | Status |
|--------------------|---------|
| instantiate_sem_record    | ✅ Debugged |
| instantiate_epi_record    | ✅ Debugged |
| instantiate_proc_record   | ✅ Debugged |
| size(@property)  | ✅ Debugged |
| get_records_by_ids     | ✅ Debugged |
| get_last_k_records | ✅ Debugged |
| is_exists    | ✅ Debugged |
| add    | ✅ Debugged |
| update   | ✅ Debugged |
| delete    | ✅ Debugged |
| query | ✅ Debugged |
| abstract_episodic_records | ✅ Debugged |
| upsert_abstract_semantic_records | ✅ Debugged |
| get_nearest_k_records    | ✅ Debugged |
| save   | ✅ Debugged |
| load  | ✅ Debugged |

## Slot (Short-term) API Examples

The snippets below assume your OpenAI credentials are exported so that `SlotProcess` can call the LLM-backed helpers.

### Shared setup
```python
from memory.api.slot_process_api import SlotProcess
from memory.memory_system import WorkingSlot

slot_process = SlotProcess()
research_slot = WorkingSlot(
    stage="analysis",
    topic="weather_robustness",
    summary="Benchmarked fog augmentation on coastal driving data.",
    attachments={
        "metrics": {"accuracy": 0.73, "baseline": 0.70},
        "notes": {"insight": "Fog helps when the baseline is sunny-only."}
    },
    tags=["research", "vision"]
)
```

- **`add_slot(slot)`** – enqueue a `WorkingSlot` for later filtering/routing.
  ```python
  slot_process.add_slot(research_slot)
  ```

- **`clear_container()`** – drop every slot from the short-term queue.
  ```python
  slot_process.clear_container()
  ```

- **`get_container_size()`** – inspect how many slots are currently buffered.
  ```python
  print(f"{slot_process.get_container_size()} slots queued for review")
  ```

- **`query(query_text, limit=5, key_words=None)`** – rank the in-memory slots (no LLM call) by overlap with your query.
  ```python
  slot_process.clear_container()
  slot_process.add_slot(research_slot)
  slot_process.add_slot(WorkingSlot(
      stage="execution",
      topic="rl_pipeline",
      summary="Policy distillation cut variance by 2% using episodic recall buffers.",
      tags=["rl", "execution"],
  ))

  hits = slot_process.query(
      query_text="episodic recall variance",
      limit=2,
      key_words=["episodic", "variance"],
  )
  for score, slot in hits:
      print(f"{score:.2f} :: {slot.topic}")
  ```

- **`filter_and_route_slots()`** – asynchronously discard low-quality slots and route the rest to a memory type.
  ```python
  import asyncio

  async def demo_filter_and_route():
      slot_process.clear_container()
      slot_process.add_slot(research_slot)
      routed = await slot_process.filter_and_route_slots()
      for bundle in routed:
          print(bundle["memory_type"], bundle["slot"].summary)

  asyncio.run(demo_filter_and_route())
  ```

- **`compress_slots(sids=None)`** – compress every slot (or a subset by ID) into a single distilled `WorkingSlot`.
  ```python
  import asyncio

  async def demo_compress():
      slot_process.clear_container()
      slot_process.add_slot(research_slot)
      slot_process.add_slot(WorkingSlot(
          stage="execution",
          topic="rl_pipeline",
          summary="Ran policy distillation with episodic recall and saw +2% stability.",
          attachments={"metrics": {"stability_index": 0.91}},
          tags=["rl", "execution"]
      ))
      compressed = await slot_process.compress_slots()
      print(compressed.to_dict())

  asyncio.run(demo_compress())
  ```

- **`transfer_slot_to_text(slot)`** – convert a slot JSON dump into a concise paragraph via the LLM.
  ```python
  import asyncio

  async def demo_transfer():
      text_summary = await slot_process.transfer_slot_to_text(research_slot)
      print(text_summary.strip())

  asyncio.run(demo_transfer())
  ```

- **`transfer_experiment_agent_context_to_working_slots(context, max_slots=50)`** – expand a `WorkflowContext` snapshot into fresh `WorkingSlot` objects.
  ```python
  import asyncio
  from src.agents.experiment_agent.sub_agents.experiment_master.workflow_state_machine import (
      WorkflowContext,
      WorkflowState,
  )

  context = WorkflowContext(
      research_input="Stress-test fog augmentations on coastal autopilot logs.",
      input_type="idea",
      current_state=WorkflowState.EXPERIMENT_ANALYSIS,
      iteration_count=3,
      max_iterations=6,
      pre_analysis_output={"insights": ["Fog data scarcity limits generalization."]},
      code_plan_output={"implementation_checklist": [
          {"title": "mine fog samples"},
          {"title": "rerun autopilot evaluation"},
      ]},
      experiment_execute_output={"metrics": {"accuracy": 0.73, "baseline": 0.70}},
      experiment_analysis_output={"insight": "Fog mix gains 3 points over baseline."},
  )

  async def context_to_slots():
      slots = await slot_process.transfer_experiment_agent_context_to_working_slots(
          context,
          max_slots=2,
      )
      for slot in slots:
          print(slot.stage, slot.topic, slot.summary)

  asyncio.run(context_to_slots())
  ```

- **`generate_long_term_memory(routed_slots)`** – convert routed slots into FAISS-ready payloads grouped by memory type.
  ```python
  import asyncio

  async def demo_generate_long_term_memory():
      slot_process.clear_container()
      slot_process.add_slot(research_slot)
      slot_process.add_slot(WorkingSlot(
          stage="execution",
          topic="fog_eval_run",
          summary="Executed autopilot across 120 fog frames; latency +5ms vs. clear.",
          attachments={"metrics": {"latency_delta_ms": 5}},
          tags=["execution", "fog"],
      ))

      routed_slots = await slot_process.filter_and_route_slots()
      long_term_inputs = await slot_process.generate_long_term_memory(routed_slots)
      for payload in long_term_inputs:
          print(payload["memory_type"], payload["input"])

  asyncio.run(demo_generate_long_term_memory())
  ```

- **`transfer_slot_to_semantic_record(slot)`** – ask the LLM to lift a slot into a durable semantic memory.
  ```python
  import asyncio

  async def demo_semantic_conversion():
      semantic_payload = await slot_process.transfer_slot_to_semantic_record(research_slot)
      print(semantic_payload)

  asyncio.run(demo_semantic_conversion())
  ```

- **`transfer_slot_to_episodic_record(slot)`** – capture Situation→Action→Result traces with structured `detail` metadata.
  ```python
  import asyncio

  execution_slot = WorkingSlot(
      stage="execution",
      topic="fog_eval_run",
      summary="Executed autopilot on the fog benchmark and logged safety violations.",
      attachments={
          "metrics": {"failure_rate": 0.11, "baseline_failure_rate": 0.15},
          "notes": {"insight": "Violations drop after fog-specific fine-tuning."},
      },
      tags=["execution", "fog"],
  )

  async def demo_episodic_conversion():
      episodic_payload = await slot_process.transfer_slot_to_episodic_record(execution_slot)
      print(episodic_payload)

  asyncio.run(demo_episodic_conversion())
  ```

- **`transfer_slot_to_procedural_record(slot)`** – document reproducible steps (and optional helper code) derived from a slot.
  ```python
  import asyncio

  procedural_slot = WorkingSlot(
      stage="execution",
      topic="fog_augmentation_pipeline",
      summary="Outlined the exact steps for blending fog augmentations into training.",
      attachments={
          "notes": {
              "preprocess": "Normalize visibilities before mixing fog levels.",
              "commands": "python train.py --weather=fog_mix",
          }
      },
      tags=["procedure", "training"],
  )

  async def demo_procedural_conversion():
      procedural_payload = await slot_process.transfer_slot_to_procedural_record(procedural_slot)
      print(procedural_payload)

  asyncio.run(demo_procedural_conversion())
  ```

- **`multi_thread_process(func, max_workers=5, **kwargs)`** – drive the entire short-term → long-term pipeline in one call. `func` must synchronously populate `slot_container` (for example, by calling `add_slot` after pulling context from an agent). Once the slots are buffered, the helper fans out filtering/routing and memory conversions with `_multi_thread_run`, stores the intermediate dicts in `memory_dict`, pushes the converted records into the configured FAISS stores, and finally prints the updated store sizes through the module-level `llm_client`. Because the routine references the global `args` namespace and the FAISS stores, make sure they are initialized before invoking it (the CLI wiring already does this).
  ```python
  from types import SimpleNamespace
  from memory.api import slot_process_api as slot_api
  from memory.api.slot_process_api import SlotProcess
  from memory.api.faiss_memory_system_api import FAISSMemorySystem
  from memory.memory_system import WorkingSlot

  slot_process = SlotProcess()

  slot_api.llm_client = SimpleNamespace(
      slot_process=slot_process,
      semantic_memory_system=FAISSMemorySystem(memory_type="semantic"),
      episodic_memory_system=FAISSMemorySystem(memory_type="episodic"),
      procedural_memory_system=FAISSMemorySystem(memory_type="procedural"),
  )
  slot_api.args = SimpleNamespace(abstract_memories=False)

  # Expose the memory stores on the SlotProcess instance for the threaded transfer step.
  slot_process.slot_process = slot_process
  slot_process.semantic_memory_system = slot_api.llm_client.semantic_memory_system
  slot_process.episodic_memory_system = slot_api.llm_client.episodic_memory_system
  slot_process.procedural_memory_system = slot_api.llm_client.procedural_memory_system

  demo_slots = [
      WorkingSlot(
          stage="analysis",
          topic="fog_ablation",
          summary="Compared fog-on vs fog-off policies; saw +3% accuracy in coastal storms.",
          tags=["analysis", "vision"],
      ),
      WorkingSlot(
          stage="execution",
          topic="fog_eval_run",
          summary="Replayed 120 autopilot clips with fog augmentations; latency +5ms.",
          tags=["execution", "fog"],
      ),
  ]

  def seed_slots(batch_size: int):
      slot_process.clear_container()
      for slot in demo_slots[:batch_size]:
          slot_process.add_slot(slot)

  slot_process.multi_thread_process(
      func=seed_slots,
      batch_size=2,     # forwarded to seed_slots(**kwargs)
      max_workers=4,    # used for both filtering/routing and conversion stages
  )
  ```

## Long-term Memory API Examples

All examples assume FAISS can write to disk (for save/load) and that the OpenAI client configured in `FAISSMemorySystem` can run when episodic abstraction is invoked.

### Shared setup
```python
from memory.api.faiss_memory_system_api import FAISSMemorySystem

semantic_store = FAISSMemorySystem(memory_type="semantic")
episodic_store = FAISSMemorySystem(memory_type="episodic")
procedural_store = FAISSMemorySystem(memory_type="procedural")
```

- **`instantiate_sem_record(**kwargs)`** – create a `SemanticRecord` before persisting it.
  ```python
  sem_record = semantic_store.instantiate_sem_record(
      summary="Fog augmentations raise coastal accuracy by ~3%.",
      detail="Baseline accuracy 0.70; fog mix training run reached 0.73.",
      tags=("augmentation", "experiment")
  )
  ```

- **`instantiate_epi_record(**kwargs)`** – construct an `EpisodicRecord` with an embedding ready for clustering.
  ```python
  epi_record = episodic_store.instantiate_epi_record(
      stage="execution",
      summary="Ran fog training job with seed 7.",
      detail={"metrics": {"accuracy": 0.73}, "notes": "Used 3 fog levels."},
      tags=("execution", "fog")
  )
  ```

- **`instantiate_proc_record(**kwargs)`** – define a reusable `ProceduralRecord`.
  ```python
  proc_record = procedural_store.instantiate_proc_record(
      name="augmentation_pipeline",
      description="Steps for injecting fog noise into the loader.",
      steps=[
          "Profile deployment weather distribution.",
          "Map weather modes to augmentation operators.",
          "Blend augmentations during training."
      ],
      code="def apply_fog(batch):\n    return fogger(batch)",
      tags=("procedure", "training")
  )
  ```

The remaining snippets reuse `sem_record`, `epi_record`, and `proc_record` defined above.

- **`size`** – read-only property showing the number of records already indexed.
  ```python
  print(f"{semantic_store.size} semantic memories stored so far")
  ```

- **`get_records_by_ids(mids)`** – fetch specific records (assuming they were previously added).
  ```python
  semantic_store.add([sem_record])
  fetched = semantic_store.get_records_by_ids([sem_record.id])
  print(fetched[0].summary)
  ```

- **`get_last_k_records(k)`** – inspect the most recently inserted memories.
  ```python
  recent_records, total_seen = semantic_store.get_last_k_records(k=3)
  for record in recent_records:
      print(record.id, record.summary)
  ```

- **`is_exists(mids)`** – check whether particular IDs are present without fetching them.
  ```python
  flags = semantic_store.is_exists([sem_record.id, "sem-missing"])
  print(flags)  # -> [True, False]
  ```

- **`add(memories)`** – persist new semantic/episodic/procedural memories in FAISS.
  ```python
  semantic_store.add([sem_record])
  episodic_store.add([epi_record])
  procedural_store.add([proc_record])
  ```

- **`add(..., agent_id='idea_agent')`** and `query(..., agent_id='idea_agent')` – scope inserts and lookups to a specific agent (`survey_agent`, `idea_agent`, or `experiment_agent`). The store appends `_{agent_id}` to every persisted record ID and filters query hits by that suffix, allowing multiple agents to share the same FAISS index without collisions.
  ```python
  idea_semantic = semantic_store.instantiate_sem_record(
      summary="Idea agent logged a fog augmentation hypothesis.",
      detail="Consider fog-aware sampling to stabilize RL rollouts.",
      tags=("idea", "fog"),
  )
  semantic_store.add([idea_semantic], agent_id="idea_agent")

  # Later, pull only the memories contributed by the idea agent.
  idea_hits = semantic_store.query(
      query_text="fog augmentation hypothesis",
      agent_id="idea_agent",
      method="embedding",
      limit=3,
  )
  for score, record in idea_hits:
      print(record.id, score)
  ```

- **`update(memories)`** – push field edits back into FAISS.
  ```python
  sem_record.detail = "Retrained with fog + rain augmentations; accuracy is now 0.75."
  semantic_store.update([sem_record])
  ```

- **`delete(mids)`** – remove records (and reset episodic clustering state when needed).
  ```python
  semantic_store.delete([sem_record.id])
  episodic_store.delete([epi_record.id])
  ```

- **`query(query_text, method='embedding', limit=5, filters=None)`** – retrieve the most relevant memories via the configured similarity strategy.
  ```python
  hits = semantic_store.query(
      query_text="fog augmentation robustness",
      method="embedding",
      limit=2,
  )
  for score, record in hits:
      print(f"{score:.3f} :: {record.summary}")
  ```

- **`abstract_episodic_records(epi_records, consistency_threshold)`** – asynchronously cluster episodic traces into semantic summaries (feature marked as not fully debugged).
  ```python
  import asyncio

  async def demo_abstraction():
      episodic_store.add([epi_record])
      semantic_summaries, cluster_map = await episodic_store.abstract_episodic_records(
          [epi_record],
          consistency_threshold=0.8,
      )
      print(cluster_map, [record.summary for record in semantic_summaries])

  asyncio.run(demo_abstraction())
  ```

- **`upsert_abstract_semantic_records(sem_records, cidmap2semrec)`** – merge new semantic abstractions into the semantic FAISS index while keeping one record per cluster ID.
  ```python
  import asyncio

  async def demo_upsert_semantic():
      episodic_store.add([epi_record])
      sem_summaries, cidmap2semrec = await episodic_store.abstract_episodic_records(
          [epi_record],
          consistency_threshold=0.8,
      )

      # New clusters are added, previously seen cluster_ids trigger an in-place update.
      semantic_store.upsert_abstract_semantic_records(sem_summaries, cidmap2semrec)
      latest_records, _ = semantic_store.get_last_k_records(k=len(sem_summaries))
      print([(record.cluster_id, record.summary) for record in latest_records])

  asyncio.run(demo_upsert_semantic())
  ```

- **`get_nearest_k_records(record, method='embedding', k=5, filters=None)`** – find neighbors for an existing memory.
  ```python
  neighbors = semantic_store.get_nearest_k_records(sem_record, k=2)
  for score, record in neighbors:
      print(f"{record.id} is similar with score {score:.3f}")
  ```

- **`save(path)`** – serialize embeddings, metadata, and mappings to disk.
  ```python
  semantic_store.save("sem_cache")
  episodic_store.save("epi_cache")
  ```

- **`load(path)`** – hydrate a new `FAISSMemorySystem` from a saved directory.
  ```python
  restored_semantic = FAISSMemorySystem(memory_type="semantic")
  restored_semantic.load("sem_cache")
  print(restored_semantic.size)
  ```

# Test for STM and LTM
```
export OPENAI_API_KEY=""
export OPENAI_BASE_URL=""

python ltm_test.py
python stm_test.py
```
