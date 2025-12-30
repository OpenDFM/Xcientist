# ResearchAgent

ResearchAgent by X-LANCE.

## Local Package Imports

Some modules are designed to be imported as standalone packages. Run the editable installs once per environment so Python can resolve `agents` and `memory` without tweaking `PYTHONPATH`.

```bash
python -m pip install -e src/agents
python -m pip install -e src/memory
```

Example usage after installation:

```python
# agents package (driven by src/agents/setup.py)
from agents.idea_agent.utils.mcts_helpers import parse_json_response

# memory package (driven by src/memory/setup.py)
from memory.memory_system.vectorstore import VectorStore
```

These setups keep the repository as the single source of truth while letting interactive sessions or downstream tools import the project modules directly.

## Project Structure

```
ResearchAgent/
├── src/
│   ├── agents/
│   │   ├── base/
│   │   ├── master_agent/
│   │   ├── survey_agent/
│   │   ├── idea_agent/
│   │   └── experiment_agent/
│   ├── config/
│   ├── memory/
│   ├── tools/
│   └── utils/
├── tests/
├── data/
├── logs/
└── scripts/
```

## Agents

- **Master Agent**: Orchestrates the research workflow
- **Survey Agent**: Literature survey and information collection
- **Idea Agent**: Research idea generation
- **Experiment Agent**: Experiment execution and evaluation
