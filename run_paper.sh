export SHOW_LLM_REASONING=1
export EXPERIMENT_AGENT_MEMORY_TOOL_LOGS=1
export EXPERIMENT_AGENT_MEMORY_ENABLED=1
export EXPERIMENT_AGENT_MEMORY_WRITEBACK=1
export AGENT_BASH_TIMEOUT_SECONDS=6000


python -m src.agents.paper_agent.main --experiment pinn --resume --template-dir "/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent/src/agents/paper_agent/latex/ICML2025_Template"