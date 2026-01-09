export SHOW_LLM_REASONING=1
export EXPERIMENT_AGENT_MEMORY_TOOL_LOGS=1
export EXPERIMENT_AGENT_MEMORY_ENABLED=1
export EXPERIMENT_AGENT_MEMORY_WRITEBACK=1
export AGENT_BASH_TIMEOUT_SECONDS=6000

# python -m src.agents.experiment_agent.prepare --experiment spectral --result-json /hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent/src/agents/experiment_agent/workspaces/spectral/result.json --force --clone-depth 1 --verbose
python -m src.agents.experiment_agent.main --experiment spectral --verbose --resume
