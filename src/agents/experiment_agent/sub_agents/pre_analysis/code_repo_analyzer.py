"""
Code Repository Analyzer - Analyzes reference code repositories.

This agent reads and analyzes each code repository in the repos directory,
generating a summary of its functionality and relevance to the research idea.
"""

import os
from typing import List, Optional
from agents import Agent, Runner, RunConfig, ModelSettings
from src.agents.experiment_agent.logger import create_verbose_hooks
from src.agents.experiment_agent.tools import get_tools_for_agent


def create_code_repo_analyzer(
    model: str = "gpt-4o",
    workspace_dir: str = None,
    tools: list = None,
) -> Agent:
    """
    Create a code repository analyzer agent.
    
    Args:
        model: The model to use for the agent
        workspace_dir: Base workspace directory containing repos/
        tools: List of tools to use (defaults to CODE_REPO_ANALYZER_TOOLS)
        
    Returns:
        Agent instance configured for code repository analysis
    """
    repos_path = os.path.join(workspace_dir, "repos") if workspace_dir else "./repos"
    
    # Use provided tools or get default tools for this agent type
    if tools is None:
        tool_config = get_tools_for_agent("pre_analysis")
        tools = tool_config.get("code_repo_analyzer", [])
    
    instructions = f"""You are a Code Repository Analyst.

YOUR TASK:
Analyze reference code repositories to understand their structure, functionality, and relevance to the research project.

WORKSPACE:
- Reference repositories are located at: {repos_path}
- Each subdirectory in repos/ is a separate code repository

WORKFLOW:
1. Use `list_directory` to discover repositories in {repos_path}
2. For each repository found:
   a. Use `list_directory` to explore its structure
   b. Use `read_file` to read key files (README, main modules, config files)
   c. Identify the repository's purpose and main functionality
3. Generate a summary for each repository

OUTPUT FORMAT:
For each repository, provide:
```
### Repository: [repo_name]
**Path**: [full_path]
**Purpose**: [Brief description of what this repository does]
**Key Components**:
- [component1]: [description]
- [component2]: [description]
**Relevance to Research**: [How this repo relates to the research idea/paper]
**Useful Code Patterns**: [Any patterns or implementations that could be reused]
```

IMPORTANT:
- Focus on understanding the HIGH-LEVEL architecture and purpose
- Identify reusable components and patterns
- Note any dependencies or requirements
- Skip binary files, large data files, and generated code
- Prioritize: README > main entry points > core modules > utils
"""
    
    return Agent(
        name="Code Repository Analyzer",
        instructions=instructions,
        tools=tools,
        model=model,
    )


class CodeRepoAnalyzerAgent:
    """
    Agent class for analyzing code repositories.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        workspace_dir: str = None,
        tools: list = None,
        verbose: bool = False,
    ):
        self.model = model
        self.verbose = verbose
        self.workspace_dir = workspace_dir
        
        self.hooks = create_verbose_hooks(
            show_llm_responses=verbose,
            show_tools=verbose,
            show_tool_args=True,
        )
        
        # Use provided tools or get default tools
        if tools is None:
            tool_config = get_tools_for_agent("pre_analysis")
            tools = tool_config.get("code_repo_analyzer", [])
        self.tools = tools
        
        self.agent = create_code_repo_analyzer(
            model=model,
            workspace_dir=workspace_dir,
            tools=tools,
        )
    
    async def analyze(self, research_context: str) -> str:
        """
        Analyze all repositories in the workspace.
        
        Args:
            research_context: Context about the research (concept + algorithm analysis)
            
        Returns:
            String containing analysis of all repositories
        """
        repos_path = os.path.join(self.workspace_dir, "repos") if self.workspace_dir else "./repos"
        
        prompt = f"""Analyze the reference code repositories for this research project.

=== RESEARCH CONTEXT ===
{research_context}

=== TASK ===
1. Explore the repositories at: {repos_path}
2. For each repository, understand its purpose and structure
3. Identify how each repository relates to the research context above
4. Generate a comprehensive summary

Please start by listing the contents of {repos_path} to discover available repositories.
"""
        
        result = await Runner.run(
            self.agent,
            prompt,
            hooks=self.hooks,
            max_turns=50,
            run_config=RunConfig(model_settings=ModelSettings(max_tokens=128000)),
        )
        
        if hasattr(result, "final_output") and isinstance(result.final_output, str):
            return result.final_output
        elif hasattr(result, "chat_history") and result.chat_history:
            return result.chat_history[-1].content
        
        return ""
    
    def analyze_sync(self, research_context: str) -> str:
        import asyncio
        return asyncio.run(self.analyze(research_context))


def create_code_repo_analyzer_agent(
    model: str = "gpt-4o",
    workspace_dir: str = None,
    tools: list = None,
    verbose: bool = False,
) -> CodeRepoAnalyzerAgent:
    """
    Factory function to create a code repository analyzer agent.
    """
    return CodeRepoAnalyzerAgent(
        model=model,
        workspace_dir=workspace_dir,
        tools=tools,
        verbose=verbose,
    )


if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = create_code_repo_analyzer_agent(
            model="gpt-4o",
            workspace_dir="/workspace",
            verbose=True,
        )
        result = await agent.analyze("Research about neural network optimization")
        print(result)
    
    asyncio.run(main())
