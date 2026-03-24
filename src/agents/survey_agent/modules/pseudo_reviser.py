import json
import os
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from utils.rich_logger import get_logger
from utils.api_call import ChatAgent
from utils.utils import extract_json


# Agent prompt for deciding operations
AGENT_OPERATE_PROMPT = """
You are an intelligent agent that refines project pseudocode by interacting with a code repository.

[Repository Information]
- Name: {repo_name}
- Title: {paper_title}
- Abstract: {paper_abstract}
- Structure:
{repo_structure}

[Core Files (for reference)]
{core_files_info}

[Main Files (for reference)]
{main_files_info}

[Current Project Pseudocode]
{current_pseudocode}

[Memory - Operation History]
{memory}

[Suggestion - Review Feedback]
{suggestions}

[Context - Retrieved Content]
{context}

[Revise Requirement]
- The recommended operation sequence is: review, get_pseudocode/get_source_code, revise, so that you can provide abundant information to reviser.
- The core operation is revise since the target is refine the pseudocode.
- You MUST call "revise" operation at least once every 3 rounds to improve the pseudocode.

---

Your task is to plan a sequence of operations to refine the pseudocode.

Available operations:
1. "get_pseudocode": Query pseudocode of a specific file by providing its path
2. "get_source_code": Query source code of a specific file in main file list or core file list by providing its path
3. "revise": Modify the current pseudocode based on the context (retrieved content) and suggestion - IMPORTANT: you must call this at least once every 3 rounds!
4. "review": Call the review skill to provide suggestions on what to do next
5. "finish": Complete the refinement process when the pseudocode is satisfactory

Output format (JSON):
{{
    "plan": [
        {{"operation": "...", "file_path": "...", "reason": "..."}},
        ...
    ]
}}

- Each item in "plan" is one operation to execute in order
- For "get_pseudocode" or "get_source_code", include "file_path"
- For "revise" or "finish", no file_path needed
- Include at least 1 operation, up to 3 operations per plan
- "review" operation does not require file_path
- You must call "revise" when getting enough information with other skills

Generate JSON directly without any other things.
"""

REVIEW_PROMPT = """
You are a code review expert. Review the current project pseudocode and provide specific suggestions.

[Paper Info]
Title: {paper_title}
Abstract: {paper_abstract}

[Repository Structure]
{repo_structure}

[Core Files (high score)]
{core_files_info}

[Main Files]
{main_files_info}

[Current Project Pseudocode]
{final_pseudocode}

---
Requirements:
1. The pseudocode should be well formatted and well structured with clear logic.
2. The pseudocode should contain all the core implementation specifically.
3. The pseudocode should be concise. No more than 7 sections and no more than 400 lines.
4. The pseudocode should be concise on or skip less important parts.

Please provide suggestions and scores (0-10) for the pseudocode.
- suggestions: List of concrete improvement suggestions
- conciseness_score: How concise the pseudocode is (higher = more concise, less redundant)
- logic_score: How clear and logical the pseudocode structure is (higher = better)
- specificity_score: How specific the pseudocode is about key implementation details (core model components, critical loss functions, key algorithms) - NOT about including trivial imports or boilerplate

Scoring Standards (0-10):
1. conciseness_score:
   - 9-10: Very concise, minimal redundancy, focuses on essential logic only
   - 7-8: Mostly concise, minor redundancies
   - 5-6: Some redundant sections or verbose descriptions
   - 3-4: Noticeable redundancies, verbose in places
   - 1-2: Highly repetitive, lots of unnecessary details

2. logic_score:
   - 9-10: Clear control flow, logical section organization, easy to follow
   - 7-8: Good structure, minor issues with flow
   - 5-6: Some logical issues, sections somewhat disconnected
   - 3-4: Confusing flow, poor section organization
   - 1-2: Very hard to understand the logic

3. specificity_score: Focus on KEY content specificity, NOT adding irrelevant details like imports or file paths
   - 9-10: Captures key details: core model components, critical loss functions, key algorithms, important data flow - without trivial imports
   - 7-8: Good detail on main components, minor missing key details
   - 5-6: Too generic, missing important specifics (e.g., just says "train model" without explaining how)
   - 3-4: Lacks important details about core algorithms/components
   - 1-2: Very vague, no meaningful implementation details

Output format (JSON):
{{
    "suggestions": ["suggestion1", "suggestion2", ...],
    "conciseness_score": 7,
    "logic_score": 8,
    "specificity_score": 8
}}

Generate JSON directly without any other things.
"""

MODIFY_PROMPT = """
TASK:

Build a well-structed pseudocode and analysis for the source code of a paper.
Given the following information, refine the project pseudocode:
[Paper Info]
{paper_title}:
{paper_abstract}

[Repository Info]
{repo_structure}

[Current Pseudocode]
{final_pseudocode}

[Retrieved Context]
{context}

[Reviewer suggestion]
{suggestions}

[Reason for modification]
{reason}

Please provide the refined pseudocode that incorporates the relevant details from the context.

REQUIRMENTS:
- Merge necessary content in context rather than simply adding the context.
- Be cautious about adding new section in the pseudocode.
- Keep the pseudocode at a reasonable length. If it gets too long, prioritize the most important additions or delete less important part.
- Output the refined pseudocode and analysis directly
"""


@dataclass
class AgentContext:
    """Structured input for the central agent."""
    repo_name: str
    paper_title: str
    paper_abstract: str
    repo_structure: Dict
    core_files: List[Dict] = field(default_factory=list)
    main_files: List[Dict] = field(default_factory=list)
    current_pseudocode: str = ""
    memory: List[Dict] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    context: str = ""
    error_log: List[str] = field(default_factory=list)
    review_scores: Dict = field(default_factory=dict)  # {"conciseness": 0-10, "logic": 0-10, "specificity": 0-10}


class PseudoReviser:
    """
    Agent-based pseudocode refiner that uses a central agent to decide operations.
    """
    
    def __init__(self, config, chat_agent: ChatAgent, repo_cache_path: str):
        self.chat_agent = chat_agent
        self.logger = get_logger("PseudoReviser")
        self.repo_cache_path = repo_cache_path
        self.default_max_steps = 20  # Maximum number of agent steps
    
    def _get_pseudocode_from_repo_structure(self, repo_structure: dict, rel_path: str) -> Optional[str]:
        """Get pseudocode from repo_structure by file path."""
        try:
            parts = rel_path.split(os.sep)
            node = repo_structure
            for p in parts[:-1]:
                if p not in node:
                    return None
                node = node[p]
            filename = parts[-1]
            if filename in node and isinstance(node[filename], dict):
                return node[filename].get("pseudocode")
        except Exception as e:
            self.logger.debug(f"Error getting pseudocode for {rel_path}: {e}")
        return None
    
    def _get_source_code(self, repo_name: str, rel_path: str) -> Optional[str]:
        """Read source code from the repository."""
        try:
            repo_path = os.path.join(self.repo_cache_path, repo_name)
            file_abs = os.path.join(repo_path, rel_path)
            if os.path.exists(file_abs):
                with open(file_abs, "r", encoding="utf-8", errors="ignore") as f:
                    # Limit to first 500 lines to avoid too much content
                    lines = f.readlines()[:500]
                    return "".join(lines)
        except Exception as e:
            self.logger.error(f"Error reading source code for {rel_path}: {e}")
        return None
    
    def _format_memory(self, memory: List[Dict]) -> str:
        """Format memory (operation history) as string."""
        if not memory:
            return "No operations performed yet."
        
        lines = []
        for i, op in enumerate(memory):
            lines.append(f"{i+1}. {op['operation']}: {op.get('reason', '')}")
            if op['operation'] == 'modify_pseudocode':
                lines.append(f"   Modified pseudocode")
        return "\n".join(lines)
    
    def _format_core_main_files(self, mainfest: dict) -> tuple:
        """Format core and main files info for the prompt."""
        # Core files
        core_files = mainfest.get("core_files", [])
        core_lines = []
        for f in core_files[:10]:  # Top 10
            path = f.get("path", "")
            scores = f.get("scores", [])
            max_score = max(scores) if scores else 0
            reason = f.get("reason", "")[:100]
            core_lines.append(f"- {path} (score: {max_score}, reason: {reason})")
        core_files_info = "\n".join(core_lines) if core_lines else "No core files"
        
        # Main files
        main_files = mainfest.get("main_files", [])
        main_lines = []
        for f in main_files[:5]:  # Top 5
            path = f.get("path", "")
            reason = f.get("reason", "")[:100]
            main_lines.append(f"- {path}: {reason}")
        main_files_info = "\n".join(main_lines) if main_lines else "No main files"
        
        return core_files_info, main_files_info
    
    def _validate_review_response(self, response: str, info_dict: dict) -> None:
        """Validate review response format."""
        result = extract_json(response)
        if not isinstance(result, dict):
            raise ValueError(f"Review response is not a dict: {type(result)}")
        if "suggestions" not in result:
            raise ValueError("Missing 'suggestions' field")
        if "conciseness_score" not in result:
            raise ValueError("Missing 'conciseness_score' field")
        if "logic_score" not in result:
            raise ValueError("Missing 'logic_score' field")
        if "specificity_score" not in result:
            raise ValueError("Missing 'specificity_score' field")
        # Validate score range
        conciseness = result.get("conciseness_score", 0)
        logic = result.get("logic_score", 0)
        specificity = result.get("specificity_score", 0)
        if not (0 <= conciseness <= 10):
            raise ValueError(f"conciseness_score out of range: {conciseness}")
        if not (0 <= logic <= 10):
            raise ValueError(f"logic_score out of range: {logic}")
        if not (0 <= specificity <= 10):
            raise ValueError(f"specificity_score out of range: {specificity}")
        return True, response
    
    def _call_review(self, agent_ctx: AgentContext) -> Dict:
        """Call review skill to get suggestions and scores."""
        core_files_info, main_files_info = self._format_core_main_files({
            "core_files": agent_ctx.core_files,
            "main_files": agent_ctx.main_files
        })
        
        prompt = REVIEW_PROMPT.format(
            paper_title=agent_ctx.paper_title,
            paper_abstract=agent_ctx.paper_abstract,
            repo_structure=self._format_repo_structure(agent_ctx.repo_structure),
            core_files_info=core_files_info,
            main_files_info=main_files_info,
            final_pseudocode=agent_ctx.current_pseudocode
        )
        
        response = self.chat_agent.remote_chat_with_retry(
            prompt, 
            validate_fn=self._validate_review_response,
            max_retry=3,
            temperature=0
        )
        
        result = extract_json(response)
        # result should be dict with suggestions and scores
        if isinstance(result, dict):
            suggestions = result.get("suggestions", [])
            scores = {
                "conciseness": result.get("conciseness_score", 0),
                "logic": result.get("logic_score", 0),
                "specificity": result.get("specificity_score", 0)
            }
            return {"suggestions": suggestions, "scores": scores}
        else:
            self.logger.warning(f"Review response is not a dict: {result}")
            return {"suggestions": [str(result)] if result else [], "scores": {"conciseness": 0, "logic": 0, "specificity": 0}}

    
    def _execute_operation(self, op: dict, agent_ctx: AgentContext) -> None:
        """Execute a single operation."""
        operation = op.get("operation", "")
        file_path = op.get("file_path", "")
        reason = op.get("reason", "")
        
        if operation == "get_pseudocode":
            if file_path:
                pseudocode = self._get_pseudocode_from_repo_structure(agent_ctx.repo_structure, file_path)
                agent_ctx.context += f"=== Pseudocode of {file_path} ===\n{pseudocode if pseudocode else 'Not available'}\n"
                self.logger.info(f"Retrieved pseudocode for: {file_path}, reason: {reason}")
            else:
                self.logger.error("get_pseudocode: file_path not provided, skipped")
                agent_ctx.error_log.append(f"file path do not have pseudocode: {file_path}")
                    
        elif operation == "get_source_code":
            if file_path:
                source_code = self._get_source_code(agent_ctx.repo_name, file_path)
                agent_ctx.context += f"=== Source code of {file_path} ===\n{source_code if source_code else 'Not available'}\n"
                self.logger.info(f"Retrieved source code for: {file_path}, reason: {reason}")
            else:
                self.logger.error("get_source_code: file_path not provided, skipped")
                agent_ctx.error_log.append(f"Unknown filepath in retrieve: {file_path}")
                    
        elif operation == "revise":
            modify_prompt = MODIFY_PROMPT.format(
                paper_title=agent_ctx.paper_title,
                paper_abstract=agent_ctx.paper_abstract,
                repo_structure=self._format_repo_structure(agent_ctx.repo_structure),
                final_pseudocode=agent_ctx.current_pseudocode,
                context=agent_ctx.context,
                suggestions="\n".join(agent_ctx.suggestions) if agent_ctx.suggestions else "No suggestion yet.",
                reason=reason,
            )
            result = self.chat_agent.remote_chat_with_retry(modify_prompt, temperature=0)
            agent_ctx.current_pseudocode = result
            agent_ctx.context = ""  # Clear context after modification
            agent_ctx.suggestions = []  # Clear suggestion after modification
            self.logger.info(f"Revised pseudocode, reason: {reason}")
                
        elif operation == "finish":
            self.logger.info(f"Operation: finish, reason: {reason}")
                
        elif operation == "review":
            review_result = self._call_review(agent_ctx)
            agent_ctx.suggestions.extend(review_result.get("suggestions", []))
            agent_ctx.review_scores = review_result.get("scores", {"conciseness": 0, "logic": 0, "specificity": 0})
            # Record scores in memory
            if agent_ctx.memory:
                agent_ctx.memory[-1]["review_scores"] = agent_ctx.review_scores
            self.logger.info(f"Review scores: conciseness={agent_ctx.review_scores['conciseness']}, logic={agent_ctx.review_scores['logic']}, specificity={agent_ctx.review_scores['specificity']}")
            self.logger.info(f"reason: {reason}")
                
        else:
            self.logger.error(f"Unknown operation: {operation}, skipped")
            agent_ctx.error_log.append(f"Unknown operation: {operation}, skipped")

    def refine_pseudocode_with_agent(
        self,
        mainfest: dict,
        current_pseudocode: str,
        max_steps: int = None,
        hard_code_revise: bool = False,
        max_rounds_without_revise: int = 3,
        last_round_revise: bool = True,
    ) -> str:
        """
        Refine pseudocode using agent-based approach with plan-based execution.
        
        Args:
            mainfest: the mainfest dict containing repo_info and core_files info
            current_pseudocode: the initial project pseudocode
            max_steps: maximum number of agent steps
            hard_code_revise: if True, force call revise after max_rounds_without_revise rounds without revise
            max_rounds_without_revise: maximum rounds without revise before forcing (default: 3)
            
        Returns:
            Final refined pseudocode
        """
        repo_info = mainfest["repo_info"]
        
        # Build AgentContext
        core_files_info, main_files_info = self._format_core_main_files(mainfest)
        
        agent_ctx = AgentContext(
            repo_name=repo_info["repo_name"],
            paper_title=repo_info["paper_title"],
            paper_abstract=repo_info["paper_abstract"],
            repo_structure=repo_info["repo_structure"],
            core_files=mainfest.get("core_files", []),
            main_files=mainfest.get("main_files", []),
            current_pseudocode=current_pseudocode,
            memory=[],
            suggestions=[],
            context="",
            error_log=[],
            review_scores={"conciseness": 0, "logic": 0, "specificity": 0}
        )
        
        max_steps = max_steps or self.default_max_steps
        
        # Track rounds since last revise for hard_code_revise feature
        rounds_since_last_revise = 0
        
        self.logger.info(f"Starting agent-based pseudocode refinement (max {max_steps} steps, hard_code_revise={hard_code_revise})")
        
        for step in range(max_steps):
            self.logger.info(f"Agent step {step + 1}/{max_steps}")

            # Format the prompt with plan-based format
            prompt = AGENT_OPERATE_PROMPT.format(
                repo_name=agent_ctx.repo_name,
                paper_title=agent_ctx.paper_title,
                paper_abstract=agent_ctx.paper_abstract,
                repo_structure=self._format_repo_structure(agent_ctx.repo_structure),
                core_files_info=core_files_info,
                main_files_info=main_files_info,
                current_pseudocode=agent_ctx.current_pseudocode,
                memory=self._format_memory(agent_ctx.memory),
                suggestions="\n".join(agent_ctx.suggestions) if agent_ctx.suggestions else "No suggestion yet.",
                context=agent_ctx.context if agent_ctx.context else "No context yet."
            )
            
            # self.logger.info(f"[PLANNER PROMPT] {prompt}")
            # Get agent's plan
            response = self.chat_agent.remote_chat(prompt, temperature=0)

            # self.logger.info(f"[PLANNER RESPONSE] {response}")
            
            # Parse the response - now expects "plan" array
            try:
                decision = extract_json(response)
                plan = decision.get("plan", [])
                if not isinstance(plan, list):
                    plan = [decision]  # Fallback: treat as single operation
            except Exception as e:
                self.logger.error(f"Failed to parse agent response: {response[:200]}..., skipped this round")
                agent_ctx.error_log.append(f"error in parsing planning json: {e}")
                continue
            
            # Execute each operation in the plan
            for op in plan:
                op_type = op.get("operation", "")
                # Record to memory
                agent_ctx.memory.append({"operation": op_type, "reason": op.get("reason", "")})
                
                # Check for finish
                if op_type == "finish":
                    self.logger.info("Agent decided to finish")
                    break
                
                # Execute operation
                self._execute_operation(op, agent_ctx)
                
                # Track revise operations
                if op_type == "revise":
                    rounds_since_last_revise = 0
                else:
                    rounds_since_last_revise += 1

            # Check if we need to force revise
            if hard_code_revise and rounds_since_last_revise >= max_rounds_without_revise:
                self.logger.info(f"Force calling revise after {rounds_since_last_revise} rounds without revise")
                # Force execute revise operation
                force_revise_op = {"operation": "revise", "reason": "Forced revise after max rounds without revise"}
                agent_ctx.memory.append({"operation": "revise", "reason": "Forced revise (hard_code)"})
                self._execute_operation(force_revise_op, agent_ctx)
                rounds_since_last_revise = 0
                continue

            if step + 1 == max_steps:
                if(last_round_revise and rounds_since_last_revise > 0):
                    self.logger.info("last round does not include revise, force calling revise")
                    force_revise_op = {"operation": "revise", "reason": "Forced revise in last round"}
                    agent_ctx.memory.append({"operation": "revise", "reason": "Forced revise (last round)"})
                    self._execute_operation(force_revise_op, agent_ctx)
                    rounds_since_last_revise = 0
            
            # Check if finished
            if any(m.get("operation") == "finish" for m in agent_ctx.memory[-len(plan):] if plan):
                self.logger.info(f"planning agent decide to finish at step {step}")
                break
                
        self.logger.info(f"Agent-based refinement completed after {step + 1} steps")
        return agent_ctx.current_pseudocode
    
    def _format_repo_structure(self, structure: dict) -> str:
        """Format repo structure as string (similar to format_repo_structure in CodeAnalyzer)."""
        lines = []
        
        def _build_tree_string(node: dict, prefix: str = ""):
            keys = sorted(node.keys())
            total_items = len(keys)
            
            for i, key in enumerate(keys):
                value = node[key]
                is_last = (i == total_items - 1)
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{key}")
                
                if isinstance(value, dict) and value.get("_is_file"):
                    pass
                else:
                    extension = "    " if is_last else "│   "
                    _build_tree_string(value, prefix + extension)
        
        _build_tree_string(structure)
        return "\n".join(lines)