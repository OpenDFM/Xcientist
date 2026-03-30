"""
PseudoWriter: Agent-based pseudocode creator that creates pseudocode from scratch
based on repository structure, paper title, and abstract.
"""

import json
import os
import tiktoken
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from utils.rich_logger import get_logger
from utils.api_call import ChatAgent
from utils.utils import extract_json


# Agent prompt for deciding operations
AGENT_OPERATE_PROMPT = """
You are an intelligent agent that creates project pseudocode by interacting with a code repository.

[Repository Information]
- Name: {repo_name}
- Title: {paper_title}
- Abstract: {paper_abstract}
- Structure:
{repo_structure}

[Current Project Pseudocode]
{current_pseudocode}

[Memory - Operation History and Results]
{memory}

[Suggestion - Review Feedback]
{suggestions}

[Context - Retrieved Content from Repository]
{context}

[Operation Requirement]
- The recommended operation sequence is: review, create, get_source_code, revise
- IMPORTANT: You MUST call "create" operation at least once before calling "revise" to establish the initial pseudocode
- You MUST call "revise" operation at least once every 3 rounds to improve the pseudocode

---

Your task is to plan a sequence of operations to create and refine the pseudocode.

Available operations:
1. "create": Create a new pseudocode based on repo_structure, title, abstract, and any retrieved context - MUST be called before "revise"
2. "get_source_code": Query source code of a specific file by providing its path
3. "revise": Modify the current pseudocode based on the context (retrieved content) and suggestion - IMPORTANT: you must call this at least once every 3 rounds!
4. "review": Call the review skill to provide suggestions on what to do next
5. "finish": Complete the creation process when the pseudocode is satisfactory

Output format (JSON):
{{
    "plan": [
        {{"operation": "...", "file_path": "...", "reason": "..."}},
        ...
    ]
}}

- Each item in "plan" is one operation to execute in order
- For "get_source_code", include "file_path"
- For "create", "revise", or "finish", no file_path needed
- Include at least 1 operation, up to 3 operations per plan
- "create" should be called early to establish initial pseudocode
- "review" operation does not require file_path

Generate JSON directly without any other things.
"""

REVIEW_PROMPT = """
You are a code review expert. Review the current project pseudocode and provide specific suggestions.

[Paper Info]
Title: {paper_title}
Abstract: {paper_abstract}

[Repository Structure]
{repo_structure}

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

CREATE_PROMPT = """
TASK:

Create a well-structured pseudocode and analysis for a research project based on its paper information and repository structure.

[Paper Info]
Title: {paper_title}
Abstract: {paper_abstract}

[Repository Structure]
{repo_structure}

[Retrieved Context from Repository Files]
{context}

Please create a comprehensive pseudocode that captures:
1. The main components and architecture of the project
2. Core algorithms and their implementations
3. Data flow and processing pipelines
4. Key functions and their purposes

REQUIREMENTS:
- The pseudocode should be well formatted and well structured with clear logic
- Use clear section headers to organize the pseudocode
- Focus on the core implementation details that are essential to understanding the project
- Be concise: no more than 7 sections and no more than 400 lines
- Prioritize important parts and skip less critical details (e.g., trivial imports, boilerplate code)
- Output only the pseudocode and brief analysis directly
"""

MODIFY_PROMPT = """
TASK:

Refine the current project pseudocode based on the retrieved context and review suggestions.

[Paper Info]
{paper_title}:
{paper_abstract}

[Repository Info]
{repo_structure}

[Current Pseudocode]
{final_pseudocode}

[Retrieved Context]
{context}

[Reviewer Suggestion]
{suggestions}

[Reason for Modification]
{reason}

Please provide the refined pseudocode that incorporates the relevant details from the context.

REQUIREMENTS:
- Merge necessary content from context rather than simply adding the context
- Be cautious about adding new sections in the pseudocode
- Keep the pseudocode at a reasonable length. If it gets too long, prioritize the most important additions or delete less important parts
- Output the refined pseudocode and analysis directly
"""


@dataclass
class AgentContext:
    """Structured input for the PseudoWriter agent."""
    repo_name: str
    paper_title: str
    paper_abstract: str
    repo_structure: Dict
    current_pseudocode: str = ""
    memory: List[Dict] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    context: str = ""
    error_log: List[str] = field(default_factory=list)
    review_scores: Dict = field(default_factory=dict)  # {"conciseness": 0-10, "logic": 0-10, "specificity": 0-10}
    has_created_initial: bool = False  # Track if create has been called


def compress_memory(memory: List[Dict], encodings=None) -> List[Dict]:
    """
    Compress memory by removing the oldest half of entries.
    This is a simple implementation that can be improved later.
    
    Args:
        memory: List of memory entries
        encodings: Optional tiktoken encodings for token counting
        
    Returns:
        Compressed memory list (oldest half removed)
    """
    if len(memory) <= 1:
        return memory
    
    # Keep the newer half (more relevant for current work)
    mid_point = len(memory) // 2
    compressed = memory[mid_point:]
    
    # Log compression info
    logger = get_logger("PseudoWriter")
    logger.info(f"Memory compressed: {len(memory)} -> {len(compressed)} entries (removed oldest {mid_point} entries)")
    
    return compressed


def count_tokens(text: str, encodings) -> int:
    """Count tokens in text using tiktoken encoding."""
    if encodings is None:
        # Fallback: estimate by character count
        return len(text) // 4
    try:
        return len(encodings.encode(text))
    except Exception:
        return len(text) // 4


class PseudoWriter:
    """
    Agent-based pseudocode creator that creates pseudocode from scratch.
    Uses a planner agent with various skills to create and refine pseudocode.
    """
    
    def __init__(self, config, chat_agent: ChatAgent, repo_cache_path: str):
        self.chat_agent = chat_agent
        self.logger = get_logger("PseudoWriter")
        self.repo_cache_path = repo_cache_path
        self.default_max_steps = 20
        self.memory_token_threshold = 50000  # Threshold for memory compression
        self.use_memory_result_uplimit = True
        self.memory_result = 2000
        self.agent_source_code_max_read_lines = 700
        
        # Initialize tiktoken encoding for token counting
        try:
            self.encodings = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.logger.warning("Failed to load tiktoken encoding, using character-based estimation")
            self.encodings = None
    
    def _get_source_code(self, repo_name: str, rel_path: str) -> Optional[str]:
        """Read source code from the repository."""
        try:
            repo_path = os.path.join(self.repo_cache_path, repo_name)
            file_abs = os.path.join(repo_path, rel_path)
            if os.path.exists(file_abs):
                with open(file_abs, "r", encoding="utf-8", errors="ignore") as f:
                    # Limit to first 500 lines to avoid too much content
                    lines = f.readlines()[:self.agent_source_code_max_read_lines]
                    return "".join(lines)
        except Exception as e:
            self.logger.error(f"Error reading source code for {rel_path}: {e}")
        return None
    
    def _truncate_text(self, text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to max_length, adding suffix if truncated."""
        if not text:
            return text
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    def _format_memory(self, memory: List[Dict]) -> str:
        """Format memory (operation history with results) as string.
        
        Different operation types have different formatting:
        - create: shows truncated pseudocode
        - get_source_code: shows file path and truncated source code
        - revise: shows truncated pseudocode
        - review: shows suggestions and scores
        - finish: shows final context/suggestions
        """
        if not memory:
            return "No operations performed yet."
        
        lines = []
        for i, op in enumerate(memory):
            op_type = op.get('operation', '')
            reason = op.get('reason', '')
            
            lines.append(f"{i+1}. {op_type}: {reason}")
            
            # Format based on operation type
            if op_type == "create":
                # Show truncated pseudocode
                pseudocode = op.get('pseudocode', op.get('result', ''))
                if self.use_memory_result_uplimit:
                    pseudocode = self._truncate_text(pseudocode, self.memory_result)
                lines.append(f"   Pseudocode: {pseudocode}")
                
            elif op_type == "get_source_code":
                # Show file path and truncated source code
                file_path = op.get('file_path', 'unknown')
                source_code = op.get('source_code_result', op.get('result', ''))
                if self.use_memory_result_uplimit:
                    source_code = self._truncate_text(source_code, self.memory_result)
                lines.append(f"   File: {file_path}")
                lines.append(f"   Source: {source_code}")
                
            elif op_type == "revise":
                # Show truncated pseudocode
                pseudocode = op.get('pseudocode', op.get('result', ''))
                if self.use_memory_result_uplimit:
                    pseudocode = self._truncate_text(pseudocode, self.memory_result)
                lines.append(f"   Revised to: {pseudocode}")
                
            elif op_type == "review":
                # Show suggestions and scores
                suggestions = op.get('suggestions', [])
                scores = op.get('review_scores', {})
                if suggestions:
                    suggestions_text = "; ".join(suggestions)  # Limit to 5 suggestions
                    if self.use_memory_result_uplimit:
                        suggestions_text = self._truncate_text(suggestions_text, self.memory_result // 2)
                    lines.append(f"   Suggestions: {suggestions_text}")
                if scores:
                    lines.append(f"   Scores: conciseness={scores.get('conciseness', 0)}, logic={scores.get('logic', 0)}, specificity={scores.get('specificity', 0)}")
                    
            elif op_type == "finish":
                # Show context and suggestions
                context = op.get('context', '')
                suggestions = op.get('suggestions', [])
                if context and self.use_memory_result_uplimit:
                    context = self._truncate_text(context, self.memory_result)
                if context:
                    lines.append(f"   Final Context: {context}")
                if suggestions:
                    suggestions_text = "; ".join(suggestions[:5])
                    if self.use_memory_result_uplimit:
                        suggestions_text = self._truncate_text(suggestions_text, self.memory_result // 2)
                    lines.append(f"   Final Suggestions: {suggestions_text}")
            
            else:
                # Generic fallback
                result = op.get('result', '')
                if result:
                    if self.use_memory_result_uplimit:
                        result = self._truncate_text(result, self.memory_result)
                    lines.append(f"   Result: {result}")
                    
        return "\n".join(lines)
    
    def _check_memory_compression(self, agent_ctx: AgentContext) -> bool:
        """Check if memory needs compression based on token count."""
        memory_str = self._format_memory(agent_ctx.memory)
        token_count = count_tokens(memory_str, self.encodings)
        
        if token_count > self.memory_token_threshold:
            self.logger.info(f"Memory token count ({token_count}) exceeds threshold ({self.memory_token_threshold}), compressing...")
            agent_ctx.memory = compress_memory(agent_ctx.memory, self.encodings)
            return True
        return False
    
    def _format_repo_structure(self, structure: dict) -> str:
        """Format repo structure as string (tree format)."""
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
        prompt = REVIEW_PROMPT.format(
            paper_title=agent_ctx.paper_title,
            paper_abstract=agent_ctx.paper_abstract,
            repo_structure=self._format_repo_structure(agent_ctx.repo_structure),
            final_pseudocode=agent_ctx.current_pseudocode
        )
        
        response = self.chat_agent.remote_chat_with_retry(
            prompt, 
            validate_fn=self._validate_review_response,
            max_retry=3,
            temperature=0
        )
        
        result = extract_json(response)
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
        
        if operation == "create":
            # Create initial pseudocode from scratch
            create_prompt = CREATE_PROMPT.format(
                paper_title=agent_ctx.paper_title,
                paper_abstract=agent_ctx.paper_abstract,
                repo_structure=self._format_repo_structure(agent_ctx.repo_structure),
                context=agent_ctx.context if agent_ctx.context else "No additional context retrieved yet. Create based on paper info and repo structure."
            )
            
            result = self.chat_agent.remote_chat_with_retry(create_prompt, temperature=0)
            agent_ctx.current_pseudocode = result
            agent_ctx.has_created_initial = True
            self.logger.info(f"Created initial pseudocode, reason: {reason}")
            
        elif operation == "get_source_code":
            if file_path:
                source_code = self._get_source_code(agent_ctx.repo_name, file_path)
                agent_ctx.context += f"=== Source code of {file_path} ===\n{source_code if source_code else 'Not available'}\n"
                self.logger.info(f"Retrieved source code for: {file_path}, reason: {reason}")
            else:
                self.logger.error("get_source_code: file_path not provided, skipped")
                agent_ctx.error_log.append(f"Unknown filepath in retrieve: {file_path}")
                    
        elif operation == "revise":
            # Check if create has been called
            if not agent_ctx.has_created_initial:
                self.logger.warning("revise called before create - calling create first")
                create_op = {"operation": "create", "reason": "Forced create before revise"}
                self._execute_operation(create_op, agent_ctx)
            
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
            self.logger.info(f"Revised pseudocode, reason: {reason}")
                
        elif operation == "finish":
            self.logger.info(f"Operation: finish, reason: {reason}")
                
        elif operation == "review":
            review_result = self._call_review(agent_ctx)
            agent_ctx.suggestions.extend(review_result.get("suggestions", []))
            agent_ctx.review_scores = review_result.get("scores", {"conciseness": 0, "logic": 0, "specificity": 0})
            self.logger.info(f"Review scores: conciseness={agent_ctx.review_scores['conciseness']}, logic={agent_ctx.review_scores['logic']}, specificity={agent_ctx.review_scores['specificity']}")
            self.logger.info(f"reason: {reason}")
                
        else:
            self.logger.error(f"Unknown operation: {operation}, skipped")
            agent_ctx.error_log.append(f"Unknown operation: {operation}, skipped")

    def create_pseudocode_with_agent(
        self,
        repo_name: str,
        paper_title: str,
        paper_abstract: str,
        repo_structure: Dict,
        initial_pseudocode: str = "",
        max_steps: int = None,
        hard_code_revise: bool = False,
        max_rounds_without_revise: int = 3,
        last_round_revise: bool = True,
    ) -> tuple:
        """
        Create pseudocode using agent-based approach with plan-based execution.
        
        Args:
            repo_name: Repository name
            paper_title: Title of the paper
            paper_abstract: Abstract of the paper
            repo_structure: Repository structure as a nested dict
            initial_pseudocode: Optional initial pseudocode (can be empty)
            max_steps: Maximum number of agent steps
            hard_code_revise: If True, force call revise after max_rounds_without_revise rounds without revise
            max_rounds_without_revise: Maximum rounds without revise before forcing (default: 3)
            last_round_revise: If True, force revise in last round
            
        Returns:
            Tuple of (final_pseudocode, initial_pseudocode) where initial_pseudocode is the first created pseudocode before any revision
        """
        # Build AgentContext
        agent_ctx = AgentContext(
            repo_name=repo_name,
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            repo_structure=repo_structure,
            current_pseudocode=initial_pseudocode,
            memory=[],
            suggestions=[],
            context="",
            error_log=[],
            review_scores={"conciseness": 0, "logic": 0, "specificity": 0},
            has_created_initial=bool(initial_pseudocode)
        )
        
        max_steps = max_steps or self.default_max_steps
        
        # Track rounds since last revise
        rounds_since_last_revise = 0
        
        # Store the first created pseudocode (before any revision)
        first_created_pseudocode = ""
        
        self.logger.info(f"Starting agent-based pseudocode creation (max {max_steps} steps)")
        
        for step in range(max_steps):
            self.logger.info(f"Agent step {step + 1}/{max_steps}")

            # Check memory compression before building prompt
            self._check_memory_compression(agent_ctx)

            # Format the prompt
            prompt = AGENT_OPERATE_PROMPT.format(
                repo_name=agent_ctx.repo_name,
                paper_title=agent_ctx.paper_title,
                paper_abstract=agent_ctx.paper_abstract,
                repo_structure=self._format_repo_structure(agent_ctx.repo_structure),
                current_pseudocode=agent_ctx.current_pseudocode if agent_ctx.current_pseudocode else "(No pseudocode yet - use create operation)",
                memory=self._format_memory(agent_ctx.memory),
                suggestions="\n".join(agent_ctx.suggestions) if agent_ctx.suggestions else "No suggestion yet.",
                context=agent_ctx.context if agent_ctx.context else "No context retrieved yet."
            )
            
            # Get agent's plan
            response = self.chat_agent.remote_chat(prompt, temperature=0)
            
            # Parse the response
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
                reason = op.get("reason", "")
                
                # Record operation to memory
                memory_entry = {"operation": op_type, "reason": reason}
                
                # Check for finish
                if op_type == "finish":
                    self.logger.info("Agent decided to finish")
                    # Store final context and suggestions for finish
                    if agent_ctx.context or agent_ctx.suggestions:
                        memory_entry["context"] = agent_ctx.context
                        memory_entry["suggestions"] = agent_ctx.suggestions.copy()
                    agent_ctx.memory.append(memory_entry)
                    break
                
                # Execute operation
                self._execute_operation(op, agent_ctx)
                
                # Add complete operation results to memory entry
                if op_type == "create":
                    # Store complete pseudocode result
                    memory_entry["pseudocode"] = agent_ctx.current_pseudocode
                    memory_entry["result"] = agent_ctx.current_pseudocode
                    # Save first created pseudocode for return value
                    if not first_created_pseudocode:
                        first_created_pseudocode = agent_ctx.current_pseudocode
                        
                elif op_type == "get_source_code":
                    # Store complete file path and retrieved content
                    file_path = op.get('file_path', 'unknown')
                    source_result = agent_ctx.context.split(f"=== Source code of {file_path} ===")[-1] if f"=== Source code of {file_path} ===" in agent_ctx.context else "Not available"
                    memory_entry["file_path"] = file_path
                    memory_entry["source_code_result"] = source_result
                    memory_entry["result"] = source_result
                    
                elif op_type == "revise":
                    # Store complete revised pseudocode
                    memory_entry["pseudocode"] = agent_ctx.current_pseudocode
                    memory_entry["result"] = agent_ctx.current_pseudocode
                    
                elif op_type == "review":
                    # Store complete review suggestions and scores
                    memory_entry["suggestions"] = agent_ctx.suggestions.copy()
                    memory_entry["review_scores"] = agent_ctx.review_scores.copy()
                    memory_entry["result"] = f"Scores: {agent_ctx.review_scores}, Suggestions: {agent_ctx.suggestions}"
                
                agent_ctx.memory.append(memory_entry)
                
                # Track revise operations
                if op_type == "revise":
                    rounds_since_last_revise = 0
                else:
                    rounds_since_last_revise += 1

            # Check if we need to force revise
            if hard_code_revise and rounds_since_last_revise >= max_rounds_without_revise:
                self.logger.info(f"Force calling revise after {rounds_since_last_revise} rounds without revise")
                force_revise_op = {"operation": "revise", "reason": "Forced revise after max rounds without revise"}
                self._execute_operation(force_revise_op, agent_ctx)
                force_memory_entry = {
                    "operation": "revise", 
                    "reason": "Forced revise (hard_code)",
                    "pseudocode": agent_ctx.current_pseudocode,
                    "result": agent_ctx.current_pseudocode
                }
                agent_ctx.memory.append(force_memory_entry)
                rounds_since_last_revise = 0
                continue

            if step + 1 == max_steps:
                if last_round_revise and rounds_since_last_revise > 0:
                    self.logger.info("Last round does not include revise, force calling revise")
                    force_revise_op = {"operation": "revise", "reason": "Forced revise in last round"}
                    self._execute_operation(force_revise_op, agent_ctx)
                    force_memory_entry = {
                        "operation": "revise", 
                        "reason": "Forced revise (last round)",
                        "pseudocode": agent_ctx.current_pseudocode,
                        "result": agent_ctx.current_pseudocode
                    }
                    agent_ctx.memory.append(force_memory_entry)
                    rounds_since_last_revise = 0
            
            # Check if finished
            if any(m.get("operation") == "finish" for m in agent_ctx.memory[-len(plan):] if plan):
                self.logger.info(f"Planning agent decided to finish at step {step}")
                break
                
        self.logger.info(f"Agent-based creation completed after {step + 1} steps")
        
        # Return both final pseudocode and the first created pseudocode (initial)
        return agent_ctx.current_pseudocode, first_created_pseudocode
