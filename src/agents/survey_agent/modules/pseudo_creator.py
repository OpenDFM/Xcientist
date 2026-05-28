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
from utils.repo_utils import format_repo_structure


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

[OPERATION REQUIREMENTS - CRITICAL]
1. You MUST call "get_source_code" at least 3-5 times before calling "create" to understand the actual code implementation
2. You MUST prioritize reading actual source code files over relying on abstract - the abstract is just context, NOT the source of truth
3. For each paper, identify the 3-5 most important source files (main entry point, core algorithms, key utilities) and read them
4. "create" should ONLY be called AFTER you have retrieved and understood the actual source code
5. You MUST call "revise" operation at least once every 3 rounds to improve the pseudocode based on source code findings

[Priority File Selection]
When selecting files to read, prioritize in this order:
1. Main entry points (e.g., main.py, app.py, __main__.py, cli.py)
2. Core algorithm files (e.g., agent.py, solver.py, model.py, engine.py)
3. Key utility files that implement important logic
4. Configuration files that reveal the architecture
5. README and docs for architectural context

---

Your task is to plan a sequence of operations to create and refine the pseudocode.

Available operations:
1. "get_source_code": Query source code of a specific file by providing its path - USE THIS EXTENSIVELY to understand actual implementation
2. "create": Create a new pseudocode based on ACTUAL SOURCE CODE you retrieved - NOT from abstract alone
3. "revise": Modify the current pseudocode based on the retrieved source code context and suggestion
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
- For "get_source_code", include "file_path" with the actual path from repo_structure
- For "create", "revise", or "finish", no file_path needed
- Include at least 1 operation, up to 3 operations per plan
- You MUST call "get_source_code" multiple times to retrieve key source files before "create"
- "review" operation does not require file_path
- Prioritize reading source code files over creating pseudocode from abstract

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
        self.config = config
        self.repo_cache_path = repo_cache_path
        self.default_max_steps = 20
        self.memory_token_threshold = 50000  # Threshold for memory compression
        self.use_memory_result_uplimit = True
        self.memory_result_max_length = 8000  # Max characters for memory results to keep in memory display
        self.agent_source_code_max_read_lines = 700
        
        # Initialize tiktoken encoding for token counting
        try:
            self.encodings = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.logger.warning("Failed to load tiktoken encoding, using character-based estimation")
            self.encodings = None
        
        self.read_readme_first = True  # Flag to read README before first agent planning round
    
    def _find_readme_path(self, repo_structure: dict) -> Optional[str]:
        """
        Find README file path from repo_structure. Prioritize root directory, then search others.
        
        Args:
            repo_structure: Repository structure as a nested dict
            
        Returns:
            Relative path to README file, or None if not found
        """
        readme_patterns = {"readme.md", "readme.rst", "readme.txt", "readme"}
        
        def search(node: dict, current_path: str = "") -> Optional[str]:
            for key in node:
                lower_key = key.lower()
                if lower_key in readme_patterns:
                    value = node[key]
                    if isinstance(value, dict) and value.get("_is_file"):
                        return os.path.join(current_path, key) if current_path else key
                value = node[key]
                if isinstance(value, dict) and not value.get("_is_file"):
                    new_path = os.path.join(current_path, key) if current_path else key
                    result = search(value, new_path)
                    if result:
                        return result
            return None
        
        # First check root directory
        for key in repo_structure:
            lower_key = key.lower()
            if lower_key in readme_patterns:
                value = repo_structure[key]
                if isinstance(value, dict) and value.get("_is_file"):
                    self.logger.info(f"Found README in root: {key}")
                    return key
        
        # Search in subdirectories
        return search(repo_structure)
    
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
    
    def _truncate_text(self, text: str, max_length: int, suffix: str = "(too long, truncated)...") -> str:
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
                    pseudocode = self._truncate_text(pseudocode, self.memory_result_max_length)
                lines.append(f"   Pseudocode: {pseudocode}")
                
            elif op_type == "get_source_code":
                # Show file path and truncated source code
                file_path = op.get('file_path', 'unknown')
                source_code = op.get('source_code_result', op.get('result', ''))
                if self.use_memory_result_uplimit:
                    source_code = self._truncate_text(source_code, self.memory_result_max_length)
                lines.append(f"   File: {file_path}")
                lines.append(f"   Source: {source_code}")
                
            elif op_type == "revise":
                # Show truncated pseudocode
                pseudocode = op.get('pseudocode', op.get('result', ''))
                if self.use_memory_result_uplimit:
                    pseudocode = self._truncate_text(pseudocode, self.memory_result_max_length)
                lines.append(f"   Revised to: {pseudocode}")
                
            elif op_type == "review":
                # Show suggestions and scores
                suggestions = op.get('suggestions', [])
                scores = op.get('review_scores', {})
                if suggestions:
                    suggestions_text = "; ".join(suggestions)  # Limit to 5 suggestions
                    if self.use_memory_result_uplimit:
                        suggestions_text = self._truncate_text(suggestions_text, self.memory_result_max_length // 2)
                    lines.append(f"   Suggestions: {suggestions_text}")
                if scores:
                    lines.append(f"   Scores: conciseness={scores.get('conciseness', 0)}, logic={scores.get('logic', 0)}, specificity={scores.get('specificity', 0)}")
                    
            elif op_type == "finish":
                # Show context and suggestions
                context = op.get('context', '')
                reason = op.get('reason', '')
                suggestions = op.get('suggestions', [])
                if context and self.use_memory_result_uplimit:
                    context = self._truncate_text(context, self.memory_result_max_length)
                if context:
                    lines.append(f"   Final Context: {context}")
                if suggestions:
                    suggestions_text = "; ".join(suggestions[:5])
                    if self.use_memory_result_uplimit:
                        suggestions_text = self._truncate_text(suggestions_text, self.memory_result_max_length // 2)
                    lines.append(f"   Final Suggestions: {suggestions_text}")
                if reason:
                    lines.append(f"   Reason: {reason}")
            
            else:
                # Generic fallback
                result = op.get('result', '')
                if result:
                    if self.use_memory_result_uplimit:
                        result = self._truncate_text(result, self.memory_result_max_length)
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
            repo_structure=format_repo_structure(agent_ctx.repo_structure),
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
    
    def _execute_operation(self, op: dict, agent_ctx: AgentContext) -> tuple:
        """
        Execute a single operation and return the complete memory_entry.
        
        Returns:
            Tuple of (op_type, memory_entry) where memory_entry is ready to append to agent_ctx.memory
        """
        operation = op.get("operation", "")
        file_path = op.get("file_path", "")
        reason = op.get("reason", "")
        
        # Initialize memory_entry with operation and reason
        memory_entry = {"operation": operation, "reason": reason}
        
        if operation == "create":
            # Create initial pseudocode from scratch
            create_prompt = CREATE_PROMPT.format(
                paper_title=agent_ctx.paper_title,
                paper_abstract=agent_ctx.paper_abstract,
                repo_structure=format_repo_structure(agent_ctx.repo_structure),
                context=agent_ctx.context if agent_ctx.context else "No additional context retrieved yet. Create based on paper info and repo structure."
            )
            
            result = self.chat_agent.remote_chat_with_retry(create_prompt, temperature=0)
            agent_ctx.current_pseudocode = result
            agent_ctx.has_created_initial = True
            self.logger.info(f"Created initial pseudocode, reason: {reason}")
            memory_entry["pseudocode"] = result

            agent_ctx.context = ""
            agent_ctx.suggestions = []
            
        elif operation == "get_source_code":
            if file_path:
                source_code = self._get_source_code(agent_ctx.repo_name, file_path)
                agent_ctx.context += f"=== Source code of {file_path} ===\n{source_code if source_code else 'Not available'}\n"
                self.logger.info(f"Retrieved source code for: {file_path}, reason: {reason}")
                memory_entry["file_path"] = file_path
                memory_entry["source_code_result"] = source_code if source_code else "Not available: file not exist or read failed"
                memory_entry["result"] = f"Retrieved source code for {file_path}: {memory_entry['source_code_result']}"
            else:
                self.logger.error("get_source_code: file_path not provided, skipped")
                agent_ctx.error_log.append(f"Unknown filepath in retrieve: {file_path}")
                memory_entry["file_path"] = "file_path not provided corrctly"
                memory_entry["source_code_result"] = "file_path not provided"
                memory_entry["result"] = f"Retrieved source code for {file_path}: {memory_entry['source_code_result']}"
                    
        elif operation == "revise":
            # Check if create has been called
            if not agent_ctx.has_created_initial:
                self.logger.warning("revise called before create - calling create first")
                create_op = {"operation": "create", "reason": "Forced create before revise"}
                create_type, create_memory_entry = self._execute_operation(create_op, agent_ctx)
                agent_ctx.memory.append(create_memory_entry)
                # Note: first_created_pseudocode should be updated by the caller if needed
            
            modify_prompt = MODIFY_PROMPT.format(
                paper_title=agent_ctx.paper_title,
                paper_abstract=agent_ctx.paper_abstract,
                repo_structure=format_repo_structure(agent_ctx.repo_structure),
                final_pseudocode=agent_ctx.current_pseudocode,
                context=agent_ctx.context,
                suggestions="\n".join(agent_ctx.suggestions) if agent_ctx.suggestions else "No suggestion yet.",
                reason=reason,
            )
            result = self.chat_agent.remote_chat_with_retry(modify_prompt, temperature=0)
            agent_ctx.current_pseudocode = result
            memory_entry["pseudocode"] = result
            memory_entry["result"] = f"Revised pseudocode: {memory_entry['pseudocode']}"
            self.logger.info(f"Revised pseudocode, reason: {reason}")

            agent_ctx.context = ""
            agent_ctx.suggestions = []
                
        elif operation == "finish":
            self.logger.info(f"Operation: finish, reason: {reason}")
            self.logger.info("Agent decided to finish")
            # Store final context and suggestions for finish
            if agent_ctx.context or agent_ctx.suggestions:
                memory_entry["context"] = agent_ctx.context
                memory_entry["suggestions"] = agent_ctx.suggestions.copy()
                
        elif operation == "review":
            review_result = self._call_review(agent_ctx)
            suggestions = review_result.get("suggestions", [])
            scores = review_result.get("scores", {"conciseness": 0, "logic": 0, "specificity": 0})
            agent_ctx.suggestions.extend(suggestions)
            agent_ctx.review_scores = scores
            self.logger.info(f"Review scores: conciseness={agent_ctx.review_scores['conciseness']}, logic={agent_ctx.review_scores['logic']}, specificity={agent_ctx.review_scores['specificity']}")
            self.logger.info(f"reason: {reason}")

            memory_entry["scores"] = scores
            memory_entry["suggestions"] = suggestions.copy()
            memory_entry["result"] = f"Scores: {memory_entry['scores']}, Suggestions: {memory_entry['suggestions']}"
                
        else:
            self.logger.error(f"Unknown operation: {operation}, skipped")
            agent_ctx.error_log.append(f"Unknown operation: {operation}, skipped")
            memory_entry["result"] = f"Unknown operation: {operation}, skipped"

        return operation, memory_entry

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
        
        # Initialize: Read README before first planning round if enabled
        if self.read_readme_first:
            readme_path = self._find_readme_path(repo_structure)
            if readme_path:
                self.logger.info(f"Reading README file: {readme_path}")
                # Use _execute_operation to handle context and memory automatically
                readme_op = {"operation": "get_source_code", "file_path": readme_path, "reason": "Initial README read"}
                _, readme_memory_entry = self._execute_operation(readme_op, agent_ctx)
                agent_ctx.memory.append(readme_memory_entry)
                self.logger.info(f"README content added to context and memory")
            else:
                self.logger.info(f"No README file found in repository structure")
        
        for step in range(max_steps):
            self.logger.info(f"Agent step {step + 1}/{max_steps}")

            # Check memory compression before building prompt
            self._check_memory_compression(agent_ctx)

            # Format the prompt
            prompt = AGENT_OPERATE_PROMPT.format(
                repo_name=agent_ctx.repo_name,
                paper_title=agent_ctx.paper_title,
                paper_abstract=agent_ctx.paper_abstract,
                repo_structure=format_repo_structure(agent_ctx.repo_structure),
                current_pseudocode=agent_ctx.current_pseudocode if agent_ctx.current_pseudocode else "(No pseudocode yet - use create operation)",
                memory=self._format_memory(agent_ctx.memory),
                suggestions="\n".join(agent_ctx.suggestions) if agent_ctx.suggestions else "No suggestion yet.",
                context=agent_ctx.context if agent_ctx.context else "No context retrieved yet."
            )

            # Get agent's plan
            try:
                plan = self.chat_agent.remote_chat_with_retry(
                    prompt=prompt, 
                    validate_fn=self._validate_plan, 
                    max_retry= self.config.ModuleInfo.CodeAnalysis.PseudoCreator.planner_max_retry, 
                    temperature=self.config.ModuleInfo.CodeAnalysis.PseudoCreator.planner_temperature
                )
            except Exception as e:
                # Escape any markup-like content in the exception to avoid Rich markup errors
                error_str = str(e).replace('[', r'\[').replace(']', r'\]')
                self.logger.error(f"Failed to parse agent response: {error_str}..., skipped this round")
                agent_ctx.error_log.append(f"error in parsing planning json: {e}")
                continue
            
            # Execute each operation in the plan
            for op in plan:
                try:
                    # Execute operation and get complete memory_entry directly
                    op_type, memory_entry = self._execute_operation(op, agent_ctx)
                    agent_ctx.memory.append(memory_entry)
                    
                    # Check for finish - handled in _execute_operation, just log and break here
                    if op_type == "finish":
                        self.logger.info("Agent decided to finish")
                        break
                except Exception as e:
                    # If an operation fails, log error and skip this operation but continue with next
                    self.logger.error(f"Failed to execute operation '{op.get('operation', 'unknown')}': {e}. Skipping this operation.")
                    agent_ctx.error_log.append(f"Operation '{op.get('operation', 'unknown')}' failed: {e}")
                    continue  # Continue with next operation in plan
                
                # Track revise operations
                if op_type == "revise":
                    rounds_since_last_revise = 0
                else:
                    rounds_since_last_revise += 1
                
                # Track first created pseudocode for return value
                if op_type == "create" and not first_created_pseudocode:
                    first_created_pseudocode = memory_entry.get("pseudocode", "")
                
                # Also check if create was called internally by revise operation
                # by looking at the memory entries after a revise operation
                if op_type == "revise" and not first_created_pseudocode:
                    # Find the most recent create operation in memory after the revise call
                    for memory_item in reversed(agent_ctx.memory):
                        if memory_item.get("operation") == "create" and memory_item.get("pseudocode"):
                            first_created_pseudocode = memory_item.get("pseudocode", "")
                            break

            # Check if we need to force revise
            if hard_code_revise and rounds_since_last_revise >= max_rounds_without_revise:
                self.logger.info(f"Force calling revise after {rounds_since_last_revise} rounds without revise")
                force_revise_op = {"operation": "revise", "reason": "Forced revise after max rounds without revise"}
                _, force_memory_entry = self._execute_operation(force_revise_op, agent_ctx)
                force_memory_entry["reason"] = "Forced revise (hard_code)"
                agent_ctx.memory.append(force_memory_entry)
                rounds_since_last_revise = 0
                continue

            if step + 1 == max_steps:
                if last_round_revise and rounds_since_last_revise > 0:
                    self.logger.info("Last round does not include revise, force calling revise")
                    force_revise_op = {"operation": "revise", "reason": "Forced revise in last round"}
                    _, force_memory_entry = self._execute_operation(force_revise_op, agent_ctx)
                    force_memory_entry["reason"] = "Forced revise (last round)"
                    agent_ctx.memory.append(force_memory_entry)
                    rounds_since_last_revise = 0
            
            # Check if finished
            if any(m.get("operation") == "finish" for m in agent_ctx.memory[-len(plan):] if plan):
                self.logger.info(f"Planning agent decided to finish at step {step}")
                break
        
        # self.logger.info(f"[Memory Debug] Final memory after agent execution:\n{self._format_memory(agent_ctx.memory)}")
        self.logger.info(f"Agent-based creation completed after {step + 1} steps")

        if not agent_ctx.current_pseudocode or not first_created_pseudocode:
            self.logger.error(f"Agents fail to generate pseudocode for {agent_ctx.repo_name}")
            raise ValueError(f"Agents fail to generate pseudocode for {agent_ctx.repo_name}")
        
        # Return both final pseudocode and the first created pseudocode (initial)
        return agent_ctx.current_pseudocode, first_created_pseudocode

    def _validate_plan(self, response: str, info_dict: dict = None):
        result = extract_json(response)
        if not isinstance(result, dict):
            raise ValueError(f"Response is not a dict: {type(result)}")
        if "plan" not in result:
            raise ValueError("Missing 'plan' field in response")
        plan = result.get("plan")
        if not isinstance(plan, list):
            raise ValueError("'plan' field is not a list")
        for op in plan:
            if "operation" not in op:
                raise ValueError("Each operation must have an 'operation' field")
        return True, plan
