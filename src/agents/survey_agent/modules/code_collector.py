import re
import requests
from typing import List, Optional, Dict
import os
import subprocess
import sys
import json

from typing import Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.rich_logger import get_logger
# from utils.rich_logger import get_logger
from utils.api_call import ChatAgent
from modules.work_collector import WorkCollector
from modules.pseudo_reviser import PseudoReviser
from modules.code_report_generator import CodeReportGenerator
from utils.utils import extract_json
import hydra


MAIN_AND_CORE_FILTER = """
You are an assistant that classifies whether a file chunk is a *core implementation* for the main algorithm/architecture
of the repository of a paper titled {paper_title}. 

Context:
- paper title and abstract:
{paper_title}
{paper_abstract}

- repository structure:
{repo_structure}
- file_path: {file_path}
- file chunk:
{chunk_text}
(chunk end)

TASK 1: Classify Core Code
Rules for is_core:
- If the chunk contains core algorithm implementations, model definitions (class names like Model, Transformer, UNet, forward/ call methods),
  important data pipeline steps, or central algorithmic logic => is_core true.
- If it is helper utils, CI configs, docs, tests, or setup scripts => likely not core.

Scoring Standard for "score" (1-10):
[Score 1-2] Pure Boilerplate & Config (is_core: false)
Pure configurations, documentation, environment setup, CI/CD scripts, basic `argparse`, or trivial dependency management (e.g., requirements.txt, Dockerfiles, README, completely empty init files).
[Score 3-4] Generic Utilities & Helpers (is_core: false)
Generic software engineering tools that do not reflect the paper's algorithm. Examples: logging (Logger), checkpoint saving/loading, visualization/plotting scripts, standard file I/O, or standard metric calculations (e.g., generic Accuracy/MSE).
[Score 5-6] Standard Pipelines & Training Loops (is_core: false)
Standard Deep Learning skeletons. Examples: standard PyTorch/TensorFlow training loops (e.g., standard `for epoch in epochs`, basic optimizer steps), generic `Dataset`/`DataLoader` definitions, or standard text/image preprocessing. While necessary to run the code, they lack the paper's specific novelties.
[Score 7-8] Key Components & Novel Sub-modules (is_core: true)
Code implementing specific, secondary innovations from the paper. Examples: custom loss functions, novel data sampling/augmentation strategies, customized attention mechanisms, or specific sub-network blocks (e.g., a custom Adapter or ResBlock variation specifically proposed in the paper). 
[Score 9-10] Absolute Core & Main Algorithm (is_core: true)
The absolute "soul" of the paper. Main model class definitions (e.g., `class ProposedModel`), the complete `forward` or `call` methods that tie the novel architecture together, or direct code translations of the paper's primary mathematical equations and pseudocode. If this chunk is removed, the paper's main contribution ceases to exist.

TASK 2: Identify Main Entry Point Code (is_main_code)
This is a SEPARATE classification from is_core. Even if is_core is false, a file can be is_main_code if it contains:
- Functions that drive the main execution flow: main()
- Scripts that orchestrate the entire pipeline
- The skeleton of the pipeline

SCORING (main_score):
Assign a `main_score` from 0 to 10 indicating how central this file is to the overall execution flow:
- 9-10: Unambiguous top-level main entry point (orchestrates the entire pipeline).
- 7-8: High-level orchestrator for a major subsystem, but maybe not the top-most script.
- 0-6: Sub-modules, core algorithm implementations (without orchestration), utilities, data loaders, etc. (For these, is_main_code MUST be false).

IMPORTANT: 
is_main_code is about WHETHER THIS FILE IS AN ENTRY POINT that orchestrates the algorithm execution or is the skeleton script of the pipeline, NOT about whether it contains the core algorithm implementation.
I will use is_main_code True file to build the skeleton of the repository pseudocode. 

Output:
- Return EXACTLY a JSON object with keys for TASK 1 and TASK 2:
  - is_core: boolean
  - score: between 0 to 10 (core level)
  - reason: short string explanation for core classification (one or two sentences)
  - is_main_code: boolean (separate from is_core)
  - main_score: between 0 to 10 (main level)
  - main_reason: short string explanation for main code classification (one or two sentences)

- Generate JSON directly without any other things.
"""

CORE_FILTER = """
You are an assistant that classifies whether a file chunk is a *core implementation* for the main algorithm/architecture
of the repository of a paper titled {paper_title}. 

Context:
- paper title and abstract:
{paper_title}
{paper_abstract}

- repository structure:
{repo_structure}
- file_path: {file_path}
- file chunk:
{chunk_text}
(chunk end)

TASK: Classify Core Code
Rules for is_core:
- If the chunk contains core algorithm implementations, model definitions (class names like Model, Transformer, UNet, forward/ call methods),
  important data pipeline steps, or central algorithmic logic => is_core true.
- If it is helper utils, CI configs, docs, tests, or setup scripts => likely not core.

Scoring Standard for "score" (1-10):
[Score 1-2] Pure Boilerplate & Config (is_core: false)
Pure configurations, documentation, environment setup, CI/CD scripts, basic `argparse`, or trivial dependency management (e.g., requirements.txt, Dockerfiles, README, completely empty init files).
[Score 3-4] Generic Utilities & Helpers (is_core: false)
Generic software engineering tools that do not reflect the paper's algorithm. Examples: logging (Logger), checkpoint saving/loading, visualization/plotting scripts, standard file I/O, or standard metric calculations (e.g., generic Accuracy/MSE).
[Score 5-6] Standard Pipelines & Training Loops (is_core: false)
Standard Deep Learning skeletons. Examples: standard PyTorch/TensorFlow training loops (e.g., standard `for epoch in epochs`, basic optimizer steps), generic `Dataset`/`DataLoader` definitions, or standard text/image preprocessing. While necessary to run the code, they lack the paper's specific novelties.
[Score 7-8] Key Components & Novel Sub-modules (is_core: true)
Code implementing specific, secondary innovations from the paper. Examples: custom loss functions, novel data sampling/augmentation strategies, customized attention mechanisms, or specific sub-network blocks (e.g., a custom Adapter or ResBlock variation specifically proposed in the paper). 
[Score 9-10] Absolute Core & Main Algorithm (is_core: true)
The absolute "soul" of the paper. Main model class definitions (e.g., `class ProposedModel`), the complete `forward` or `call` methods that tie the novel architecture together, or direct code translations of the paper's primary mathematical equations and pseudocode. If this chunk is removed, the paper's main contribution ceases to exist.

Output:
- Return EXACTLY a JSON object with keys for TASK 1 and TASK 2:
  - is_core: boolean
  - score: between 0 to 10 (core level)
  - reason: short string explanation for core classification (one or two sentences)
- Output json example:
  {{
    "is_core": False,
    "score": 5,
    "reason": ".....",
  }}
- Generate JSON directly without any other things.
"""

MAIN_FILTER = """
You are an assistant that classifies whether a file chunk is the main code for the algorithm/architecture
of the repository of a paper titled {paper_title}. 

Context:
- paper title and abstract:
{paper_title}
{paper_abstract}

- repository structure:
{repo_structure}
- file_path: {file_path}
- file chunk:
{chunk_text}
(chunk end)


TASK: Identify Main Entry Point Code (is_main_code) and Assign a Score
A file can be is_main_code if it contains:
- Functions that drive the main execution flow: main()
- Scripts that orchestrate the entire pipeline
- The skeleton of the pipeline

IMPORTANT STRICTNESS REQUIREMENT (CRITICAL):
You must be CAUTIOUS when setting `is_main_code` to true. 
is_main_code is about WHETHER THIS FILE IS AN ENTRY POINT that orchestrates the algorithm execution or is the skeleton script of the pipeline, NOT about whether it contains the core algorithm implementation.
I will concatenate ALL `is_main_code=True` files at once to build the SKELETON of the repository pseudocode. Too many files or including sub-modules/utilities will exceed the context window. 
- Only the absolute top-level entry points (the "conductor" of the orchestra, e.g., main.py, train.py) should be marked True.
- When in doubt, or if the file is just a module called by the main script, set `is_main_code` to false.
- Be cautious about marking the data-collect code and evaluation code as main code because sometimes they are unrelated to the core pipeline of repo.

SCORING (main_score):
Assign a `main_score` from 0 to 10 indicating how central this file is to the overall execution flow:
- 9-10: Unambiguous top-level main entry point (orchestrates the entire pipeline).
- 7-8: High-level orchestrator for a major subsystem, but maybe not the top-most script.
- 0-6: Sub-modules, core algorithm implementations (without orchestration), utilities, data loaders, etc. (For these, is_main_code MUST be false).

Output:
- Return EXACTLY a JSON object with the following keys:
  - is_main_code: boolean (Extremely strict! Only true for the absolute main skeleton/entry point)
  - main_score: integer (0-10, representing the confidence and centrality of the file)
  - main_reason: short string explanation for the classification and score (one or two sentences)

- Generate JSON directly without any markdown formatting (do not use ```json) or other text.
"""

PSEUDOCODE_WRITER = """
Analyze the following code file from a research paper repository.

[File Info]
- File: {file_path}

- Core File
Core file means the file contains core algorithm implementations, model definitions, important data pipeline steps, or central logic.
Core score measure the core level of the file.
You need to focus on the key points of the core files and reflect them in the analysis and pseudocode
-- is_core_file: {is_core_file}
-- Core Score: {core_score}/10 (higher = more central to paper's contribution)
-- Reason for the score: {reason}

- Main File
Main file means the file orchestrates the algorithm execution or is the skeleton script of the pipeline.
The pseudocode for main file will serve as the foundation pseudocode when building pseudocode for the whole repo.
You need to clearly present the logic and structure of the main code in pseudocode and analysis.
-- is_main_file: {is_main_code}
-- Main Score: {main_score}/10
-- Reason for is_main_file: {main_reason}


[Paper Context]
Title: {paper_title}
Abstract: {paper_abstract}

[Code]
{file_content}

---

Generate a concise analysis with:
1. One-sentence summary of what this file does
2. Pseudocode (algorithm-style, focus on core logic, skip boilerplate)
3. Any additional insights you think are valuable (be concise)
"""

# Template for generating project pseudocode from main files
PROJECT_PSEUDOCODE_FROM_MAIN = """
You are an expert at analyzing research paper repositories and creating high-level pseudocode.

Your task is to create a project-level pseudocode framework based on the main entry point files of a repository.

[Paper Context]
Title: {paper_title}
Abstract: {paper_abstract}

[Repository Structure]
{repo_structure}

[Main Files Pseudocode and Analysis]
The following are the main entry point files, ordered by their main_score (higher = more central to the overall execution flow):
{main_files_pseudocode}

---

TASK: Create a project-level pseudocode framework

Based on the main files above, create a comprehensive project pseudocode that:
1. Shows the overall execution flow and pipeline structure
2. Identifies key components and their interactions
3. Serves as a foundation that can be refined with core implementation details

Requirements:
- Structure the pseudocode to reflect the main execution flow.
- You should build a well structed pseudocode with clear logic rather than simply listing pseudocode form main file.
- Use clear section divisions (e.g., for different phases like initialization, training, evaluation)
- The section number should be no more than 5.
- Do NOT include actual code - use algorithm-style pseudocode
- You are encourage to annotate file name (source code or called code) parsed by <> besides the corresponding pseudocode.
- You can also refine the analysis after pseudocode but must keep conciseness.

Output format:
Provide a clear, structured pseudocode with section headers, explanations. and concise analysis.
"""

# Template for refining project pseudocode with core files (batch mode)
REFINE_PROJECT_PSEUDOCODE_WITH_CORE = """
You are an expert at analyzing research paper repositories and refining pseudocode.

Your task is to refine the existing project pseudocode by incorporating core implementation details.

[Paper Context]
Title: {paper_title}
Abstract: {paper_abstract}

[Repository Structure]
{repo_structure}

[Current Project Pseudocode Framework]
{current_pseudocode}

[Core Files Batch - Pseudocode and Analysis]
The following is a batch of core implementation files (with their pseudocode and analysis):
{core_files_pseudocode}

---

TASK: Refine the project pseudocode

Incorporate the core implementation details from the above files into the existing project pseudocode:
1. Add specific algorithm implementations where relevant
2. Update function/class definitions with actual logic from core files
3. Fill in any placeholder sections with concrete details
4. Maintain the overall structure while adding depth

Requirements:
- Preserve the existing structure and flow
- Add meaningful implementation details from the core files
- Focus on the most important details that contribute to understanding the paper's contribution
- Keep the pseudocode at a reasonable length - if it gets too long, prioritize the most important additions
- You should build a well structed pseudocode with clear logic rather than simply listing pseudocode from core file.
- Use clear section divisions (e.g., for different phases like initialization, training, evaluation)
- You are encourage to annotate file name (source code or called code) parsed by <> besides the corresponding pseudocode.
- Keep the pseudocode concise. Only add necessary content to the existing pseudocode. The section number should be no more than 5.
- You can also refine the analysis after pseudocode but must keep conciseness.

Output format:
Provide the refined project pseudocode with the new details incorporated.
"""

class CodeCollector:
    def __init__(self, config):
        self.logger = get_logger("CodeCollector")
        self.pwc_api_base = "https://paperswithcode.com/api/v1"
        self.repo_cache_path = "./database/code_repo_cache"
        os.makedirs(self.repo_cache_path, exist_ok=True)

    def extract_code_links(self, paper_id: str, paper_markdown: str, valid_num: int = 2) -> List[str]:
        """
        Extracts GitHub repository links for a given paper using PapersWithCode API 
        and Markdown Regex parsing.
        """
        self.logger.info(f"Attempting to extract code links for Paper ID: {paper_id}")
        extracted_links = set()

        if paper_markdown:
            regex_links = self._get_links_from_markdown(paper_markdown)
            if regex_links:
                self.logger.info(f"Found {len(regex_links)} links via Markdown Regex.")
                extracted_links.update(regex_links)

        # Clean and deduplicate
        cleaned_links = {self._clean_github_url(url) for url in extracted_links if url}
        
        # Filter for valid GitHub repos (e.g., ignoring 'https://github.com/microsoft' without a repo name)
        valid_repos = [url for url in cleaned_links if len(url.strip("/").split("/")) >= 5]

        if not valid_repos:
            self.logger.warning(f"No valid GitHub links found for Paper ID: {paper_id}")
            
        valid_repos = [valid_repos[i] for i in range(min(valid_num,len(valid_repos)))]
        return valid_repos

    ### the paperwithcode api has broken
    def _get_links_from_paperswithcode(self, paper_id: str) -> List[str]:
        """
        Queries the PapersWithCode API using an ArXiv ID.
        """
        # Clean the ID in case it has 'arxiv:' prefix or version numbers (e.g., '2408.08464v1')
        clean_id = re.sub(r"(?i)^arxiv:?", "", paper_id).split('v')[0]
        
        try:
            # Step 1: Search for the paper to get the internal PWC ID
            search_url = f"{self.pwc_api_base}/papers/?arxiv_id={clean_id}"
            response = requests.get(search_url, timeout=10)
            
            if response.status_code != 200:
                return []
                
            data = response.json()
            if not data.get("results"):
                return []
                
            pwc_paper_id = data["results"][0]["id"]
            
            # Step 2: Fetch repositories associated with that PWC ID
            repo_url = f"{self.pwc_api_base}/papers/{pwc_paper_id}/repositories/"
            repo_response = requests.get(repo_url, timeout=10)
            
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                # Ensure it's marked as official if possible, but return all linked repos
                return [repo["url"] for repo in repo_data.get("results", []) if "github.com" in repo["url"]]
                
        except Exception as e:
            self.logger.error(f"PapersWithCode API failed for {paper_id}: {e}")
            
        return []

    def _get_links_from_markdown(self, markdown_text: str) -> List[str]:
        """
        Extracts GitHub URLs from raw markdown using Regex.
        Matches formats like https://github.com/user/repo
        
        Note: Content after # References section (case-insensitive) is excluded
        to avoid extracting GitHub links from the bibliography.
        """
        # Remove content after # References (case-insensitive)
        # Match # References or # REFERENCES or # references, etc.
        references_pattern = r"(?i)^#\s*references.*$"
        match = re.search(references_pattern, markdown_text, re.MULTILINE)
        
        if match:
            markdown_text = markdown_text[:match.start()]
            self.logger.info("Removed content after # References section before extracting links")
        
        # Pattern looks for github.com/ followed by two path segments (user/repo)
        # It handles optional trailing slashes but stops at whitespace, parentheses, or brackets.
        pattern = r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
        matches = re.findall(pattern, markdown_text)
        return matches

    def _clean_github_url(self, url: str) -> str:
        """
        Cleans trailing punctuation that might get caught in the regex from markdown parsing.
        """
        # Remove trailing punctuation like dots, commas, or closing brackets
        url = re.sub(r"[.,;)\]]+$", "", url.strip())
        
        # Strip .git extension if present
        if url.endswith(".git"):
            url = url[:-4]
            
        return url

    def _extract_repo_name(self, repo_url: str):
        repo_name = repo_url.rstrip("/").split("/")[-1]

        # remove .git if present
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        return repo_name


    def _clone_repo(self, repo_url: str, depth: int = 1, max_retry: int = 3) -> Optional[str]:
        """
        Clone a GitHub repo into base_dir.

        Args:
            repo_url: GitHub repository url
            base_dir: directory where repos are stored
            depth: git clone depth (default shallow clone)

        Returns:
            local repo path if success, otherwise None
        """
        repo_name = self._extract_repo_name(repo_url)

        target_path = os.path.join(self.repo_cache_path, repo_name)

        # skip if already cloned
        if os.path.exists(target_path):
            self.logger.info(f"Repo already exists: {target_path}")
            return target_path

        os.makedirs(self.repo_cache_path, exist_ok=True)

        retry = 0
        while retry <= max_retry:
            try:
                cmd = [
                    "git",
                    "clone",
                    "--depth",
                    str(depth),
                    repo_url,
                    target_path
                ]

                subprocess.run(cmd, check=True)

                self.logger.info(f"Cloned repo to {target_path}")
                return target_path

            except subprocess.CalledProcessError as e:
                retry += 1
                if retry <= max_retry:
                    self.logger.warning(f"Failed to clone repo {repo_url}: {e}, retrying for {retry}/{max_retry}")
                else:
                    self.logger.error(f"Failed to clone repo {repo_url}: {e}, max retry reached")

        return None

class CodeAnalyzer():
    def __init__(
            self, config, 
            code_collector: CodeCollector, 
            work_collector: WorkCollector
        ):
        self.chat_agent = ChatAgent(config)
        self.logger = get_logger("CodeAnalyzer")
        self.filter_code_only = True
        self.code_extensions = {".py", ".ipynb", ".yaml", ".yml", ".json", ".cfg", ".toml", ".sh"}
        self.ignore_dirs = {".git", "__pycache__", "node_modules", "build", "dist", ".idea", ".vscode"}
        self.ignore_filenames = {"__init__.py"}
        self.heur_keywords = {"train", "model", "main", "run", "inference", "predict", "forward"}
        self.code_collector = code_collector
        self.work_collector = work_collector
        self.MAX_FILE_SIZE = 10 * 1024 * 1024
        self.repo_cache_path = self.code_collector.repo_cache_path
        self.CHUNK_SIZE = 40000
        self.CHUNK_OVERLAP = 10000
        self.CORE_SCORE_THRESHOLD = 8
        self.chunk_file = False
        self.cache_result_json = True
        self.force_regenerate = False
        self.filter_in_steps = True
        

    def _list_repo_files(self, repo_name):
        repo_path = os.path.join(self.repo_cache_path, repo_name)
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"repo not found: {repo_path}")

        files = []
        for root, dirs, filenames in os.walk(repo_path):
            # filter out large irrelevant dirs
            dirs[:] = [d for d in dirs if d not in self.ignore_filenames]
            for fname in filenames:
                if (fname.endswith(tuple(self.code_extensions)) or not self.filter_code_only) and fname not in self.ignore_filenames:
                    full = os.path.join(root, fname)
                    files.append(full)
        return files

    def _read_file(self, path: str) -> Optional[str]:
        try:
            size = os.path.getsize(path)
            if size > self.MAX_FILE_SIZE:
                self.logger.warning(f"reading very large file: {path} ({size} bytes/maximum {self.MAX_FILE_SIZE}), chunking...")
                with open(path, "rb") as f:
                    return f.read(self.MAX_FILE_SIZE)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"[Error] cannot read {path}: {e}")
            return None

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        L = len(text)
        while start < L:
            end = start + self.CHUNK_SIZE
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= L:
                break
            start = end - self.CHUNK_OVERLAP  # overlap
        return chunks

    
    def scan_repo_structure(self, repo_name: str) -> Optional[dict]:
        """
        Scan a cloned repo in cache and return its file tree structure.

        Args:
            repo_name: repository name (folder name in cache)

        Returns:
            dict representing repo tree
        """

        repo_path = os.path.join(self.repo_cache_path, repo_name)

        if not os.path.exists(repo_path):
            self.logger.error(f"Repo not found in cache: {repo_path}")
            return None

        tree = {}

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            # relative path from repo root
            rel_path = os.path.relpath(root, repo_path)

            # navigate tree
            current = tree
            if rel_path != ".":
                for part in rel_path.split(os.sep):
                    current = current.setdefault(part, {})

            # add directories
            for d in dirs:
                current.setdefault(d, {})

            # add files
            for f in files:
                full_path = os.path.join(root, f)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    size = 0
                
                _, ext = os.path.splitext(f)
                
                current[f] = {
                    "_is_file": True,
                    "size": size,
                    "type": ext.lower() if ext else "file"
                }

        return tree

    def format_repo_structure(self, structure: dict) -> str:
        """
        Returns a string representing the repository structure like the 'tree' command.
        Files show only names.
        """
        lines = []

        def _build_tree_string(node: dict, prefix: str = ""):
            keys = sorted(node.keys())
            total_items = len(keys)
            
            for i, key in enumerate(keys):
                value = node[key]
                
                # The last element in the list needs different sign
                is_last = (i == total_items - 1)
                connector = "└── " if is_last else "├── "   
                lines.append(f"{prefix}{connector}{key}")
                
                if isinstance(value, dict) and value.get("_is_file"):
                    # file, stop recursion
                    if value.get("is_core_code"):
                        lines[-1] += " (core code)"
                    if value.get("is_main_code"):  # 缺少这个
                        lines[-1] += " (main code)"
                    pass
                else:
                    # dirctory, recurse
                    extension = "    " if is_last else "│   "
                    _build_tree_string(value, prefix + extension)

        _build_tree_string(structure)
        
        return "\n".join(lines)

    def build_mainfest(self, 
                       paper_id: str,
                       repo_url: str,
                       min_core_score: float = None,
                       per_file_score_method: str = "avg"
                       ):
        repo_name = self.code_collector._extract_repo_name(repo_url)

        if not self.force_regenerate:
            mainfest = self.load_cached_mainfest(repo_name)
            if self.validate_mainfest(mainfest):
                return mainfest
            else:
                self.logger.warning("mainfest invalid. Regenerate.")

        mainfest = self.identify_and_collect_core_code(paper_id, repo_url, repo_name, min_core_score, per_file_score_method)

        mainfest = self.generate_pseudocode(mainfest)

        if self.cache_result_json:
            self.cache_mainfest(mainfest)

        return mainfest
            
        
    def _validate_filter_result(self, input, info_dict: dict = None):
        input_dict = extract_json(input)
        keys = ["is_core", "score", "reason", "is_main_code","main_score", "main_reason"]
        for key in keys:
            if key not in input_dict.keys():
                raise ValueError(f"key {key} not in result original text: <{input}>")
        if not isinstance(input_dict["is_core"], bool) or not isinstance(input_dict["is_main_code"], bool):
            raise ValueError(f"key is_core or is_main_code not boolean")
        return input, True

    def _validate_filter_result_core(self, input, info_dict: dict = None):
        """Validation function for core filter results (step 1 of 2)."""
        input_dict = extract_json(input)
        keys = ["is_core", "score", "reason"]
        for key in keys:
            if key not in input_dict.keys():
                raise ValueError(f"key {key} not in result: original text: <{input}>")
        if not isinstance(input_dict["is_core"], bool):
            raise ValueError(f"key is_core not boolean")
        return input, True

    def _validate_filter_result_main(self, input, info_dict: dict = None):
        """Validation function for main filter results (step 2 of 2)."""
        input_dict = extract_json(input)
        keys = ["is_main_code", "main_reason", "main_score"]
        for key in keys:
            if key not in input_dict.keys():
                raise ValueError(f"key {key} not in result original text: <{input}>")
        if not isinstance(input_dict["is_main_code"], bool):
            raise ValueError(f"key is_main_code not boolean")
        return input, True

    def _filter_call_model_in_steps(self,
                                files_chunks: dict,
                                paper_title: str,
                                paper_abstract: str,
                                repo_structure: dict,
                                max_retry: int = 5,
                                ):
        all_chunks_prompts_core = []
        all_chunks_prompts_main = []

        for file_chunks in files_chunks:
            chunks = file_chunks["chunks"]
            rel_path = file_chunks["path"]
            for chunk in chunks:
                prompt_core = CORE_FILTER.format(
                    paper_title = paper_title,
                    paper_abstract = paper_abstract,
                    file_path = rel_path,
                    repo_structure = self.format_repo_structure(repo_structure),
                    chunk_text = chunk
                )
                all_chunks_prompts_core.append(prompt_core)

            for chunk in chunks:
                prompt_main = MAIN_FILTER.format(
                    paper_title = paper_title,
                    paper_abstract = paper_abstract,
                    file_path = rel_path,
                    repo_structure = self.format_repo_structure(repo_structure),
                    chunk_text = chunk
                )
                all_chunks_prompts_main.append(prompt_main)
        
        main_results = self.chat_agent.batch_remote_chat_with_retry(
            prompts = all_chunks_prompts_main, 
            validate_fn = self._validate_filter_result_main, 
            max_retry = max_retry,
            desc = "filtering out main code",
            temperature = 0
        )
        
        core_results = self.chat_agent.batch_remote_chat_with_retry(
            prompts = all_chunks_prompts_core, 
            validate_fn = self._validate_filter_result_core, 
            max_retry = max_retry,
            desc = "filtering out core code",
            temperature = 0
        )
        
        # Parse JSON strings into dictionaries
        results = []
        for core_result_str, main_result_str in zip(core_results, main_results):
            core_dict = extract_json(core_result_str)
            main_dict = extract_json(main_result_str)
            result = {
                "is_core": core_dict["is_core"], 
                "score": core_dict["score"], 
                "reason": core_dict["reason"],
                "is_main_code": main_dict["is_main_code"],
                "main_score": main_dict["main_score"],
                "main_reason": main_dict["main_reason"],
            }
            results.append(result)

        return results

    def _filter_call_model_one_step(self,
                                files_chunks: dict,
                                paper_title: str,
                                paper_abstract: str,
                                repo_structure: dict,
                                max_retry: int = 5,
                                ):
        
        all_chunks_prompts = []

        for file_chunks in files_chunks:
            chunks = file_chunks["chunks"]
            rel_path = file_chunks["path"]
            for chunk in chunks:
                prompt = MAIN_AND_CORE_FILTER.format(
                    paper_title = paper_title,
                    paper_abstract = paper_abstract,
                    file_path = rel_path,
                    repo_structure = self.format_repo_structure(repo_structure),
                    chunk_text = chunk
                )
                all_chunks_prompts.append(prompt)

        result_strings = self.chat_agent.batch_remote_chat_with_retry(
            prompts = all_chunks_prompts, 
            validate_fn = self._validate_filter_result, 
            max_retry = max_retry,
            desc = "filtering out main and core code",
            temperature = 0
        )
        
        # Parse JSON strings into dictionaries
        results = [extract_json(result_str) for result_str in result_strings]

        return results

        

    def identify_and_collect_core_code(self,
                                       paper_id: str, 
                                       repo_url: str,
                                       repo_name: str,
                                       min_core_score: float = None,
                                       per_file_score_method: str = "avg" , ## or max
                                       max_retry: int = 8,
                                       ):
        
        repo_path = os.path.join(self.repo_cache_path, repo_name)
        if not os.path.isdir(repo_path):
            self.logger.warning(f"repo not found: {repo_path}, cloneing...")
            self.code_collector._clone_repo(repo_url, 1, 3)
        all_files = self._list_repo_files(repo_name)
        paper_title, paper_abstract = self.work_collector.get_paper_title_abstract(paper_id)

        mainfest = {
                    "repo_info": {
                        "repo_url": repo_url,
                        "repo_name": repo_name,
                        "paper_title": paper_title,
                        "paper_abstract": paper_abstract,
                        "repo_structure": {},
                    },
                    "scanned_files": 0, 
                    "core_files": [],
                    "main_files": [],  # Main entry point files (subset of core_files)
                    "edge_files": [] , 
                    "skipped": [] ,
                    "pseudocode_input": []
                }
        repo_structure = self.scan_repo_structure(repo_name)

        files_chunks = []
        file_num = len(all_files)

        for file_abs in all_files:
            rel_path = os.path.relpath(file_abs, repo_path)
            content = self._read_file(file_abs)
            if content is None:
                mainfest["skipped"].append({"path": rel_path, "reason": "unreadable/too large"})
                file_num -= 1
                continue

            mainfest["scanned_files"] += 1

            # quick filename heuristics to short-circuit some obvious cores
            fname = os.path.basename(file_abs).lower()
            if any(k in fname for k in self.heur_keywords):
                pass
            
            if self.chunk_file:
                chunks = self._chunk_text(content)
            else:
                chunks = [content]
            if not chunks:
                ## !! The file orders in "scanned" and "skipped" are uncertain !! ##
                mainfest["skipped"].append({"path": rel_path, "reason": "empty or no chunks"})
                file_num -= 1
                continue

            files_chunks.append({"path": rel_path, "chunks": chunks})

        self.logger.info(f"file number: {file_num}")
        
        if self.filter_in_steps:
            results = self._filter_call_model_in_steps(files_chunks = files_chunks,
                                                       paper_title = paper_title,
                                                       paper_abstract = paper_abstract,
                                                       repo_structure = repo_structure,
                                                       max_retry = max_retry)
        else:
            results = self._filter_call_model_one_step(files_chunks = files_chunks,
                                                       paper_title = paper_title,
                                                       paper_abstract = paper_abstract,
                                                       repo_structure = repo_structure,
                                                       max_retry = max_retry)

        index = 0
        for file_idx, chunks_file_info in enumerate(files_chunks):
            chunks = chunks_file_info["chunks"]
            chunk_results = []
            for chunk_idx in range(len(chunks)):
                result = results[index]
                chunk_results.append(result)
                index += 1

            # aggregate chunk scores to file score
            scores = [c["score"] for c in chunk_results]
            main_scores = [c["main_score"] for c in chunk_results]
            is_core_flags = [c["is_core"] for c in chunk_results]

            if per_file_score_method == "avg":
                file_score = float(sum(scores) / max(1, len(scores)))
            else:
                file_score = float(max(scores))

            # Use default threshold if not provided
            effective_threshold = min_core_score if min_core_score is not None else self.CORE_SCORE_THRESHOLD
            
            # determine final boolean: 
            any_core_chunk = any(c["is_core"] and c["score"] >= effective_threshold for c in chunk_results)
            is_core_file = (file_score >= effective_threshold) or any_core_chunk
            is_main_code = any(c.get("is_main_code", False) for c in chunk_results)

            rel_path = chunks_file_info["path"]
            reason = sorted(chunk_results, key=lambda x: x["score"], reverse=True)[0]["reason"]
            main_chunk = next((c for c in chunk_results if c.get("is_main_code", False)), None)
            main_reason = main_chunk["main_reason"] if main_chunk else chunk_results[-1]["main_reason"]

            pseudocode_entry = {
                "path": rel_path,
                "is_core_file": is_core_file,
                "scores": scores, 
                "reason": reason,
                "is_main_file": is_main_code,
                "main_scores": main_scores,
                "main_reason": main_reason
            }

            if is_core_file or is_main_code:
                mainfest["pseudocode_input"].append(pseudocode_entry)

            if is_core_file:
                core_file_entry = {
                    "path": rel_path, 
                    "scores": scores, 
                    "reason": reason,
                }
                mainfest["core_files"].append(core_file_entry)
                try:
                    parts = rel_path.split(os.sep)
                    node = repo_structure
                    for p in parts[:-1]:
                        node = node[p]
                    filename = parts[-1]
                    if filename in node and isinstance(node[filename], dict):
                        node[filename]["is_core_code"] = True
                    else:
                        self.logger.debug(f"Could not set is_core_code for {rel_path} (not found in repo_structure)")
                except Exception as e:
                    self.logger.debug(f"Error marking repo_structure for {rel_path}: {e}")
            else:
                mainfest["edge_files"].append({"path": rel_path, "scores": scores, "reason": reason})

            if is_main_code:
                # Add to main_files even if not core
                main_file_entry = {
                    "path": rel_path,
                    "reason": main_chunk["main_reason"],
                }
                mainfest["main_files"].append(main_file_entry)

                # add core file information in the structure for cache
                try:
                    parts = rel_path.split(os.sep)
                    node = repo_structure
                    for p in parts[:-1]:
                        node = node[p]
                    filename = parts[-1]
                    if filename in node and isinstance(node[filename], dict):
                        node[filename]["is_main_code"] = True
                    else:
                        self.logger.debug(f"Could not set is_main_code for {rel_path} (not found in repo_structure)")
                except Exception as e:
                    self.logger.debug(f"Error marking repo_structure for {rel_path}: {e}")

        mainfest["repo_info"]["repo_structure"] = repo_structure

        return mainfest

    def format_mainfest(self, mainfest: dict, include_pseudocode = False) -> str:
        lines = []
        lines.append("==== mainfest ====\n")
        lines.append(f"repository name: {mainfest['repo_info']['repo_name']}")
        lines.append(f"scanned files: {mainfest['scanned_files']}")
        lines.append(f"core files: {len(mainfest['core_files'])}")
        lines.append(f"skip files: {len(mainfest['skipped'])}")

        lines.append(f"--- skipped_files ----")
        for skipped_file in mainfest["skipped"]:
            lines.append(f"skipped file path{skipped_file['path']}")
            lines.append(f"reason {skipped_file['reason']}")

        lines.append(f"--- main_files (entry points) ----")
        main_files = mainfest.get("main_files", [])
        for main_file in main_files:
            lines.append(f"main file path: {main_file['path']}")
            lines.append(f"reason: {main_file.get('reason', '')}")

        lines.append(f"--- core_files ----")
        for core_file in mainfest["core_files"]:
            lines.append(f"core file path{core_file['path']}")
            lines.append(f"core level: {core_file['scores']}")
            lines.append(f"reason {core_file['reason']}")

        lines.append(f"--- edge_files ----")
        for edge_file in mainfest["edge_files"]:
            lines.append(f"edge file path{edge_file['path']}")
            lines.append(f"core level: {edge_file['scores']}")
            lines.append(f"reason {edge_file['reason']}")

        if include_pseudocode:
            lines.append(f"--- pseudocode_files ----")
            for pseudo_file in mainfest["pseudocode_input"]:
                lines.append(f"{pseudo_file['path']}")
                lines.append(f"{pseudo_file['is_core_file']}")
                lines.append(f"{pseudo_file['is_main_file']}")
                lines.append(f"PSEUDOCODE:")
                lines.append(f"{pseudo_file.get('pseudocode', 'no pseudocode')}")

        return "\n".join(lines)

    def validate_mainfest(self, mainfest: dict) -> bool:
        """
        Validate if the mainfest structure is complete and contains all required fields.
        
        Required structure:
        - repo_info: {repo_name, paper_title, paper_abstract, repo_structure}
        - scanned_files: int
        - core_files: list (each with path, scores, reason)
        - edge_files: list
        - skipped: list
        
        For core_files, each entry must have:
        - path: str
        - scores: list of numbers
        - reason: str
        - pesudocode: str, no less than 50 chars
        
        Returns True if mainfest is valid and complete, False otherwise.
        """
        if not mainfest or not isinstance(mainfest, dict):
            return False


        self.logger.info("Loading and validating mainfest from cache....")

        required_keys = ["repo_info", "scanned_files", "core_files", "main_files", "edge_files", "skipped", "pseudocode_input"]
        repo_info_keys = ["repo_name", "paper_title", "paper_abstract", "repo_structure"]
        core_file_keys = ["path", "scores", "reason"]
        
        # Check top-level keys
        for key in required_keys:
            if key not in mainfest:
                self.logger.warning(f"mainfest missing required key: {key}")
                return False
        
        # Check repo_info keys
        for key in repo_info_keys:
            if key not in mainfest["repo_info"]:
                self.logger.warning(f"mainfest repo_info missing required key: {key}")
                return False
        
        # Check that core_files is not empty
        if not mainfest.get("core_files"):
            self.logger.warning("mainfest has no core_files")
            return False
        
        # Check each core_file has required fields
        for idx, core_file in enumerate(mainfest["core_files"]):
            for key in core_file_keys:
                if key not in core_file:
                    self.logger.warning(f"Core file {idx} missing required key: {key}")
                    return False
            
            # Validate scores is a non-empty list of numbers
            scores = core_file.get("scores", [])
            if not isinstance(scores, list) or not scores:
                self.logger.warning(f"Core file {idx} has invalid scores: {scores}")
                return False
            
            if not all(isinstance(s, (int, float)) for s in scores):
                self.logger.warning(f"Core file {idx} scores contain non-numeric values")
                return False

        for idx, pseudocode_file in enumerate(mainfest["pseudocode_input"]):
            if not pseudocode_file or not isinstance(pseudocode_file, dict):
                self.logger.warning(f"Core file {idx} has wrong pseudo input format, not dict but {type(pseudocode_file)}")
                return False

            if "pseudocode" not in pseudocode_file.keys():
                self.logger.warning(f"Core file {idx} does not have pseudocode")
                return False

            # Check if pseudocode is a valid string before checking length
            pseudocode_value = pseudocode_file["pseudocode"]
            if not isinstance(pseudocode_value, str):
                self.logger.warning(f"Core file {idx} pseudocode is not a string, got {type(pseudocode_value)}: {pseudocode_value}")
                return False

            if len(pseudocode_value) < 30:
                self.logger.warning(f"Core file {idx} pseudocode too short")
                return False
        return True

    def load_cached_mainfest(self, repo_name: str) -> Optional[dict]:
        model_name = getattr(self.chat_agent, "model_name", "default")
        cache_path = os.path.join(self.repo_cache_path, repo_name, f"mainfest_{model_name}.json")
        
        if not os.path.exists(cache_path):
            self.logger.info(f"Cache file not found: {cache_path}")
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                mainfest = json.load(f)
            
            self.logger.info(f"Loaded cached mainfest from {cache_path}")
            return mainfest
        except Exception as e:
            self.logger.warning(f"Failed to load cached mainfest: {e}")
            return None

    def generate_pseudocode(self, mainfest: dict) -> dict:
        """
        Internal method to generate pseudocode using LLM.
        
        For each core file:
        1. Read the file content
        2. Build a prompt using PSEUDOCODE_WRITER template
        3. Use batch_remote_chat to get the analysis
        4. Add the pseudocode result to the core_file entry
        5. Save to cache
        """
        repo_name = mainfest["repo_info"]["repo_name"]
        paper_title = mainfest["repo_info"]["paper_title"]
        paper_abstract = mainfest["repo_info"]["paper_abstract"]
        repo_path = os.path.join(self.repo_cache_path, repo_name)
        
        prompts = []
        pseudo_input_indices = []
        pseudo_input_list = mainfest["pseudocode_input"]
        
        for idx, input_file in enumerate(pseudo_input_list):
            rel_path = input_file["path"]
            file_abs = os.path.join(repo_path, rel_path)
            
            # Read file content
            content = self._read_file(file_abs)
            if content is None:
                self.logger.warning(f"Could not read core file: {rel_path}, skipping...")
                input_file["pseudocode"] = {"error": "Could not read file"}
                continue
            
            # Get the score reason from the core file info
            score_reason = input_file.get("reason", "No reason provided")
            main_reason = input_file.get("main_reason", "No reason provided")
            
            # Get the highest score for this file
            scores = input_file.get("scores", [])
            main_scores = input_file.get("main_scores", [])
            core_score = max(scores) if scores else 0
            main_score = max(main_scores) if main_scores else 0
            is_main_code = input_file.get("is_main_file", "Unknown")
            is_core_file = input_file.get("is_core_code", "Unknown")
            
            # Build the prompt using PSEUDOCODE_WRITER template
            prompt = PSEUDOCODE_WRITER.format(
                file_path=rel_path,
                core_score=core_score,
                is_core_file = is_core_file,
                reason=score_reason,
                is_main_code = is_main_code,
                main_score = main_score,
                main_reason = main_reason,
                paper_title=paper_title,
                paper_abstract=paper_abstract,
                file_content=content
            )
            
            prompts.append(prompt)
            pseudo_input_indices.append(idx)

        if not prompts:
            self.logger.info("No prompts to process for pseudocode generation")
            return mainfest
        
        self.logger.info(f"Generating pseudocode for {len(prompts)} core files...")
        
        # Use batch_remote_chat to get the analysis
        results = self.chat_agent.batch_remote_chat(prompts, desc="Generating pseudocode for core and main files...")
        
        # Process results and add to mainfest
        for i, result in enumerate(results):
            idx = pseudo_input_indices[i]
            mainfest["pseudocode_input"][idx]["pseudocode"] = result
            input_file = pseudo_input_list[idx]

            rel_path = input_file["path"]
            repo_structure = mainfest["repo_info"]["repo_structure"]

            try:
                parts = rel_path.split(os.sep)
                node = repo_structure
                for p in parts[:-1]:
                    node = node[p]
                filename = parts[-1]
                if filename in node and isinstance(node[filename], dict):
                    node[filename]["pseudocode"] = result
                else:
                    self.logger.debug(f"Could not set pseudocode result for {rel_path} (not found in repo_structure)")
            except Exception as e:
                self.logger.debug(f"Error marking repo_structure for {rel_path}: {e}")
        
        return mainfest

    def cache_mainfest(self, mainfest):
        try:
            with open(f"{self.repo_cache_path}/{mainfest['repo_info']['repo_name']}/mainfest_{self.chat_agent.model_name}.json", "w") as f:
                json.dump(mainfest, f)
        except Exception as e:
            self.logger.error(f"error in mainfest cache for {mainfest.get('repo_info', {}).get('repo_name', 'Unknown')}: {e}")
        return

    def _get_pseudocode_from_repo_structure(self, repo_structure: dict, rel_path: str) -> Optional[str]:
        """
        Helper method to retrieve pseudocode from repo_structure by file path.
        
        Args:
            repo_structure: the repo_structure dict from mainfest
            rel_path: relative path to the file
            
        Returns:
            pseudocode string if found, None otherwise
        """
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

    def _get_main_files_sorted_by_score(self, mainfest: dict, k: int = 5) -> List[dict]:
        """
        Get main files sorted by main_score, returning top-k if needed.
        
        Args:
            mainfest: the mainfest dict containing main_files info
            k: default number of main files to include (default 5)
            
        Returns:
            list of main file entries sorted by main_score descending
        """
        repo_structure = mainfest["repo_info"]["repo_structure"]
        
        # Get main files with their scores from main_files
        main_files_with_score = []
        for main_file in mainfest.get("main_files", []):
            rel_path = main_file["path"]
            # Find the corresponding entry in pseudocode_input to get main_scores
            pseudo_entry = next((p for p in mainfest.get("pseudocode_input", []) 
                                if p["path"] == rel_path), None)
            if pseudo_entry:
                main_scores = pseudo_entry.get("main_scores", [])
                max_main_score = max(main_scores) if main_scores else 0
                main_files_with_score.append({
                    "path": rel_path,
                    "main_score": max_main_score,
                    "reason": main_file.get("reason", "")
                })
        
        # Sort by main_score descending
        main_files_with_score.sort(key=lambda x: x["main_score"], reverse=True)
        
        # Return all if count <= k, otherwise return top k
        if len(main_files_with_score) <= k:
            return main_files_with_score
        else:
            return main_files_with_score[:k]

    def generate_project_pseudocode_from_main_files(
        self, 
        mainfest: dict, 
        k: int = 5,
        max_retry: int = 3
    ) -> str:
        """
        Function 1: Generate project pseudocode framework from main files.
        
        If the number of main files is <= k (default 5), all are input.
        Otherwise, select the top-k main files with highest main_score.
        
        Args:
            mainfest: the mainfest dict containing repo_info and main_files info
            k: default threshold for number of main files (default 5)
            max_retry: maximum number of retries for API calls
            
        Returns:
            Initial project pseudocode string
        """
        self.logger.info("Generating project pseudocode from main files...")
        
        # Extract necessary info from mainfest
        paper_title = mainfest["repo_info"]["paper_title"]
        paper_abstract = mainfest["repo_info"]["paper_abstract"]
        repo_structure = mainfest["repo_info"]["repo_structure"]
        
        # Get main files sorted by main_score
        main_files_sorted = self._get_main_files_sorted_by_score(mainfest, k)
        
        if not main_files_sorted:
            self.logger.warning("No main files found in mainfest")
            return ""
        
        self.logger.info(f"Using {len(main_files_sorted)} main files to build project pseudocode")
        
        # Build the main files pseudocode content
        main_files_pseudocode = []
        for i, main_file in enumerate(main_files_sorted):
            rel_path = main_file["path"]
            main_score = main_file["main_score"]
            reason = main_file.get("reason", "")
            
            # Get pseudocode from repo_structure
            pseudocode = self._get_pseudocode_from_repo_structure(repo_structure, rel_path)
            
            if pseudocode:
                entry = f"=== Main File {i+1}: {rel_path} (main_score: {main_score}/10) ===\n"
                entry += f"Reason: {reason}\n"
                entry += "Pseudocode:\n"
                entry += f"{pseudocode}"
            else:
                entry = f"=== Main File {i+1}: {rel_path} (main_score: {main_score}/10) ===\n"
                entry += f"Reason: {reason}\n"
                entry += "Pseudocode:\n"
                entry += f"[Not avaliable]"
            main_files_pseudocode.append(entry)
        
        main_files_content = "\n".join(main_files_pseudocode)
        
        # Build the prompt
        prompt = PROJECT_PSEUDOCODE_FROM_MAIN.format(
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            repo_structure=self.format_repo_structure(repo_structure),
            main_files_pseudocode=main_files_content
        )
        
        # Call the LLM to generate project pseudocode
        self.logger.info("Calling LLM to generate project pseudocode framework...")
        result = self.chat_agent.remote_chat(
            text_content=prompt,
            temperature=0
        )
        
        return result

    def refine_project_pseudocode_with_core_files(
        self,
        mainfest: dict,
        current_pseudocode: str,
        batch_size: int = 3,
        max_retry: int = 3
    ) -> str:
        """
        Function 2: Refine project pseudocode by incorporating core files in batches.
        
        This function processes core files in batches and progressively refines
        the project pseudocode by incorporating each batch's pseudocode.
        
        Args:
            mainfest: the mainfest dict containing repo_info and core_files info
            current_pseudocode: the current project pseudocode to refine
            batch_size: number of core files to process in each batch (default 3)
            max_retry: maximum number of retries for API calls
            
        Returns:
            Final refined project pseudocode string
        """
        self.logger.info("Refining project pseudocode with core files (batch mode)...")
        
        # Extract necessary info from mainfest
        paper_title = mainfest["repo_info"]["paper_title"]
        paper_abstract = mainfest["repo_info"]["paper_abstract"]
        repo_structure = mainfest["repo_info"]["repo_structure"]
        
        # Get core files sorted by score
        core_files_sorted = []
        for core_file in mainfest.get("core_files", []):
            rel_path = core_file["path"]
            scores = core_file.get("scores", [])
            max_score = max(scores) if scores else 0
            core_files_sorted.append({
                "path": rel_path,
                "score": max_score,
                "reason": core_file.get("reason", "")
            })
        
        # Sort by score descending
        core_files_sorted.sort(key=lambda x: x["score"], reverse=True)
        
        if not core_files_sorted:
            self.logger.warning("No core files found in mainfest")
            return current_pseudocode
        
        self.logger.info(f"Processing {len(core_files_sorted)} core files in batches of {batch_size}")
        
        # Initialize the current pseudocode
        refined_pseudocode = current_pseudocode
        
        # Process in batches
        for batch_idx in range(0, len(core_files_sorted), batch_size):
            batch = core_files_sorted[batch_idx:batch_idx + batch_size]
            self.logger.info(f"Processing batch {batch_idx // batch_size + 1}: files {batch_idx + 1} to {batch_idx + len(batch)}")
            
            # Build core files pseudocode content for this batch
            core_files_pseudocode = []
            for i, core_file in enumerate(batch):
                rel_path = core_file["path"]
                score = core_file["score"]
                reason = core_file.get("reason", "")
                
                # Get pseudocode from repo_structure
                pseudocode = self._get_pseudocode_from_repo_structure(repo_structure, rel_path)
                
                if pseudocode:
                    entry = f"--- Core File {i+1}: {rel_path} (core_score: {score}/10) ---"
                    entry += f"Reason: {reason}\n"
                    entry += "Pseudocode:\n"
                    entry += f"{pseudocode}"
                else:
                    entry = f"--- Core File {i+1}: {rel_path} (core_score: {score}/10) ---"
                    entry += f"Reason: {reason}\n"
                    entry += "Pseudocode:\n"
                    entry += f"[Not Available]"
                core_files_pseudocode.append(entry)
            
            core_files_content = "\n".join(core_files_pseudocode)
            
            # Build the prompt for refinement
            prompt = REFINE_PROJECT_PSEUDOCODE_WITH_CORE.format(
                paper_title=paper_title,
                paper_abstract=paper_abstract,
                repo_structure=self.format_repo_structure(repo_structure),
                current_pseudocode=refined_pseudocode,
                core_files_pseudocode=core_files_content
            )
            
            # Call the LLM to refine the pseudocode
            self.logger.info(f"Calling LLM to refine pseudocode for batch {batch_idx // batch_size + 1}...")
            result = self.chat_agent.remote_chat(
                text_content=prompt,
                temperature=0
            )
            
            # Update the refined pseudocode
            refined_pseudocode = result
        
        self.logger.info("Finished refining project pseudocode with all core files")
        return refined_pseudocode

    def cache_repo_pseudocode(self, repo_name, pseudocode, concise = False):
        model_name = self.chat_agent.model_name
        if not concise:
            cache_path = os.path.join(self.repo_cache_path, repo_name, f"pseudocode_{model_name}.json") 
        else:
            cache_path = os.path.join(self.repo_cache_path, repo_name, f"concise_pseudocode_{model_name}.json") 

        with open(cache_path, "w") as f:
            f.write(pseudocode)

    def read_repo_pseudocode_cache(self, repo_name, include_concise = True):
        model_name = self.chat_agent.model_name
        cache_path = os.path.join(self.repo_cache_path, repo_name, f"pseudocode_{model_name}.json") 
        concise_cache_path = os.path.join(self.repo_cache_path, repo_name, f"concise_pseudocode_{model_name}.json") 

        concise_pseudocode = None
        pseudocode = None

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                pseudocode = f.read()
        
        if include_concise and os.path.exists(concise_cache_path):
            with open(concise_cache_path, "r") as f:
                concise_pseudocode = f.read()

        return pseudocode, concise_pseudocode

    def refine_project_pseudocode_with_agent(
        self,
        mainfest: dict,
        current_pseudocode: str,
        max_steps: int = 20,
        hard_code_revise: bool = False,
        max_rounds_without_revise: int = 3,
        last_round_revise: bool = True,
    ) -> str:
        """
        Function 3 (Agent mode): Refine project pseudocode using agent-based approach.
        
        A central agent decides operations:
        - get_pseudocode: Query pseudocode of a specific file
        - get_source_code: Query source code of a specific file
        - revise: Modify current pseudocode based on context
        - review: Call review skill to provide suggestions
        - finish: Complete the refinement
        
        Args:
            mainfest: the mainfest dict containing repo_info and core_files info
            current_pseudocode: the current project pseudocode to refine
            max_steps: maximum number of agent steps (default 20)
            hard_code_revise: if True, force call revise after max_rounds_without_revise rounds without revise
            max_rounds_without_revise: maximum rounds without revise before forcing (default: 3)
            
        Returns:
            Final refined project pseudocode string
        """
        self.logger.info(f"Refining project pseudocode with agent mode (hard_code_revise={hard_code_revise})...")
        
        # Create PseudoReviser instance
        revisor = PseudoReviser(
            config=None,  # Already have chat_agent
            chat_agent=self.chat_agent,
            repo_cache_path=self.repo_cache_path
        )
        
        # Call the agent-based refinement
        final_pseudocode = revisor.refine_pseudocode_with_agent(
            mainfest=mainfest,
            current_pseudocode=current_pseudocode,
            max_steps=max_steps,
            hard_code_revise=hard_code_revise,
            max_rounds_without_revise=max_rounds_without_revise,
            last_round_revise = last_round_revise,
        )
        
        self.logger.info("Agent-based pseudocode refinement completed")
        return final_pseudocode

    def execute(self, papers: List[str], batch_refine: bool = True, agent_refine: bool = True, batch_size: int = 3):
        papers_sites = []
        site_num = 0
        for paper in papers:
            try:
                paper_markdown = self.work_collector.get_paper_raw_markdown(paper)
            except Exception as e:
                papers_sites.append([])
                continue
            sites =  self.code_collector.extract_code_links(paper_id = paper, paper_markdown=paper_markdown)
            papers_sites.append(sites)
            site_num += len(sites)
        self.logger.info(f"[GENERAL] site number: {site_num}")


        paper_mainfests = []
        for paper, sites in zip(papers, papers_sites):
            self.logger.info(f"processing sites for {paper}")
            paper_pseudocode = []
            paper_repo_names = []
            paper_repo_mainfests = []
            paper_concise_pseudocode = []
            for site in sites:
                repo_name = self.code_collector._extract_repo_name(site)
                pseudocode, initial_pseudocode = self.read_repo_pseudocode_cache(repo_name, True)

                if not pseudocode or not initial_pseudocode:
                    main_fest = self.build_mainfest(paper, site, 8)
                    initial_pseudocode = self.generate_project_pseudocode_from_main_files(main_fest)

                    pseudocode = initial_pseudocode
                    # Refine
                    if batch_refine:
                        pseudocode = self.refine_project_pseudocode_with_core_files(mainfest = main_fest,
                                                                                        current_pseudocode=pseudocode,
                                                                                        batch_size = batch_size)
                    if agent_refine:
                        pseudocode = self.refine_project_pseudocode_with_agent(mainfest = main_fest,
                                                                                        current_pseudocode=pseudocode,
                                                                                        max_steps = 20,
                                                                                        hard_code_revise = True,
                                                                                        max_rounds_without_revise = 3,
                                                                                        last_round_revise = True)
                    pseudocode = f"[REPOSITORY NAME]:{repo_name}: \n" + pseudocode
                    initial_pseudocode = f"[REPOSITORY NAME]:{repo_name}: \n" + initial_pseudocode

                paper_pseudocode.append(pseudocode)
                paper_concise_pseudocode.append(initial_pseudocode)
                paper_repo_names.append(repo_name)
                paper_repo_mainfests.append(main_fest)

                self.cache_repo_pseudocode(repo_name, pseudocode)
                self.cache_repo_pseudocode(repo_name, initial_pseudocode, concise=True)
                
            if len(paper_pseudocode) > 0:
                paper_mainfests.append({
                    "paper_id": paper,
                    "paper_title": paper_repo_mainfests[0]["repo_info"].get("paper_title", "title loss"),
                    "paper_abstract": paper_repo_mainfests[0]["repo_info"].get("paper_abstract", "abstract loss"),
                    "pseudocodes":paper_pseudocode,
                    "concise_pseudocodes": paper_concise_pseudocode,
                    "repo_urls": sites,
                    "repo_names": paper_repo_names,
                    "mainfests": paper_repo_mainfests,
                })

        self.logger.info(f"PAPER PSEUDOCODE SUCCESS NUM: {len(paper_mainfests)}")
        repo_num = 0
        for repo in paper_mainfests["repo_names"]:
            repo_num += len(repo)
        self.logger.info(f"REPO PSEUDOCODE SUCCESS NUM: {repo_num}")

        return paper_mainfests

def single_paper_test(config):
    # with open("/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/database/parsed_papers/2602.09372/auto/2602.09372.md", "r") as f:
    #     md_text = f.read()
    code_collector = CodeCollector(config)
    work_collector = WorkCollector(config)
    code_analyzer = CodeAnalyzer(config, code_collector, work_collector)

    md_text = work_collector.get_paper_raw_markdown("2406.10252")
    print(code_collector.extract_code_links("2406.10252" ,md_text))
    for site in code_collector.extract_code_links("2406.10252", md_text):
        code_collector._clone_repo(site)
        repo_name = code_collector._extract_repo_name(site)
        tree = code_analyzer.scan_repo_structure(repo_name)
        print(code_analyzer.format_repo_structure(tree))

        mainfest = code_analyzer.build_mainfest("2406.10252", site, 8)
        print(code_analyzer.format_repo_structure(mainfest["repo_info"]["repo_structure"]))
        print(code_analyzer.format_mainfest(mainfest, True))

        initial_pseudocode = code_analyzer.generate_project_pseudocode_from_main_files(mainfest)
        print(f"============= initial pseudocode ================")
        print(initial_pseudocode)
        final_pseudocode = code_analyzer.refine_project_pseudocode_with_core_files(mainfest = mainfest,
                                                                                    current_pseudocode=initial_pseudocode,
                                                                                    batch_size = 8)
        final_pseudocode = code_analyzer.refine_project_pseudocode_with_agent(mainfest = mainfest,
                                                                                    current_pseudocode=initial_pseudocode,
                                                                                    max_steps = 10)
        print(f"============= final pseudocode ================")
        print(final_pseudocode)

        code_analyzer.cache_repo_pseudocode(repo_name, final_pseudocode)

@hydra.main(config_path="../config", config_name="deep_survey_batch_xiaomi", version_base=None)
def main(config):
    code_collector = CodeCollector(config)
    work_collector = WorkCollector(config)
    code_analyzer = CodeAnalyzer(config, code_collector, work_collector)
    code_report_generator = CodeReportGenerator(config, work_collector=work_collector, code_collector=code_collector, code_analyzer=code_analyzer)

    test_topics_paper = ["2406.10252", "2509.18661", "2510.07733", "2508.17647", "2503.04629", "2502.14776", "2504.05732v2"]
    topic = "LLM automated academic survey/review generation framework"
    paper_mainfests = code_analyzer.execute(test_topics_paper, batch_refine=True, agent_refine=True, batch_size=2)
    print("Genrating pseudocode report for test papers...")
    print("-------------------------------------------")
    print(code_report_generator.generate_report(papers = paper_mainfests, topic = topic, batch_size = 2))
    print("-------------------------------------------")
    print()
    print("-------------------------------------------")
    print("Genrating env report for test papers...")
    print(code_report_generator.generate_framework_env_report(paper_mainfests = paper_mainfests, topic = topic, batch_size = 2))
    print("-------------------------------------------")


if __name__ == "__main__":
    main()