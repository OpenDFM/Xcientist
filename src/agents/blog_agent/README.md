# Blog Agent

AI-powered blog generation agent using OpenHands SDK. Generates high-quality technical blog articles from research projects (experiment_agent) with automated image generation and quality scoring.

## Architecture

```
run.py (Entry)
    │
    ├─ Step 1: IdeaAgent      → Explore project, find papers, create outline
    ├─ Step 2: WriteAgent      → Write blog + generate graph method files
    ├─ Step 3: AnalyzeAgent    → Analyze & score (quality, SEO, E-E-A-T)
    ├─ Step 4: RefineAgent      → Loop improvement (until score > 90 or 3 iterations)
    └─ Step 5: ImageGen        → Generate images + replace <graphN> placeholders
```

## Project Structure

```
blog_agent/
├── agent/
│   └── new_base_agent.py      # BaseAgent using OpenHands SDK
├── config/
│   ├── config.yaml            # Configuration (API keys, settings)
│   └── loader.py              # Config loader
├── scripts/
│   └── run.py                 # Main entry point
├── skills/                    # 4 skill modules
│   ├── workspace-navigator/  # Project exploration & paper search
│   ├── blog-writer/          # Article writing
│   ├── blog-analyze/         # Quality analysis & scoring
│   └── blog-refine/          # Iterative improvement
├── tools/                     # OpenHands format tools
│   ├── search_core_tool.py          # Search knowledge graph for papers
│   ├── search_paper_abstract_tool.py # Get paper abstract from Semantic Scholar
│   ├── download_paper_pdf_tool.py   # Download paper PDF
│   ├── count_words_tool.py           # Count words in markdown
│   ├── gengraph.py                   # Image generation via AI
│   ├── illustrate.py                  # High-level illustrate function
│   └── ocr.py                        # OCR text removal
└── utils/
    ├── search_core.py           # Knowledge graph search utilities
    ├── semantic_scholar.py      # Semantic Scholar API utilities
    └── deeperaser.py           # Deep learning text removal
```

## Quick Start

### Setup
```bash
1. cd <research-agent-root>/src/agents
2. mkdir blog_agent
```
3. Put these files into that folder
4. Move `run_blog.sh` to ResearchAgent root directory
5. Fill in the `blog:` block in `src/config/default.yaml`
6. Place `graph.db` (knowledge graph for paper search) into `blog_agent/utils/` folder
7. (Optional) Edit `scripts/run.py` to change the `MODEL` parameter - this controls which model is actually used

> **Important**: Blog Agent reads experiment outputs from `blog.source_workspace_root` in `src/config/default.yaml`, using `<source_workspace_root>/<project_name>` by default.

### Run

```bash
cd <research-agent-root>

# Run full pipeline
xcientist blog --experiment <project_name>

# Explicitly use the default experiment workspace root
xcientist blog --experiment <project_name> --source-workspace /hpc_stor03/sjtu_home/zijian.wang/Xcientist-os/Xcientist/workspace/<project_name>

# Resume from last checkpoint
xcientist blog --experiment <project_name> --resume

# Use an experiment workspace outside the default source path
xcientist blog --experiment <project_name> --source-workspace /abs/path/to/experiment_workspace
```

## Resume / Checkpoint

The agent saves its progress after each step to `workflow_status.json`. If the process is interrupted (e.g., network error, manual stop), you can resume from where it left off.

### How It Works

1. **Status File**: After completing each step, the agent saves progress to:
   ```
   workspaces/<project_name>/workflow_status.json
   ```

   Example content:
   ```json
   {
     "experiment": "dcr_v1",
     "current_step": 3,
     "current_loop": 2,
     "steps_completed": ["IDEAAgent", "WRITEAgent", "ANALYZEAgent", "REFINEAgent"],
     "last_updated": "2026-03-09T10:30:00"
   }
   ```

2. **Step Numbers**:
   - Step 0: Initialized (not started)
   - Step 1: IDEAAgent completed
   - Step 2: WRITEAgent completed
   - Step 3: ANALYZEAgent completed
   - Step 4: REFINEAgent loop completed
   - Step 5: Image generation completed

3. **Resume Logic**: When `--resume` is passed, the agent reads `current_step` and skips all steps less than or equal to that value:
   ```python
   if resume:
       start_step = status.get("current_step", 0)
   ```

### Example

If the agent stops during Step 3 (ANALYZEAgent), the status file shows `current_step: 3`. Running with `--resume` will:
- Skip Step 1 and Step 2 (re-executes but skips internally)
- Continue from Step 3 onwards
- Restart the REFINE loop if needed

## Configuration

All configuration is in the `blog:` block of `src/config/default.yaml`.

### API Keys

```yaml
blog:
  minimax:
    api_key: "your-minimax-api-key"
    base_url: "https://api.minimaxi.com/v1"

  openai:
    api_key: "your-open-ai-api-key"
    base_url: "https://api-2.xi-ai.cn/v1"

  gemini:
    api_key: "your-gemini-api-key"
    base_url: "https://api.xi-ai.cn/v1"
```

- **minimax**: Default LLM provider for agents
- **openai**: Alternative LLM provider
- **gemini**: Alternative LLM provider

### Model Settings

```yaml
blog:
  model: "MiniMax-M2.5"  # Default model for all agents
```

### Image Generation (gengraph)

```yaml
blog:
  gengraph:
    provider: "xiai"  # Image generation provider
    api_key: "your-image-api-key"

    providers:
      xiai:
        base_url: "https://api.xi-ai.cn/v1"
        model: "gemini-3-pro-image-preview"
      openrouter:
        base_url: "https://openrouter.ai/api/v1"
        model: "google/gemini-3-pro-image-preview"
      bianxie:
        base_url: "https://api.bianxie.ai/v1"
        model: "gemini-3-pro-image-preview"
      gemini:
        base_url: "https://generativelanguage.googleapis.com/v1beta"
        model: "gemini-3-pro-image-preview"
```

- **provider**: Which AI provider to use for generating academic figures
- **providers**: Configuration for each available provider

### OCR / Text Removal

```yaml
blog:
  deeperaser:
    use_cuda: false  # Whether to use GPU for DeepEraser (text removal from images)
```

- **use_cuda**: Set to `true` if you have NVIDIA GPU and want faster OCR processing

### Semantic Scholar API

```yaml
blog:
  semantic_scholar:
    api_key: ""  # Optional: your Semantic Scholar API key for paper abstract retrieval and PDF download
```

- **api_key**: Optional. Without it, you can still search papers but may have rate limits.

### Blog Illustration

```yaml
blog:
  illustrate:
    only_gen_img: true  # true: only generate images, false: generate + OCR cleanup
```

- **only_gen_img**:
  - `true`: Faster, skip OCR cleanup step
  - `false`: Slower, but removes generated text from images using OCR

### Adding New Model Providers

To add support for new LLM providers, modify the `get_openhands_config` function in `agent/new_base_agent.py`:

```python
def get_openhands_config(model: str) -> dict:
    if is_minimax_model(model):
        return {
            "api_key": MINIMAX_API_KEY,
            "model": 'minimax/'+model,
            "base_url": MINIMAX_API_BASE,
        }
    elif is_gemini_model(model):
        return {
            "api_key": GEMINI_API_KEY,
            "model": 'gemini/'+model,
            "base_url": GEMINI_API_BASE,
        }
    # Add new provider here
    elif is_new_provider(model):
        return {
            "api_key": NEW_PROVIDER_API_KEY,
            "model": 'new_provider/'+model,
            "base_url": NEW_PROVIDER_API_BASE,
        }
    else:
        # Default to OpenAI
        return {
            "api_key": OPENAI_API_KEY,
            "model": model,
            "base_url": OPENAI_API_BASE,
        }
```

You also need to:
1. Add the new provider to the `blog:` block in `src/config/default.yaml`
2. Add model detection function (e.g., `is_new_provider`)

## Workflow

1. **IdeaAgent** - Explores source project, identifies key files and structure, searches academic papers for citation, creates outline
2. **WriteAgent** - Writes blog article based on outline, creates `<graphN>` placeholders + graph method files (graph1.md, graph2.md, etc.)
3. **AnalyzeAgent** - Scores article (0-100), provides improvement suggestions across 5 dimensions:
   - Content quality
   - SEO optimization
   - E-E-A-T signals (Experience, Expertise, Authoritativeness, Trustworthiness)
   - Technical elements
   - AI citation readiness
4. **RefineAgent** - Iteratively improves based on analysis (max 3 loops)
5. **Image Generation** - Generates images from graph method files, replaces `<graphN>` placeholders with actual markdown images

## Output

After running, the following files are generated in `workspaces/<project_name>/`:

```
<project_name>/
├── blog_idea.md        # Initial blog idea and outline
├── blog_article.md    # Final blog article with images embedded
├── blog_analysis.md   # Quality analysis report with score
└── test_output/
    ├── graph1.md     # Image method file 1 (description for AI)
    ├── figure1.png   # Generated image 1
    ├── graph2.md
    ├── figure2.png
    └── ...
```

## How It Works

1. IdeaAgent explores project, searches papers via knowledge graph + Semantic Scholar, downloads PDFs
2. WriteAgent creates article with `<graphN>` placeholders where images should go
3. Each `<graphN>` corresponds to a `graph{N}.md` file describing what image to generate
4. AnalyzeAgent scores the article and suggests improvements
5. RefineAgent loops to improve until score > 90 or max iterations reached
6. Images are generated from graph method files and placeholders are replaced
