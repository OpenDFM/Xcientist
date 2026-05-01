---
name: blog-analyze
description: Analyze and audit technical blog posts with engineering-focused scoring, source fidelity verification, and AI detection optimized for jargon-rich content
triggers:
  - analyze
  - audit
  - score
  - quality
  - 检查
  - 分析
  - 评分
---

# Blog Analyzer Skill

You are a technical blog auditor. When asked to analyze, audit, or score a blog post, follow these steps.

## Project Paths

- **Source workspace**: `/home/zhang/projects/ResearchAgent-main/src/agents/experiment_agent/workspaces/{project_name}`
- **Blog article**: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/blog_article.md`
- **Graph method files**: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/test_output/graph{N}.md`
- **Analysis output**: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/blog_analysis.md`

## Input Handling

1. **Identify Project Name**: Extract project folder from user input (e.g., "analyze dcr_v1" → project_name = "dcr_v1")
2. **Read Blog Article**: Read `blog_article.md` from the workspace
3. **Locate Source Code**: Navigate to `/home/zhang/projects/ResearchAgent-main/src/agents/experiment_agent/workspaces/{project_name}`

## Scoring Process

### Step 1: Content Extraction

Read the blog post and extract:
- Frontmatter (title, description, date, author, tags)
- Heading structure (H1, H2, H3 hierarchy)
- Paragraph count and word counts (use CountWords tool for accurate word counting)
- Code blocks (language, line count, content)
- Function names, class names mentioned
- Parameter names and default values
- Statistics and claims
- Images, diagrams
- Links

### Step 2: Score Article Content

Score on a 0-100 scale across 6 categories:

#### Content Quality (22 points)
| Check | Points | Pass Criteria |
|-------|--------|---------------|
| Depth/comprehensiveness | 6 | Covers topic thoroughly, no major gaps |
| Readability | 5 | Clear explanations, appropriate complexity |
| Originality/unique value | 4 | Original data, case studies, first-hand experience |
| Sentence & paragraph structure | 4 | Varied sentence length, well-structured paragraphs |
| Engagement elements | 3 | TL;DR, callouts, code examples, diagrams |

#### Engineering Depth (20 points)
| Check | Points | Pass Criteria |
|-------|--------|---------------|
| Code accuracy | 6 | Code snippets are syntactically correct, variable names match source |
| Architecture mapping | 6 | Accurately describes component relationships and causality |
| Technical precision | 5 | Correct use of terminology, accurate API references |
| Chart generation | 3 | Experimental section has charts/visualizations generated |

#### Source Fidelity (26 points)
| Check | Points | Pass Criteria |
|-------|--------|---------------|
| Function existence | 10 | All mentioned function names exist in project source |
| Parameter alignment | 11 | Default values, types match actual source code |
| File reference accuracy | 5 | File paths and structure match actual project |
**Important**: If any critical mismatch found, apply **additional penalty**. A single serious error can result in 0 points for this category.

#### Research Integrity (10 points)
| Check | Points | Pass Criteria |
|-------|--------|---------------|
| Paper existence | 5 | PDFs exist in workspace/{project_name}/ directory |
| Citation authenticity | 5 | Claims in citations match content from referenced PDFs |
**IMPORTANT - Big Penalty**: If papers are missing or citations cannot be verified from PDFs, apply **-20 point penalty** to final score.

#### E-E-A-T Signals (4 points)
| Check | Points | Pass Criteria |
|-------|--------|---------------|
| Source citations | 4 | Credible sources, zero fabricated statistics |

#### AI Citation Readiness (18 points)
| Check | Points | Pass Criteria |
|-------|--------|---------------|
| Passage-level citability | 5 | Self-contained sections with actionable insights |
| Entity clarity | 4 | Consistent terminology, unambiguous references |
| Pseudocode quality | 5 | Plain text pseudocode, no LaTeX or executable code, explains logic clearly |
| Jargon density | 4 | Appropriate use of domain-specific terminology |

### Step 3: Source Fidelity Verification

**Critical**: Verify blog content against actual source code.

#### Function Existence Check
1. Extract all function/class names from blog (e.g., `compute_contract`, `FallbackActivationPolicy`)
2. Search in source: `grep -r "def compute_contract" /home/zhang/projects/ResearchAgent-main/src/agents/experiment_agent/workspaces/{project_name}`
3. Flag any function names that don't exist in source

#### Parameter Alignment Check
1. Extract mentioned parameters (e.g., `min_samples=100`)
2. Find actual default values in source code
3. Flag mismatches between blog claims and source

#### File Reference Check
1. Extract file paths mentioned in blog
2. Verify files exist in project directory
3. Flag references to non-existent files

### Step 4: AI Content Detection (Optimized for Technical Content)

Analyze for AI-generated content risk, with adjustments for technical writing:

#### Burstiness Score (sentence length variance)
- Calculate standard deviation of sentence lengths
- Human technical writing: high variance (concise comments + detailed explanations)
- AI writing: low variance (consistently medium-length)
- Score: 0-10 (10 = human-like)

#### Jargon Density (replaces naive AI phrase detection)
- Identify domain-specific terms (e.g., Pydantic, JSONL, Backbone Caching, TCIR, AsyncIterator)
- Calculate jargon token ratio: jargon_words / total_words
- High jargon density (>5%) → Lower AI probability (technical content naturally repetitive)
- Do NOT flag "Furthermore", "In conclusion" as AI indicators in technical context

#### Vocabulary Analysis
- Calculate Type-Token Ratio (TTR): unique_words / total_words
- Technical writing TTR 0.25-0.4 is acceptable (repetition of function names is normal)
- Only flag if TTR < 0.2 AND low jargon density

#### AI Risk Assessment
- Combine: burstiness + jargon_density + TTR
- High jargon + moderate burstiness = low AI probability
- Flag if: low jargon + low TTR + low burstiness (genuine AI risk)

### Step 5: Image Evaluation

Find `<graph1>`, `<graph2>`, etc. in article and read corresponding graph{N}.md files:

#### Quality Scoring
| Dimension | Points | Criteria |
|-----------|--------|----------|
| Description Quality | 25 | Graph method file has clear Visual Concept, Main Elements, Key Message; description is specific and complete |
| Position | 25 | Image appears before/after relevant text, at logical breaking point, not orphaned |
| Content Match | 25 | Image accurately represents and complements the written content; not decorative or tangential |
| Uniqueness | 25 | Each image serves a distinct purpose; no redundant or repetitive images |

**Quantity Check** :
- >600 words per image → mark as HIGH priority issue
- 300-600 words per image → OK
- <300 words per image → mark as HIGH priority issue

**Distribution Check**:
- Check if any two images appear adjacent (no text between them)
- If `<graph1>` is immediately followed by `<graph2>` → **Critical issue**
- Images should be evenly distributed throughout the article with text in between

### Step 5b: Pseudocode Check

Check if code blocks use proper pseudocode format:

**GOOD (full score)**:
- Plain text pseudocode (e.g., `FUNCTION name:`, `FOR each item:`)
- No LaTeX algorithm markup (no `\begin{algorithm}`, `\SetKwFunction`, etc.)
- Simple numbered steps or indentation-based flow

**BAD (deduct points)**:
- LaTeX algorithm format (should not render in markdown)
- Actual Python/Java code with real function names
- Code that looks like executable code rather than pseudocode
- Hard to understand

| Check | Deduction |
|-------|-----------|
| Uses LaTeX algorithm format | -10 points |
| Uses actual executable code (not pseudocode) | -5 points per occurrence |
| Mixes real function names in pseudocode | -3 points per occurrence |

Apply deductions to **Engineering Depth** score.

### Step 5c: Experimental Charts/Tables Check

Check if experimental results section has proper visualizations:

**Check**:
1. Identify if article discusses experimental results (metrics, benchmarks, comparisons)
2. Look for tables, charts, or visualizations in the article
3. Check for markdown tables, or graph placeholders
4. **Position Check**: Experimental charts/tables must appear AFTER the "Experimental Results" or "Experiments" section heading. If a graph placeholder appears before the experiment section → **Critical issue**
5. **Placeholder Verification**: If `<graphN>` placeholders exist in experiment section, read `test_output/graph{N}.md`. If it's about experimental data, then the requirement is satisfied. But if it's not, then it still needs markdown tables or graph placeholders

**Scoring**:
- **MUST HAVE**: If experiment data is discussed, there MUST be tables or charts → **Critical issue if missing**
- If tables/charts provided with experiment data → OK
- ⚠️ **STRICT**: No experiment discussion without visualizations!

**Quantity Check**:
- Experiment section should have at most 1 chart or table per sub-section to avoid clutter
- If more than 1 visualization in a single experiment sub-section → mark as Medium priority issue

**Visualization types to look for**:
- Markdown tables with metrics
- `<graphN>` image placeholders
- Actual images of charts/graphs
- Bar charts, line charts, confusion matrices

### Step 6: Determine Rating

1. Calculate Article Score (from Step 2 to 4): [X]/100
2. Calculate Image Score (from Step 5): [Y]/100
3. Calculate Final Score: `Final = Article * 0.8 + Image * 0.2`

| Final Score | Rating | Action |
|-------------|--------|--------|
| 90-100 | Exceptional | Publish as-is, flagship content |
| 80-89 | Strong | Minor polish, ready for publication |
| 70-79 | Acceptable | Targeted improvements needed |
| 60-69 | Below Standard | Significant rework required |
| < 60 | Rewrite | Fundamental issues |

### Step 7: Generate Report

Write to: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/blog_analysis.md`

```markdown
# Blog Quality Report: [Title]

**Score: [X]/100** -- [Rating]

## Score Breakdown

| Category | Score | Max |
|----------|-------|-----|
| Content Quality | X | 22 |
| Engineering Depth | X | 20 |
| Source Fidelity | X | 26 |
| Research Integrity | X | 10 |
| E-E-A-T Signals | X | 4 |
| AI Citation Readiness | X | 18 |
| **Article Score** | **X** | **100** |
| **Image Score** | **X** | **100** |
| **Penalty** | -X | (if any) |
| **Final (0.8*Article + 0.2*Image - Penalty)** | **X** | **100** |

## Source Fidelity Check

### Function Existence
- [ ] `function_name_1` - Found / NOT FOUND
- [ ] `function_name_2` - Found / NOT FOUND

### Parameter Alignment
- [ ] `param_name=value` - Matches source / MISMATCH
- [ ] `param_name=value` - Matches source / MISMATCH

## Research Integrity Check

### Paper Existence
- [ ] PDF files exist in: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/`
- Note: This is where the navigator downloaded the papers. Check this folder first!

### Citation Authenticity
For each citation in the article, verify against PDF (check the folder above):
- [ ] Citation [1]: Verified from PDF / NOT VERIFIED
- [ ] Citation [2]: Verified from PDF / NOT VERIFIED
- Note: Must read actual PDF to verify claims

### Citation Completeness (CRITICAL)
Check if every paper in the References section is actually cited in the article body:
1. List all papers in References section
2. Search article body for citations of each paper (e.g., "<sup>[1]</sup>" )
3. If any reference is NOT cited in body → **Critical issue with -15 point penalty**

### Penalty
- [ ] Apply -20 point penalty if papers missing or citations unverified
- [ ] Apply -15 point penalty if references listed but not cited in body

### Issues Found

#### Critical (Must Fix)
- [ ] Source fidelity issue: [function/parameter not found in source]
- [ ] Code accuracy: [code snippet incorrect]
- [ ] Research integrity: PDF papers not found in workspace/{project_name}/
- [ ] Research integrity: Citation claims cannot be verified from referenced PDFs
- [ ] Citation completeness: Papers listed in References section but NOT cited in article body
- [ ] Experimental results: Experiment data discussed but no tables/charts provided
- [ ] Image: Two images appear adjacent (no text between them)

#### High Priority
- [ ] Engineering depth: [architectural relationship unclear]
- [ ] Technical precision: [terminology error]
- [ ] Image: Graph{N} position inappropriate (orphan or wrong placement)
- [ ] Image: Too few (>600 words/image) or too many (<300 words/image) images
- [ ] Code: Uses LaTeX algorithm format instead of plain pseudocode
- [ ] Code: Uses actual executable code instead of pseudocode

#### Medium Priority
- [ ] Readability: [explanation unclear]
- [ ] Structure: [section organization]
- [ ] Image: Graph{N} description unclear or missing elements
- [ ] Image: Graph{N} could be more supplementary to text

#### Low Priority
- [ ] Minor improvements

## AI Content Analysis
- **Burstiness**: [X]/10
- **Jargon density**: [X]% (acceptable for technical content)
- **Vocabulary diversity (TTR)**: [X] (expected range for technical writing)
- **AI probability**: Low / Medium / High
- **Flagged passages**: [if any]

## Quick Stats
- Word count: [N]
- Pseudocode blocks: [N]
- Functions mentioned: [N]
- Source files referenced: [N]
- Statistics: [N] sourced
- Images (graph placeholders): [N]
- Graph method files: [N] found
- Words per image: [N] (target: 300-600)
- Pseudocode issues: [N] (LaTeX format/executable code found)
- PDF papers available: [N]
- Citations in article: [N]
- Citations verified: [N] / [total]
- Charts in experimental section: [N] (expected if results discussed)

## Recommended Actions
1. [Priority fix]
2. [Second priority]
3. [Third priority]
```

## Export Options

### JSON Format (`--format json`)
For programmatic analysis:
```json
{
  "file": "blog_article.md",
  "title": "...",
  "score": 78,
  "rating": "Acceptable",
  "categories": {
    "content_quality": { "score": 18, "max": 22 },
    "engineering_depth": { "score": 16, "max": 20 },
    "source_fidelity": { "score": 24, "max": 26 },
    "research_integrity": { "score": 8, "max": 10 },
    "eeat_signals": { "score": 4, "max": 4 },
    "ai_citation_readiness": { "score": 14, "max": 18 }
  },
  "penalty": 0,
  "final_score": 84,
  "source_check": {
    "functions_found": ["func1", "func2"],
    "functions_missing": [],
    "parameter_mismatches": []
  },
  "ai_analysis": {
    "burstiness": 6.5,
    "jargon_density": 8.2,
    "ttr": 0.32,
    "ai_probability": "Low"
  }
}
```

### Table Format (`--format table`)
Compact summary:
```
File            | Score | Rating     | Quality | Eng | Fidelity | Integrity | EEAT | AI-Ready
blog_article.md |    78 | Acceptable |   18/22 | 16/20 |   24/26 |    8/10   |  4/4 |   14/18
```