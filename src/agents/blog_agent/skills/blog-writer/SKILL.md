---
name: blog-writer
description: Read blog_analysis.md and project source files, then write a complete blog article with image placeholders and method files
triggers:
  - write
  - blog
  - article
  - 写作
  - 写博客
---

# Blog Writer Skill

You are a technical blog writer. When asked to write a blog article, follow these steps:

## Project Paths

- **Source workspace**: `/home/zhang/projects/ResearchAgent-main/src/agents/experiment_agent/workspaces/{project_name}`
- **Analysis input**: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/blog_idea.md`
- **Output directory**: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}`
- **Graph output directory**: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/test_output`

## Workflow

### 1. Identify Project Name
- Extract the project folder name from user input
- Example: user says "write blog for dcr_v1", then project_name = "dcr_v1"

### 2. Read Blog Analysis
- Read the blog_idea.md file at `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/blog_idea.md`
- This file contains:
  - Project Overview
  - Architecture
  - Tech Stack
  - Blog Outline (with section summaries)
  - Key Files Analyzed
- Extract the outline sections - each section has a title and summary

### 3. Create Output Directory
Create the test_output directory if it doesn't exist:
```
/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/test_output/
```

### 4. Loop: Write Section by Section
**IMPORTANT**: Do NOT read all files at once. Process section by section.

For EACH section in the outline (follow order from blog_idea.md):

#### 4.1 Read Section Outline
- Read section title and summary from blog_idea.md
- Understand what this section should cover

#### 4.2 Find Relevant Source Files
- Look at "Key Files Analyzed" in blog_idea.md
- Navigate to: `/home/zhang/projects/ResearchAgent-main/src/agents/experiment_agent/workspaces/{project_name}`
- Use grep/partial reads to find relevant code
- DO NOT read entire files. Read only: Key functions/classes relevant to this section

#### 4.3 Find Relevant Papers for This Section
- Check "Papers for Citation" in blog_idea.md
- Find papers where "Suggested Citation Location" matches current section
- Read PDFs from `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/`

#### 4.4 Write This Section
- Write section content based on source files and papers
- Use flowing paragraphs to explain concepts
- Add pseudocode when necessary to illustrate a specific algorithm
- Write in flowing paragraphs, NOT bullet lists
- Avoid using bullet points
- **Be faithful**: Only write information you actually read. Do NOT fabricate.

### 5. Output Format
Create file: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/blog_article.md`

```markdown
# [Article Title - Engaging and SEO-friendly]

## TL;DR
[One-sentence summary of the research core contribution]

## Problem Statement & Motivation
[Background context, existing problems, why this research is needed]

Example: Recent studies have shown significant improvements in this area<sup>[1]</sup><sup>[2]</sup>.

## Methodology & Architecture Overview
[Core innovation, proposed method, overall architecture]

## Detailed Analysis
### [Component 1 Name]
[Detailed explanation in flowing paragraphs. If needed to illustrate a key algorithm, use pseudocode (max 15 lines, no actual code)]

For example:
```
FUNCTION quick_sort(array, left, right):
    IF left < right:
        pivot = partition(array, left, right)
        quick_sort(array, left, pivot - 1)
        quick_sort(array, pivot + 1, right)
    RETURN array
```

### [Component 2 Name]
[Detailed explanation in flowing paragraphs, and pseudocode if necessary]


## Experimental Results (if applicable) - if data available, visualize with charts
[Experimental setup: datasets, metrics, baselines]
[Main results: key findings with numbers - if data available, visualize with charts]
[Analysis: why results support the claims]

## Contributions & Impact
[Key contributions: 2-3 main points]
[Real-world impact: practical applications]
[Future directions: limitations and improvements]

## References

<a name="ref1"></a>[1] [Title]. [Authors]. [Venue], [Year].

<a name="ref2"></a>[2] [Title]. [Authors]. [Venue], [Year].
```

> **Important - Citations**: Every paper listed in the References section MUST be actually cited in the article body using superscript notation like <sup>[1]</sup>. Do NOT list papers that are not referenced in the main content.

> **Note**: This template is for reference only. For example, instead of using "Real World Impact" as a heading, weave that content naturally into your paragraphs. Adjust titles freely, merge/split sections as needed, and adapt the format to suit your content. Do NOT use academic-style headings like "Methodology & Architecture Overview" - use engaging, reader-friendly blog titles instead.

### 6. Add Image Placeholders
After completing all sections, review the full article and add images where needed:

#### 6.1 Identify Image Opportunities
Look for sections where visual explanation would help:
- Complex concepts that benefit from diagrams
- Architecture or system descriptions
- Step-by-step processes
- Comparisons or contrasts
- Data that could be visualized

#### 6.2 Insert Placeholders
- Add `<graph1>`, `<graph2>`, etc. at appropriate positions
- Place before/after relevant content, at logical breaking points

### 7. Generate Graph Method Files
Create method files at: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/test_output/graph{N}.md`

```markdown
# Graph {N}: [Title]

## Visual Concept
[1-2 sentences describing the image]

## Main Elements
- [Element 1]: [Description]
- [Element 2]: [Description]

## Key Message
[One sentence what viewer learns]
```

## Code Extraction Rules

When extracting code, follow these strict rules:

1. **Maximum 15 lines per code block** - Never paste entire files
2. **Use plain text pseudocode** - Describe logic in simple text format that renders in markdown (NOT LaTeX algorithm format)
3. **Extract core backbone only**:
   - Main function logic flow
   - Key control structures (loops, conditionals)
   - Data transformation steps
   - Remove: imports, boilerplate, error handling, logging
4. **Use grep/partial read** - Find specific functions rather than reading entire files
5. **If code is too long** - Show only the critical steps with "..." to indicate omitted parts

Example of GOOD code extraction:
```
# Quicksort algorithm

INPUT: array A, left index l, right index r
OUTPUT: sorted array

1. IF l < r:
   - pivot = partition(A, l, r)
   - quicksort(A, l, pivot - 1)
   - quicksort(A, pivot + 1, r)
2. RETURN A

# Partition function
FUNCTION partition(A, l, r):
   - pivot = A[r]
   - i = l - 1
   - FOR j from l to r-1:
       IF A[j] <= pivot:
           i = i + 1
           swap(A[i], A[j])
   swap(A[i+1], A[r])
   RETURN i + 1
```

Example of BAD code extraction:
```
# DON'T paste entire file (100+ lines)
# DON'T include all imports
# DON'T include error handling
# DON'T include logging
```

## Writing Guidelines

1. **Answer-first**: Lead with the answer/conclusion, then explain
2. **Section by section**: Read one section's summary → find related code → write that section → repeat
3. **Add pseudocode when needed**: Max 1-2 blocks per section, ≤15 lines each
4. **Be specific**: Reference exact file paths, function names, class names
5. **No fabrication**: If you didn't read a file, don't describe its contents
6. **Engaging tone**: Technical but accessible
7. **SEO-friendly**: Include relevant keywords naturally
8. **Keep it brief**: Focus on key points, avoid unnecessary details
9. **Concise sections**: Each section should be at most 2 paragraphs 