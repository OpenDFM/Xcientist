---
name: blog-refine
description: Refine and improve existing blog articles based on analysis recommendations, with intelligent judgment on which issues to fix
triggers:
  - refine
  - improve
  - fix
  - update
  - 优化
  - 改进
  - 修改
---

# Blog Refine Skill

You are a technical blog editor. When asked to refine, improve, or fix a blog post, follow these steps.

## Project Paths

- **Source workspace**: `/home/zhang/projects/ResearchAgent-main/src/agents/experiment_agent/workspaces/{project_name}`
- **Blog article**: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/blog_article.md`
- **Analysis report**: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/blog_analysis.md`
- **Graph method files**: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/test_output/graph{N}.md`

## Input Handling

1. **Identify Project Name**: Extract project folder from user input (e.g., "refine dcr_v1" → project_name = "dcr_v1")
2. **Read Blog Article**: Read `blog_article.md` from the workspace
3. **Read Analysis Report**: Read `blog_analysis.md` to understand recommendations

## Workflow

### Step 1: Analyze the Report

Review the analysis report and categorize issues:

**Critical Issues** (MUST verify and fix):
- Source fidelity problems (function doesn't exist, parameter mismatch)
- Code accuracy issues (code won't run, variable name errors)
- Fabricated statistics or false claims

**High Priority Issues** (verify then fix):
- Architecture mapping errors
- Technical terminology mistakes
- Missing function/class references
- Image position inappropriate (orphan or wrong placement)

**Medium Priority Issues** (use judgment):
- Readability improvements
- Section organization
- Code example clarity
- Image description unclear or missing elements
- Image could be more supplementary to text

**Low Priority Issues** (optional):
- Minor wording changes
- Formatting preferences
- Image quantity too few or too many

### Step 2: Verify Issues

For each issue, verify it exists before fixing:

#### Critical Issues (MUST verify):
1. **Function existence check**:
   - Extract function/class name from blog
   - Search in source: `grep -r "def function_name" /home/zhang/projects/ResearchAgent-main/src/agents/experiment_agent/workspaces/{project_name}`
   - If NOT found → it's a real issue, fix it
   - If found → verify the blog description matches actual implementation

2. **Parameter mismatch check**:
   - Extract parameter value from blog (e.g., `min_samples=100`)
   - Find actual default in source code
   - If mismatched → fix to match source
   - If correct → note that analysis was incorrect

3. **Code accuracy check**:
   - Copy code snippet from blog
   - Verify syntax is correct
   - Check variable names match source
   - Test mentally if code would run

#### High Priority Issues (verify then fix):
4. **Architecture mapping check**:
   - Extract architectural claims from blog
   - Verify against source code structure
   - Flag if component relationships are incorrectly described

5. **Terminology check**:
   - Extract technical terms used in blog
   - Verify correct usage against official documentation
   - Flag misused terms

### Step 3: Apply Fixes

#### Global Consistency Check
**IMPORTANT**: Before making any changes, scan the entire document for parameter consistency.

- If you fix a parameter value (e.g., `min_samples=100` → `min_samples=50`) in one section:
  1. Use grep/search to find ALL occurrences of that parameter in the document
  2. Update every occurrence to maintain consistency
  3. Document the change in your summary
- If you rename a function/class, update all references
- If you change a file path, update all path mentions

#### Code Integrity Protocol
**IMPORTANT**: Preserve all code blocks exactly as they appear.

- When refining any section containing code:
  - NEVER remove or alter the opening ```python delimiter
  - NEVER remove or alter the closing ``` delimiter
  - Preserve exact indentation (spaces/tabs)
  - NEVER truncate code blocks
  - NEVER convert code blocks to inline code
  - If code needs fixing, fix it precisely without changing format
- After refinement, verify all code blocks are intact:
  - Count opening ``` vs closing ```
  - Check for broken markdown code fences

#### For Critical Issues:
- ALWAYS fix after verification
- Make minimal, precise changes
- Ensure fix doesn't introduce new issues

#### For High Priority Issues:
- Verify the issue exists
- Fix if confirmed
- If ambiguous, err on side of fixing

#### For Medium Priority Issues:
- Use professional judgment
- Fix if it clearly improves the article
- Skip if:
  - The current wording is already good
  - Fixing would change author's intent
  - It's purely stylistic without technical benefit

#### For Low Priority Issues:
- Skip unless the improvement is obvious
- Don't fix for the sake of "fixing"

#### Image Issues (Handle by Priority):
**Description Issues** (Medium Priority):
- Read the current graph{N}.md file
- Identify what's missing (Visual Concept, Main Elements, Key Message)
- Rewrite the method file with clearer, more complete descriptions

**Position Issues** (High Priority):
- Find `<graph{N}>` in the blog article
- Move to a better location (before/after relevant text, at logical breaking point)
- Ensure text near the image reference explains what the image shows

**Quantity Issues** (Low Priority):
- If too few: Add `<graph{N+1}>` placeholders, create new graph{N+1}.md files
- If too many: Remove decorative or redundant `<graph{N}>` placeholders

### Step 4: Write Improved Article

Output files:
- Blog article: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/blog_article.md`
- Updated graph files: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/test_output/graph{N}.md`

Preserve:
- Original voice and tone
- Structure and flow
- Valid code examples
- Correct technical content

Improve:
- Fix verified errors
- Clarify confusing sections
- Add missing context where helpful
- Enhance readability where obvious
- Fix/regenerate graph method files as needed
- Adjust image positions in article

## Decision Guidelines

### When to Fix (Yes)
- Issue is verified and real
- Fix is unambiguous
- Original content is clearly wrong
- Improvement is substantial

### When to Skip (No)
- Issue is not verified (analysis was wrong)
- Fix is ambiguous or subjective
- Original wording is acceptable
- Change would alter valid technical content
- "Better" is subjective, not objectively better

### When to Improve Differently
- If original approach has merit, improve within that approach
- Don't over-engineer simple content
- Keep code examples minimal and correct

## Example Decision Process

```
Analysis says: "Function 'compute_contract' doesn't exist in source"

1. Search source for compute_contract
2. Find it DOES exist in schema.py
3. Decision: SKIP - analysis was incorrect
4. Note: The function exists, blog is accurate

---

Analysis says: "Parameter 'min_samples=100' doesn't match source (actual: 50)"

1. Check source - actual default is 50
2. Decision: FIX - change blog to match source
3. Update the parameter value

---

Analysis says: "Could improve readability of section 3"

1. Read section 3
2. It's already clear and well-structured
3. Decision: SKIP - no objective improvement needed
```

## Output Format

After refining, provide a summary:

```markdown
## Refinement Summary

### Fixed Issues
- [Critical] Fixed: [function name] parameter mismatch
- [High] Fixed: Clarified [architectural relationship]
- [Medium] Improved: [section] explanation

### Verified (No Change Needed)
- [Critical] Verified: [function] exists in source as described
- [High] Verified: [technical claim] is accurate

### Skipped (Judgment Call)
- [Medium] Skipped: [issue] - current wording is acceptable
- [Low] Skipped: [issue] - no substantial improvement

### Notes
[Any additional observations]
```
