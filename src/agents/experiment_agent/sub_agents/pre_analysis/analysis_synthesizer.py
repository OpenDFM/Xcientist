"""
Analysis Synthesizer - Synthesizes concept and algorithm analysis results.

This agent merges the outputs from concept and algorithm analyzers into a 
structured JSON output matching PreAnalysisOutput schema.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.pre_analysis.output_schemas import (
    PreAnalysisOutput,
)


# Hand-written JSON output instruction for PreAnalysisOutput
PRE_ANALYSIS_JSON_INSTRUCTION = """
## Required JSON Output Format: PreAnalysisOutput

You MUST output a JSON object with this EXACT structure:

```json
{
  "input_type": "paper",
  "system_architecture": "The system uses a hierarchical encoder-decoder architecture with attention mechanisms. Main components: 1) Multi-scale encoder for feature extraction, 2) Cross-attention module for feature fusion, 3) Decoder with skip connections.",
  "conceptual_framework": "Based on variational inference theory, the model learns a latent representation that captures semantic features. Key principles: information bottleneck, disentanglement, reconstruction fidelity.",
  "key_innovations": "1. Novel attention mechanism that reduces complexity from O(n²) to O(n log n)\\n2. Hierarchical latent space with progressive refinement\\n3. Contrastive loss for better feature separation",
  "algorithms": "Algorithm 1: Training Loop\\n1. Sample batch from dataset\\n2. Encode to latent space z = E(x)\\n3. Apply attention: z' = Attention(z, z)\\n4. Decode: x' = D(z')\\n5. Compute loss: L = MSE(x, x') + KL(z||prior)\\n6. Backpropagate and update",
  "mathematical_formulations": "Loss function: L = E[||x - D(E(x))||²] + β·KL(q(z|x)||p(z))\\nWhere E is encoder, D is decoder, β is weighting factor\\nAttention: A(Q,K,V) = softmax(QK^T/√d)V",
  "technical_specifications": "Input: 224×224 RGB images\\nEncoder: ResNet-18 backbone, output 512-dim features\\nLatent dim: 128\\nDecoder: 4 transposed conv layers with batch norm\\nTraining: Adam optimizer, lr=1e-4, batch_size=64",
  "summary": "This paper presents a novel VAE architecture with efficient attention for image synthesis. Key contribution is the O(n log n) attention mechanism that enables scaling to high-resolution images while maintaining quality.",
  "implementation_guidance": "1. Start with standard VAE implementation\\n2. Add attention module after encoder\\n3. Use pretrained ResNet as encoder backbone\\n4. Implement progressive training: start with low resolution, gradually increase\\n5. Monitor KL divergence to avoid posterior collapse",
  "code_repos_info": "Reference repos:\\n1. pytorch-vae (repos/pytorch-vae): Standard VAE implementation, useful for base architecture\\n2. attention-models (repos/attention): Contains efficient attention implementations"
}
```

### Field Descriptions:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_type` | string | YES | "paper" or "idea" |
| `system_architecture` | string | YES | High-level architecture and design patterns |
| `conceptual_framework` | string | YES | Theoretical foundations and principles |
| `key_innovations` | string | YES | Novel contributions and innovations |
| `algorithms` | string | YES | Core algorithms and procedures |
| `mathematical_formulations` | string | YES | Mathematical models and formulas |
| `technical_specifications` | string | YES | Implementation-level details |
| `summary` | string | YES | Executive summary of analysis |
| `implementation_guidance` | string | YES | Strategic implementation advice |
| `code_repos_info` | string | NO | Analysis of reference repositories |

### Content Guidelines:
- Preserve ALL technical details, LaTeX formulas, and code snippets
- Do NOT truncate or over-summarize
- `summary` and `implementation_guidance` should be YOUR synthesis
- Other fields should be EXTRACTED from input analyses

⚠️ **CRITICAL**: Output ONLY valid JSON, no markdown explanations!
"""


def create_analysis_synthesizer(model: str = "gpt-4o") -> Agent:
    """
    Create a synthesizer agent that merges analysis results and outputs JSON.

    Args:
        model: The model to use for the agent

    Returns:
        Agent instance configured for analysis synthesis with JSON output
    """
    instructions = f"""You are an expert research synthesizer that outputs structured JSON.

YOUR TASK:
Synthesize Concept Analysis, Algorithm Analysis, and Code Repository Analysis into a unified JSON output matching the PreAnalysisOutput schema.

INPUT FORMAT:
You will receive:
1. INPUT_TYPE: "paper" or "idea"
2. CONCEPT ANALYSIS: System architecture, conceptual framework, key innovations
3. ALGORITHM ANALYSIS: Algorithms, mathematical formulations, technical details
4. CODE REPOSITORIES ANALYSIS: Repository information and relevance

OUTPUT REQUIREMENTS:
You MUST output a valid JSON object with these exact fields:

{{
    "input_type": "<paper or idea>",
    "system_architecture": "<extracted from concept analysis>",
    "conceptual_framework": "<extracted from concept analysis>",
    "key_innovations": "<extracted from concept analysis>",
    "algorithms": "<extracted from algorithm analysis>",
    "mathematical_formulations": "<extracted from algorithm analysis>",
    "technical_specifications": "<extracted from algorithm analysis>",
    "summary": "<your executive summary synthesizing all analyses>",
    "implementation_guidance": "<strategic implementation advice incorporating code repo insights>",
    "code_repos_info": "<extracted from code repositories analysis>"
}}

CRITICAL RULES:
1. Output ONLY the JSON object - no markdown, no explanation
2. Preserve all technical details, LaTeX formulas, and code snippets from the input
3. Do NOT truncate or summarize the extracted content - preserve full detail
4. The summary and implementation_guidance fields should be YOUR synthesis
5. All other fields should be EXTRACTED from the corresponding input sections

{PRE_ANALYSIS_JSON_INSTRUCTION}
"""
    return Agent(
        name="Analysis Synthesizer",
        instructions=instructions,
        model=model,
    )


# Default agent instance
analysis_synthesizer = create_analysis_synthesizer()
