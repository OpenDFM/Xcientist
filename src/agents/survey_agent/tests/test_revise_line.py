# script_line_based_fixed.py
# pip install openai nltk

import json
import re
from openai import OpenAI
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# ========== CONFIG ==========
client = OpenAI(
    api_key="sk-c1n2jxt2llyucdaac9g09zflyijszu803gr7hhzv083oj2gq",
    base_url="https://api.xiaomimimo.com/v1/"
)
MODEL = "mimo-v2-flash"
# ============================
# ========== CONFIG ==========
# client = OpenAI(
#     api_key="sk-UCfwgl63Xg27JF8W33D746F3B80d4862979c82A51951485f",
#     base_url="https://api.xi-ai.cn/v1/"
# )
# MODEL = "gpt-4o-mini"
# ============================

original_text = """
    1. The Modality Interface: From Linear Projections to Deep
    Fusion and Generative Unification\n\nThe architectural design of the modality       
    interface—the bridge that connects visual encoders with Large Language Models       
    (LLMs)—has evolved into a central battleground for defining the capabilities of     
    Multimodal Large Language Models (MLLMs). While early efforts, such as **<Visual    
    Instruction Tuning>**, demonstrated the viability of connecting a visual encoder to 
    an LLM using a simple linear projection layer, the field has rapidly shifted toward 
    more sophisticated architectures. These newer designs aim to handle the "modality   
    gap" more effectively, resulting in a spectrum of interfaces ranging from           
    specialized compression modules to deep, interleaved fusion strategies.\n\nA        
    prominent approach involves the use of specialized modules to map visual tokens into
    the semantic space of the LLM. **<MiniGPT-4>** utilizes Q-Former (inspired by       
    BLIP-2) to learn a set of learnable queries that aggregate visual features before   
    they are fed into the LLM. This method proved that treating visual input as a       
    sequence of compressed, highly relevant features allows an otherwise frozen LLM to  
    understand complex modalities, unlocking emergent capabilities like website         
    generation from sketches. Similarly, **<KOSMOS-1>** adopts a Perceiver-like         
    architecture to handle interleaved multimodal inputs within a single Transformer. By
    employing a mapping network and special boundary tokens, it effectively unifies     
    modalities. However, models relying on complex, pre-trained feature compressors like
    Q-Former (as seen in **<BLIVA>**, which integrates Q-Former-style embeddings with   
    patch embeddings to better handle text-rich images) often introduce significant     
    parameter overhead and latency.\n\nIn reaction to this complexity, recent research  
    has sought to simplify the interface without sacrificing performance.               
    **<mPLUG-Owl2>** introduces a "modality-adaptive module" that preserves             
    modality-specific features, arguing that strict fusion can degrade text-only        
    capabilities. It decouples the visual pathway while still sharing a universal       
    language decoder. Conversely, **<FIBER>** argues for "Fusion in the Backbone."      
    Instead of a separate connector, it interleaves cross-attention layers directly into
    the LLM’s encoder and decoder context. This deep fusion allows the model to reason  
    about visual dependencies dynamically throughout the generation process, rather than
    relying on a static pre-fusion step. This trade-off—between specialized, compressed 
    encoders (**<MiniGPT-4>**) and deep, integrated fusion (**<FIBER>**)—defines the    
    current tension in interface design.\n\nSimultaneously, a radical shift in          
    philosophy has emerged, epitomized by **<VisionLLM>**. This work proposes treating  
    images as a "foreign language." Rather than designing complex connectors, it aligns 
    vision-centric tasks directly with the LLM’s language processing capabilities. It   
    uses a language-guided image tokenizer and treats outputs (like bounding boxes) as  
    text tokens, effectively unifying the architecture into a pure generative text      
    decoder. This approach suggests that the modality interface might eventually        
    disappear entirely, with the LLM handling all modalities natively. **<GIT>** further
    supports this generative unification, achieving state-of-the-art performance in     
    image-to-text generation with a simple encoder-decoder architecture that requires no
    complex modality-specific modules. This evolution from simple linear projections    
    (**<Visual Instruction Tuning>**) to specialized compressors (**<MiniGPT-4>**,      
    **<KOSMOS-1>**), deep fusion (**<FIBER>**, **<mPLUG-Owl2>**), and finally generative
    unification (**<VisionLLM>**, **<GIT>**) represents the field’s search for the most 
    efficient way to bridge the visual and linguistic worlds.\n\nHowever, architectural 
    innovation cannot compensate for insufficient training data, a point emphasized by  
    **<Static Augmentation>**. This paper argues that standard supervised fine-tuning   
    (SFT) data is often too narrow, limiting the model\'s ability to generalize across  
    different instruction styles. This highlights that the "interface" is not just      
    architecture; it is also the instructional data that teaches the model how to       
    utilize that architecture. Even the most elegant connector, such as the "Spatial    
    Vision Aggregator (SVA)" proposed in **<Cambrian-1>**, relies on a massive,         
    high-quality dataset to fine-tune the interaction between visual backbones and the  
    LLM.\n\nFurthermore, the industry is trending toward architectures that prioritize  
    "visual grounding"—explicitly linking text to visual regions. **<KOSMOS-2>**        
    introduced grounding capabilities by training on grounded image-text pairs, allowing
    the model to output `<region>` tags corresponding to bounding boxes. While not      
    strictly an architectural connector change, it alters the output interface to       
    enforce alignment between the visual and textual streams. Similarly, **<UniTAB>**   
    Unifies Text and Box outputs, proposing a shared token sequence that allows for     
    grounded captioning. The inclusion of an `<obj>` token represents a structural      
    change to the interface, enabling the model to explicitly align generated words with
    visual locations.\n\nThe trajectory of research suggests that while early interfaces
    focused on "how to connect" (via linear layers or projectors), the current focus is 
    on "how to integrate." **<mPLUG-Owl2>** demonstrates that preserving                
    modality-specific features is crucial for general intelligence, while **<FIBER>**   
    suggests that dynamic fusion is key for fine-grained tasks. As we look at the       
    evolution from **<Visual Instruction Tuning>**\'s linear layer to **<VisionLLM>**\'s
    generative decoder, the interface is transforming from a static bridge into a       
    dynamic, intelligent router that determines which modality to prioritize and how to 
    fuse context effectively.\n\nThe limitations of current interfaces are also becoming
    apparent in specific domains. For example, **<BLIVA>** addresses the failure of     
    generalist connectors in text-rich images by explicitly injecting patch embeddings  
    alongside learned query embeddings. This suggests that "one size fits all"          
    interfaces may not be optimal. As the line between the visual encoder and the       
    language model blurs, exemplified by **<Downscaling Intelligence>**\'s proposal for 
    visual extraction tuning, the modality interface is effectively becoming part of the
    reasoning engine itself, rather than a mere data conduit.\n\nIn conclusion, the     
    modality interface has moved from a trivial linear layer to a core architectural    
    component defining the model\'s reasoning capabilities. The choice between heavy,   
    pre-trained connectors and deep, integrated fusion reflects a fundamental debate on 
    whether visual processing should be heuristically compressed before reaching the LLM
    or treated as an integral part of the reasoning process. Works like **<VisionLLM>** 
    and **<GIT>** suggest the ultimate interface may be the absence of one—a unified    
    generative space—while current practical architectures like **<Cambrian-1>** and    
    **<FIBER>** are pushing the boundaries of sophisticated, multi-stage fusion to      
    maximize performance on complex reasoning tasks.\n\n(Word count: ~1650 words)'
    """

# 1) Split into sentences (one sentence per 'line')
lines = sent_tokenize(original_text)
lines_text = ""
for i, line in enumerate(lines):
    lines_text += f"'line {i+1}': '{line}'\n"
input = {}
for i, line in enumerate(lines):
    input[f"line{i}"] = line
# 2) Build prompt (English) — we pass the JSON array of lines so the model sees line indexes clearly
prompt = f"""
You are a revise assistant. The document has been split into sentences (one per line), indexed from 0.

Document lines (JSON array):
{lines_text}

Revision requests:
- This text is too short for a scientific survey introduction. Please expand and improve it by expanding on the challenges and recent developments in the field.
- There are some error and annotation like word count, remove them.
- The correct form for citation is <title>(like <Attention is All you need). Correct the wrong citation forms. Do not omit any citation.
- Improve academic tone.
- Modify the necessary lines to achieve the above.
- Output in strict JSON format.

Output ONLY a JSON array of edits (no prose). Each edit must be an object with:
{{ "type": "replace" | "insert_before" | "insert_after" | "delete",
   "line": <int>,
   "newText": "<string>"
}}

Example (allowed):
[{{"type":"replace","line":0,"newText":"..."}}, ...]
"""

print("=== PROMPT ===")
print(prompt)

# 3) Call the model
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role":"user","content":prompt}],
    temperature=0
)

raw = resp.choices[0].message.content

# Helper: extract first JSON array from model text (robust to ``` fences or extra commentary)
def extract_first_json_array(s: str):
    # find first '[' and matching ']' (simple approach)
    start = s.find('[')
    if start == -1:
        raise ValueError("No JSON array found in model response.")
    # try to find a balanced bracket end
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '[':
            depth += 1
        elif s[i] == ']':
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    raise ValueError("Could not extract balanced JSON array.")

# 4) Parse edits JSON (with fallback)
try:
    edits_json = json.loads(raw)
except Exception:
    try:
        arr_text = extract_first_json_array(raw)
        edits_json = json.loads(arr_text)
    except Exception as e:
        raise RuntimeError("Failed to parse JSON from model response.") from e

# 5) Apply edits (validate line indices, apply in descending order)
def apply_line_edits(lines, edits):
    # allowed ops: replace, insert_before, insert_after, delete
    # sort by line desc so changes don't affect earlier indices in this simple model
    applied = []
    blocked = []
    for e in sorted(edits, key=lambda x: x.get("line", 0), reverse=True):
        typ = e.get("type")
        ln = e.get("line")
        new = e.get("newText", "")
        if not isinstance(ln, int) or ln < 0 or ln >= len(lines):
            e["error"] = "invalid_line_index"
            blocked.append(e)
            continue
        if typ == "replace":
            lines[ln] = new
            applied.append(e)
        elif typ == "insert_before":
            lines[ln] = new + " " + lines[ln]
            applied.append(e)
        elif typ == "insert_after":
            lines[ln] = lines[ln] + " " + new
            applied.append(e)
        elif typ == "delete":
            lines[ln] = ""
            applied.append(e)
        else:
            e["error"] = "unsupported_op"
            blocked.append(e)
    return lines, applied, blocked

lines_after, applied, blocked = apply_line_edits(lines.copy(), edits_json)

# 6) Final text: join with a single space to form a paragraph
final_text = " ".join([ln.strip() for ln in lines_after if ln.strip() != ""])

print("=== APPLIED EDITS ===")
print(json.dumps(applied, ensure_ascii=False, indent=2))
print("\n=== BLOCKED EDITS ===")
print(json.dumps(blocked, ensure_ascii=False, indent=2))
print("\n=== FINAL TEXT ===\n")
print(final_text)
