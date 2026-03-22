# script_char_based.py
# pip install openai

from openai import OpenAI
import json
# ========== CONFIG ==========
client = OpenAI(
    api_key="sk-c1n2jxt2llyucdaac9g09zflyijszu803gr7hhzv083oj2gq",
    base_url="https://api.xiaomimimo.com/v1/"
)
MODEL = "mimo-v2-flash"
# ============================
# ========== CONFIG ==========
client = OpenAI(
    api_key="sk-UCfwgl63Xg27JF8W33D746F3B80d4862979c82A51951485f",
    base_url="http://122.193.22.114:8889/v1/"
)
MODEL = "gpt-4o-mini"
# ============================

text = """Large language models (LLMs) have shown remarkable capabilities in natural language processing tasks. 
Recent studies such as GPT-4 and PaLM-2 demonstrate strong performance across benchmarks like MMLU and BIG-Bench. 
However, these models still suffer from hallucination, reasoning instability, and high computational cost. 
For example, a model may generate fluent but factually incorrect statements when prompted with ambiguous questions. 
Therefore, improving reliability and efficiency remains an open research challenge.
"""

prompt = f"""
You are a revise assistant.

Document:
{text}

Revision request:
- Reduce absolute claims
- Improve academic tone
- Make wording more concise

Output ONLY a JSON array of edits.
Each edit must follow this schema:
{{
  "type": "replace",
  "start": <int>,     // zero-based character index
  "end": <int>,       // exclusive
  "originalText": "<string>",
  "newText": "<string>"
}}

Use character positions based on the exact input text.
"""

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

edits = json.loads(resp.choices[0].message.content)

# apply edits (descending order)
for e in sorted(edits, key=lambda x: x["start"], reverse=True):
    s, t = e["start"], e["end"]
    assert text[s:t] == e["originalText"], "Original text mismatch!"
    text = text[:s] + e["newText"] + text[t:]

print("==== FINAL TEXT (CHAR-BASED) ====\n")
print(text)
