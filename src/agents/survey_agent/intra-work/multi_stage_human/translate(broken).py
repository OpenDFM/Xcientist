"""
translate_schema_extraction.py

Usage:
    - Set schema_dir and extraction_dir to your input folders
    - Set out_schema_dir and out_extraction_dir to desired output folders (will be created)
    - Ensure ChatAgent is importable and you can instantiate chat_agent = ChatAgent()
    - Run: python translate_schema_extraction.py
"""

import json
import os
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Tuple

# ---- Configuration: edit these paths ----
schema_dir = Path("./schemas")
extraction_dir = Path("./gt")
out_schema_dir = Path("./schemas_eng")
out_extraction_dir = Path("./gts_eng")
# -----------------------------------------

# Ensure output dirs exist
out_schema_dir.mkdir(parents=True, exist_ok=True)
out_extraction_dir.mkdir(parents=True, exist_ok=True)

# Instantiate your ChatAgent here (adjust to your environment)
import sys 
sys.path.append("..") 
from ChatAgent import ChatAgent  # <-- change to your actual import
chat_agent = ChatAgent()


# ---- PROMPT TEMPLATES (English) ----
SCHEMA_TRANSLATION_PROMPT = (
    "Translate the following Chinese text into fluent, natural English.\n"
    "This text is the content of a schema field (either 'description' or 'rationale').\n"
    "Requirements:\n"
    " - Output ONLY the translated text and nothing else (no explanations, no quotes).\n"
    " - Preserve punctuation, code-like tokens, percent signs, backslashes, and JSON-safe characters.\n"
    " - Keep the translation concise and domain-appropriate (scientific/technical register).\n\n"
    "Source text:\n"
    "<<<\n{source}\n>>>\n\n"
    "Return the translated text now."
)

VALUE_TRANSLATION_PROMPT = (
    "Translate the following Chinese text into fluent, natural English.\n"
    "This text is the 'value' field extracted from an academic paper JSON. It may be a sentence, paragraph, or short phrase.\n"
    "Requirements:\n"
    " - Output ONLY the translated text and nothing else (no explanations, no quotes).\n"
    " - Preserve punctuation, inline code tokens, percent signs, backslashes, and other special characters.\n"
    " - If the input is a short list item or an author name, preserve capitalization/ordering meaningfully.\n\n"
    "Source text:\n"
    "<<<\n{source}\n>>>\n\n"
    "Return the translated text only."
)
# -------------------------------------

# Utility: walk and collect files
def list_json_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("**/*.json") if p.is_file()])


# Schema: collect all translatable texts (description & rationale)
def collect_schema_texts(schema_obj: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    """
    Return list of tuples (text_to_translate, location_path)
    location_path is a list describing how to find it in the schema dict, e.g.
      ["schema_proposal", "method", "rationale"]
    """
    res = []
    sp = schema_obj.get("schema_proposal", {})
    for key, meta in sp.items():
        # description
        if isinstance(meta, dict):
            if "description" in meta and isinstance(meta["description"], str):
                res.append((meta["description"], ["schema_proposal", key, "description"]))
            if "rationale" in meta and isinstance(meta["rationale"], str):
                res.append((meta["rationale"], ["schema_proposal", key, "rationale"]))
            # if this field defines sub_keys, iterate them
            sub_keys = meta.get("sub_keys")
            if isinstance(sub_keys, dict):
                for sk, sk_meta in sub_keys.items():
                    if isinstance(sk_meta, dict):
                        if "description" in sk_meta and isinstance(sk_meta["description"], str):
                            res.append((sk_meta["description"], ["schema_proposal", key, "sub_keys", sk, "description"]))
                        if "rationale" in sk_meta and isinstance(sk_meta["rationale"], str):
                            res.append((sk_meta["rationale"], ["schema_proposal", key, "sub_keys", sk, "rationale"]))
    return res


# Extraction: recursively find all "value" keys and collect translatable ones
def collect_extraction_values(obj: Any, base_path: List[str] = None) -> List[Tuple[str, List[str]]]:
    """
    Traverse extraction object and return list of tuples (text_to_translate, location_path)
    location_path: list of keys / indices to locate the value (e.g., ["extraction", "method", "hyperparams", "lr", "value"])
    Only collect string values or lists of strings (for lists, we will produce separate entries per element
    and mark their path with an index).
    """
    if base_path is None:
        base_path = []

    collected = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = base_path + [k]
            if k == "value":
                # v could be string, list, number, None, dict
                if isinstance(v, str):
                    collected.append((v, new_path))
                elif isinstance(v, list):
                    # collect each string element
                    for idx, item in enumerate(v):
                        if isinstance(item, str):
                            collected.append((item, new_path + [str(idx)]))
                elif isinstance(v, dict):
                    # If value itself is object, recurse inside it
                    collected.extend(collect_extraction_values(v, new_path))
                else:
                    # numbers / null / bool -> skip
                    pass
            else:
                collected.extend(collect_extraction_values(v, new_path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            collected.extend(collect_extraction_values(item, base_path + [str(idx)]))
    return collected


# Helper: write translated text back to schema
def write_back_schema_translation(schema_obj: Dict[str, Any], path: List[str], translated: str):
    # Navigate and set the value at path
    cur = schema_obj
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = translated


# Helper: write translated text back to extraction
def write_back_extraction_translation(extraction_obj: Dict[str, Any], path: List[str], translated: str):
    # Path elements might include numeric indices (as strings). We need to handle lists.
    cur = extraction_obj
    for p in path[:-1]:
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = cur[p]
    last = path[-1]
    if last.isdigit():
        cur[int(last)] = translated
    else:
        cur[last] = translated



# Batch send prompts to LLM and return responses
def batch_translate(prompts_and_index: List[Tuple[str, Any]], desc: str, max_retries: int = 3):
    """
    prompts_and_index: list of tuples (prompt_str, meta)
    returns list of tuples (meta, translated_text) in the same order
    """
    results = []
    remaining = prompts_and_index.copy()
    attempt = 0
    while remaining and attempt < max_retries:
        prompts = [p for p, _ in remaining]
        metas = [m for _, m in remaining]
        # call your ChatAgent batch API
        res_list = chat_agent.batch_remote_chat(prompts, desc=desc)
        # res_list should be parallel to prompts
        new_remaining = []
        for res, meta in zip(res_list, metas):
            text = (res or "").strip()
            if text:
                results.append((meta, text))
            else:
                # schedule for retry
                new_remaining.append((SCHEMA_TRANSLATION_PROMPT.format(source=meta["source_text"]) if meta.get("kind")=="schema" else VALUE_TRANSLATION_PROMPT.format(source=meta["source_text"]), meta))
        remaining = new_remaining
        attempt += 1
    # Note: results may be unordered relative to inputs; return in mapping order
    return results


# Main processing functions
def process_schema_file(path: Path, out_path: Path):
    with open(path, "r", encoding="utf-8") as f:
        schema_obj = json.load(f)
    to_translate = collect_schema_texts(schema_obj)  # list of (text, path)
    prompts_and_index = []
    # prepare prompts with meta info
    for text, loc in to_translate:
        prompt = SCHEMA_TRANSLATION_PROMPT.format(source=text)
        meta = {"file": str(path), "path": loc, "source_text": text, "kind": "schema"}
        prompts_and_index.append((prompt, meta))
    translated_pairs = []
    if prompts_and_index:
        # We will batch in chunks to avoid huge requests
        CHUNK = 40
        for i in range(0, len(prompts_and_index), CHUNK):
            chunk = prompts_and_index[i : i + CHUNK]
            prompts = [p for p, _ in chunk]
            metas = [m for _, m in chunk]
            res_list = chat_agent.batch_remote_chat(prompts, desc="Translating schema description/rationale to English")
            for res, meta in zip(res_list, metas):
                translated_pairs.append((meta, (res or "").strip()))
    # write back translations
    new_schema = deepcopy(schema_obj)
    for meta, translated in translated_pairs:
        write_back_schema_translation(new_schema, meta["path"], translated)
    # save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_schema, f, ensure_ascii=False, indent=2)


def process_extraction_file(path: Path, out_path: Path):
    with open(path, "r", encoding="utf-8") as f:
        extraction_obj = json.load(f)
    to_translate = collect_extraction_values(extraction_obj)
    prompts_and_index = []
    for text, loc in to_translate:
        prompt = VALUE_TRANSLATION_PROMPT.format(source=text)
        meta = {"file": str(path), "path": loc, "source_text": text, "kind": "value"}
        prompts_and_index.append((prompt, meta))
    translated_pairs = []
    if prompts_and_index:
        CHUNK = 40
        for i in range(0, len(prompts_and_index), CHUNK):
            chunk = prompts_and_index[i : i + CHUNK]
            prompts = [p for p, _ in chunk]
            metas = [m for _, m in chunk]
            res_list = chat_agent.batch_remote_chat(prompts, desc="Translating extraction 'value' fields to English")
            for res, meta in zip(res_list, metas):
                translated_pairs.append((meta, (res or "").strip()))
    # write back into copy
    new_extraction = deepcopy(extraction_obj)
    print("DEBUG: ", new_extraction.keys())
    for meta, translated in translated_pairs:
        write_back_extraction_translation(new_extraction, meta["path"], translated)
    # save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_extraction, f, ensure_ascii=False, indent=2)


def main():
    schema_files = list_json_files(schema_dir)
    extraction_files = list_json_files(extraction_dir)

    print(f"Found {len(schema_files)} schema files and {len(extraction_files)} extraction files.")

    # Process schemas
    # for p in schema_files:
    #     rel = p.relative_to(schema_dir)
    #     out_p = out_schema_dir / rel
    #     out_p.parent.mkdir(parents=True, exist_ok=True)
    #     print(f"Processing schema: {p} -> {out_p}")
    #     process_schema_file(p, out_p)

    # Process extractions
    for p in extraction_files:
        rel = p.relative_to(extraction_dir)
        out_p = out_extraction_dir / rel
        out_p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Processing extraction: {p} -> {out_p}")
        process_extraction_file(p, out_p)

    print("Done.")


if __name__ == "__main__":
    main()
