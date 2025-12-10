import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from copy import deepcopy
import math

from ChatAgent import ChatAgent
# -------------------------------------------------
task_base = "../outputs/multi_stage_test/10_papers_human_schemas_eng_prompt_v1.3"
human_extraction_dir = Path("../multi_stage_human/test_10papers_gts_eng")
llm_extraction_dir = Path(f"{task_base}/papers/attris")
out_llm_dir = Path(f"{task_base}/papers/human_alignment")
output_txt = f"{task_base}/human_alignment.txt"
out_llm_dir.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 40
# Prompt template (English) - instruct LLM to return ONLY "1" or "0"
JUDGE_PROMPT = (
    "You are a strict evaluator. Given two text values extracted from the same paper field,\n"
    "decide whether the LLM-extracted value (LLM) is EQUIVALENT TO OR CONTAINS the human-extracted value (HUMAN).\n\n"
    "Rules:\n"
    "1) If HUMAN is null/empty, return 0.\n"
    "2) Comparison should be robust: consider typical paraphrases or re-orderings as possible matches, "
    "3) For lists, consider LLM contains HUMAN if any element of LLM contains an element/text that matches HUMAN.\n"
    "4) Output ONLY a single character: 1 (if LLM equals or contains HUMAN) or 0 (otherwise). No extra text, no punctuation.\n\n"
    "5) If HUMAN extracts more deep insights of the field than LLM, please set the score to 0\n\n"
    "6) The PATH reflects the position of the text in the complete extraction dictionary reflecting which aspect of the text is extracted from the paper.  \n\n"
    "HUMAN:\n<<<\n{human}\n>>>\n\n"
    "LLM:\n<<<\n{llm}\n>>>\n\n"
    "PATH:\n<<<\n{path}\n>>>\n\n"
    "Return 1 or 0 only."
)
def parse_arguments():
    parser = argparse.ArgumentParser(description="argments for retrieve paper")

    parser.add_argument('-p', '--base_dir', type=str, default="/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/intra-work", help="input pdf path")
    parser.add_argument('-t', '--task_id', type=str, default="semantic_scholar_test", help="input pdf path")
    parser.add_argument('-q', '--query', type=str, default="AI automatic overview/survey generation", help="input pdf path")
    parser.add_argument('-b', '--batch_size', type=int, default=40, help="output markdown path")

    return parser.parse_args()

def list_json_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("**/*.json") if p.is_file()])


def collect_value_paths(obj: Any, base_path: List[str] = None) -> List[List[str]]:
    """
    Path format: list of keys and numeric indices-as-strings if any.
    Example: ["extraction", "methods", "0", "hyperparam", "value"]
    """
    if base_path is None:
        base_path = []

    paths: List[List[str]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_base = base_path + [k]
            if k == "value":
                paths.append(new_base)
            else:
                paths.extend(collect_value_paths(v, new_base))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            paths.extend(collect_value_paths(item, base_path + [str(idx)]))
    return paths


def get_by_path(root: Any, path: List[str]) -> Any:
    """
    Robust get by path. Accepts root either being the whole JSON (with "extraction")
    or the inner dict extraction_obj["extraction"]. If the path starts with 'extraction'
    but root doesn't contain it, strip the leading token.
    """
    p = list(path)
    cur = root
    for token in p:
        if isinstance(cur, dict):
            if token not in cur:
                return None
            cur = cur[token]
        elif isinstance(cur, list):
            if not token.isdigit():
                return None
            idx = int(token)
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur


def set_sibling_score(root: Any, path: List[str], score: int):
    """
    Set "human_alignment_score" as a sibling of the "value" node located by path.
    Example: if path ends with ...,"value", then set parent["human_alignment_score"] = score.
    Robust to root being whole JSON or inner extraction dict (handles optional leading 'extraction').
    """
    p = list(path)

    if p[-1] != "value":
        pass

    # Navigate to parent of "value"
    cur = root
    for token in p[:-1]:
        if isinstance(cur, dict):
            if token not in cur:
                raise KeyError(f"Key '{token}' not found while setting score. Available keys: {list(cur.keys())}")
            cur = cur[token]
        elif isinstance(cur, list):
            if not token.isdigit():
                raise KeyError(f"Expected numeric index in path but got '{token}'")
            idx = int(token)
            if idx < 0 or idx >= len(cur):
                raise IndexError(f"Index {idx} out of range when setting score.")
            cur = cur[idx]
        else:
            raise TypeError("Cannot traverse into object of type {}".format(type(cur)))

    # cur should be a dict that contains "value"
    if not isinstance(cur, dict):
        raise TypeError("Parent of 'value' is not a dict; cannot set sibling score.")
    # Set the sibling key (overwrite only this key)
    cur["human_alignment_score"] = score


def prepare_text_for_prompt(val: Any) -> str:
    """
    Convert a value into a string suitable for prompt. For list -> JSON dumps,
    for dict -> JSON dumps, for string -> as-is, for None -> empty string.
    """
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


def process_pairwise_evaluation(human_json: Dict[str, Any], llm_json: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
    llm_new = deepcopy(llm_json)
    # Collect all value paths from human extraction (they should align with llm)
    paths = collect_value_paths(human_json)
    total = 0
    positives = 0
    chat_agent = ChatAgent()

    # Prepare prompts in order, with meta for writing back
    prompts_meta = []  # list of tuples (prompt_text, meta_dict)
    for path in paths:
        human_val = get_by_path(human_json, path)
        llm_val = get_by_path(llm_json, path)

        human_text = prepare_text_for_prompt(human_val)
        llm_text = prepare_text_for_prompt(llm_val)

        meta = {"path": path, "human_text": human_text, "llm_text": llm_text}
        prompts_meta.append(meta)
        total += 1

    # Batch the prompts and send to LLM
    results_by_index = [None] * len(prompts_meta)
    for i in range(0, len(prompts_meta), BATCH_SIZE):
        chunk = prompts_meta[i : i + BATCH_SIZE]
        prompts = []
        for m in chunk:
            # Short-circuit: if human_text is empty, we decide 0 without calling model
            if (m["human_text"] is None) or (str(m["human_text"]).strip() == ""):
                prompts.append(None)
            else:
                prompts.append(JUDGE_PROMPT.format(human=m["human_text"], llm=m["llm_text"], path = m["path"]))
        # Call batch API only for non-None prompts
        # We need to maintain order; create mapping index->response
        # Prepare list of prompts_to_call and map positions
        map_positions = []
        prompts_to_call = []
        for idx_p, p in enumerate(prompts):
            if p is not None:
                map_positions.append(idx_p)
                prompts_to_call.append(p)
        if prompts_to_call:
            res_list = chat_agent.batch_remote_chat(prompts_to_call, desc="judging human-llm alignment")
            # res_list corresponds to prompts_to_call order
            # place responses back to results_by_index accordingly
            for pos_idx, res_text in zip(map_positions, res_list):
                results_by_index[i + pos_idx] = (res_text or "").strip()
        # fill None responses (short-circuited)
        for idx_p, p in enumerate(prompts):
            if p is None:
                results_by_index[i + idx_p] = "0"  # by rule if human empty -> 0

    # Now parse responses and write back scores
    for idx, meta in enumerate(prompts_meta):
        raw_resp = results_by_index[idx] or ""
        score = 0
        # Normalize response: extract first digit 0/1 if possible
        extracted_digit = None
        for ch in raw_resp:
            if ch in ("0", "1"):
                extracted_digit = ch
                break
        if extracted_digit is not None:
            score = int(extracted_digit)
        else:
            # fallback: do simple heuristic: if llm_text contains human_text as substring -> 1
            h = meta["human_text"] or ""
            l = meta["llm_text"] or ""
            try:
                if h and h in l:
                    score = 1
                else:
                    score = 0
            except Exception:
                score = 0

        try:
            set_sibling_score(llm_new, meta["path"], score)
        except Exception as e:
            print(f"[WARN] Failed to set score at path {meta['path']}: {e}")

        if score == 1:
            positives += 1

    return llm_new, total, positives


def process_all_files():
    human_files = list_json_files(human_extraction_dir)
    llm_files = list_json_files(llm_extraction_dir)

    # Build map of human files by relative path
    human_map = {p.relative_to(human_extraction_dir): p for p in human_files}
    llm_map = {p.relative_to(llm_extraction_dir): p for p in llm_files}

    paired = []
    for rel, llm_path in llm_map.items():
        human_path = human_map.get(rel)
        if human_path:
            paired.append((human_path, llm_path))
        else:
            print(f"[WARN] No matching human file for {llm_path}; skipping.")

    grand_total = 0
    grand_positive = 0
    processed = 0

    outcome_logs = ""
    for human_path, llm_path in paired:
        print(f"Processing pair: {human_path} <-> {llm_path}")
        with open(human_path, "r", encoding="utf-8") as f:
            human_json = json.load(f)
        with open(llm_path, "r", encoding="utf-8") as f:
            llm_json = json.load(f)

        llm_new, total, positives = process_pairwise_evaluation(human_json, llm_json)
        grand_total += total
        grand_positive += positives
        processed += 1

        # write out modified llm file to out_llm_dir, preserving relative path
        rel = llm_path.relative_to(llm_extraction_dir)
        out_path = out_llm_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(llm_new, f, ensure_ascii=False, indent=2)

        print(f" -> values: {total}, positives: {positives}, portion: {positives/total},  written to {out_path}")

        outcome = f"Processing pair: {human_path} <-> {llm_path}\n"
        outcome += f" -> values: {total}, positives: {positives}, portion: {positives/total},  written to {out_path}\n\n"
        outcome_logs += outcome

    overall_pct = (grand_positive / grand_total * 100.0) if grand_total > 0 else 0.0
    print("======================================")
    print(f"Processed {processed} file pairs.")
    print(f"Total values compared: {grand_total}")
    print(f"Total positives (score=1): {grand_positive}")
    print(f"Overall alignment rate: {overall_pct:.2f}%")
    print("Modified LLM extractions saved under:", out_llm_dir)

    with open(output_txt, 'w') as f:
        f.write(outcome_logs)


if __name__ == "__main__":
    args = parse_arguments()
    process_all_files()
