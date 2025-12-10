import os
import json
from pathlib import Path
from ChatAgent import ChatAgent  # 假设你的 ChatAgent 类在这个模块下

# ========== 配置部分 ==========
task_base = "../outputs/multi_stage_test/10_papers_human_schemas_eng_prompt_v1.3"
extraction_dir = Path(f"{task_base}/papers/attris")  
paper_dir = Path("../references/test_10papers_mds")            
scored_dir = Path(f"{task_base}/papers/LLM_evaluated") 
output_txt = f"{task_base}/LLM_eval.txt"
scored_dir.mkdir(parents=True, exist_ok=True)

# 读取 prompt 文件（或直接在此定义）
PROMPT_PATH = Path("./prompts/multi_stage/prompt_llm_evaluation.md")  # 就是你上面写的prompt文本
prompt_text = PROMPT_PATH.read_text(encoding="utf-8")

chat_agent = ChatAgent()

# ========== 辅助函数：遍历并收集 score ==========
def collect_scores(obj):
    """
    Recursively traverse obj and collect all score dicts.
    Return lists: consist_list, insight_list, integrity_list (may be empty).
    """
    consist = []
    insight = []
    integrity = []

    if isinstance(obj, dict):
        # if this dict contains a 'score' key with expected ints, collect it
        if "score" in obj and isinstance(obj["score"], dict):
            s = obj["score"]
            # use safe access and ensure ints
            c = s.get("consistency")
            i = s.get("insight")
            g = s.get("integrity")
            if isinstance(c, int):
                consist.append(c)
            if isinstance(i, int):
                insight.append(i)
            if isinstance(g, int):
                integrity.append(g)
        # recurse into all values
        for v in obj.values():
            c2, i2, g2 = collect_scores(v)
            consist.extend(c2)
            insight.extend(i2)
            integrity.extend(g2)
    elif isinstance(obj, list):
        for item in obj:
            c2, i2, g2 = collect_scores(item)
            consist.extend(c2)
            insight.extend(i2)
            integrity.extend(g2)

    return consist, insight, integrity

# ========== 核心执行部分 ==========
def evaluate_extraction_with_llm(extraction_path: Path, paper_path: Path):
    # 读取 extraction JSON
    with open(extraction_path, "r", encoding="utf-8") as f:
        extraction_json = json.load(f)

    # 读取论文文本
    paper_text = Path(paper_path).read_text(encoding="utf-8")

    # 构造输入内容：prompt + 两个变量
    full_input = (
        f"{prompt_text}\n\n"
        f"=== PAPER TEXT START ===\n{paper_text}\n=== PAPER TEXT END ===\n\n"
        f"=== EXTRACTION JSON START ===\n{json.dumps(extraction_json, ensure_ascii=False, indent=2)}\n=== EXTRACTION JSON END ==="
    )

    # 调用模型
    response = chat_agent.remote_chat(
        full_input,
        temperature=0,
        debug=False
    )

    try:
        scored_json = json.loads(response)
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to parse JSON for {extraction_path.name}")
        debug_path = scored_dir / f"{extraction_path.stem}_raw_output.txt"
        debug_path.write_text(response, encoding="utf-8")
        return None

    output_path = scored_dir / extraction_path.name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scored_json, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Scored file saved: {output_path}")

    # 计算并打印本篇文章三个维度的平均值
    consist_list, insight_list, integrity_list = collect_scores(scored_json)
    def mean_or_none(lst):
        return (sum(lst) / len(lst)) if lst else None

    mean_consist = mean_or_none(consist_list)
    mean_insight = mean_or_none(insight_list)
    mean_integrity = mean_or_none(integrity_list)

    return_score = list((mean_consist, mean_insight, mean_integrity))
    # Format output: if None (no scores found), print notice
    if mean_consist is None and mean_insight is None and mean_integrity is None:
        print(f"[WARN] No scores found in {extraction_path.name}")
    else:
        print(f"[STATS] {extraction_path.name} averages -> "
              f"consistency: {mean_consist:.3f}" if mean_consist is not None else f"consistency: N/A",
              f", insight: {mean_insight:.3f}" if mean_insight is not None else f", insight: N/A",
              f", integrity: {mean_integrity:.3f}" if mean_integrity is not None else f", integrity: N/A")

    return scored_json, return_score


# ========== 主循环 ==========
def main():
    extraction_files = sorted(extraction_dir.glob("*.json"))
    total_score = list((0.0, 0.0, 0.0))
    file_num = 0

    output_logs = ""
    for extraction_file in extraction_files:
        paper_file_md = paper_dir / f"{extraction_file.stem}.md"
        paper_file_txt = paper_dir / f"{extraction_file.stem}.txt"

        # 匹配 markdown 或 txt 文件
        if paper_file_md.exists():
            paper_file = paper_file_md
        elif paper_file_txt.exists():
            paper_file = paper_file_txt
        else:
            print(f"[WARN] Paper file not found for {extraction_file.name}")
            continue

        _ , mean_score = evaluate_extraction_with_llm(extraction_file, paper_file)

        total_score = [a + b for a, b in zip(total_score, mean_score)]
        file_num += 1
        output_logs += f"[STATS] {extraction_file.name} averages -> \n"
        output_logs += f"consistency: {mean_score[0]:.3f}" if mean_score[0] is not None else f"consistency: N/A"
        output_logs += f", insight: {mean_score[1]:.3f}" if mean_score[1] is not None else f", insight: N/A"
        output_logs += f", integrity: {mean_score[2]:.3f}\n" if mean_score[2] is not None else f", integrity: N/A\n"
        output_logs += "\n"

    print(total_score)
    print(file_num)
    total_mean_score = [score/file_num for score in total_score]
    print(total_mean_score)

    overall_log = ""
    overall_log += f"[STATS] OVERALL averages -> \n"
    overall_log += f"consistency: {total_mean_score[0]:.3f}" if total_mean_score[0] is not None else f"consistency: N/A"
    overall_log += f", insight: {total_mean_score[1]:.3f}" if total_mean_score[1] is not None else f", insight: N/A"
    overall_log += f", integrity: {total_mean_score[2]:.3f}\n" if total_mean_score[2] is not None else f", integrity: N/A\n"
    print(overall_log)
    output_logs += overall_log

    with open(output_txt, 'w') as f:
        f.write(output_logs)

if __name__ == "__main__":
    main()
