import json
import argparse
from pathlib import Path

def escape_latex(text: str, escape_ampersand: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}',
        '\\': '\\textbackslash{}',
    }
    if escape_ampersand:
        replacements['&'] = '\\&'
    
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text

def generate_intra_cluster_latex(json_path: str, output_path: str = None) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clusters = data if isinstance(data, list) else []
    
    latex_parts = []
    latex_parts.append("\\begin{landscape}")
    latex_parts.append("\\section{Cluster Analysis}")
    latex_parts.append("\\subsection{Intra-Cluster Analysis}")
    
    for idx, cluster in enumerate(clusters):
        cluster_id = idx + 1
        
        if not isinstance(cluster, list) or len(cluster) == 0:
            continue
        
        for item in cluster:
            if not isinstance(item, dict):
                continue
            
            question = item.get('question', '')
            related_papers = item.get('related_papers', [])
            answer = item.get('answer', '')
            
            latex_parts.append(f"\\begin{{frame}}{{Cluster {cluster_id}}}")
            latex_parts.append(f"\\textbf{{Question:}} {question}")
            latex_parts.append("")
            latex_parts.append(f"\\textbf{{Related Papers:}} \\newline")
            for i, paper_id in enumerate(related_papers):
                latex_parts.append(f"\\quad \\texttt{{{paper_id}}}" + (f" ({i+1})" if i > 0 else ""))
            latex_parts.append("")
            latex_parts.append("\\textbf{{Answer:}}")
            latex_parts.append(answer)
            latex_parts.append("\\end{frame}")
            latex_parts.append("")
    
    latex_parts.append("\\end{landscape}")
    
    latex_output = "\n".join(latex_parts)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_output)
    
    return latex_output

def generate_compact_latex(json_path: str, output_path: str = None) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clusters = data if isinstance(data, list) else []
    
    latex_parts = []
    latex_parts.append("\\section{Intra-Cluster Analysis Results}")
    latex_parts.append("\\label{appendix:cluster-analysis}")
    latex_parts.append("")
    
    caption_text = "Cluster Analysis Results"
    label_text = "tab:cluster-analysis"
    cont_table_text = "\\textbf{续表~\\thetable}"
    
    col_format = "*{3}{p{0.25\\textwidth}}"
    
    latex_parts.append("\\begin{ThreePartTable}")
    latex_parts.append(f"\\begin{{longtable}}[c]{{{col_format}}}")
    latex_parts.append(f"\\caption{{{caption_text}}}")
    latex_parts.append(f"\\label{{{label_text}}} \\\\")
    latex_parts.append("\\toprule")
    latex_parts.append("\\multicolumn{1}{c}{\\textbf{Cluster ID}} & "
                      "\\multicolumn{1}{c}{\\textbf{Key Question}} & "
                      "\\multicolumn{1}{c}{\\textbf{Key Insights}}\\\\")
    latex_parts.append("\\midrule")
    latex_parts.append("\\endfirsthead")
    latex_parts.append(f"\\multicolumn{{3}}{{l}}{{{cont_table_text}}}\\\\")
    latex_parts.append("\\toprule")
    latex_parts.append("\\multicolumn{1}{c}{\\textbf{Cluster ID}} & "
                      "\\multicolumn{1}{c}{\\textbf{Key Question}} & "
                      "\\multicolumn{1}{c}{\\textbf{Key Insights}}\\\\")
    latex_parts.append("\\midrule")
    latex_parts.append("\\endhead")
    latex_parts.append("\\hline")
    latex_parts.append("\\multicolumn{3}{r}{}")
    latex_parts.append("\\endfoot")
    latex_parts.append("\\endlastfoot")
    
    for idx, cluster in enumerate(clusters):
        cluster_id = idx + 1
        
        if not isinstance(cluster, list) or len(cluster) == 0:
            continue
        
        first_item = cluster[0]
        if not isinstance(first_item, dict):
            continue
        
        question = first_item.get('question', '')
        answer = first_item.get('answer', '')
        
        question_words = question.split()[:150]
        answer_words = answer.split()[:300]
        short_question = ' '.join(question_words) + "..." if len(question.split()) > 150 else question
        short_answer = ' '.join(answer_words) + "..." if len(answer.split()) > 300 else answer
        
        question_escaped = escape_latex(short_question)
        answer_escaped = escape_latex(short_answer)
        
        latex_parts.append(f"\\texttt{{Cluster-{cluster_id}}} & {question_escaped} & {answer_escaped} \\\\")
        latex_parts.append("\\hline")
    
    latex_parts.append("\\bottomrule")
    latex_parts.append("\\end{longtable}")
    latex_parts.append("\\end{ThreePartTable}")
    
    latex_output = "\n".join(latex_parts)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_output)
    
    return latex_output

def generate_table_latex(json_path: str, output_path: str = None) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clusters = data if isinstance(data, list) else []
    
    latex_parts = []
    latex_parts.append("\\section{Detailed Cluster Analysis}")
    latex_parts.append("\\subsection*{Cluster Overview}")
    latex_parts.append("")
    
    caption_text = "Cluster Overview"
    label_text = "tab:cluster-overview"
    cont_table_text = "\\textbf{续表~\\thetable}"
    num_cols_text = "3"
    
    latex_parts.append("\\begin{ThreePartTable}")
    latex_parts.append("\\begin{longtable}[c]{lcr}")
    latex_parts.append(f"\\caption{{{caption_text}}}")
    latex_parts.append(f"\\label{{{label_text}}} \\\\")
    latex_parts.append("\\toprule")
    latex_parts.append("\\multicolumn{1}{c}{\\textbf{Cluster ID}} & "
                      "\\multicolumn{1}{c}{\\textbf{\\# Papers}} & "
                      "\\multicolumn{1}{c}{\\textbf{Primary Focus}}\\\\")
    latex_parts.append("\\midrule")
    latex_parts.append("\\endfirsthead")
    latex_parts.append(f"\\multicolumn{{{num_cols_text}}}{{l}}{{{cont_table_text}}}\\\\")
    latex_parts.append("\\toprule")
    latex_parts.append("\\multicolumn{1}{c}{\\textbf{Cluster ID}} & "
                      "\\multicolumn{1}{c}{\\textbf{\\# Papers}} & "
                      "\\multicolumn{1}{c}{\\textbf{Primary Focus}}\\\\")
    latex_parts.append("\\midrule")
    latex_parts.append("\\endhead")
    latex_parts.append("\\hline")
    latex_parts.append("\\multicolumn{3}{r}{}")
    latex_parts.append("\\endfoot")
    latex_parts.append("\\endlastfoot")
    
    for idx, cluster in enumerate(clusters):
        cluster_id = idx + 1
        
        if not isinstance(cluster, list) or len(cluster) == 0:
            continue
        
        paper_count = 0
        first_question = ""
        
        for item in cluster:
            if isinstance(item, dict):
                papers = item.get('related_papers', [])
                paper_count += len(papers)
                if not first_question and item.get('question'):
                    first_question = item.get('question', '')
        
        words = first_question.split()
        truncated = ' '.join(words[:30]) + ('...' if len(words) > 30 else '')
        question_escaped = escape_latex(truncated)
        latex_parts.append(f"{cluster_id} & {paper_count} & \\textit{{{question_escaped}}} \\\\")
        latex_parts.append("\\hline")
    
    latex_parts.append("\\bottomrule")
    latex_parts.append("\\end{longtable}")
    latex_parts.append("\\end{ThreePartTable}")
    
    latex_output = "\n".join(latex_parts)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_output)
    
    return latex_output

def generate_relation_table_latex(json_path: str, output_path: str = None) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    latex_parts = []
    latex_parts.append("\\section{Relation Analysis}")
    latex_parts.append("\\subsection{Relationship Comparison Table}")
    latex_parts.append("\\label{appendix:relation-table}")
    latex_parts.append("")
    
    for category, category_data in data.items():
        category_clean = escape_latex(category)
        latex_parts.append(f"\\subsubsection*{{{category_clean}}}")
        latex_parts.append("")
        
        comparison_dimensions = category_data.get('comparison_dimensions', [])
        table_data = category_data.get('table_data', [])
        
        if not table_data or not comparison_dimensions:
            continue
        
        num_cols = len(comparison_dimensions) + 1
        col_format = "|" + "|".join(["c"] * num_cols) + "|"
        
        cont_table_text = "\\textbf{续表~\\thetable}"
        num_cols_str = str(num_cols)
        
        latex_parts.append("\\begin{landscape}")
        latex_parts.append("\\begin{ThreePartTable}")
        latex_parts.append(f"\\begin{{longtable}}[c]{{{col_format}}}")
        latex_parts.append(f"\\caption{{{category_clean} - Comparison}}\\\\")
        latex_parts.append("\\toprule")
        
        header_row = "\\textbf{Paper ID} & " + " & ".join([f"\\textbf{{{escape_latex(dim, escape_ampersand=False)}}}" for dim in comparison_dimensions]) + "\\\\"
        latex_parts.append(header_row)
        latex_parts.append("\\midrule")
        latex_parts.append("\\endfirsthead")
        latex_parts.append(f"\\multicolumn{{{num_cols_str}}}{{l}}{{{cont_table_text}}}\\\\")
        latex_parts.append("\\toprule")
        latex_parts.append(header_row)
        latex_parts.append("\\midrule")
        latex_parts.append("\\endhead")
        latex_parts.append("\\hline")
        latex_parts.append(f"\\multicolumn{{{num_cols_str}}}{{r}}{{}}")
        latex_parts.append("\\endfoot")
        latex_parts.append("\\endlastfoot")
        
        for item in table_data:
            paper_id = escape_latex(item.get('paper_id', 'N/A'))
            paper_title = escape_latex(item.get('paper_title', 'Unknown'))
            
            row_cells = [f"\\texttt{{{paper_id}}}"]
            
            columns = item.get('columns', {})
            for dim in comparison_dimensions:
                cell_text = escape_latex(columns.get(dim, 'N/A'))
                row_cells.append(cell_text)
            
            latex_parts.append(" & ".join(row_cells) + "\\\\")
            latex_parts.append("\\hline")
        
        latex_parts.append("\\bottomrule")
        latex_parts.append("\\end{longtable}")
        latex_parts.append("\\end{ThreePartTable}")
        latex_parts.append("\\end{landscape}")
        latex_parts.append("")
    
    latex_output = "\n".join(latex_parts)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_output)
    
    return latex_output

def generate_relation_table_compact(json_path: str, output_path: str = None) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    latex_parts = []
    latex_parts.append("\\section{System Comparison}")
    latex_parts.append("\\label{appendix:system-comparison}")
    latex_parts.append("")
    
    for category, category_data in data.items():
        category_clean = escape_latex(category)
        latex_parts.append(f"\\subsection*{{{category_clean}}}")
        latex_parts.append("")
        
        comparison_dimensions = category_data.get('comparison_dimensions', [])
        table_data = category_data.get('table_data', [])
        
        if not table_data:
            continue
        
        max_items = min(len(table_data), 15)
        num_cols = min(len(comparison_dimensions), 5) + 2
        
        col_format = "|" + "|".join(["c"] * num_cols) + "|"
        
        cont_table_text = "\\textbf{续表~\\thetable}"
        num_cols_str = str(num_cols)
        
        latex_parts.append("\\begin{ThreePartTable}")
        latex_parts.append(f"\\begin{{longtable}}[c]{{{col_format}}}")
        latex_parts.append(f"\\caption{{{category_clean}}} \\\\")
        latex_parts.append("\\toprule")
        
        header_row = "\\textbf{ID} & \\textbf{Title} & " + " & ".join([
            f"\\textbf{{{escape_latex(comparison_dimensions[i], escape_ampersand=False)}}}" 
            for i in range(min(len(comparison_dimensions), 5))
        ]) + "\\\\"
        latex_parts.append(header_row)
        latex_parts.append("\\midrule")
        latex_parts.append("\\endfirsthead")
        latex_parts.append(f"\\multicolumn{{{num_cols_str}}}{{l}}{{{cont_table_text}}}\\\\")
        latex_parts.append("\\toprule")
        latex_parts.append(header_row)
        latex_parts.append("\\midrule")
        latex_parts.append("\\endhead")
        latex_parts.append("\\hline")
        latex_parts.append(f"\\multicolumn{{{num_cols_str}}}{{r}}{{}}")
        latex_parts.append("\\endfoot")
        latex_parts.append("\\endlastfoot")
        
        for idx, item in enumerate(table_data[:max_items]):
            paper_id = escape_latex(item.get('paper_id', 'N/A')[:15])
            paper_title = escape_latex(item.get('paper_title', 'Unknown')[:40])
            
            row = [f"\\texttt{{{paper_id}}}", f"\\small{{{paper_title}}}"]
            
            columns = item.get('columns', {})
            for i, dim in enumerate(comparison_dimensions[:5]):
                cell_text = escape_latex(columns.get(dim, 'N/A'))[:40]
                row.append(f"\\small{{{cell_text}}}")
            
            latex_parts.append(" & ".join(row) + "\\\\")
            latex_parts.append("\\hline")
        
        latex_parts.append("\\bottomrule")
        latex_parts.append("\\end{longtable}")
        latex_parts.append("\\end{ThreePartTable}")
        latex_parts.append("")
    
    latex_output = "\n".join(latex_parts)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_output)
    
    return latex_output

def generate_paper_list_latex(json_path: str, output_path: str = None) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clusters = data if isinstance(data, list) else []
    
    latex_parts = []
    latex_parts.append("\\section{Cluster Papers Detail}")
    latex_parts.append("\\subsection*{Papers in Each Cluster}")
    latex_parts.append("")
    
    caption_text = "Papers in Clusters"
    label_text = "tab:cluster-papers"
    cont_table_text = "\\textbf{续表~\\thetable}"
    num_cols_text = "3"
    
    latex_parts.append("\\begin{ThreePartTable}")
    latex_parts.append("\\begin{longtable}[c]{lcp{0.5\\textwidth}}")
    latex_parts.append(f"\\caption{{{caption_text}}}")
    latex_parts.append(f"\\label{{{label_text}}} \\\\")
    latex_parts.append("\\toprule")
    latex_parts.append("\\multicolumn{1}{c}{\\textbf{Cluster}} & "
                      "\\multicolumn{1}{c}{\\textbf{Index}} & "
                      "\\multicolumn{1}{c}{\\textbf{Paper ID}}\\\\")
    latex_parts.append("\\midrule")
    latex_parts.append("\\endfirsthead")
    latex_parts.append(f"\\multicolumn{{{num_cols_text}}}{{l}}{{{cont_table_text}}}\\\\")
    latex_parts.append("\\toprule")
    latex_parts.append("\\multicolumn{1}{c}{\\textbf{Cluster}} & "
                      "\\multicolumn{1}{c}{\\textbf{Index}} & "
                      "\\multicolumn{1}{c}{\\textbf{Paper ID}}\\\\")
    latex_parts.append("\\midrule")
    latex_parts.append("\\endhead")
    latex_parts.append("\\hline")
    latex_parts.append(f"\\multicolumn{{{num_cols_text}}}{{r}}{{}}")
    latex_parts.append("\\endfoot")
    latex_parts.append("\\endlastfoot")
    
    for idx, cluster in enumerate(clusters):
        cluster_id = idx + 1
        
        if not isinstance(cluster, list) or len(cluster) == 0:
            continue
        
        all_papers = []
        for item in cluster:
            if isinstance(item, dict):
                papers = item.get('related_papers', [])
                all_papers.extend(papers)
        
        all_papers = list(dict.fromkeys(all_papers))
        
        for paper_idx, paper_id in enumerate(all_papers, 1):
            paper_escaped = escape_latex(paper_id)
            latex_parts.append(f"\\texttt{{Cluster-{cluster_id}}} & {paper_idx} & \\texttt{{{paper_escaped}}} \\\\")
            latex_parts.append("\\hline")
    
    latex_parts.append("\\bottomrule")
    latex_parts.append("\\end{longtable}")
    latex_parts.append("\\end{ThreePartTable}")
    
    latex_output = "\n".join(latex_parts)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_output)
    
    return latex_output

def generate_problem_latex(json_path: str, output_path: str = None) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clusters = data if isinstance(data, list) else []
    
    latex_parts = []
    latex_parts.append("\\section{Cluster Problems}")
    latex_parts.append("\\subsection*{Problems in Each Cluster}")
    latex_parts.append("")
    
    caption_text = "Cluster Problems"
    label_text = "tab:cluster-problems"
    cont_table_text = "\\textbf{续表~\\thetable}"
    num_cols_text = "3"
    
    latex_parts.append("\\begin{ThreePartTable}")
    latex_parts.append("\\begin{longtable}[c]{lcp{0.55\\textwidth}}")
    latex_parts.append(f"\\caption{{{caption_text}}}")
    latex_parts.append(f"\\label{{{label_text}}} \\\\")
    latex_parts.append("\\toprule")
    latex_parts.append("\\multicolumn{1}{c}{\\textbf{Cluster ID}} & "
                      "\\multicolumn{1}{c}{\\textbf{Problem Index}} & "
                      "\\multicolumn{1}{c}{\\textbf{Problem Description}}\\\\")
    latex_parts.append("\\midrule")
    latex_parts.append("\\endfirsthead")
    latex_parts.append(f"\\multicolumn{{{num_cols_text}}}{{l}}{{{cont_table_text}}}\\\\")
    latex_parts.append("\\toprule")
    latex_parts.append("\\multicolumn{1}{c}{\\textbf{Cluster ID}} & "
                      "\\multicolumn{1}{c}{\\textbf{Problem Index}} & "
                      "\\multicolumn{1}{c}{\\textbf{Problem Description}}\\\\")
    latex_parts.append("\\midrule")
    latex_parts.append("\\endhead")
    latex_parts.append("\\hline")
    latex_parts.append(f"\\multicolumn{{{num_cols_text}}}{{r}}{{}}")
    latex_parts.append("\\endfoot")
    latex_parts.append("\\endlastfoot")
    
    for idx, cluster in enumerate(clusters):
        cluster_id = idx + 1
        
        if not isinstance(cluster, list) or len(cluster) == 0:
            continue
        
        problem_idx = 0
        for item in cluster:
            if isinstance(item, dict):
                question = item.get('question', '')
                if question:
                    problem_idx += 1
                    short_question = question[:500] + "..." if len(question) > 500 else question
                    question_escaped = escape_latex(short_question)
                    latex_parts.append(f"\\texttt{{Cluster-{cluster_id}}} & {problem_idx} & {question_escaped} \\\\")
                    latex_parts.append("\\hline")
    
    latex_parts.append("\\bottomrule")
    latex_parts.append("\\end{longtable}")
    latex_parts.append("\\end{ThreePartTable}")
    
    latex_output = "\n".join(latex_parts)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_output)
    
    return latex_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate LaTeX from intra-cluster analysis JSON')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to survey_intra_cluster_analysis.json')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output LaTeX file path')
    parser.add_argument('--mode', '-m', type=str, choices=['full', 'compact', 'table', 'papers', 'problems'], 
                        default='full', help='Generation mode')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        result = generate_intra_cluster_latex(args.input, args.output)
    elif args.mode == 'table':
        result = generate_table_latex(args.input, args.output)
    elif args.mode == 'papers':
        result = generate_paper_list_latex(args.input, args.output)
    elif args.mode == 'problems':
        result = generate_problem_latex(args.input, args.output)
    else:
        result = generate_compact_latex(args.input, args.output)
    
    print(f"LaTeX output {'saved to ' + args.output if args.output else 'printed to stdout'}")
    print(result)