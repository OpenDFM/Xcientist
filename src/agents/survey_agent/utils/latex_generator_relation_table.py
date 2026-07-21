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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate LaTeX from relation table JSON')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to survey_relation_table.json')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output LaTeX file path')
    parser.add_argument('--mode', '-m', type=str, choices=['full', 'compact'], 
                        default='compact', help='Generation mode')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        result = generate_relation_table_latex(args.input, args.output)
    else:
        result = generate_relation_table_compact(args.input, args.output)
    
    print(f"LaTeX output {'saved to ' + args.output if args.output else 'printed to stdout'}")
    print(result)