import argparse
import os
import sys

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))  # 回到 deep-survey
sys.path.append(project_root)

from utils.mineru_utils import parse_doc 

def parse_arguments():
    parser = argparse.ArgumentParser(description="denote md conversion's input and output path")

    parser.add_argument('-p', '--base_dir', type=str, default="/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/intra-work", help="input pdf path")
    parser.add_argument('-t', '--task_id', type=str, default="semantic_scholar_test", help="input pdf path")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    pdf_dir = f"{args.base_dir}/outputs/pipeline/{args.task_id}/references/pdfs"
    output_dir = f"{args.base_dir}/outputs/pipeline/{args.task_id}/references/mds"

    os.makedirs(output_dir)

    pdf_paths = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")
    ]

    # 3. 逐个调用parse_doc
    for pdf_path in pdf_paths:
        print(f"正在处理: {pdf_path}")
        parse_doc(
            path_list=[pdf_path],
            output_dir=output_dir,
            lang="ch",
            backend="pipeline",
            method="auto",
        )