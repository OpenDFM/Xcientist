from mineru_utils import parse_doc
from pathlib import Path
import os

pdf_path = "/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/baselines/human/pdfs"
output_path = "/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/baselines/human/md"

pdf_paths_list = []

for filename in os.listdir(pdf_path):
    if filename.endswith(".pdf"):
        pdf_paths_list.append(Path(os.path.join(pdf_path, filename)))

parse_doc(
    path_list=pdf_paths_list,
    output_dir=output_path,
    lang="ch",
    backend="pipeline",
    method="auto",
)