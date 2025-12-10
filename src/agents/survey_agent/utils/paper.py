import os
import glob
from typing import List, Dict
import json
import re


class Paper:
    def __init__(self, tex_or_mineru_path: str):
        assert os.path.isdir(
            tex_or_mineru_path
        ), "The provided path is not a directory."
        self.path = tex_or_mineru_path
        self.sections = self.extract_sections()

    def extract_sections(self) -> Dict:
        """Extract sections from the paper based on its format.

        Returns:
            Dict: A dictionary containing extracted sections.
        """
        self.sections = dict()

        # TODO: cache the extracted sections

        if os.path.exists(os.path.join(self.path, "auto")):
            md_files = glob.glob(os.path.join(self.path, "auto", "*.md"))
        else:
            raise NotImplementedError("Only mineru format is supported currently.")
        assert (
            len(md_files) == 1
        ), "Expected exactly one markdown file in the directory."
        md_file = md_files[0]

        toc = []
        stack = []
        current_node = None

        with open(md_file, "r", encoding="utf-8") as reader:
            for line in reader:
                match = re.match(r"^(#+)\s+(.*)", line)
                if match:
                    level = len(match.group(1))
                    title = match.group(2).strip()

                    node = {
                        "title": title,
                        "level": level,
                        "content": "",
                        "children": [],
                    }

                    # parent node
                    while stack and stack[-1]["level"] >= level:
                        stack.pop()

                    if not stack:
                        toc.append(node)
                    else:
                        stack[-1]["children"].append(node)

                    stack.append(node)
                    current_node = node
                else:
                    # ignore text before the title
                    if current_node is not None:
                        current_node["content"] += line + "\n"

        def trim(node):
            node["content"] = node["content"].strip()
            for c in node["children"]:
                trim(c)

        for n in toc:
            trim(n)

        return toc

    @property
    def title(self) -> str:
        return self.sections[0]["title"]

    @property
    def abstract(self) -> str:
        for section in self.sections[0]["children"]:
            if "abstract" in section["title"].lower():
                return section["content"]
        return None

    @property
    def reference(self) -> List[str]:
        ref_nodes = []
        keywords = [
            "reference",
            "references",
            "bibliography",
            "works cited",
        ]

        def search(nodes):
            for node in nodes:
                title_lower = node["title"].lower()
                if any(re.search(rf"\b{kw}\b", title_lower) for kw in keywords):
                    ref_nodes.append(node)
                # 递归搜索子节点
                search(node["children"])

        search(self.sections)

        def extract_references(text):
            lines = text.splitlines()
            refs = []
            current_ref = []
            for line in lines:
                # 检测新参考文献开始
                if re.match(r"^\s*\[\d+\]", line):
                    if current_ref:
                        refs.append(" ".join(current_ref).strip())
                        current_ref = []
                # 追加行
                if line.strip():
                    current_ref.append(line.strip())
            if current_ref:
                refs.append(" ".join(current_ref).strip())
            return refs

        refs = []

        for node in ref_nodes:
            refs.extend(extract_references(node["content"]))

        return refs

    def survey_recognition(self):
        pass


if __name__ == "__main__":
    paper_path = "/hpc_stor03/sjtu_home/da.ma/src/AI_survey/asserts/test_output/attention_is_all_you_need"
    paper = Paper(paper_path)
    for ref in paper.reference:
        print(ref)
