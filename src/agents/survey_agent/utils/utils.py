import hashlib
import re, json
from typing import Dict
import os
import pypdfium2 as pdfium

def get_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def extract_json(text):
    # remove ```json fences
    text = re.sub(r"```[\w]*", "", text).replace("```", "")
    text = text.strip()

    # match list or dict
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])$", text)
    if not m:
        raise ValueError("No JSON found")
    return json.loads(m.group())

def is_valid_pdf(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < 2048:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(5)
            size = os.path.getsize(path)
            f.seek(max(size - 20, 0), os.SEEK_SET)
            tail = f.read()
        if not (head == b"%PDF-" and b"%%EOF" in tail):
            return False
        pdfium.PdfDocument(path)  # 深一层校验
        return True
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return False


if __name__ == "__main__":
    json_data = """```json
    {
        "cluster_name": "Taxonomy and Structural Frameworks for Knowledge Organization",
        "summary": "This cluster includes methodologies for automatic taxonomy generation and knowledge organization, focusing on structured frameworks to enhance navigation and usefulness of scholarly content.",
        "papers": [
            {
                "id": "2510.17263",
                "title": "TAXOALIGN: Automating Scholarly Taxonomy Generation",
                "tldr": "The paper presents TAXOALIGN, an innovative approach for automating the generation of scholarly taxonomies using large language models, significantly outperforming existing methods in structural alignment and semantic coherence."
            }
        ]
    }
```"""
    data = extract_json(json_data)
    for cluster in data:
        print("Cluster Name:", cluster["cluster_name"])
        print("Summary:", cluster["summary"])
        print("Papers:")
        for paper in cluster["papers"]:
            print(f"  - ID: {paper['id']}")
            print(f"    Title: {paper['title']}")
            print(f"    TL;DR: {paper['tldr']}")
        print()
