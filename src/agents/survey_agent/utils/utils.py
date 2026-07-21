import hashlib
import re, json
from typing import Dict, Any
import os
import pypdfium2 as pdfium

def get_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def extract_json(text: str) -> Any:
    """Extract JSON from LLM response, handling various edge cases."""
    if not text:
        raise ValueError("Empty text in json extraction function")
    
    # Remove ```json fences
    text = re.sub(r"```[\w]*", "", text).replace("```", "")
    text = text.strip()

    # Match list or dict
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])$", text)
    if not m:
        raise ValueError("No JSON found")
    json_str = m.group()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common escape sequence issues
        # Fix invalid escape sequences like \c, \x, etc.
        fixed_str = _fix_invalid_escapes(json_str)
        try:
            return json.loads(fixed_str)
        except json.JSONDecodeError:
            # If still failing, re-raise original error
            raise e


def _fix_invalid_escapes(json_str: str) -> str:
    """Fix common invalid escape sequences in JSON strings."""
    # Pattern to match invalid escape sequences (backslash followed by non-standard chars)
    # Valid escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    # Invalid: \c, \x, \ followed by anything else that's not a valid escape
    
    result = []
    i = 0
    while i < len(json_str):
        if json_str[i] == '\\' and i + 1 < len(json_str):
            next_char = json_str[i + 1]
            # Keep valid escapes as-is
            if next_char in '"\\/\b\f\n\r\t' or next_char == 'u':
                result.append(json_str[i:i+2])
                i += 2
            else:
                # Fix invalid escape: remove the backslash
                # e.g., \c -> c, \x -> x
                result.append(next_char)
                i += 2
        else:
            result.append(json_str[i])
            i += 1
    
    return ''.join(result)



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
        pdfium.PdfDocument(path)
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
