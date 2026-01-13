import re

def get_unique_paper_ids_from_raw( text: str):
    pattern = re.compile(
        r"<Paper ID:\s*([^\s>]+)\s*>"
        r"|\(Paper ID:\s*([^\s\)]+)\s*\)"
        r"|<Paper\s*ID\s*:\s*([^\s>]+)\s*>"
        r"|\(Paper\s*ID\s*([^\s\)]+)\s*\)"
        r"|<Paper\s*([^\s>]+)\s*>"
        r"|\(Paper\s*([^\s\)]+)\s*\)"
        r"|<Paper\s*<\s*([^\s>]+)\s*>\s*>"
        r"|<Paper ID:\s*([^>]+?)\s*>",  # e.g., '<Paper ID: 2408.08464, Paper ID: 2406.09324>'",
        flags=re.IGNORECASE,
    )

    matches = pattern.findall(text or "")
    ids = set()
    ordered_ids = []
    for m in matches:
        raw = next((grp for grp in m if grp), "").strip()
        if not raw:
            continue
        # handle combined forms like "2408.08464, Paper ID: 2406.09324"
        parts = [p.strip() for p in re.split(r",|and", raw) if p.strip()]
        for p in parts:
            # remove leading 'Paper ID:' if present
            p = re.sub(r"^(?i:paper\s*id:?)\s*", "", p).strip()
            if p not in ids:
                ids.add(p)
                ordered_ids.append(p)
    return ordered_ids

with open("./draft.md", "r") as f:
    draft_content = f.read()
    print("Unique Paper IDs referenced:", len(get_unique_paper_ids_from_raw(draft_content)) )   
