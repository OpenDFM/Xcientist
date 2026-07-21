import re
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup

# 可选库：trafilatura, readability, html2text
import trafilatura

from readability import Document
import html2text


# ---------- 辅助函数 ----------
def _normalize_whitespace(s: str) -> str:
    s = re.sub(r'\r\n', '\n', s)
    s = re.sub(r'\n[ \t]+\n', '\n\n', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    s = re.sub(r'[ \t]{2,}', ' ', s)
    return s.strip()

def _safe_text(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode('utf-8', errors='ignore')
        except:
            return x.decode('latin-1', errors='ignore')
    return str(x)

# ---------- 主体抽取：尝试多种策略 ----------
def extract_main_text(html: str) -> str:
    """
    trafilatura -> readability -> heuristic
    """
    html = _safe_text(html)
    # 1) trafilatura
    if trafilatura is not None:
        try:
            txt = trafilatura.extract(html, include_comments=False, favor_recall=True)
            if txt and len(txt) > 200:  # 合理阈值
                return _normalize_whitespace(txt)
        except Exception:
            pass

    # 2) readability (Document.summary())
    if Document is not None:
        try:
            doc = Document(html)
            summary_html = doc.summary()
            soup = BeautifulSoup(summary_html, "lxml")
            text = soup.get_text(separator="\n")
            if text and len(text) > 200:
                return _normalize_whitespace(text)
        except Exception:
            pass

    # 3) remove <head>, script, style，then get main/article or longest div/section
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav", "form"]):
        tag.decompose()
    # find main/article
    main = soup.find("main") or soup.find("article")
    if main:
        text = main.get_text(separator="\n")
        if len(text) > 200:
            return _normalize_whitespace(text)
    # find maximal div/section by text length
    candidates = soup.find_all(["div", "section"], recursive=True)
    if candidates:
        best = max(candidates, key=lambda t: len(t.get_text()), default=soup)
        text = best.get_text(separator="\n")
        return _normalize_whitespace(text)

    # fallback: complete
    text = soup.get_text(separator="\n")
    return _normalize_whitespace(text)

# ---------- extract metadata（title, authors, abstract） ----------
def extract_metadata(html: str) -> dict:
    """
    try to extract title, authors(list), abstract in arxiv html structure
    use html2text to convert html to md(not debug yet)
    """
    soup = BeautifulSoup(_safe_text(html), "lxml")
    meta = {"title": "", "authors": [], "abstract": ""}

    # title: <title> or meta og:title or h1
    if soup.title and soup.title.string:
        meta["title"] = soup.title.string.strip()
    if not meta["title"]:
        og = soup.find("meta", {"property": "og:title"}) or soup.find("meta", {"name": "twitter:title"})
        if og and og.get("content"):
            meta["title"] = og["content"].strip()
    if not meta["title"]:
        h1 = soup.find("h1")
        if h1:
            meta["title"] = h1.get_text(separator=" ").strip()

    # authors: meta citation_author or meta name="author" or div.class contains author
    authors = []
    for m in soup.find_all("meta", attrs={"name": "citation_author"}):
        if m.get("content"):
            authors.append(m["content"].strip())
    if not authors:
        ma = soup.find("meta", {"name": "author"})
        if ma and ma.get("content"):
            authors = [a.strip() for a in re.split(r'[;,]', ma["content"]) if a.strip()]
    if not authors:
        # arXiv-like: div.authors or p.authors
        el = soup.find(class_=re.compile(r"author", re.I))
        if el:
            # join/link names
            text = el.get_text(separator=",")
            authors = [a.strip() for a in re.split(r'[,\n;]', text) if a.strip()]

    meta["authors"] = authors

    # abstract: look for blockquote.abstract, div#abstract, meta description
    abstract = ""
    # common patterns
    sel = soup.find("blockquote", class_=re.compile(r"abstract", re.I))
    if sel:
        abstract = sel.get_text(separator=" ").strip()
    if not abstract:
        div_abs = soup.find(id=re.compile(r"abstract", re.I)) or soup.find(class_=re.compile(r"abstract", re.I))
        if div_abs:
            abstract = div_abs.get_text(separator=" ").strip()
    if not abstract:
        mdesc = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
        if mdesc and mdesc.get("content"):
            abstract = mdesc["content"].strip()
    # heuristic: sometimes abstract starts with 'Abstract' heading
    if not abstract:
        for hdr in soup.find_all(re.compile(r"h[1-6]")):
            if "abstract" in hdr.get_text().lower():
                # next sibling paragraphs
                texts = []
                nxt = hdr.find_next_siblings(limit=3)
                for n in nxt:
                    texts.append(n.get_text(separator=" "))
                if texts:
                    abstract = " ".join(texts).strip()
                    break

    meta["abstract"] = _normalize_whitespace(abstract)
    return meta

# ---------- html -> markdown ----------
def html_to_markdown(html: str) -> str:
    """
    use html2text to convert html to md(not debug yet)
    """
    if html2text is None:
        # fallback: just return cleaned text
        return extract_main_text(html)
    h = html2text.HTML2Text()
    h.ignore_images = False
    h.ignore_links = False
    h.body_width = 0
    # optional: tune other settings e.g. h.escape_snob = True
    md = h.handle(html)
    md = _normalize_whitespace(md)
    return md

# ---------- overall extract ----------
def parse_html_document(html: str) -> dict:
    # meta = extract_metadata(html)
    text = extract_main_text(html)
    # md = html_to_markdown(html)
    return text

if __name__ == "__main__":
    html_path = "../database/htmls/1601.03896.html"

    html = open(html_path, "rb").read()

    text = parse_html_document(html)

    with open("./debug.txt", 'w') as f:
        f.write(text)