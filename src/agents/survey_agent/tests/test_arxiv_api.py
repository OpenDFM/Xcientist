import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

# 构造 API URL
url = "http://export.arxiv.org/api/query?search_query=all:survey+AND+cat:cs*&max_results=100"

# 获取数据
response = urllib.request.urlopen(url)
data = response.read().decode("utf-8")

# 解析 XML
root = ET.fromstring(data)

ns = {"atom": "http://www.w3.org/2005/Atom"}

papers = []

for entry in root.findall("atom:entry", ns):
    title = entry.find("atom:title", ns).text.strip()
    summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
    published = entry.find("atom:published", ns).text[:10]
    authors = ", ".join(
        [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
    )
    link = entry.find("atom:id", ns).text

    print(title)

    papers.append(
        {
            "title": title,
            "authors": authors,
            "published": published,
            "summary": summary,
            "url": link,
        }
    )

# # 转成 DataFrame 并保存
# df = pd.DataFrame(papers)
# df.to_csv("cs_survey_arxiv.csv", index=False)

# print(f"抓取完成，共 {len(df)} 篇论文，已保存到 cs_survey_arxiv.csv")
