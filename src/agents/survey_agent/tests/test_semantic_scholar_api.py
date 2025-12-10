import requests

API_KEY = "1EzJeomTxpaiYyR5cJbCoaZThZTgFkph707DvYzJ"  # 替换成你实际的 API Key
API_URL = "https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.03762"
FIELDS = "title,authors,year,references"

headers = {
    "x-api-key": API_KEY
}

response = requests.get(f"{API_URL}?fields={FIELDS}", headers=headers)

if response.status_code == 200:
    data = response.json()
    print("API 请求成功！")
    print("论文标题:", data.get("title"))
    print("年份:", data.get("year"))
    print("作者:", [a['name'] for a in data.get("authors", [])])
    print("参考文献:", [r['paperId'] for r in data.get("references", [])])
else:
    print("请求失败，状态码:", response.status_code)
    print(response.text)