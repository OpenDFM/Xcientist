import requests

url = "https://arxiv.org/pdf/2505.11711.pdf"

resp = requests.get(url, stream=True)
total = int(resp.headers.get("Content-Length", 0))
print(total)
