import requests
import os
from semanticscholar import SemanticScholar
from FlagEmbedding import FlagAutoModel
import numpy as np
from tqdm import tqdm
import hashlib
import json
import sys
from typing import Optional, Tuple, Union


TimeoutType = Union[float, Tuple[float, float]]

class Semantic:
    def __init__(self, api_key=None, timeout: Optional[TimeoutType] = None):
        self.api_key = api_key or os.getenv("S2_API_KEY")
        self.sch = SemanticScholar(api_key=self.api_key) if self.api_key else SemanticScholar()
        self.timeout: TimeoutType = timeout if timeout is not None else self._parse_timeout_env(os.getenv("S2_API_TIMEOUT"))
        self.embed_model = FlagAutoModel.from_finetuned("BAAI/bge-large-en-v1.5",
                                     query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                     use_fp16=True,
                                     devices=['cpu'])

    def _parse_timeout_env(self, value: Optional[str]) -> TimeoutType:
        """Parse timeout from env.

        Supported formats:
        - "30" -> 30.0 seconds
        - "10,60" -> (connect=10.0, read=60.0)
        """
        default: TimeoutType = (10.0, 60.0)
        if not value:
            return default
        raw = value.strip()
        try:
            if "," in raw:
                a, b = raw.split(",", 1)
                return (float(a.strip()), float(b.strip()))
            return float(raw)
        except Exception:
            return default

    def _get_timeout(self, timeout: Optional[TimeoutType]) -> TimeoutType:
        return timeout if timeout is not None else (self.timeout if self.timeout is not None else (10.0, 60.0))

    def search_papers(self, query, limit=10, timeout: Optional[TimeoutType] = None):
        # prepare cache directory
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache", "semantic_scholar_search")
        os.makedirs(cache_dir, exist_ok=True)

        # include limit in the hash so different limits produce different caches
        hash_input = f"{query}||limit={limit}"
        cache_key = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")

        # if cached file exists, return cached results
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                print(f"📥 Loaded cached results for query (hash={cache_key})")
                return cached['data']
            except Exception as e:
                print(f"Failed to read cache {cache_path}: {e} — will fetch from API")

        # after this point the code will call the API; after obtaining `data` (below),
        # write the relevant part to the cache:
        # with open(cache_path, "w", encoding="utf-8") as f:
        #     json.dump(data.get("data", []), f, ensure_ascii=False)
        URL = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {
            "x-api-key": self.api_key
        }
        params = {
            "query": query,
            "limit": limit,         
            "fields": "title,authors,year,abstract,url,tldr"  
        }
        # import pdb; pdb.set_trace()
        try:
            response = requests.get(URL, headers=headers, params=params, timeout=self._get_timeout(timeout))
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            print(f"⏱️ Semantic Scholar request timed out (timeout={self._get_timeout(timeout)})")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Semantic Scholar request failed: {e}")
            return []
        except ValueError as e:
            print(f"Failed to parse Semantic Scholar response as JSON: {e}")
            return []
        # import pdb; pdb.set_trace()
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            return data['data']
        except KeyError:
            print(data.get('message', 'Semantic Scholar returned unexpected response.'))
            return []

    def recommend_papers(self, positive_paper_id_list, negative_paper_id_list, limit=500, return_num=10, sort_by="citationCount", timeout: Optional[TimeoutType] = None):
        print("Fetching recommended papers...\n")
        url = "https://api.semanticscholar.org/recommendations/v1/papers"
        query_params = {
            "fields": "title,url,citationCount,authors,abstract,year",
            "limit": limit
        }

        data = {
            "positivePaperIds": positive_paper_id_list,
            "negativePaperIds": negative_paper_id_list
        }

        api_key = self.api_key
        headers = {"x-api-key": api_key}
        try:
            response = requests.post(url, params=query_params, json=data, headers=headers, timeout=self._get_timeout(timeout))
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.Timeout:
            print(f"⏱️ Semantic Scholar request timed out (timeout={self._get_timeout(timeout)})")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Semantic Scholar request failed: {e}")
            return []
        except ValueError as e:
            print(f"Failed to parse Semantic Scholar response as JSON: {e}")
            return []

        papers = response_json.get("recommendedPapers", [])
        print(f"Total recommended papers fetched: {len(papers)}\n")
        

        if sort_by == "citationCount":
            papers.sort(key=lambda paper: paper["citationCount"], reverse=True)
        if sort_by == "latest":
            papers = sorted(
                [paper for paper in papers if getattr(paper, "year", None) is not None],
                key=lambda paper: getattr(paper, "year", 0),
                reverse=True
            )
        if sort_by == "relevance":
            pass
            # query_paper = self.sch.get_paper(paper_id)
            # if hasattr(query_paper, "title"):
            #     query_text = query_paper.title + " " + (getattr(query_paper, "abstract", "") or "")
            # else:
            #     query_text = getattr(query_paper, "abstract", "") or ""
            # candidate_texts = [
            #     getattr(paper, "title", "") + " " + (getattr(paper, "abstract", "") or "") for paper in results
            # ]
            # print("Computing embeddings for relevance sorting...\n")
            # query_embedding = self.embed_model.encode([query_text])
            # candidate_embeddings = []
            # for text in tqdm(candidate_texts, desc="Encoding candidate papers"):
            #     emb = self.embed_model.encode([text])
            #     candidate_embeddings.append(emb[0])
            # candidate_embeddings = np.stack(candidate_embeddings)
            # similarity = np.dot(query_embedding, candidate_embeddings.T)[0]
            # # print(similarity)
            # sorted_indices = np.argsort(-similarity)
            # results = [results[i] for i in sorted_indices]
       
        return papers[0:return_num]
    
# Example usage:
if __name__ == "__main__":
    semantic = Semantic()

    # Search for papers
    search_words = "agentic systems that generate research ideas"
    search_results = semantic.search_papers(search_words, limit=5)
    # print("Search Results:")
    # for paper in search_results.get("data", []):
    #     print(f"Title: {paper['title']}")
    #     print(f"Authors: {[a['name'] for a in paper['authors']]}")
    #     print(f"Year: {paper['year']}")
    #     print(f"URL: {paper['url']}\n")
    #     # print(f"Abstract: {paper['abstract']}\n")

    # # Recommend papers
    # positive_ids = [
    #     "02138d6d094d1e7511c157f0b1a3dd4e5b20ebee", 
    #     "018f58247a20ec6b3256fd3119f57980a6f37748"
    # ]
    # negative_ids = [
    #     "0045ad0c1e14a4d1f4b011c92eb36b8df63d65bc"
    # ]
    # recommended_results = semantic.recommend_papers(positive_ids, negative_ids, limit=500, sort_by="citationCount", return_num=10)
    # print("\nRecommended Papers:")
    # for paper in recommended_results:
    #     print(f"- {paper['title']} (Citations: {paper['citationCount']}, Year: {paper.get('year', 'N/A')})")
