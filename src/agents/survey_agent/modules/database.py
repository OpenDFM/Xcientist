import os, torch
from sentence_transformers import SentenceTransformer, util
from utils.rich_logger import get_logger

class Database:
    def __init__(self, config, work_collector):
        self.config = config
        self.logger = get_logger("Database")
        self.work_collector = work_collector
        self.valid_paper_ids = set()
        self.model = SentenceTransformer(config.ModuleInfo.WorkCollector.sentence_transformer_model)
        self.use_title_in_draft = True
        try:
            self.model = self.model.cuda()
            self.device = "cuda"
        except Exception:
            self.device = "cpu"
        # self.index_path = os.path.join(config.BasicInfo.cache_path, "paper_embedding_index.pt")
        # self.title_index_path = os.path.join(config.BasicInfo.cache_path, "paper_title_embedding_index.pt")
        self.emb_dict = None
        self.title_emb_dict = None

    def build_with_graph(self):
        self.valid_paper_ids = set()
        paper_ids = list(self.work_collector.graph_paper_ids)
        self.logger.info(f"Building database with {len(paper_ids)} papers from the reference graph.")
        self.build(paper_ids)

    def _text_for(self, pid: str) -> str:
        if not self.work_collector.expand_in_local_paper_graph:
            node = self.work_collector.reference_graph.nodes.get(pid, {})
            title = node.get("title", "")
            abstract = node.get("abstract", "")
        else:
            abstract = ""
            title = ""
        if not abstract:
            try:
                title, abstract = self.work_collector.get_paper_title_abstract(pid)
            except Exception as e:
                self.logger.error(f"Error getting title and abstract for paper ID: {pid}: {e}")
                title, abstract = "", ""

        if not abstract or not title:
            self.logger.warning(f"No abstract found for paper ID: {pid}")
            return None, None, None
        return f"Title: {title}\nAbstract: {abstract}", title, abstract

    def build(self, paper_ids):
        # Reset state for a clean rebuild
        self.valid_paper_ids.clear()

        texts = []
        titles = []
        valid_paper_embed_ids = []
        for pid in paper_ids:
            text, title_only, abstract_only = self._text_for(pid)
            if text is not None:
                texts.append(text)
                self.valid_paper_ids.add(pid)
                valid_paper_embed_ids.append(pid)
                titles.append(title_only)

        if not texts:
            self.logger.warning("No valid texts found for embedding. Database build aborted.")
            return
        self.logger.info(f"Encoding {len(valid_paper_embed_ids)} paper abstracts for database.")
        embs = self.model.encode(
            texts,
            convert_to_tensor=True,
            batch_size=self.config.ModuleInfo.WorkCollector.sentence_transformer_batch_size,
            show_progress_bar=True,
        )
        # Title-only embeddings stay aligned with the same ids
        title_embs = self.model.encode(
            titles,
            convert_to_tensor=True,
            batch_size=self.config.ModuleInfo.WorkCollector.sentence_transformer_batch_size,
            show_progress_bar=False,
        )

        self.emb_dict = {"ids": valid_paper_embed_ids, "embs": embs.cpu()}
        self.title_emb_dict = {"ids": valid_paper_embed_ids, "embs": title_embs.cpu()}

    def query(self, query_text: str, top_k: int = None):
        if self.emb_dict is None:
            raise RuntimeError("Database not built; call build/build_with_graph first.")
        if top_k is None:
            top_k = self.config.ModuleInfo.Database.default_top_k
        data = self.emb_dict
        embs = data["embs"].to(self.device)
        q = self.model.encode([query_text], convert_to_tensor=True)

        scores = util.pytorch_cos_sim(q, embs)[0]
        
        top_k = min(top_k, scores.shape[0])
        vals, idxs = torch.topk(scores, k=top_k)
        return [data["ids"][i] for i in idxs.cpu().tolist()], vals.cpu().tolist()

    def query_titles(self, query_text: str, top_k: int = None):
        """Query against the title-only embedding index."""
        if self.title_emb_dict is None:
            raise RuntimeError("Database not built; call build/build_with_graph first.")
        if top_k is None:
            top_k = self.config.ModuleInfo.Database.default_top_k
        data = self.title_emb_dict
        embs = data["embs"].to(self.device)
        q = self.model.encode([query_text], convert_to_tensor=True)

        scores = util.pytorch_cos_sim(q, embs)[0]

        top_k = min(top_k, scores.shape[0])
        vals, idxs = torch.topk(scores, k=top_k)
        return [data["ids"][i] for i in idxs.cpu().tolist()], vals.cpu().tolist()

    def query_and_text(self, query_text: str, top_k: int = None, include_paper_id: bool = False):
        if top_k is None:
            top_k = self.config.ModuleInfo.Database.default_top_k
        paper_ids, _ = self.query(query_text, top_k=top_k)
        # if self.config.BasicInfo.debug:
        #     self.logger.info(f"Database query for '{query_text}' returned paper IDs: {paper_ids}")
        texts = ""
        for pid in paper_ids:
            text, _, _ = self._text_for(pid)
            if not text:
                continue
            if self.use_title_in_draft and not include_paper_id:
                texts += f"{text}\n\n"
            else:
                texts += f"Paper_id: {pid}\n{text}\n\n"
        return texts

    def resolve_title_to_paper_id(self, title_text: str, min_title_similarity: float = 0.0):
        """Given a title string, return (paper_id, matched_title, similarity) if confident; raise if likely hallucination."""
        if not isinstance(title_text, str) or not title_text.strip():
            raise ValueError("title_text must be a non-empty string")

        # Find nearest paper using the title-only index to avoid abstract influence.
        paper_ids, sims = self.query_titles(title_text, top_k=1)
        if not paper_ids:
            raise ValueError("No papers available to match title")

        top_pid = paper_ids[0]
        top_sim = sims[0]

        try:
            matched_title, _ = self.work_collector.get_paper_title_abstract(top_pid)
            matched_title = matched_title or ""
        except Exception as e:
            raise ValueError(f"No title found for matched paper ID {top_pid}")

        # Recompute similarity between input title text and the retrieved title for a sanity check.
        q_vec = self.model.encode([title_text], convert_to_tensor=True, device=self.device)
        t_vec = self.model.encode([matched_title], convert_to_tensor=True, device=self.device)
        title_sim = util.cos_sim(q_vec, t_vec)[0][0].item()

        # if self.config.BasicInfo.debug:
        #     self.logger.info(
        #         f"Title similarity check: input '{title_text}' vs retrieved '{matched_title}' (ID {top_pid}) = {title_sim:.3f}"
        #     )


        if title_sim < min_title_similarity:
            raise ValueError(
                f"Title mismatch: input '{title_text}' vs retrieved '{matched_title}' (ID {top_pid}) with similarity {title_sim:.3f} < {min_title_similarity}"
            )

        return top_pid, matched_title, title_sim