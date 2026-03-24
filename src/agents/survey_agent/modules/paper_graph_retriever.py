import sqlite3
from typing import List, Dict, Optional, Tuple
import hydra
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.rich_logger import get_logger
from utils.step2v2_extractor import (
    build_main_extraction_prompt,
    build_baseline_extraction_prompt,
    extract_regex_candidates,
    clean_title_latex,
    aggregate_results,
    validate_and_parse_main,
    validate_and_parse_baseline,
    format_extraction_result,
)
from utils.api_call import ChatAgent
from modules.data_manager import DataManager
import re


class PaperGraphRetriever:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger("PaperGraphRetriever")
        self.db_path = self.config.ModuleInfo.PaperGraphRetriever.db_path
        self.chat_agent = ChatAgent(config)
        self.data_manager = DataManager(config)

    def search_by_paper_title(self, title_query: str, limit: int = 20):
        """
        根据 paper_title 模糊搜索论文
        
        Args:
            title_query: 标题关键词，支持模糊匹配
            limit: 返回结果数量限制，默认20
        
        Returns:
            list: 匹配的论文节点列表，每项为包含节点信息的字典
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        pattern = f"%{title_query}%"
        sql = """
            SELECT id, node_type, paper_id, paper_title, pub_year, 
                source_venue, full_name, acronym, summary
            FROM nodes
            WHERE paper_title LIKE ?
            ORDER BY 
                CASE node_type
                    WHEN 'Core' THEN 1
                    WHEN 'Baseline' THEN 2
                    WHEN 'Dataset' THEN 3
                    ELSE 4
                END,
                pub_year DESC
            LIMIT ?
        """
        cursor.execute(sql, (pattern, limit))
        rows = cursor.fetchall()
        
        results = [dict(row) for row in rows]
        
        conn.close()
        return results


    def search_by_paper_id(self, paper_id: str, limit: int = 20):
        """
        根据 paper_id 精确搜索论文
        
        Args:
            paper_id: 论文ID（如 'vlm', 'nlp' 等）
            limit: 返回结果数量限制，默认20
        
        Returns:
            list: 匹配的论文节点列表，每项为包含节点信息的字典
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        sql = """
            SELECT id, node_type, paper_id, paper_title, pub_year,
                source_venue, full_name, acronym, summary
            FROM nodes
            WHERE paper_id = ?
            ORDER BY 
                CASE node_type
                    WHEN 'Core' THEN 1
                    WHEN 'Baseline' THEN 2
                    WHEN 'Dataset' THEN 3
                    ELSE 4
                END,
                pub_year DESC
            LIMIT ?
        """
        cursor.execute(sql, (paper_id, limit))
        rows = cursor.fetchall()
        
        results = [dict(row) for row in rows]
        
        conn.close()
        return results


    def get_adjacent_nodes(self, node_id: str, edge_types: list = None, include_edges: bool = False):
        """
        获取与指定节点相邻的所有节点
        
        Args:
            node_id: 节点ID（可以是论文的 id 字段）
            edge_types: 可选的边类型过滤列表
            include_edges: 是否同时返回边信息
        
        Returns:
            如果 include_edges=False: 返回相邻节点的信息列表
            如果 include_edges=True: 返回 (相邻节点列表, 边列表) 的元组
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        out_edges_sql = """
            SELECT source, target, edge_type, summary, keywords, insight
            FROM edges
            WHERE source = ?
        """
        in_edges_sql = """
            SELECT source, target, edge_type, summary, keywords, insight
            FROM edges
            WHERE target = ?
        """
        
        params = [node_id]
        if edge_types:
            placeholders = ','.join(['?' for _ in edge_types])
            out_edges_sql += f" AND edge_type IN ({placeholders})"
            in_edges_sql += f" AND edge_type IN ({placeholders})"
            params = [node_id] + edge_types
        
        out_edges = cursor.execute(out_edges_sql, params).fetchall()
        in_edges = cursor.execute(in_edges_sql, params).fetchall()
        
        neighbor_ids = set()
        all_edges = []
        
        for e in out_edges:
            neighbor_ids.add(e['target'])
            all_edges.append(dict(e))
        
        for e in in_edges:
            neighbor_ids.add(e['source'])
            all_edges.append(dict(e))
        
        if neighbor_ids:
            placeholders = ','.join(['?' for _ in neighbor_ids])
            nodes_sql = f"""
                SELECT id, node_type, paper_id, paper_title, pub_year,
                    source_venue, full_name, acronym, summary
                FROM nodes
                WHERE id IN ({placeholders})
                ORDER BY 
                    CASE node_type
                        WHEN 'Core' THEN 1
                        WHEN 'Baseline' THEN 2
                        WHEN 'Dataset' THEN 3
                        ELSE 4
                    END
            """
            cursor.execute(nodes_sql, list(neighbor_ids))
            neighbors = [dict(row) for row in cursor.fetchall()]
        else:
            neighbors = []
        
        conn.close()
        
        if include_edges:
            return neighbors, all_edges
        return neighbors


    def debug_table(self, table_name: str, limit: int = 10, where_clause: str = None, where_params: tuple = None):
        """Debug函数：查看任意表的数据结构和内容"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [dict(row) for row in cursor.fetchall()]
        
        if where_clause:
            sql = f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT ?"
            params = (where_params if where_params else ()) + (limit,)
        else:
            sql = f"SELECT * FROM {table_name} LIMIT ?"
            params = (limit,)
        
        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "columns": columns,
            "rows": rows,
            "total_columns": len(columns),
            "total_rows_fetched": len(rows)
        }


    def print_debug_table(self, table_name: str, limit: int = 5, where_clause: str = None, where_params: tuple = None):
        result = self.debug_table(table_name, limit, where_clause, where_params)
        
        print(f"\n{'='*60}")
        print(f"表名: {table_name}")
        print(f"{'='*60}")
        
        print(f"\n【列结构】(共 {result['total_columns']} 列)")
        print("-" * 40)
        for col in result['columns']:
            print(f"  {col['name']:30} | {col['type']:10} | nullable: {col['notnull']}")
        
        print(f"\n【数据】(显示前 {result['total_rows_fetched']} 条)")
        print("-" * 40)
        
        if not result['rows']:
            print("  (无数据)")
        else:
            for i, row in enumerate(result['rows']):
                print(f"\n  --- 第 {i+1} 条 ---")
                for key, value in row.items():
                    value_str = str(value) if value is not None else "NULL"
                    if len(value_str) > 80:
                        value_str = value_str[:80] + "..."
                    print(f"    {key}: {value_str}")
        
        print(f"\n{'='*60}\n")
        return result


    def print_all_tables(self, limit: int = 5):
        """打印数据库中所有表的列信息和前若干行数据"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row['name'] for row in cursor.fetchall()]
        
        if not tables:
            print("数据库中没有表")
            conn.close()
            return
        
        print(f"\n{'='*80}")
        print(f"数据库: {self.db_path}")
        print(f"包含 {len(tables)} 个表")
        print(f"{'='*80}")
        
        for table_name in tables:
            print(f"\n{'='*80}")
            print(f"表名: {table_name}")
            print(f"{'='*80}")
            
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [dict(row) for row in cursor.fetchall()]
            
            print(f"\n【列结构】(共 {len(columns)} 列)")
            print("-" * 60)
            for col in columns:
                pk = " (PK)" if col['pk'] else ""
                nullable = "" if col['notnull'] else " (nullable)"
                print(f"  {col['cid']:3} | {col['name']:30} | {col['type']:15}{pk}{nullable}")
            
            cursor.execute(f"SELECT * FROM {table_name} LIMIT ?", (limit,))
            rows = cursor.fetchall()
            total_rows = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            print(f"\n【数据】(共 {total_rows} 条，显示前 {min(limit, len(rows))} 条)")
            print("-" * 60)
            
            if not rows:
                print("  (无数据)")
            else:
                for i, row in enumerate(rows):
                    row_dict = dict(row)
                    print(f"\n  --- 第 {i+1} 条 ---")
                    for key, value in row_dict.items():
                        value_str = str(value) if value is not None else "NULL"
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        print(f"    {key}: {value_str}")
        
        conn.close()
        print(f"\n{'='*80}\n")

    def expand_nodes(self, node_ids: List[str], max_step: int = 2):
        visited = set(node_ids)
        current_layer = list(node_ids)
        
        for step in range(max_step):
            next_layer = []
            for node_id in current_layer:
                neighbors, _ = self.get_adjacent_nodes(node_id, include_edges=True)
                for node in neighbors:
                    if node['id'] not in visited:
                        visited.add(node['id'])
                        next_layer.append(node['id'])
            
            if not next_layer:
                break
            current_layer = next_layer
        
        return list(visited)

    def format_details(self, details: dict):
        if not details:
            return "No details found."
        
        formatted = f"Paper Title: {details.get('paper_title', 'N/A')}\n"
        formatted += f"Paper Type: {details.get('paper_type', 'N/A')}\n"
        formatted += f"Domain: {details.get('paper_domain', 'N/A')}\n"
        formatted += f"Quote: {details.get('quote', 'N/A')}\n"
        formatted += f"Summary: {details.get('summary', 'N/A')}\n"
        formatted += f"Keywords: {details.get('keywords', 'N/A')}\n"
        formatted += f"Insight: {details.get('insight', 'N/A')}\n"
        
        return formatted

    def title_to_id(self, title: str) -> str:
        """
        Convert paper title to node id.
        
        Args:
            title: Paper title (exact match or fuzzy match)
            
        Returns:
            Node id string
            
        Raises:
            ValueError: If title not found
        """
        # Use existing search function
        results = self.search_by_paper_title(title, limit=1)
        
        if not results:
            raise ValueError(f"Title not found in paper graph: {title}, cannot convert to id")
        
        return results[0]['id']

    def id_to_title(self, node_id: str) -> str:
        """
        Convert node id to paper title.
        
        Args:
            node_id: Node id in the graph
            
        Returns:
            Paper title string
            
        Raises:
            ValueError: If id not found
        """
        # Use existing search function
        results = self.search_by_paper_id(node_id, limit=1)
        
        if not results or not results[0].get('paper_title'):
            raise ValueError(f"Node id not found in paper graph: {node_id}, cannot find title")
        
        return results[0]['paper_title']

    def get_paper_details(self, paper_id: str):
        details = self.search_by_paper_id(paper_id, 1)
        if not details:
            return None
        return details[0]

    def get_paper_keynote(self, paper_ids: List[str], use_graph_id: bool = False):
        """
        Get paper keynote information.
        If node_type is 'baseline' or node info is missing, extract information.
        
        Args:
            paper_ids: List of paper/node IDs
            
        Returns:
            List of formatted keynote strings (same order as paper_ids)
        """
        error_ids = []
        if not use_graph_id:
            graph_ids = []
            for paper_id in paper_ids:
                try:
                    title = self.data_manager.get_paper_title(paper_id)
                    graph_id = self.title_to_id(title)
                    graph_ids.append(graph_id)
                except Exception as e:
                    self.logger.warning(f"Fail to convert ds_id to graph_id before retrieve in graph: {e}")
                    graph_ids.append(None)
        else:
            graph_ids = paper_ids
            
        nodes_to_be_constructed = []
        nodes_indices = []
        results = [None] * len(graph_ids)
        
        for idx, paper_id in enumerate(graph_ids):
            if not paper_id:
                error_ids.append(paper_id)
                continue

            details = self.get_paper_details(paper_id)
            if details and details.get('node_type', 'baseline').lower() != 'baseline':
                results[idx] = self.format_details(details)
            else:
                if details and details.get('node_type', '').lower() == 'baseline':
                    self.logger.info("baseline node. Need to extract information")
                else:
                    self.logger.info("lack in paper graph. Need to extract information")
                nodes_to_be_constructed.append(paper_id)
                nodes_indices.append(idx)

        if nodes_to_be_constructed:
            extraction_results = self.extract_node_info(nodes_to_be_constructed)
            for i, extraction in enumerate(extraction_results):
                results[nodes_indices[i]] = format_extraction_result(extraction) if extraction else None
                
        for idx, result in enumerate(results):
            if not result:
                error_ids.append(paper_ids[idx])

        return results
    
    def get_paper_markdown(self, paper_id: str) -> Optional[str]:
        """Get the raw markdown text for a paper via data_manager."""
        try:
            paper_title = self.id_to_title(paper_id)
            ds_id = self.data_manager.get_paper_with_title(paper_title)
            return self.data_manager.get_paper_raw_markdown(ds_id)
        except Exception as e:
            self.logger.warning(f"Failed to get markdown for paper {paper_id}: {e}")
            return None

    def get_source_info(self, paper_id: str) -> dict:
        """Get source info (venue, year) for a paper from database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT source_venue, pub_year FROM nodes WHERE id = ?",
            (paper_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'source_venue': row['source_venue'] or 'Unknown',
                'pub_year': str(row['pub_year']) if row['pub_year'] else '2024'
            }
        return {'source_venue': 'Unknown', 'pub_year': '2024'}

    def extract_node_info(self, paper_ids: List[str]) -> List[dict]:
        """
        Extract key information from papers that need re-extraction.
        
        Args:
            paper_ids: List of paper/node IDs to extract information for
            
        Returns:
            List of extraction results
        """
        # Step 1: Collect markdown texts and source info for all papers
        papers = {}
        source_info = {}
        for paper_id in paper_ids:
            md = self.get_paper_markdown(paper_id)
            if md:
                papers[paper_id] = md
                source_info[paper_id] = self.get_source_info(paper_id)
            else:
                self.logger.warning(f"Markdown not found for paper {paper_id}, skipping")
        
        if not papers:
            self.logger.error("No markdown found for any of the requested papers")
            return []
        
        self.logger.info(f"Extracting info for {len(papers)} papers via batch_remote_chat_with_retry")
        
        # Step 2: Extract with retry
        final_results = self._extract_with_retry(papers, source_info)
        
        self.logger.info(f"Extracted info for {len(final_results)} papers")
        return final_results
    
    def _extract_with_retry(self, papers: Dict[str, str], source_info: Dict[str, dict]) -> List[dict]:
        """Extract paper info using batch_remote_chat_with_retry."""
        paper_ids = list(papers.keys())
        markdowns = list(papers.values())
        
        # ===== Step 1: Main Extraction =====
        main_prompts, main_metadata = self._build_main_prompts(paper_ids, markdowns)
        self.logger.info(f"Calling batch_remote_chat_with_retry for main extraction ({len(main_prompts)} papers)")
        
        # Build info_dict with metadata list for validation function
        main_info_dict = {'metadata': main_metadata}
        
        main_results = self.chat_agent.batch_remote_chat_with_retry(
            prompts=main_prompts,
            validate_fn=self._make_main_validate_fn(main_metadata),
            max_retry=3,
            desc="Main extraction for information missing nodes in graph",
            temperature=0.05,
            info_dict=main_info_dict
        )
        
        # Build core names map for baseline extraction
        core_names_map = {}
        for res in main_results:
            if res:
                core_names_map[res['paper_id']] = res.get('core_names', [])
        
        # ===== Step 2: Baseline Extraction =====
        baseline_prompts, baseline_metadata = self._build_baseline_prompts(paper_ids, markdowns, core_names_map)
        self.logger.info(f"Calling batch_remote_chat_with_retry for baseline extraction ({len(baseline_prompts)} papers)")
        
        baseline_info_dict = {'metadata': baseline_metadata}
        
        baseline_results = self.chat_agent.batch_remote_chat_with_retry(
            prompts=baseline_prompts,
            validate_fn=self._make_baseline_validate_fn(baseline_metadata),
            max_retry=3,
            desc="Baseline extraction",
            temperature=0.05,
            info_dict=baseline_info_dict
        )
        
        # ===== Step 3: Aggregate =====
        return aggregate_results(main_results, baseline_results, source_info)
    
    def _make_main_validate_fn(self, metadata_list: List[dict]):
        """Create a validate function with closure for main extraction."""
        def validate_fn(response: str, info_dict: dict = None) -> Tuple[bool, dict]:
            idx = info_dict.get('idx', 0) if info_dict else 0
            paper_meta = metadata_list[idx] if idx < len(metadata_list) else {}
            return validate_and_parse_main(response, paper_meta)
        return validate_fn
    
    def _make_baseline_validate_fn(self, metadata_list: List[dict]):
        """Create a validate function with closure for baseline extraction."""
        def validate_fn(response: str, info_dict: dict = None) -> Tuple[bool, dict]:
            idx = info_dict.get('idx', 0) if info_dict else 0
            paper_meta = metadata_list[idx] if idx < len(metadata_list) else {}
            return validate_and_parse_baseline(response, paper_meta)
        return validate_fn
    
    def _build_main_prompts(self, paper_ids: List[str], markdowns: List[str]) -> Tuple[List[str], List[dict]]:
        """Build main extraction prompts for papers."""
        prompts = []
        metadata = []
        
        for paper_id, markdown in zip(paper_ids, markdowns):
            # Extract title from markdown
            title = "Unknown Title"
            for line in markdown.split('\n')[:20]:
                match = re.match(r'^#\s+(.+)$', line.strip())
                if match:
                    title = clean_title_latex(match.group(1))
                    break
            
            regex_candidates = extract_regex_candidates(markdown)
            system, user = build_main_extraction_prompt(markdown, title, regex_candidates)
            
            prompts.append(f"{system}\n\n{user}")
            metadata.append({'paper_id': paper_id, 'title': title, 'regex_candidates': regex_candidates})
        
        return prompts, metadata
    
    def _build_baseline_prompts(
        self, 
        paper_ids: List[str], 
        markdowns: List[str],
        core_names_map: Dict[str, List[str]]
    ) -> Tuple[List[str], List[dict]]:
        """Build baseline extraction prompts for papers."""
        prompts = []
        metadata = []
        
        for paper_id, markdown in zip(paper_ids, markdowns):
            core_names = core_names_map.get(paper_id, [])
            regex_candidates = extract_regex_candidates(markdown)
            system, user = build_baseline_extraction_prompt(markdown, core_names, regex_candidates)
            
            prompts.append(f"{system}\n\n{user}")
            metadata.append({'paper_id': paper_id, 'core_names': core_names})
        
        return prompts, metadata


@hydra.main(config_path="../config", config_name="deep_survey_batch_others_huoshan", version_base=None)
def main(config):
    retriever = PaperGraphRetriever(config)
    # retriever.print_debug_table("nodes", limit = 10)
    # ids = ["Robust Training Methods", "Evaluation Protocol", "TWIST", "BUGFARM", "Multilingual Domain Adaptation with Adapters", "VAENAR-TTS", 
    # "Uniform-Sum Compression Ratio Sampling", "SAQA", "Two-Branch Swin Block", "MixA-Q"]
    id_list = retriever.expand_nodes([retriever.title_to_id("Attention Is All You Need")], max_step = 2)
    returned = retriever.get_paper_keynote(id_list, True)
    returned = [r for r in returned if not r is None]
    print(len(returned))
    print(returned)



if __name__ == "__main__":
    main()
