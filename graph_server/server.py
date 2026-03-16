import os
import json
import sqlite3
from fastapi import FastAPI, Query
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DB_PATH = BASE_DIR / "graph.db"

app = FastAPI(title="Graph Server")

def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def normalize_key(name):
    if not name:
        return ""
    s = str(name).lower().strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch == "+":
            out.append(ch)
    return "".join(out)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/stats")
def stats():
    conn = get_conn()
    cur = conn.cursor()
    nodes = cur.execute("select count(*) from nodes").fetchone()[0]
    edges = cur.execute("select count(*) from edges").fetchone()[0]
    core = cur.execute("select count(*) from nodes where node_type='Core'").fetchone()[0]
    baseline = cur.execute("select count(*) from nodes where node_type='Baseline'").fetchone()[0]
    dataset = cur.execute("select count(*) from nodes where node_type='Dataset'").fetchone()[0]
    conn.close()
    return {
        "nodes": nodes,
        "edges": edges,
        "core": core,
        "baseline": baseline,
        "dataset": dataset
    }

@app.get("/node/{node_id}")
def get_node(node_id: str):
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute("select raw_json from nodes where id = ?", (node_id,)).fetchone()
    conn.close()
    if not row:
        return {"found": False}
    return {"found": True, "node": json.loads(row["raw_json"])}

@app.get("/paper/{paper_id}")
def get_paper_nodes(paper_id: str):
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute("""
    select raw_json from nodes where paper_id = ?
    order by node_type, id
    """, (paper_id,)).fetchall()
    conn.close()
    return {
        "paper_id": paper_id,
        "count": len(rows),
        "nodes": [json.loads(r["raw_json"]) for r in rows]
    }

@app.get("/neighbors/{node_id}")
def get_neighbors(node_id: str, limit: int = 100):
    conn = get_conn()
    cur = conn.cursor()

    out_rows = cur.execute("""
    select e.raw_json as edge_json, n.raw_json as node_json
    from edges e
    join nodes n on e.target = n.id
    where e.source = ?
    limit ?
    """, (node_id, limit)).fetchall()

    in_rows = cur.execute("""
    select e.raw_json as edge_json, n.raw_json as node_json
    from edges e
    join nodes n on e.source = n.id
    where e.target = ?
    limit ?
    """, (node_id, limit)).fetchall()

    conn.close()

    return {
        "node_id": node_id,
        "out_neighbors": [
            {"edge": json.loads(r["edge_json"]), "node": json.loads(r["node_json"])}
            for r in out_rows
        ],
        "in_neighbors": [
            {"edge": json.loads(r["edge_json"]), "node": json.loads(r["node_json"])}
            for r in in_rows
        ]
    }

@app.get("/alias")
def resolve_alias(q: str):
    norm = normalize_key(q)
    conn = get_conn()
    cur = conn.cursor()

    rows = cur.execute("""
    select alias, normalized_alias, target_id
    from aliases
    where normalized_alias = ?
    limit 20
    """, (norm,)).fetchall()

    if rows:
        target_ids = list(dict.fromkeys([r["target_id"] for r in rows]))
        nodes = []
        for tid in target_ids:
            n = cur.execute("select raw_json from nodes where id = ?", (tid,)).fetchone()
            if n:
                nodes.append(json.loads(n["raw_json"]))
        conn.close()
        return {
            "query": q,
            "normalized": norm,
            "matched": True,
            "targets": target_ids,
            "nodes": nodes
        }

    rows = cur.execute("""
    select alias, normalized_alias, target_id
    from aliases
    where normalized_alias like ?
    limit 20
    """, (norm + "%",)).fetchall()

    conn.close()
    return {
        "query": q,
        "normalized": norm,
        "matched": False,
        "candidates": [dict(r) for r in rows]
    }

@app.get("/search")
def search_nodes(
    q: str = Query(""),
    node_type: str = Query(""),
    limit: int = Query(20)
):
    conn = get_conn()
    cur = conn.cursor()

    if q.strip():
        sql = """
        select n.raw_json
        from node_fts f
        join nodes n on n.id = f.id
        where node_fts match ?
        """
        params = [q]
        if node_type:
            sql += " and n.node_type = ?"
            params.append(node_type)
        sql += " limit ?"
        params.append(limit)
        rows = cur.execute(sql, params).fetchall()
    else:
        if node_type:
            rows = cur.execute("""
            select raw_json from nodes
            where node_type = ?
            limit ?
            """, (node_type, limit)).fetchall()
        else:
            rows = cur.execute("""
            select raw_json from nodes
            limit ?
            """, (limit,)).fetchall()

    conn.close()
    return {
        "query": q,
        "node_type": node_type,
        "count": len(rows),
        "results": [json.loads(r["raw_json"]) for r in rows]
    }

@app.get("/search_simple")
def search_simple(
    q: str = Query(""),
    node_type: str = Query(""),
    limit: int = Query(20)
):
    conn = get_conn()
    cur = conn.cursor()

    pattern = "%" + q + "%"
    if node_type:
        rows = cur.execute("""
        select raw_json from nodes
        where node_type = ?
          and (
            id like ? or
            full_name like ? or
            acronym like ? or
            paper_title like ? or
            summary like ?
          )
        limit ?
        """, (node_type, pattern, pattern, pattern, pattern, pattern, limit)).fetchall()
    else:
        rows = cur.execute("""
        select raw_json from nodes
        where
            id like ? or
            full_name like ? or
            acronym like ? or
            paper_title like ? or
            summary like ?
        limit ?
        """, (pattern, pattern, pattern, pattern, pattern, limit)).fetchall()

    conn.close()
    return {
        "query": q,
        "node_type": node_type,
        "count": len(rows),
        "results": [json.loads(r["raw_json"]) for r in rows]
    }