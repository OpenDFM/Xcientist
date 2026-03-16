import os
import re
import json
import html
import sqlite3
import argparse
import webbrowser
from collections import deque
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DB_PATH = BASE_DIR / "graph.db"
OUT_DIR = BASE_DIR

def normalize_key(name):
    if not name:
        return ""
    s = str(name).lower().strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch == "+":
            out.append(ch)
    return "".join(out)

def safe_json_loads(x):
    if x is None:
        return None
    if isinstance(x, (dict, list, int, float, bool)):
        return x
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s:
        return x
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")) or s == "null":
        try:
            return json.loads(s)
        except:
            return x
    return x

def clean_filename(s):
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:80] if s else "subgraph"

def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def resolve_query_node(cur, query):
    norm = normalize_key(query)

    rows = cur.execute("""
    select target_id
    from aliases
    where normalized_alias = ?
    limit 20
    """, (norm,)).fetchall()

    if rows:
        target_ids = []
        seen = set()
        for r in rows:
            tid = r["target_id"]
            if tid not in seen:
                seen.add(tid)
                target_ids.append(tid)

        scored = []
        for tid in target_ids:
            row = cur.execute("select raw_json from nodes where id = ?", (tid,)).fetchone()
            if row:
                obj = json.loads(row["raw_json"])
                score = 0
                if obj.get("node_type") == "Core":
                    score += 100
                elif obj.get("node_type") == "Baseline":
                    score += 50
                elif obj.get("node_type") == "Dataset":
                    score += 20
                name = str(obj.get("full_name", "") or obj.get("id", "")).lower()
                if name == query.lower():
                    score += 30
                if normalize_key(obj.get("id", "")) == norm:
                    score += 20
                scored.append((score, obj))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

    pattern = "%" + query + "%"

    for node_type_weight in [("Core", 100), ("Baseline", 50), ("Dataset", 20), ("", 0)]:
        node_type, weight = node_type_weight
        if node_type:
            rows = cur.execute("""
            select raw_json
            from nodes
            where node_type = ?
              and (
                id like ?
                or full_name like ?
                or acronym like ?
                or paper_title like ?
                or summary like ?
              )
            limit 50
            """, (node_type, pattern, pattern, pattern, pattern, pattern)).fetchall()
        else:
            rows = cur.execute("""
            select raw_json
            from nodes
            where
                id like ?
                or full_name like ?
                or acronym like ?
                or paper_title like ?
                or summary like ?
            limit 50
            """, (pattern, pattern, pattern, pattern, pattern)).fetchall()

        if rows:
            scored = []
            for r in rows:
                obj = json.loads(r["raw_json"])
                score = weight
                text_id = str(obj.get("id", "")).lower()
                text_name = str(obj.get("full_name", "")).lower()
                text_acro = str(obj.get("acronym", "")).lower()
                if text_name == query.lower():
                    score += 50
                if text_id == query.lower():
                    score += 40
                if text_acro == query.lower():
                    score += 35
                if normalize_key(text_name) == norm:
                    score += 25
                if normalize_key(text_id) == norm:
                    score += 20
                if normalize_key(text_acro) == norm:
                    score += 20
                if query.lower() in text_name:
                    score += 10
                if query.lower() in text_id:
                    score += 8
                scored.append((score, obj))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

    return None

def fetch_node(cur, node_id):
    row = cur.execute("select raw_json from nodes where id = ?", (node_id,)).fetchone()
    if not row:
        return None
    return json.loads(row["raw_json"])

def fetch_out_edges(cur, node_id):
    rows = cur.execute("""
    select raw_json
    from edges
    where source = ?
    """, (node_id,)).fetchall()
    return [json.loads(r["raw_json"]) for r in rows]

def fetch_in_edges(cur, node_id):
    rows = cur.execute("""
    select raw_json
    from edges
    where target = ?
    """, (node_id,)).fetchall()
    return [json.loads(r["raw_json"]) for r in rows]

def score_edge_for_expansion(edge):
    edge_type = edge.get("edge_type", "")
    score = 0
    if edge_type == "core_relation":
        score += 5
    elif edge_type == "baseline_comparison":
        score += 4
    elif edge_type == "evaluated_on":
        score += 3
    else:
        score += 1
    summary = str(edge.get("summary", "") or "")
    if summary:
        score += 1
    return score

def bfs_subgraph(cur, center_id, hops=2, max_nodes=180, max_edges=500, per_hop_expand=25):
    visited = set([center_id])
    q = deque([(center_id, 0)])

    node_ids = set([center_id])
    edges = []
    edge_keys = set()

    while q:
        current, depth = q.popleft()
        if depth >= hops:
            continue
        if len(node_ids) >= max_nodes or len(edges) >= max_edges:
            break

        out_edges = fetch_out_edges(cur, current)
        in_edges = fetch_in_edges(cur, current)

        all_edges = out_edges + in_edges
        all_edges.sort(key=score_edge_for_expansion, reverse=True)
        if per_hop_expand > 0:
            all_edges = all_edges[:per_hop_expand]

        for e in all_edges:
            if len(node_ids) >= max_nodes or len(edges) >= max_edges:
                break

            s = e.get("source")
            t = e.get("target")
            if not s or not t:
                continue

            other = t if s == current else s

            if other not in node_ids and len(node_ids) >= max_nodes:
                continue

            if other not in node_ids:
                node_ids.add(other)

            key = (
                s,
                t,
                e.get("edge_type", ""),
                e.get("summary", "")
            )
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append(e)

            if other not in visited:
                visited.add(other)
                q.append((other, depth + 1))

    filtered_edges = []
    final_edge_keys = set()
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if s in node_ids and t in node_ids:
            key = (
                s,
                t,
                e.get("edge_type", ""),
                e.get("summary", "")
            )
            if key not in final_edge_keys:
                final_edge_keys.add(key)
                filtered_edges.append(e)
        if len(filtered_edges) >= max_edges:
            break

    nodes = []
    for nid in node_ids:
        obj = fetch_node(cur, nid)
        if obj:
            nodes.append(obj)

    return nodes, filtered_edges

def simplify_value(v, max_len=1200):
    v = safe_json_loads(v)
    if isinstance(v, str) and len(v) > max_len:
        return v[:max_len] + " ..."
    return v

def split_node_fields(n):
    main_keys = [
        "id", "node_type", "full_name", "acronym", "paper_id",
        "pub_year", "paper_title", "paper_domain", "paper_type",
        "core_type", "source_venue", "summary"
    ]
    complex_keys = [
        "keywords", "aliases", "metrics", "structured_summary", "components",
        "problems", "innovations", "limitations", "future_work",
        "insight", "quote", "code_url"
    ]

    main = {}
    extra = {}

    for k, v in n.items():
        sv = simplify_value(v)
        if k in main_keys:
            main[k] = sv
        elif k in complex_keys:
            extra[k] = sv
        else:
            extra[k] = sv

    return main, extra

def split_edge_fields(e):
    main_keys = ["source", "target", "edge_type", "summary"]
    complex_keys = ["keywords", "metrics", "insight", "quote"]

    main = {}
    extra = {}

    for k, v in e.items():
        sv = simplify_value(v)
        if k in main_keys:
            main[k] = sv
        elif k in complex_keys:
            extra[k] = sv
        else:
            extra[k] = sv

    return main, extra

def prepare_nodes_for_vis(nodes, center_id):
    out = []
    for n in nodes:
        node_type = n.get("node_type", "Other")
        if node_type == "Core":
            color = {"background": "#ffd166", "border": "#cc9a06"}
            size = 24
        elif node_type == "Baseline":
            color = {"background": "#7bdff2", "border": "#1f8ea5"}
            size = 18
        elif node_type == "Dataset":
            color = {"background": "#b2f7b6", "border": "#2a8f38"}
            size = 18
        else:
            color = {"background": "#d9d9d9", "border": "#8d8d8d"}
            size = 16

        if n.get("id") == center_id:
            size = 34
            color = {"background": "#ff8fab", "border": "#c0395e"}

        main, extra = split_node_fields(n)

        out.append({
            "id": n.get("id"),
            "label": n.get("full_name") or n.get("id"),
            "group": node_type,
            "size": size,
            "color": color,
            "title": html.escape(str(n.get("summary", "") or "")[:200]),
            "main": main,
            "extra": extra
        })
    return out

def prepare_edges_for_vis(edges):
    out = []
    for idx, e in enumerate(edges):
        edge_type = e.get("edge_type", "")
        if edge_type == "baseline_comparison":
            color = {"color": "#3a86ff", "highlight": "#1d4ed8"}
        elif edge_type == "evaluated_on":
            color = {"color": "#2a9d8f", "highlight": "#1f6f66"}
        elif edge_type == "core_relation":
            color = {"color": "#e76f51", "highlight": "#b94f33"}
        else:
            color = {"color": "#999999", "highlight": "#666666"}

        main, extra = split_edge_fields(e)

        out.append({
            "id": "edge_" + str(idx),
            "from": e.get("source"),
            "to": e.get("target"),
            "label": edge_type,
            "arrows": "to",
            "color": color,
            "width": 1.4,
            "main": main,
            "extra": extra
        })
    return out

def build_html(query, center_node, nodes_vis, edges_vis, hops, max_nodes, max_edges):
    template = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Subgraph Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    font-family: Arial, Helvetica, sans-serif;
    background: #f5f7fb;
}
#app {
    display: flex;
    width: 100%;
    height: 100vh;
    min-height: 0;
}
#left {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
}
#topbar {
    padding: 12px 16px;
    background: #ffffff;
    border-bottom: 1px solid #d9e0ea;
}
#topbar .title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 6px;
}
#topbar .meta {
    color: #516173;
    font-size: 13px;
    line-height: 1.5;
}
#toolbar {
    margin-top: 10px;
}
button {
    border: 1px solid #d0d7e2;
    background: #fff;
    border-radius: 8px;
    padding: 6px 10px;
    cursor: pointer;
    margin-right: 8px;
}
button:hover {
    background: #f3f6fb;
}
#graph-wrap {
    position: relative;
    flex: 1 1 auto;
    min-height: 0;
    height: 100%;
    overflow: hidden;
    background: #ffffff;
}
#network {
    width: 100%;
    height: 100%;
    display: block;
    background: #ffffff;
}
#right {
    width: 420px;
    max-width: 46vw;
    min-width: 340px;
    border-left: 1px solid #d9e0ea;
    background: #fbfcfe;
    overflow-y: auto;
    overscroll-behavior: contain;
}
#panel {
    padding: 16px;
}
.panel-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 10px;
}
.panel-sub {
    color: #6b7b8d;
    font-size: 13px;
    margin-bottom: 14px;
    line-height: 1.6;
}
.card {
    background: #ffffff;
    border: 1px solid #e0e7f0;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
}
.card h3 {
    margin: 0 0 8px 0;
    font-size: 15px;
}
.kv-item {
    border-top: 1px solid #eef2f7;
    padding: 8px 0;
}
.kv-item:first-child {
    border-top: none;
    padding-top: 0;
}
.kv-key {
    font-weight: 700;
    color: #243447;
    margin-bottom: 5px;
    word-break: break-word;
}
.kv-value {
    color: #314355;
    line-height: 1.55;
    white-space: pre-wrap;
    word-break: break-word;
}
.tag {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 999px;
    background: #edf2ff;
    color: #2f5bd2;
    font-size: 12px;
    margin-right: 6px;
    margin-bottom: 6px;
}
.section-title {
    font-size: 14px;
    font-weight: 700;
    margin: 14px 0 8px 0;
    color: #243447;
}
details {
    border: 1px solid #e7edf5;
    border-radius: 8px;
    padding: 8px 10px;
    margin-top: 8px;
    background: #fcfdff;
}
summary {
    cursor: pointer;
    font-weight: 700;
    color: #243447;
}
#loading {
    position: absolute;
    left: 24px;
    top: 24px;
    z-index: 10;
    background: rgba(255,255,255,0.92);
    border: 1px solid #d9e0ea;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 13px;
    color: #445566;
}
</style>
</head>
<body>
<div id="app">
    <div id="left">
        <div id="topbar">
            <div class="title">Subgraph Viewer</div>
            <div class="meta">
                query: <b>__QUERY__</b><br>
                center: <b>__CENTER_LABEL__</b><br>
                hops: <b>__HOPS__</b> &nbsp; max_nodes: <b>__MAX_NODES__</b> &nbsp; max_edges: <b>__MAX_EDGES__</b> &nbsp; nodes: <b>__NODE_COUNT__</b> &nbsp; edges: <b>__EDGE_COUNT__</b>
            </div>
            <div id="toolbar">
                <button onclick="fitGraph()">Fit</button>
                <button onclick="togglePhysics()">Toggle Physics</button>
                <button onclick="focusCenter()">Focus Center</button>
            </div>
        </div>
        <div id="graph-wrap">
            <div id="loading">Rendering graph...</div>
            <div id="network"></div>
        </div>
    </div>
    <div id="right">
        <div id="panel">
            <div class="panel-title">Details</div>
            <div class="panel-sub">Click a node or edge to inspect fields. Large structured fields are folded by default.</div>
            <div class="card">
                <h3>Legend</h3>
                <div>
                    <span class="tag">Core</span>
                    <span class="tag">Baseline</span>
                    <span class="tag">Dataset</span>
                </div>
            </div>
            <div id="details"></div>
        </div>
    </div>
</div>

<script>
const NODE_DATA = __NODE_DATA__;
const EDGE_DATA = __EDGE_DATA__;
const CENTER_ID = __CENTER_ID__;

const nodes = new vis.DataSet(NODE_DATA);
const edges = new vis.DataSet(EDGE_DATA);
const container = document.getElementById("network");

let physicsEnabled = false;

const network = new vis.Network(container, {
    nodes: nodes,
    edges: edges
}, {
    autoResize: true,
    interaction: {
        hover: true,
        navigationButtons: true,
        keyboard: false
    },
    physics: {
        enabled: physicsEnabled,
        stabilization: {
            enabled: false
        }
    },
    layout: {
        improvedLayout: true
    },
    edges: {
        smooth: false,
        font: {
            size: 10,
            align: "middle"
        },
        selectionWidth: 2
    },
    nodes: {
        shape: "dot",
        borderWidth: 1.5,
        font: {
            size: 13
        }
    }
});

function escapeHtml(text) {
    if (text === null || text === undefined) return "";
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderPrimitive(value) {
    if (value === null) return '<div class="kv-value">null</div>';
    if (value === undefined) return '<div class="kv-value">undefined</div>';
    return '<div class="kv-value">' + escapeHtml(String(value)) + '</div>';
}

function renderValue(value) {
    if (typeof value === "string") {
        const s = value.trim();
        if ((s.startsWith("{") && s.endsWith("}")) || (s.startsWith("[") && s.endsWith("]")) || s === "null") {
            try {
                const parsed = JSON.parse(s);
                return renderValue(parsed);
            } catch (e) {}
        }
    }

    if (Array.isArray(value)) {
        if (!value.length) return '<div class="kv-value">[]</div>';
        let out = "";
        for (let i = 0; i < value.length; i++) {
            out += '<details><summary>item ' + (i + 1) + '</summary>' + renderValue(value[i]) + '</details>';
        }
        return out;
    }

    if (value !== null && typeof value === "object") {
        const keys = Object.keys(value);
        if (!keys.length) return '<div class="kv-value">{}</div>';
        let out = "";
        for (const k of keys) {
            out += '<div class="kv-item">';
            out += '<div class="kv-key">' + escapeHtml(k) + '</div>';
            out += renderValue(value[k]);
            out += '</div>';
        }
        return out;
    }

    return renderPrimitive(value);
}

function renderSection(title, obj, folded=false) {
    if (!obj || Object.keys(obj).length === 0) return "";
    let inner = "";
    for (const k of Object.keys(obj)) {
        inner += '<div class="kv-item">';
        inner += '<div class="kv-key">' + escapeHtml(k) + '</div>';
        inner += renderValue(obj[k]);
        inner += '</div>';
    }
    if (folded) {
        return '<details><summary>' + escapeHtml(title) + '</summary>' + inner + '</details>';
    }
    return '<div class="section-title">' + escapeHtml(title) + '</div><div>' + inner + '</div>';
}

function showNodeDetails(nodeObj) {
    const el = document.getElementById("details");
    let h = '';
    h += '<div class="card">';
    h += '<h3>Node</h3>';
    h += '<div><span class="tag">' + escapeHtml(nodeObj.group || '') + '</span></div>';
    h += renderSection("Main", nodeObj.main || {}, false);
    h += renderSection("Structured fields", nodeObj.extra || {}, true);
    h += '</div>';
    el.innerHTML = h;
}

function showEdgeDetails(edgeObj) {
    const el = document.getElementById("details");
    let h = '';
    h += '<div class="card">';
    h += '<h3>Edge</h3>';
    h += '<div><span class="tag">' + escapeHtml(edgeObj.label || '') + '</span></div>';
    h += renderSection("Main", edgeObj.main || {}, false);
    h += renderSection("Structured fields", edgeObj.extra || {}, true);
    h += '</div>';
    el.innerHTML = h;
}

function hideLoading() {
    const loading = document.getElementById("loading");
    if (loading) loading.style.display = "none";
}

function fitGraph() {
    network.fit({
        animation: {
            duration: 500,
            easingFunction: "easeInOutQuad"
        }
    });
}

function focusCenter() {
    network.focus(CENTER_ID, {
        scale: 1.1,
        animation: {
            duration: 500,
            easingFunction: "easeInOutQuad"
        }
    });
}

function togglePhysics() {
    physicsEnabled = !physicsEnabled;
    network.setOptions({
        physics: {
            enabled: physicsEnabled,
            stabilization: {
                enabled: false
            },
            barnesHut: {
                gravitationalConstant: -3000,
                centralGravity: 0.15,
                springLength: 120,
                springConstant: 0.04,
                damping: 0.22,
                avoidOverlap: 0.5
            }
        }
    });
}

network.on("click", function(params) {
    if (params.nodes.length > 0) {
        const nodeObj = nodes.get(params.nodes[0]);
        showNodeDetails(nodeObj);
        return;
    }
    if (params.edges.length > 0) {
        const edgeObj = edges.get(params.edges[0]);
        showEdgeDetails(edgeObj);
        return;
    }
});

setTimeout(function() {
    network.redraw();
    hideLoading();
    fitGraph();
    const centerNode = nodes.get(CENTER_ID);
    if (centerNode) {
        showNodeDetails(centerNode);
    }
}, 300);

window.addEventListener("resize", function() {
    network.redraw();
    fitGraph();
});
</script>
</body>
</html>
"""
    html_text = template
    html_text = html_text.replace("__QUERY__", html.escape(str(query)))
    html_text = html_text.replace("__CENTER_LABEL__", html.escape(str(center_node.get("full_name") or center_node.get("id"))))
    html_text = html_text.replace("__HOPS__", str(hops))
    html_text = html_text.replace("__MAX_NODES__", str(max_nodes))
    html_text = html_text.replace("__MAX_EDGES__", str(max_edges))
    html_text = html_text.replace("__NODE_COUNT__", str(len(nodes_vis)))
    html_text = html_text.replace("__EDGE_COUNT__", str(len(edges_vis)))
    html_text = html_text.replace("__NODE_DATA__", json.dumps(nodes_vis, ensure_ascii=False))
    html_text = html_text.replace("__EDGE_DATA__", json.dumps(edges_vis, ensure_ascii=False))
    html_text = html_text.replace("__CENTER_ID__", json.dumps(center_node.get("id"), ensure_ascii=False))
    return html_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max_nodes", type=int, default=180)
    parser.add_argument("--max_edges", type=int, default=500)
    parser.add_argument("--per_hop_expand", type=int, default=25)
    parser.add_argument("--open_browser", type=int, default=1)
    args = parser.parse_args()

    hops = max(1, min(5, args.hops))

    conn = get_conn()
    cur = conn.cursor()

    center = resolve_query_node(cur, args.query)
    if not center:
        print("not found:", args.query)
        return

    print("center:", center.get("id"))

    nodes, edges = bfs_subgraph(
        cur,
        center.get("id"),
        hops=hops,
        max_nodes=max(20, args.max_nodes),
        max_edges=max(50, args.max_edges),
        per_hop_expand=max(5, args.per_hop_expand)
    )

    conn.close()

    nodes_vis = prepare_nodes_for_vis(nodes, center.get("id"))
    edges_vis = prepare_edges_for_vis(edges)

    html_text = build_html(
        args.query,
        center,
        nodes_vis,
        edges_vis,
        hops,
        args.max_nodes,
        args.max_edges
    )

    filename = "subgraph_" + clean_filename(args.query) + "_h" + str(hops) + ".html"
    out_path = os.path.join(OUT_DIR, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    print("saved:", out_path)

    if args.open_browser:
        webbrowser.open("file:///" + out_path.replace("\\", "/"))

if __name__ == "__main__":
    main()