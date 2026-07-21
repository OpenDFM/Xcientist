import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path

EDGE_COLORS = {
    'Extension': '#e74c3c',
    'Background': '#3498db',
    'Comparison': '#2ecc71',
    'Alternative': '#9b59b6',
    'Methodology': '#f39c12',
    'Evaluation': '#1abc9c',
}

EDGE_STYLES = {
    'Extension': 'solid',
    'Background': 'dashed',
    'Comparison': 'dotted',
    'Alternative': 'dashdot',
    'Methodology': 'solid',
    'Evaluation': 'dashed',
}

def load_graph_data(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_graph(data: dict):
    G = nx.DiGraph()
    
    category_graphs = {}
    
    for category, category_data in data.items():
        G_cat = nx.DiGraph()
        G_cat.graph['category'] = category
        
        nodes = category_data.get('nodes', [])
        edges = category_data.get('edges', [])
        
        for node in nodes:
            G_cat.add_node(node)
            G.add_node(node, category=category)
        
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            edge_type = edge.get('type', 'Background')
            analysis = edge.get('analysis', '')
            
            if source and target:
                G_cat.add_edge(source, target, type=edge_type, analysis=analysis)
                G.add_edge(source, target, type=edge_type, analysis=analysis, category=category)
        
        category_graphs[category] = G_cat
    
    return G, category_graphs

def draw_category_graph(G_cat: nx.DiGraph, ax: plt.Axes, title: str = None):
    if len(G_cat.nodes) == 0:
        return
    
    pos = nx.kamada_kawai_layout(G_cat)
    
    node_colors = ['#3498db' for _ in G_cat.nodes()]
    node_sizes = [min(6000, 3000 + 500 * G_cat.degree(n)) for n in G_cat.nodes()]
    
    edge_types = [G_cat.edges[e].get('type', 'Background') for e in G_cat.edges()]
    edge_colors = [EDGE_COLORS.get(et, '#95a5a6') for et in edge_types]
    edge_styles = [EDGE_STYLES.get(et, 'solid') for et in edge_types]
    
    nx.draw_networkx_nodes(G_cat, pos, ax=ax, node_color=node_colors, 
                           node_size=node_sizes, alpha=0.9,
                           edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G_cat, pos, ax=ax, font_size=8, font_weight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    for (u, v), color, style in zip(G_cat.edges(), edge_colors, edge_styles):
        nx.draw_networkx_edges(G_cat, pos, ax=ax, edgelist=[(u, v)],
                               edge_color=color, style=style, 
                               arrows=True, arrowsize=15,
                               connectionstyle="arc3,rad=0.1",
                               width=2)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.axis('off')

def plot_full_graph(G: nx.DiGraph, output_path: str = None, figsize=(20, 16)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    categories = list(set(nx.get_node_attributes(G, 'category').values()))
    color_map = plt.cm.Set2(range(len(categories)))
    category_colors = {cat: color_map[i] for i, cat in enumerate(categories)}
    
    node_colors = [category_colors.get(G.nodes[n].get('category', ''), '#95a5a6') 
                   for n in G.nodes()]
    
    # pos = nx.kamada_kawai_layout(G)
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
    #                        node_size=4000, alpha=0.9,
    #                        edgecolors='white', linewidths=2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                            node_size=4000, alpha=0.9,
                            edgecolors='white', linewidths=2)
    # nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold',
    #                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold',
                            alpha=0.8)
    
    for edge_type, color in EDGE_COLORS.items():
        edges_of_type = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == edge_type]
        if edges_of_type:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges_of_type,
                                   edge_color=color, style=EDGE_STYLES.get(edge_type, 'solid'),
                                   arrows=True, arrowsize=12,
                                   connectionstyle="arc3,rad=0.1", alpha=0.7,
                                   width=2)
    
    legend_patches = [mpatches.Patch(color=color, label=edge_type) 
                      for edge_type, color in EDGE_COLORS.items()]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=18)
    
    ax.set_title('Research Paper Relationship Graph', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Graph saved to: {output_path}")
    
    return fig

def plot_category_graphs(data: dict, output_dir: str = None, figsize=(12, 10)):
    num_categories = len(data)
    cols = 2
    rows = (num_categories + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
    axes = axes.flatten() if num_categories > 1 else [axes]
    
    for idx, (category, category_data) in enumerate(data.items()):
        G_cat, _ = create_graph({category: category_data})
        draw_category_graph(G_cat, axes[idx], title=category)
    
    for idx in range(num_categories, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'category_graphs.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Category graphs saved to: {output_path}")
    
    return fig

def generate_latex_integration(G: nx.Graph, output_path: str = None) -> str:
    latex_template = """
\\section{Relationship Visualization}
\\label{appendix:graphs}

The following figures illustrate the relationships between papers in the generated survey.

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.9\\textwidth]{{graphs/full_graph.png}}
\\caption{Overall Research Paper Relationship Graph}
\\label{fig:full-graph}
\\end{figure}

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.9\\textwidth]{{graphs/category_graphs.png}}
\\caption{Relationship Graphs by Category}
\\label{fig:category-graphs}
\\end{figure}

"""
    latex_output = latex_template
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_output)
    
    return latex_output

def plot_edge_type_summary(G: nx.DiGraph, output_path: str = None, figsize=(10, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    edge_types = [d.get('type', 'Unknown') for _, _, d in G.edges(data=True)]
    type_counts = {}
    for et in edge_types:
        type_counts[et] = type_counts.get(et, 0) + 1
    
    types = list(type_counts.keys())
    counts = list(type_counts.values())
    colors = [EDGE_COLORS.get(t, '#95a5a6') for t in types]
    
    bars = ax.barh(types, counts, color=colors, edgecolor='black', linewidth=1)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=10)
    
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Relationship Type', fontsize=12)
    ax.set_title('Distribution of Paper Relationships', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(counts) * 1.2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Edge type summary saved to: {output_path}")
    
    return fig

def plot_node_degree_distribution(G: nx.DiGraph, output_path: str = None, figsize=(10, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    degrees = [G.degree(n) for n in G.nodes()]
    
    ax.hist(degrees, bins=range(min(degrees), max(degrees) + 2), 
            color='#3498db', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Degree (Number of Connections)', fontsize=12)
    ax.set_ylabel('Number of Papers', fontsize=12)
    ax.set_title('Paper Connectivity Distribution', fontsize=14, fontweight='bold')
    
    mean_degree = sum(degrees) / len(degrees) if degrees else 0
    ax.axvline(mean_degree, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_degree:.2f}')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Degree distribution saved to: {output_path}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize survey relation graph')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to survey_relation_graph.json')
    parser.add_argument('--output-dir', '-o', type=str, default=None, help='Output directory for figures')
    parser.add_argument('--mode', '-m', type=str, 
                        choices=['full', 'categories', 'summary', 'latex'],
                        default='full', help='Visualization mode')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or './graph_output'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    data = load_graph_data(args.input)
    G, category_graphs = create_graph(data)
    
    print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    if args.mode == 'full':
        plot_full_graph(G, output_path=str(Path(output_dir) / 'full_graph.png'))
        plot_category_graphs(data, output_dir=output_dir)
    elif args.mode == 'categories':
        plot_category_graphs(data, output_dir=output_dir)
    elif args.mode == 'summary':
        plot_edge_type_summary(G, output_path=str(Path(output_dir) / 'edge_type_summary.png'))
        plot_node_degree_distribution(G, output_path=str(Path(output_dir) / 'degree_distribution.png'))
    elif args.mode == 'latex':
        generate_latex_integration(G, output_path=str(Path(output_dir) / 'graph_latex.tex'))
    
    plt.show()