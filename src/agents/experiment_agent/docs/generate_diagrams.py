#!/usr/bin/env python3
"""
SuperAgent Architecture Diagram Generator

Generates beautiful architecture diagrams using matplotlib.
Run this script to regenerate the diagrams in docs/images/

Requirements:
    pip install matplotlib numpy

Usage:
    python generate_diagrams.py
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
    Circle,
    Rectangle,
    Polygon,
)
import numpy as np


# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color Palette (Modern Dark Theme)
COLORS = {
    "bg": "#0d1117",
    "bg_light": "#161b22",
    "code_layer": "#1f6feb",
    "code_layer_light": "#388bfd",
    "science_layer": "#238636",
    "science_layer_light": "#3fb950",
    "accent": "#f78166",
    "accent2": "#a371f7",
    "text": "#c9d1d9",
    "text_muted": "#8b949e",
    "border": "#30363d",
    "success": "#3fb950",
    "warning": "#d29922",
    "error": "#f85149",
    "arrow": "#58a6ff",
}


# Font scaling (for diagrams)
FONT_SCALE = 1.35


def fs(x: float) -> float:
    """Scale font sizes consistently across diagrams."""
    try:
        return max(8.0, float(x) * float(FONT_SCALE))
    except Exception:
        return float(x)


def create_rounded_box(
    ax,
    x,
    y,
    width,
    height,
    color,
    text,
    fontsize=fs(10),
    text_color="white",
    alpha=0.9,
    border_color=None,
):
    """Create a rounded rectangle box with centered text."""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor=border_color or color,
        alpha=alpha,
        linewidth=2,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
        zorder=3,
    )
    return box


def create_arrow(
    ax,
    start,
    end,
    color=None,
    style="->",
    connectionstyle="arc3,rad=0.1",
    linewidth=2,
    **kwargs,
):
    """Create a curved arrow between two points."""
    if color is None:
        color = COLORS["arrow"]
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=connectionstyle,
        arrowstyle=style,
        mutation_scale=15,
        color=color,
        linewidth=linewidth,
        zorder=1,
        **kwargs,
    )
    ax.add_patch(arrow)
    return arrow


def generate_main_architecture():
    """Generate the main SuperAgent architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Title
    ax.text(
        8,
        11.5,
        "SuperAgent Architecture",
        ha="center",
        va="center",
        fontsize=24,
        color=COLORS["text"],
        fontweight="bold",
    )
    ax.text(
        8,
        10.9,
        "Dual-Layer AI System for Automated Scientific Discovery",
        ha="center",
        va="center",
        fontsize=12,
        color=COLORS["text_muted"],
    )

    # ========== CODE LAYER ==========
    # Background box for Code Layer
    code_bg = FancyBboxPatch(
        (0.5, 5.5),
        15,
        4.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["code_layer"],
        alpha=0.8,
        linewidth=3,
        zorder=0,
    )
    ax.add_patch(code_bg)
    ax.text(
        1.2,
        9.6,
        "CODE LAYER (Engineering)",
        fontsize=14,
        color=COLORS["code_layer_light"],
        fontweight="bold",
    )

    # Code Layer Agents
    code_agents = [
        (2.5, 7.5, "Architect\nSystem Design"),
        (6, 7.5, "Manager\nTask Scheduling"),
        (9.5, 7.5, "Workers\nImplementation"),
        (13, 7.5, "Integrator\nVerification"),
    ]

    for x, y, text in code_agents:
        create_rounded_box(
            ax,
            x,
            y,
            2.5,
            1.5,
            COLORS["code_layer"],
            text,
            fontsize=9,
            border_color=COLORS["code_layer_light"],
        )

    # Code Layer Arrows
    create_arrow(
        ax,
        (3.8, 7.5),
        (4.7, 7.5),
        COLORS["code_layer_light"],
        connectionstyle="arc3,rad=0",
    )
    create_arrow(
        ax,
        (7.3, 7.5),
        (8.2, 7.5),
        COLORS["code_layer_light"],
        connectionstyle="arc3,rad=0",
    )
    create_arrow(
        ax,
        (10.8, 7.5),
        (11.7, 7.5),
        COLORS["code_layer_light"],
        connectionstyle="arc3,rad=0",
    )

    # Blueprint, FileSpec, Code labels
    ax.text(4.25, 8.2, "Blueprint", fontsize=8, color=COLORS["text_muted"], ha="center")
    ax.text(7.75, 8.2, "FileSpec", fontsize=8, color=COLORS["text_muted"], ha="center")
    ax.text(11.25, 8.2, "Code", fontsize=8, color=COLORS["text_muted"], ha="center")

    # ========== SCIENCE LAYER ==========
    # Background box for Science Layer
    science_bg = FancyBboxPatch(
        (0.5, 0.5),
        15,
        4.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["science_layer"],
        alpha=0.8,
        linewidth=3,
        zorder=0,
    )
    ax.add_patch(science_bg)
    ax.text(
        1.2,
        4.6,
        "SCIENCE LAYER (Experimentation)",
        fontsize=14,
        color=COLORS["science_layer_light"],
        fontweight="bold",
    )

    # Science Layer Agents
    science_agents = [
        (2.5, 2.5, "Architect\nExperiment Design"),
        (6, 2.5, "Manager\nExecution Schedule"),
        (9.5, 2.5, "Workers\nRun Experiments"),
        (13, 2.5, "Integrator\nResult Analysis"),
    ]

    for x, y, text in science_agents:
        create_rounded_box(
            ax,
            x,
            y,
            2.5,
            1.5,
            COLORS["science_layer"],
            text,
            fontsize=9,
            border_color=COLORS["science_layer_light"],
        )

    # Science Layer Arrows
    create_arrow(
        ax,
        (3.8, 2.5),
        (4.7, 2.5),
        COLORS["science_layer_light"],
        connectionstyle="arc3,rad=0",
    )
    create_arrow(
        ax,
        (7.3, 2.5),
        (8.2, 2.5),
        COLORS["science_layer_light"],
        connectionstyle="arc3,rad=0",
    )
    create_arrow(
        ax,
        (10.8, 2.5),
        (11.7, 2.5),
        COLORS["science_layer_light"],
        connectionstyle="arc3,rad=0",
    )

    # Plan, Task, Results labels
    ax.text(4.25, 3.2, "Plan", fontsize=8, color=COLORS["text_muted"], ha="center")
    ax.text(7.75, 3.2, "Task", fontsize=8, color=COLORS["text_muted"], ha="center")
    ax.text(11.25, 3.2, "Results", fontsize=8, color=COLORS["text_muted"], ha="center")

    # ========== PROTOCOLS ==========
    # CHP Arrow (Code -> Science)
    create_arrow(
        ax,
        (13, 6.7),
        (2.5, 3.3),
        COLORS["accent"],
        connectionstyle="arc3,rad=-0.3",
        linewidth=3,
    )
    ax.text(
        7,
        5.2,
        "CHP\nCodeManifest",
        fontsize=10,
        color=COLORS["accent"],
        ha="center",
        fontweight="bold",
    )

    # ORP Arrow (Science -> Code)
    create_arrow(
        ax,
        (13, 3.3),
        (6, 6.7),
        COLORS["accent2"],
        connectionstyle="arc3,rad=-0.3",
        linewidth=3,
    )
    ax.text(
        11,
        5.2,
        "ORP\nOptimizationTicket",
        fontsize=10,
        color=COLORS["accent2"],
        ha="center",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "architecture.png"),
        dpi=150,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.close()
    print("[OK] Generated: architecture.png")


def generate_information_flow():
    """Generate the information flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 16), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # Title
    ax.text(
        7,
        15.5,
        "Information Flow",
        ha="center",
        va="center",
        fontsize=22,
        color=COLORS["text"],
        fontweight="bold",
    )

    # ========== INPUT ==========
    create_rounded_box(
        ax,
        7,
        14.5,
        4,
        1,
        COLORS["accent2"],
        "Research Proposal\n(idea.md)",
        fontsize=10,
    )

    # ========== PHASE 1: ENGINEERING ==========
    phase1_bg = FancyBboxPatch(
        (0.5, 10),
        13,
        3.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["code_layer"],
        alpha=0.8,
        linewidth=2,
        zorder=0,
    )
    ax.add_patch(phase1_bg)
    ax.text(
        1.2,
        13.1,
        "PHASE 1: ENGINEERING",
        fontsize=12,
        color=COLORS["code_layer_light"],
        fontweight="bold",
    )

    create_rounded_box(
        ax, 2.5, 11.5, 2.5, 1.2, COLORS["code_layer"], "Proposal\nParsed", fontsize=9
    )
    create_rounded_box(
        ax,
        7,
        11.5,
        2.5,
        1.2,
        COLORS["code_layer"],
        "Blueprint\nDAG of Files",
        fontsize=9,
    )
    create_rounded_box(
        ax, 11.5, 11.5, 2.5, 1.2, COLORS["code_layer"], "Codebase\nTested", fontsize=9
    )

    create_arrow(
        ax,
        (3.8, 11.5),
        (5.7, 11.5),
        COLORS["code_layer_light"],
        connectionstyle="arc3,rad=0",
    )
    create_arrow(
        ax,
        (8.3, 11.5),
        (10.2, 11.5),
        COLORS["code_layer_light"],
        connectionstyle="arc3,rad=0",
    )

    # ========== CHP ==========
    create_rounded_box(
        ax, 7, 9, 5, 0.8, COLORS["accent"], "CodeManifest (CHP)", fontsize=10
    )
    create_arrow(
        ax, (7, 14), (7, 12.1), COLORS["text_muted"], connectionstyle="arc3,rad=0"
    )
    create_arrow(
        ax, (11.5, 10.9), (7, 9.4), COLORS["accent"], connectionstyle="arc3,rad=0.2"
    )

    # ========== PHASE 2: EXPERIMENTATION ==========
    phase2_bg = FancyBboxPatch(
        (0.5, 5),
        13,
        3.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["science_layer"],
        alpha=0.8,
        linewidth=2,
        zorder=0,
    )
    ax.add_patch(phase2_bg)
    ax.text(
        1.2,
        8.1,
        "PHASE 2: EXPERIMENTATION",
        fontsize=12,
        color=COLORS["science_layer_light"],
        fontweight="bold",
    )

    create_rounded_box(
        ax, 2.5, 6.5, 2.5, 1.2, COLORS["science_layer"], "Plan\nExperiments", fontsize=9
    )
    create_rounded_box(
        ax, 7, 6.5, 2.5, 1.2, COLORS["science_layer"], "Results\nMetrics", fontsize=9
    )
    create_rounded_box(
        ax,
        11.5,
        6.5,
        2.5,
        1.2,
        COLORS["science_layer"],
        "Analysis\nFindings",
        fontsize=9,
    )

    create_arrow(
        ax,
        (3.8, 6.5),
        (5.7, 6.5),
        COLORS["science_layer_light"],
        connectionstyle="arc3,rad=0",
    )
    create_arrow(
        ax,
        (8.3, 6.5),
        (10.2, 6.5),
        COLORS["science_layer_light"],
        connectionstyle="arc3,rad=0",
    )
    create_arrow(
        ax, (7, 8.6), (2.5, 7.1), COLORS["accent"], connectionstyle="arc3,rad=0.2"
    )

    # ========== PHASE 3: OPTIMIZATION ==========
    phase3_bg = FancyBboxPatch(
        (0.5, 1),
        13,
        3.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["warning"],
        alpha=0.8,
        linewidth=2,
        zorder=0,
    )
    ax.add_patch(phase3_bg)
    ax.text(
        1.2,
        4.1,
        "PHASE 3: OPTIMIZATION LOOP",
        fontsize=12,
        color=COLORS["warning"],
        fontweight="bold",
    )

    # Decision diamond
    diamond = Polygon(
        [(7, 3.2), (8, 2.5), (7, 1.8), (6, 2.5)],
        facecolor=COLORS["warning"],
        edgecolor=COLORS["warning"],
        linewidth=2,
        zorder=2,
    )
    ax.add_patch(diamond)
    ax.text(
        7,
        2.5,
        "Goal?",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
        zorder=3,
    )

    create_rounded_box(
        ax, 3, 2.5, 2.5, 1, COLORS["error"], "Tickets\nOptimization", fontsize=9
    )
    create_rounded_box(
        ax, 11.5, 2.5, 2.5, 1, COLORS["success"], "SUCCESS\nValidated", fontsize=9
    )

    create_arrow(
        ax, (11.5, 5.9), (7, 3.2), COLORS["warning"], connectionstyle="arc3,rad=0"
    )
    create_arrow(
        ax, (6, 2.5), (4.3, 2.5), COLORS["error"], connectionstyle="arc3,rad=0"
    )
    create_arrow(
        ax, (8, 2.5), (10.2, 2.5), COLORS["success"], connectionstyle="arc3,rad=0"
    )
    ax.text(5.2, 2.9, "No", fontsize=9, color=COLORS["error"])
    ax.text(9, 2.9, "Yes", fontsize=9, color=COLORS["success"])

    # Loop back arrow
    create_arrow(
        ax,
        (3, 3),
        (3, 10.3),
        COLORS["error"],
        connectionstyle="arc3,rad=-0.3",
        linewidth=2,
    )
    create_arrow(
        ax,
        (3, 10.3),
        (5.5, 11.5),
        COLORS["error"],
        connectionstyle="arc3,rad=-0.1",
        linewidth=2,
    )
    ax.text(
        1.5,
        7,
        "Fix &\nRetry",
        fontsize=9,
        color=COLORS["error"],
        ha="center",
        rotation=90,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "information_flow.png"),
        dpi=150,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.close()
    print("[OK] Generated: information_flow.png")


def generate_dag_scheduling():
    """Generate the DAG scheduling diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        7,
        9.5,
        "DAG-Based Parallel Execution",
        ha="center",
        va="center",
        fontsize=20,
        color=COLORS["text"],
        fontweight="bold",
    )

    # Wave backgrounds
    wave_colors = ["#1e3a5f", "#1e4d3d", "#3d2e1e", "#3d1e3d"]
    wave_labels = [
        "Wave 1 (Parallel)",
        "Wave 2 (Parallel)",
        "Wave 3",
        "Wave 4 (Parallel)",
    ]
    wave_y = [8, 6, 4, 2]

    for i, (y, color, label) in enumerate(zip(wave_y, wave_colors, wave_labels)):
        bg = FancyBboxPatch(
            (0.5, y - 0.8),
            13,
            1.6,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=color,
            edgecolor=COLORS["border"],
            alpha=0.6,
            linewidth=1,
            zorder=0,
        )
        ax.add_patch(bg)
        ax.text(1.5, y, label, fontsize=10, color=COLORS["text_muted"], va="center")

    # Wave 1 nodes
    nodes_w1 = [
        (4, 8, "utils.py", COLORS["code_layer"]),
        (7, 8, "config.py", COLORS["code_layer"]),
        (10, 8, "types.py", COLORS["code_layer"]),
    ]

    # Wave 2 nodes
    nodes_w2 = [
        (5, 6, "data_loader.py", COLORS["science_layer"]),
        (9, 6, "model.py", COLORS["science_layer"]),
    ]

    # Wave 3 nodes
    nodes_w3 = [
        (7, 4, "train.py", COLORS["accent"]),
    ]

    # Wave 4 nodes
    nodes_w4 = [
        (5, 2, "eval.py", COLORS["accent2"]),
        (9, 2, "main.py", COLORS["accent2"]),
    ]

    # Draw all nodes
    all_nodes = nodes_w1 + nodes_w2 + nodes_w3 + nodes_w4
    for x, y, text, color in all_nodes:
        create_rounded_box(ax, x, y, 2.2, 0.8, color, text, fontsize=9)

    # Draw dependencies
    deps = [
        # Wave 1 -> Wave 2
        ((4, 7.6), (5, 6.4)),
        ((7, 7.6), (5, 6.4)),
        ((7, 7.6), (9, 6.4)),
        ((10, 7.6), (9, 6.4)),
        # Wave 2 -> Wave 3
        ((5, 5.6), (7, 4.4)),
        ((9, 5.6), (7, 4.4)),
        # Wave 3 -> Wave 4
        ((7, 3.6), (5, 2.4)),
        ((7, 3.6), (9, 2.4)),
    ]

    for start, end in deps:
        create_arrow(
            ax, start, end, COLORS["arrow"], connectionstyle="arc3,rad=0", linewidth=1.5
        )

    # Legend
    ax.text(12.5, 8.5, "Parallel", fontsize=9, color=COLORS["text_muted"], ha="center")
    ax.text(12.5, 8.1, "execution", fontsize=9, color=COLORS["text_muted"], ha="center")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "dag_scheduling.png"),
        dpi=150,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.close()
    print("[OK] Generated: dag_scheduling.png")


def generate_cache_workflow():
    """Generate the cache workflow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Title
    ax.text(
        8,
        11.5,
        "Cache & State Management Workflow",
        ha="center",
        va="center",
        fontsize=20,
        color=COLORS["text"],
        fontweight="bold",
    )

    # ========== LEFT: Cache System ==========
    cache_bg = FancyBboxPatch(
        (0.5, 3),
        7,
        7.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["accent"],
        alpha=0.8,
        linewidth=2,
        zorder=0,
    )
    ax.add_patch(cache_bg)
    ax.text(
        1,
        10.1,
        "Blueprint Cache",
        fontsize=14,
        color=COLORS["accent"],
        fontweight="bold",
    )

    # Cache flow
    create_rounded_box(ax, 4, 9, 3, 0.8, COLORS["code_layer"], "Proposal", fontsize=10)
    create_rounded_box(
        ax, 4, 7.5, 3, 0.8, COLORS["warning"], "hash_proposal()", fontsize=10
    )
    create_rounded_box(ax, 4, 6, 3, 0.8, COLORS["accent2"], "MD5 Hash Key", fontsize=10)

    # Decision
    diamond = Polygon(
        [(4, 4.7), (5.2, 4), (4, 3.3), (2.8, 4)],
        facecolor=COLORS["warning"],
        edgecolor=COLORS["warning"],
        linewidth=2,
        zorder=2,
    )
    ax.add_patch(diamond)
    ax.text(
        4,
        4,
        "Cache\nHit?",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        fontweight="bold",
        zorder=3,
    )

    # Cache hit/miss paths
    create_rounded_box(ax, 1.8, 4, 1.5, 0.7, COLORS["success"], "Load", fontsize=9)
    create_rounded_box(ax, 6.2, 4, 1.5, 0.7, COLORS["error"], "Generate", fontsize=9)

    # Arrows
    create_arrow(ax, (4, 8.6), (4, 7.9), COLORS["arrow"], connectionstyle="arc3,rad=0")
    create_arrow(ax, (4, 7.1), (4, 6.4), COLORS["arrow"], connectionstyle="arc3,rad=0")
    create_arrow(ax, (4, 5.6), (4, 4.7), COLORS["arrow"], connectionstyle="arc3,rad=0")
    create_arrow(
        ax, (2.8, 4), (2.55, 4), COLORS["success"], connectionstyle="arc3,rad=0"
    )
    create_arrow(ax, (5.2, 4), (5.45, 4), COLORS["error"], connectionstyle="arc3,rad=0")

    ax.text(2.3, 4.4, "Yes", fontsize=8, color=COLORS["success"])
    ax.text(5.5, 4.4, "No", fontsize=8, color=COLORS["error"])

    # Cache structure
    ax.text(1.2, 3.2, "cached/blueprints/", fontsize=9, color=COLORS["text_muted"])
    ax.text(
        1.4,
        2.8,
        "+-- {hash1}.json",
        fontsize=8,
        color=COLORS["text_muted"],
        family="monospace",
    )
    ax.text(
        1.4,
        2.5,
        "+-- {hash2}.json",
        fontsize=8,
        color=COLORS["text_muted"],
        family="monospace",
    )
    ax.text(
        1.4, 2.2, "+-- ...", fontsize=8, color=COLORS["text_muted"], family="monospace"
    )

    # ========== RIGHT: State System ==========
    state_bg = FancyBboxPatch(
        (8.5, 3),
        7,
        7.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["science_layer"],
        alpha=0.8,
        linewidth=2,
        zorder=0,
    )
    ax.add_patch(state_bg)
    ax.text(
        9,
        10.1,
        "Execution State (StateManager)",
        fontsize=14,
        color=COLORS["science_layer_light"],
        fontweight="bold",
    )

    # State phases
    phases = [
        (12, 9, "INIT", COLORS["text_muted"]),
        (12, 8, "PLANNING", COLORS["code_layer"]),
        (12, 7, "EXECUTION", COLORS["science_layer"]),
        (12, 6, "VERIFICATION", COLORS["warning"]),
        (12, 5, "REFINEMENT", COLORS["accent"]),
        (12, 4, "COMPLETED", COLORS["success"]),
    ]

    for x, y, text, color in phases:
        create_rounded_box(ax, x, y, 2.5, 0.6, color, text, fontsize=9)

    # Phase arrows
    for i in range(len(phases) - 1):
        create_arrow(
            ax,
            (12, phases[i][1] - 0.3),
            (12, phases[i + 1][1] + 0.3),
            COLORS["arrow"],
            connectionstyle="arc3,rad=0",
            linewidth=1.5,
        )

    # Refinement loop
    create_arrow(
        ax, (13.3, 5), (13.8, 5), COLORS["accent"], connectionstyle="arc3,rad=0"
    )
    create_arrow(
        ax, (13.8, 5), (13.8, 7), COLORS["accent"], connectionstyle="arc3,rad=0"
    )
    create_arrow(
        ax, (13.8, 7), (13.3, 7), COLORS["accent"], connectionstyle="arc3,rad=0"
    )
    ax.text(
        14.2, 6, "Loop", fontsize=8, color=COLORS["accent"], rotation=90, va="center"
    )

    # State structure
    ax.text(9.2, 3.2, "cached/execution/", fontsize=9, color=COLORS["text_muted"])
    ax.text(
        9.4,
        2.8,
        "+-- state.json (head)",
        fontsize=8,
        color=COLORS["text_muted"],
        family="monospace",
    )
    ax.text(
        9.4,
        2.5,
        "+-- steps/",
        fontsize=8,
        color=COLORS["text_muted"],
        family="monospace",
    )
    ax.text(
        9.7,
        2.2,
        "+-- step_0000.json",
        fontsize=8,
        color=COLORS["text_muted"],
        family="monospace",
    )
    ax.text(
        9.7,
        1.9,
        "+-- step_0001.json",
        fontsize=8,
        color=COLORS["text_muted"],
        family="monospace",
    )
    ax.text(
        9.7, 1.6, "+-- ...", fontsize=8, color=COLORS["text_muted"], family="monospace"
    )

    # Connection
    ax.text(
        8,
        1,
        "Resume: --resume flag loads state.json -> restores step_index -> continues execution",
        fontsize=10,
        color=COLORS["text"],
        ha="center",
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "cache_workflow.png"),
        dpi=150,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.close()
    print("[OK] Generated: cache_workflow.png")


def generate_validation_pipeline():
    """Generate the validation pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        7,
        9.5,
        "Code Integrator Validation Pipeline",
        ha="center",
        va="center",
        fontsize=20,
        color=COLORS["text"],
        fontweight="bold",
    )

    # Phases
    phases = [
        (2.5, 8, "[1] Syntax Check", "ast.parse() all .py files", COLORS["code_layer"]),
        (
            2.5,
            6.5,
            "[2] Import Validation",
            "Resolve all imports",
            COLORS["code_layer"],
        ),
        (2.5, 5, "[3] Entry Point Test", "Verify main entry", COLORS["code_layer"]),
        (2.5, 3.5, "[4] Compilation", "compileall project", COLORS["code_layer"]),
        (2.5, 2, "[5] Unit Tests", "pytest tests/", COLORS["code_layer"]),
    ]

    for x, y, title, desc, color in phases:
        create_rounded_box(ax, x, y, 3.5, 1, color, title, fontsize=10)
        ax.text(4.5, y + 0.1, desc, fontsize=9, color=COLORS["text_muted"], va="center")

    # Arrows between phases
    for i in range(len(phases) - 1):
        create_arrow(
            ax,
            (2.5, phases[i][1] - 0.5),
            (2.5, phases[i + 1][1] + 0.5),
            COLORS["arrow"],
            connectionstyle="arc3,rad=0",
        )

    # Pass/Fail branches
    for x, y, _, _, _ in phases:
        # Pass arrow to right
        create_arrow(
            ax, (4.3, y), (6.5, y), COLORS["success"], connectionstyle="arc3,rad=0"
        )
        ax.text(5.4, y + 0.25, "PASS", fontsize=8, color=COLORS["success"])

        # Fail arrow further right
        create_arrow(
            ax,
            (4.3, y - 0.15),
            (9.5, y - 0.15),
            COLORS["error"],
            connectionstyle="arc3,rad=0",
        )
        ax.text(6.5, y - 0.4, "FAIL", fontsize=8, color=COLORS["error"])

    # Continue boxes
    for i, (x, y, _, _, _) in enumerate(phases[:-1]):
        create_rounded_box(
            ax, 7.5, y, 2, 0.7, COLORS["success"], "Continue", fontsize=9
        )

    # Final output
    create_rounded_box(ax, 7.5, 2, 2, 0.7, COLORS["success"], "SUCCESS", fontsize=9)

    # Error output box
    create_rounded_box(
        ax,
        11.5,
        5,
        2.5,
        3.5,
        COLORS["error"],
        "FixBlueprint\n\n- file_path\n- issue_type\n- message\n- suggestion",
        fontsize=9,
    )

    # Error arrows to FixBlueprint
    for y in [8, 6.5, 5, 3.5, 2]:
        create_arrow(
            ax,
            (9.5, y - 0.15),
            (10.2, 5),
            COLORS["error"],
            connectionstyle="arc3,rad=0.15",
            linewidth=1,
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "validation_pipeline.png"),
        dpi=150,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.close()
    print("[OK] Generated: validation_pipeline.png")


def generate_experiment_agent_framework():
    """Generate the ExperimentAgent framework diagram (current implementation)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def _anchor(box, side: str):
        x0 = float(box.get_x())
        y0 = float(box.get_y())
        w = float(box.get_width())
        h = float(box.get_height())
        side = (side or "").strip().lower()
        if side == "left":
            return (x0, y0 + h / 2)
        if side == "right":
            return (x0 + w, y0 + h / 2)
        if side == "top":
            return (x0 + w / 2, y0 + h)
        if side == "bottom":
            return (x0 + w / 2, y0)
        return (x0 + w / 2, y0 + h / 2)

    def _offset(pt, dx: float = 0.0, dy: float = 0.0):
        return (float(pt[0]) + float(dx), float(pt[1]) + float(dy))

    # Code Layer (make boxes larger and centered)
    code_bg = FancyBboxPatch(
        (0.6, 7.3),
        16.8,
        3.6,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["code_layer"],
        alpha=0.85,
        linewidth=2.5,
        zorder=0,
    )
    ax.add_patch(code_bg)
    ax.text(
        1.0,
        10.5,
        "CODE LAYER",
        fontsize=12,
        color=COLORS["code_layer_light"],
        fontweight="bold",
    )

    code_arch_box = create_rounded_box(
        ax,
        4.5,
        9.25,
        4.0,
        1.2,
        COLORS["code_layer"],
        "CodeArchitect\nBlueprint",
        fontsize=10,
    )
    code_mgr_box = create_rounded_box(
        ax,
        9.0,
        9.25,
        4.0,
        1.2,
        COLORS["code_layer"],
        "CodeManager\nDAG + Workers",
        fontsize=10,
    )
    code_int_box = create_rounded_box(
        ax,
        13.5,
        9.25,
        4.0,
        1.2,
        COLORS["code_layer"],
        "CodeIntegrator\nVerify + FixBlueprint",
        fontsize=10,
    )

    create_arrow(
        ax,
        _anchor(code_arch_box, "right"),
        _anchor(code_mgr_box, "left"),
        COLORS["code_layer_light"],
        connectionstyle="arc3,rad=0",
    )
    create_arrow(
        ax,
        _anchor(code_mgr_box, "right"),
        _anchor(code_int_box, "left"),
        COLORS["code_layer_light"],
        connectionstyle="arc3,rad=0",
    )
    create_arrow(
        ax,
        _anchor(code_int_box, "bottom"),
        _anchor(code_mgr_box, "bottom"),
        COLORS["warning"],
        connectionstyle="arc3,rad=-0.35",
        linewidth=2,
    )
    _code_fb_mid_x = (
        _anchor(code_int_box, "bottom")[0] + _anchor(code_mgr_box, "bottom")[0]
    ) / 2
    _code_fb_mid_y = (
        min(_anchor(code_int_box, "bottom")[1], _anchor(code_mgr_box, "bottom")[1])
        - 0.55
    )
    ax.text(
        _code_fb_mid_x,
        _code_fb_mid_y,
        "FixBlueprint",
        fontsize=9,
        color=COLORS["warning"],
        ha="center",
        va="top",
        fontweight="bold",
        bbox={
            "facecolor": COLORS["bg_light"],
            "edgecolor": "none",
            "alpha": 0.85,
            "boxstyle": "round,pad=0.25,rounding_size=0.15",
        },
    )

    # Science Layer (make boxes larger and centered)
    sci_bg = FancyBboxPatch(
        (0.6, 3.6),
        16.8,
        3.2,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["science_layer"],
        alpha=0.85,
        linewidth=2.5,
        zorder=0,
    )
    ax.add_patch(sci_bg)
    ax.text(
        1.0,
        6.55,
        "SCIENCE LAYER",
        fontsize=12,
        color=COLORS["science_layer_light"],
        fontweight="bold",
    )

    sci_arch_box = create_rounded_box(
        ax,
        4.5,
        5.25,
        4.0,
        1.2,
        COLORS["science_layer"],
        "ExpArchitect\nExperimentPlan",
        fontsize=10,
    )
    sci_mgr_box = create_rounded_box(
        ax,
        9.0,
        5.25,
        4.0,
        1.2,
        COLORS["science_layer"],
        "ExpManager\nDAG + ExpWorker",
        fontsize=10,
    )
    sci_int_box = create_rounded_box(
        ax,
        13.5,
        5.25,
        4.0,
        1.2,
        COLORS["science_layer"],
        "ExpIntegrator\nScienceAnalysis",
        fontsize=10,
    )
    create_arrow(
        ax,
        _anchor(sci_arch_box, "right"),
        _anchor(sci_mgr_box, "left"),
        COLORS["science_layer_light"],
        connectionstyle="arc3,rad=0",
    )
    create_arrow(
        ax,
        _anchor(sci_mgr_box, "right"),
        _anchor(sci_int_box, "left"),
        COLORS["science_layer_light"],
        connectionstyle="arc3,rad=0",
    )
    # NOTE: ActionPlan box intentionally omitted.
    create_arrow(
        ax,
        _anchor(sci_int_box, "bottom"),
        _anchor(sci_mgr_box, "bottom"),
        COLORS["warning"],
        connectionstyle="arc3,rad=-0.35",
        linewidth=2,
    )
    _sci_fb_mid_x = (
        _anchor(sci_int_box, "bottom")[0] + _anchor(sci_mgr_box, "bottom")[0]
    ) / 2
    _sci_fb_mid_y = (
        min(_anchor(sci_int_box, "bottom")[1], _anchor(sci_mgr_box, "bottom")[1]) - 0.55
    )
    ax.text(
        _sci_fb_mid_x,
        _sci_fb_mid_y,
        "Corrected ExperimentPlan",
        fontsize=9,
        color=COLORS["warning"],
        ha="center",
        va="top",
        fontweight="bold",
        bbox={
            "facecolor": COLORS["bg_light"],
            "edgecolor": "none",
            "alpha": 0.85,
            "boxstyle": "round,pad=0.25,rounding_size=0.15",
        },
    )

    # Shared Infrastructure (slightly larger boxes, centered)
    shared_bg = FancyBboxPatch(
        (0.6, 0.6),
        16.8,
        2.7,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["border"],
        alpha=0.75,
        linewidth=1.5,
        zorder=0,
    )
    ax.add_patch(shared_bg)
    ax.text(1.0, 2.8, "SHARED", fontsize=12, color=COLORS["text"], fontweight="bold")

    create_rounded_box(
        ax,
        3.8,
        1.85,
        4.0,
        1.2,
        COLORS["accent"],
        "Cache\ncached/blueprints/",
        fontsize=10,
    )
    create_rounded_box(
        ax,
        8.2,
        1.85,
        4.4,
        1.2,
        COLORS["accent2"],
        "StateManager\ncached/execution/steps/",
        fontsize=10,
    )
    create_rounded_box(
        ax,
        12.9,
        1.85,
        4.4,
        1.2,
        COLORS["text_muted"],
        "SecurityContext + Tools\nbash / grep / file_viewer",
        fontsize=10,
    )

    # Cross-layer communication paths (draw arrows only; do not label as CHP/ORP)
    create_arrow(
        ax,
        _anchor(code_int_box, "bottom"),
        _anchor(sci_arch_box, "top"),
        COLORS["accent"],
        connectionstyle="arc3,rad=-0.25",
        linewidth=2.5,
    )
    create_arrow(
        ax,
        _anchor(sci_int_box, "top"),
        _anchor(code_mgr_box, "bottom"),
        COLORS["accent2"],
        connectionstyle="arc3,rad=0.2",
        linewidth=2.5,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "experiment_agent_framework.png"),
        dpi=160,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close()
    print("[OK] Generated: experiment_agent_framework.png")


def generate_experiment_agent_main_flow():
    """Generate the ExperimentAgent end-to-end main flow diagram (main.py + key branches)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 13), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13)
    ax.axis("off")

    # Title
    ax.text(
        9,
        12.5,
        "ExperimentAgent Main Flow",
        ha="center",
        va="center",
        fontsize=24,
        color=COLORS["text"],
        fontweight="bold",
    )
    ax.text(
        9,
        11.9,
        "main.py orchestration: engineering -> science -> optimization loop",
        ha="center",
        va="center",
        fontsize=12,
        color=COLORS["text_muted"],
    )

    # Start / Args
    create_rounded_box(
        ax,
        3.0,
        10.8,
        4.8,
        0.9,
        COLORS["accent2"],
        "Parse args\n--experiment/--resume/--fresh",
        fontsize=10,
    )
    create_rounded_box(
        ax,
        3.0,
        9.6,
        4.8,
        0.9,
        COLORS["accent"],
        "ensure_experiment_dirs()\nCache.initialize()",
        fontsize=10,
    )
    create_arrow(
        ax, (3.0, 10.35), (3.0, 10.05), COLORS["arrow"], connectionstyle="arc3,rad=0"
    )

    # Resume decision (diamond)
    resume_diamond = Polygon(
        [(3.0, 8.7), (4.1, 8.0), (3.0, 7.3), (1.9, 8.0)],
        facecolor=COLORS["warning"],
        edgecolor=COLORS["warning"],
        linewidth=2,
        zorder=2,
    )
    ax.add_patch(resume_diamond)
    ax.text(
        3.0,
        8.0,
        "Resume\nScience?",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
        zorder=3,
    )
    create_arrow(
        ax, (3.0, 9.15), (3.0, 8.7), COLORS["arrow"], connectionstyle="arc3,rad=0"
    )

    # Resume branch
    create_rounded_box(
        ax,
        7.3,
        8.9,
        5.8,
        1.1,
        COLORS["science_layer"],
        "Resume branch\nLoad proposal + recover Blueprint\nBuild CodeManifest.from_blueprint()",
        fontsize=9,
    )
    create_arrow(
        ax, (4.1, 8.0), (5.4, 8.55), COLORS["success"], connectionstyle="arc3,rad=0.12"
    )
    ax.text(4.9, 8.65, "Yes", fontsize=9, color=COLORS["success"])

    # Fresh / normal branch
    create_rounded_box(
        ax,
        3.0,
        6.2,
        5.6,
        1.2,
        COLORS["code_layer"],
        "Engineering Layer\nrun_code_generation_loop()\nArchitect -> Manager -> Integrator(+fix loop)\n=> CodeManifest",
        fontsize=9,
    )
    create_arrow(
        ax, (3.0, 7.3), (3.0, 6.8), COLORS["error"], connectionstyle="arc3,rad=0"
    )
    ax.text(2.2, 7.55, "No", fontsize=9, color=COLORS["error"])

    # Merge: CodeManifest
    create_rounded_box(
        ax,
        9.7,
        6.2,
        4.8,
        0.9,
        COLORS["accent"],
        "CodeManifest\n(CHP payload)",
        fontsize=10,
    )
    create_arrow(
        ax, (6.0, 6.2), (7.3, 6.2), COLORS["arrow"], connectionstyle="arc3,rad=0"
    )
    create_arrow(
        ax, (7.3, 8.35), (9.7, 6.65), COLORS["arrow"], connectionstyle="arc3,rad=-0.2"
    )

    # Main loop box
    loop_bg = FancyBboxPatch(
        (0.8, 0.8),
        16.4,
        4.7,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["border"],
        alpha=0.85,
        linewidth=1.8,
        zorder=0,
    )
    ax.add_patch(loop_bg)
    ax.text(
        1.2,
        5.2,
        "MAIN LOOP (for loop_num in [start_loop..max_loops])",
        fontsize=11,
        color=COLORS["text"],
        fontweight="bold",
    )

    # Science stage within loop
    create_rounded_box(
        ax,
        4.2,
        4.1,
        6.8,
        1.2,
        COLORS["science_layer"],
        "Science Layer\nrun_science_cycle():\nDesign -> Execute(DAG) -> Analyze\nmay propose next_experiments",
        fontsize=9,
    )
    create_arrow(
        ax, (9.7, 5.75), (6.2, 4.75), COLORS["arrow"], connectionstyle="arc3,rad=0.15"
    )

    # Decision: tickets?
    tickets_diamond = Polygon(
        [(12.9, 4.1), (14.2, 3.3), (12.9, 2.5), (11.6, 3.3)],
        facecolor=COLORS["warning"],
        edgecolor=COLORS["warning"],
        linewidth=2,
        zorder=2,
    )
    ax.add_patch(tickets_diamond)
    ax.text(
        12.9,
        3.3,
        "Tickets?",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
        zorder=3,
    )
    create_arrow(
        ax, (7.6, 4.1), (11.6, 3.3), COLORS["arrow"], connectionstyle="arc3,rad=0"
    )

    # No tickets -> success
    create_rounded_box(
        ax, 15.9, 4.3, 2.6, 0.9, COLORS["success"], "DONE\nGoal achieved", fontsize=9
    )
    create_arrow(
        ax, (14.2, 3.3), (14.9, 4.1), COLORS["success"], connectionstyle="arc3,rad=0.2"
    )
    ax.text(14.6, 3.75, "No", fontsize=9, color=COLORS["success"])

    # Tickets -> optimization
    create_rounded_box(
        ax,
        12.9,
        1.7,
        6.6,
        1.2,
        COLORS["code_layer"],
        "Optimization Loop (ORP)\nrun_optimization(): CodeManager.fix_files()\n=> update code, then next loop",
        fontsize=9,
    )
    create_arrow(
        ax, (12.9, 2.5), (12.9, 2.25), COLORS["error"], connectionstyle="arc3,rad=0"
    )
    ax.text(13.35, 2.7, "Yes", fontsize=9, color=COLORS["error"])

    # Loop-back arrow
    create_arrow(
        ax,
        (10.2, 1.7),
        (4.2, 3.5),
        COLORS["error"],
        connectionstyle="arc3,rad=0.25",
        linewidth=2,
    )
    ax.text(
        7.4,
        1.2,
        "apply fixes -> re-run science",
        fontsize=9,
        color=COLORS["text_muted"],
        ha="center",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "experiment_agent_main_flow.png"),
        dpi=160,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close()
    print("[OK] Generated: experiment_agent_main_flow.png")


def generate_code_architect_workflow():
    """Generate a Code Architect workflow diagram (Inputs → Tool-aided design → Blueprint)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(
        9,
        11.4,
        "Code Architect",
        ha="center",
        va="center",
        fontsize=fs(22),
        color=COLORS["text"],
        fontweight="bold",
    )
    ax.text(
        9,
        10.85,
        "System design with constrained tools (no implementation)",
        ha="center",
        va="center",
        fontsize=fs(12),
        color=COLORS["text_muted"],
    )

    # Left: Inputs board
    left = FancyBboxPatch(
        (0.6, 6.4),
        6.2,
        4.2,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["accent2"],
        alpha=0.9,
        linewidth=2,
    )
    ax.add_patch(left)
    ax.text(
        1.0, 10.2, "Inputs", fontsize=fs(14), color=COLORS["accent2"], fontweight="bold"
    )

    create_rounded_box(
        ax,
        3.7,
        9.35,
        5.5,
        0.9,
        COLORS["accent2"],
        "idea.md / idea.json\nResearch proposal (goal, method, constraints)",
        fontsize=fs(10),
    )
    create_rounded_box(
        ax,
        3.7,
        8.15,
        5.5,
        0.9,
        COLORS["warning"],
        "dataset_dir\nInspect file formats & splits (runtime)",
        fontsize=fs(10),
    )
    create_rounded_box(
        ax,
        3.7,
        6.95,
        5.5,
        0.9,
        COLORS["science_layer"],
        "reference repos\nPatterns, best practices, APIs",
        fontsize=fs(10),
    )

    # Center: Tool-aided exploration + synthesis
    mid = FancyBboxPatch(
        (7.2, 3.1),
        6.2,
        7.5,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["code_layer"],
        alpha=0.9,
        linewidth=2.5,
    )
    ax.add_patch(mid)
    ax.text(
        7.6,
        10.2,
        "Architect loop",
        fontsize=fs(14),
        color=COLORS["code_layer_light"],
        fontweight="bold",
    )

    # Kanban-ish columns
    col_x = [7.8, 10.3, 12.6]
    col_titles = ["Explore", "Synthesize", "Validate"]
    for x, t in zip(col_x, col_titles):
        ax.text(
            x,
            9.6,
            t,
            fontsize=fs(11),
            color=COLORS["text"],
            fontweight="bold",
            ha="center",
        )

    # Cards
    cards = [
        (7.8, 8.6, "tree/grep\nopen key files", COLORS["code_layer"]),
        (7.8, 7.5, "inspect dataset\n(ls/find)", COLORS["warning"]),
        (10.3, 8.6, "define file tree\nmodules + tests", COLORS["code_layer"]),
        (10.3, 7.5, "define interfaces\nclasses/functions", COLORS["accent"]),
        (12.6, 8.6, "DAG check\nacyclic deps", COLORS["success"]),
        (12.6, 7.5, "schema check\nPydantic strict", COLORS["success"]),
    ]
    for x, y, txt, c in cards:
        create_rounded_box(ax, x, y, 2.1, 0.95, c, txt, fontsize=fs(9))

    # Tools callout (top-right inside mid)
    tool_box = FancyBboxPatch(
        (9.0, 3.6),
        4.0,
        2.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="#1a1f2e",
        edgecolor=COLORS["accent"],
        alpha=0.95,
        linewidth=1.8,
    )
    ax.add_patch(tool_box)
    ax.text(
        9.2,
        5.8,
        "Allowed tools",
        fontsize=fs(12),
        color=COLORS["accent"],
        fontweight="bold",
    )
    ax.text(
        9.2,
        5.2,
        "- bash(command)",
        fontsize=fs(9),
        color=COLORS["text"],
        family="monospace",
    )
    ax.text(
        9.2,
        4.7,
        "- file_viewer(path, start, end)",
        fontsize=fs(9),
        color=COLORS["text"],
        family="monospace",
    )
    ax.text(
        9.2,
        4.2,
        "No write/edit in architect",
        fontsize=fs(9),
        color=COLORS["warning"],
        style="italic",
    )

    # Right: Blueprint output schema panel
    right = FancyBboxPatch(
        (13.8, 1.0),
        3.6,
        9.6,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["success"],
        alpha=0.9,
        linewidth=2,
    )
    ax.add_patch(right)
    ax.text(
        14.1,
        10.2,
        "Output",
        fontsize=fs(14),
        color=COLORS["success"],
        fontweight="bold",
    )
    ax.text(
        14.1,
        9.55,
        "Blueprint JSON",
        fontsize=fs(11),
        color=COLORS["text"],
        fontweight="bold",
    )

    schema_lines = [
        "file_tree: [paths...]",
        "files: [FileSpec...]",
        "shared_data_structures: {}",
        "entry_point: '...'",
        "handover: {entry_points...}",
        "",
        "FileSpec:",
        "- file_path, description",
        "- dependencies (DAG)",
        "- classes/functions signatures",
        "- tests at end of DAG",
    ]
    y = 9.1
    for line in schema_lines:
        if line == "":
            y -= 0.25
            continue
        color = COLORS["text"] if not line.endswith(":") else COLORS["code_layer_light"]
        ax.text(14.1, y, line, fontsize=fs(8.5), color=color, family="monospace")
        y -= 0.38

    # Flows
    create_arrow(
        ax,
        (6.8, 8.8),
        (7.2, 8.8),
        COLORS["accent2"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )
    create_arrow(
        ax,
        (6.8, 7.9),
        (7.2, 7.9),
        COLORS["warning"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )
    create_arrow(
        ax,
        (6.8, 7.0),
        (7.2, 7.0),
        COLORS["science_layer_light"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )
    create_arrow(
        ax,
        (13.4, 6.8),
        (13.8, 6.8),
        COLORS["success"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )

    # Footer note
    ax.text(
        9,
        0.4,
        "Architect outputs design only (no code): workers implement later.",
        ha="center",
        fontsize=fs(10),
        color=COLORS["text_muted"],
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "code_architect_workflow.png"),
        dpi=160,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close()
    print("[OK] Generated: code_architect_workflow.png")


def generate_code_manager_workflow():
    """Generate the Code Manager diagram (DAG scheduling + validation bottleneck + state)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 11), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.text(
        9,
        10.5,
        "Code Manager",
        ha="center",
        fontsize=fs(22),
        color=COLORS["text"],
        fontweight="bold",
    )
    ax.text(
        9,
        9.95,
        "DAG scheduling + worker pool + validation bottleneck",
        ha="center",
        fontsize=fs(12),
        color=COLORS["text_muted"],
    )

    # Top: DAG waves (like a scheduler board)
    wave_colors = ["#1e3a5f", "#214a66", "#2a3b52", "#233042"]
    wave_labels = ["Wave 1", "Wave 2", "Wave 3", "Wave 4"]
    wave_y = [8.4, 7.0, 5.6, 4.2]
    for y, c, label in zip(wave_y, wave_colors, wave_labels):
        bg = FancyBboxPatch(
            (0.7, y - 0.65),
            11.0,
            1.2,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=c,
            edgecolor=COLORS["border"],
            alpha=0.7,
            linewidth=1,
        )
        ax.add_patch(bg)
        ax.text(
            1.1,
            y,
            label,
            fontsize=fs(11),
            color=COLORS["text"],
            va="center",
            fontweight="bold",
        )

    # Nodes in each wave
    nodes = [
        (3.2, 8.4, "utils.py"),
        (5.7, 8.4, "config.py"),
        (8.2, 8.4, "types.py"),
        (4.4, 7.0, "data_loader.py"),
        (7.0, 7.0, "model.py"),
        (5.7, 5.6, "train.py"),
        (4.4, 4.2, "eval.py"),
        (7.0, 4.2, "main.py"),
    ]
    for x, y, t in nodes:
        create_rounded_box(ax, x, y, 2.0, 0.75, COLORS["code_layer"], t, fontsize=fs(9))

    # Dependencies arrows (thin)
    deps = [
        ((3.2, 8.0), (4.4, 7.4)),
        ((5.7, 8.0), (4.4, 7.4)),
        ((5.7, 8.0), (7.0, 7.4)),
        ((8.2, 8.0), (7.0, 7.4)),
        ((4.4, 6.6), (5.7, 6.0)),
        ((7.0, 6.6), (5.7, 6.0)),
        ((5.7, 5.2), (4.4, 4.6)),
        ((5.7, 5.2), (7.0, 4.6)),
    ]
    for s, e in deps:
        create_arrow(
            ax, s, e, COLORS["arrow"], connectionstyle="arc3,rad=0", linewidth=1.5
        )

    # Right: Worker pool (swimlane)
    pool = FancyBboxPatch(
        (12.2, 6.7),
        5.1,
        2.9,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["accent"],
        alpha=0.9,
        linewidth=2,
    )
    ax.add_patch(pool)
    ax.text(
        12.6,
        9.2,
        "Worker Pool",
        fontsize=fs(14),
        color=COLORS["accent"],
        fontweight="bold",
    )
    for i, y in enumerate([8.4, 7.7, 7.0]):
        create_rounded_box(
            ax,
            14.8,
            y,
            3.8,
            0.6,
            COLORS["accent"],
            f"CodeWorker #{i+1}",
            fontsize=fs(9),
        )
    # (intentionally omit the small max_parallel_workers note to keep the panel clean)

    # Validation bottleneck gate
    gate = FancyBboxPatch(
        (12.2, 4.6),
        5.1,
        1.6,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor="#2b1e1e",
        edgecolor=COLORS["error"],
        alpha=0.9,
        linewidth=2,
    )
    ax.add_patch(gate)
    ax.text(
        12.6,
        5.8,
        "Validation Bottleneck",
        fontsize=fs(13),
        color=COLORS["error"],
        fontweight="bold",
    )
    ax.text(
        12.6,
        5.2,
        "validate_code_against_spec()\nrun_linter()\nextract_interface_stub()",
        fontsize=fs(9),
        color=COLORS["text"],
        family="monospace",
    )

    # State persistence panel
    state = FancyBboxPatch(
        (12.2, 1.2),
        5.1,
        3.0,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["accent2"],
        alpha=0.9,
        linewidth=2,
    )
    ax.add_patch(state)
    ax.text(
        12.6,
        3.8,
        "State/Resume",
        fontsize=fs(14),
        color=COLORS["accent2"],
        fontweight="bold",
    )
    ax.text(
        12.6,
        3.2,
        "StateManager\nsteps/step_XXXX.json",
        fontsize=fs(9),
        color=COLORS["text"],
        family="monospace",
    )
    ax.text(
        12.6,
        2.5,
        "- task status\n- retries\n- last_error",
        fontsize=fs(9),
        color=COLORS["text_muted"],
        family="monospace",
    )

    # Flow arrows: DAG → pool → gate → outputs
    create_arrow(
        ax,
        (11.7, 7.0),
        (12.2, 7.9),
        COLORS["accent"],
        connectionstyle="arc3,rad=-0.2",
        linewidth=3,
    )
    create_arrow(
        ax,
        (14.8, 6.7),
        (14.8, 6.2),
        COLORS["error"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )
    create_arrow(
        ax,
        (14.8, 4.6),
        (14.8, 4.2),
        COLORS["accent2"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )

    # Output summary
    out = FancyBboxPatch(
        (0.7, 1.0),
        11.0,
        2.6,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["success"],
        alpha=0.85,
        linewidth=2,
    )
    ax.add_patch(out)
    ax.text(
        1.1,
        3.1,
        "Outputs",
        fontsize=fs(14),
        color=COLORS["success"],
        fontweight="bold",
    )
    ax.text(
        1.1,
        2.4,
        "- generated files written under project_root\n- interface stubs cached for dependency context\n- failures become FixBlueprint tasks",
        fontsize=fs(10),
        color=COLORS["text"],
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "code_manager_workflow.png"),
        dpi=160,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close()
    print("[OK] Generated: code_manager_workflow.png")


def generate_code_worker_workflow():
    """Generate the Code Worker workflow diagram with stubs example (Horizontal 3-Panel Layout)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 11), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Title
    ax.text(
        9,
        10.5,
        "Code Worker: Single File Implementation",
        ha="center",
        fontsize=fs(20),
        color=COLORS["text"],
        fontweight="bold",
    )

    # ========== LEFT PANEL: INPUT (FileSpec + Dependency Stubs) ==========
    left_bg = FancyBboxPatch(
        (0.3, 0.5),
        5.5,
        9.5,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["code_layer"],
        alpha=0.9,
        linewidth=2.5,
        zorder=0,
    )
    ax.add_patch(left_bg)
    ax.text(
        0.7,
        9.7,
        "INPUT",
        fontsize=fs(14),
        color=COLORS["code_layer_light"],
        fontweight="bold",
    )

    # FileSpec section
    ax.text(
        0.7,
        9.2,
        "FileSpec (Assignment):",
        fontsize=fs(10),
        color=COLORS["text"],
        fontweight="bold",
    )
    ax.text(
        0.9,
        8.8,
        "file_path: model/trainer.py",
        fontsize=fs(8),
        color=COLORS["text"],
        family="monospace",
    )
    ax.text(
        0.9,
        8.5,
        "description: Training loop",
        fontsize=fs(8),
        color=COLORS["text"],
        family="monospace",
    )
    ax.text(
        0.9,
        8.2,
        "dependencies: [",
        fontsize=fs(8),
        color=COLORS["text"],
        family="monospace",
    )
    ax.text(
        1.1,
        7.9,
        "'model/network.py',",
        fontsize=fs(7.5),
        color=COLORS["text_muted"],
        family="monospace",
    )
    ax.text(
        1.1,
        7.6,
        "'data/loader.py'",
        fontsize=fs(7.5),
        color=COLORS["text_muted"],
        family="monospace",
    )
    ax.text(0.9, 7.3, "]", fontsize=fs(8), color=COLORS["text"], family="monospace")

    # Stubs section with example code
    ax.text(
        0.7,
        6.8,
        "Dependency Stubs (Interfaces):",
        fontsize=fs(10),
        color=COLORS["accent2"],
        fontweight="bold",
    )

    stub_box = FancyBboxPatch(
        (0.6, 1.0),
        4.9,
        5.4,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#1a1f2e",
        edgecolor=COLORS["accent2"],
        alpha=0.95,
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(stub_box)

    stub_code = [
        "# model/network.py (stub)",
        "class Network:",
        "    def __init__(self, config: Dict) -> None:",
        "        ...",
        "    def forward(self, x: Tensor) -> Tensor:",
        "        ...",
        "    def backward(self, loss: Tensor) -> None:",
        "        ...",
        "",
        "# data/loader.py (stub)",
        "class DataLoader:",
        "    def __iter__(self) -> Iterator[Batch]:",
        "        ...",
        "    def __len__(self) -> int:",
        "        ...",
    ]

    y_code = 6.2
    for line in stub_code:
        if line.startswith("#"):
            ax.text(
                0.8,
                y_code,
                line,
                fontsize=fs(7.5),
                color=COLORS["success"],
                family="monospace",
                weight="bold",
            )
        elif "class" in line or "def" in line:
            ax.text(
                0.8,
                y_code,
                line,
                fontsize=fs(7),
                color=COLORS["code_layer_light"],
                family="monospace",
            )
        elif line == "":
            y_code -= 0.1
            continue
        else:
            ax.text(
                0.8,
                y_code,
                line,
                fontsize=fs(7),
                color=COLORS["text_muted"],
                family="monospace",
            )
        y_code -= 0.33

    ax.text(
        3.0,
        1.55,
        "↑ Worker only sees signatures,",
        fontsize=fs(7),
        color=COLORS["warning"],
        style="italic",
        ha="center",
    )
    ax.text(
        3.0,
        1.15,
        "NOT implementation details",
        fontsize=fs(7),
        color=COLORS["warning"],
        style="italic",
        ha="center",
    )

    # ========== CENTER PANEL: WORKER PROCESS (Circular Flow) ==========
    center_bg = FancyBboxPatch(
        (6.1, 0.5),
        5.8,
        9.5,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["code_layer"],
        alpha=0.9,
        linewidth=2.5,
        zorder=0,
    )
    ax.add_patch(center_bg)
    ax.text(
        6.5,
        9.7,
        "WORKER PROCESS",
        fontsize=fs(14),
        color=COLORS["code_layer_light"],
        fontweight="bold",
    )

    # Circular process visualization
    center_x, center_y = 9.0, 5.5
    radius = 2.2

    # Circle steps
    angles = [90, 45, 0, -45, -90, -135, 180, 135]
    step_labels = [
        "Read\\nSpec",
        "Explore\\nRefs",
        "Write\\nCode",
        "Lint",
        "Fix\\nErrors",
        "Validate",
        "Submit",
        "Loop",
    ]
    step_colors = [COLORS["code_layer"]] * 6 + [COLORS["success"], COLORS["warning"]]

    for i, (angle, label, color) in enumerate(zip(angles, step_labels, step_colors)):
        rad = np.radians(angle)
        x = center_x + radius * np.cos(rad)
        y = center_y + radius * np.sin(rad)

        # Step circle
        step_circle = Circle(
            (x, y),
            0.5,
            facecolor=color,
            edgecolor="white",
            linewidth=2,
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(step_circle)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fs(6.5),
            color="white",
            fontweight="bold",
            zorder=3,
        )

        # Connecting arrows
        if i < len(angles) - 1:
            next_rad = np.radians(angles[i + 1])
            next_x = center_x + radius * np.cos(next_rad)
            next_y = center_y + radius * np.sin(next_rad)

            # Calculate arrow positions (from edge of circle to edge of next circle)
            start_x = x + 0.5 * np.cos(next_rad)
            start_y = y + 0.5 * np.sin(next_rad)
            end_x = next_x - 0.5 * np.cos(next_rad)
            end_y = next_y - 0.5 * np.sin(next_rad)

            arrow = FancyArrowPatch(
                (start_x, start_y),
                (end_x, end_y),
                arrowstyle="->",
                mutation_scale=15,
                color=COLORS["code_layer_light"],
                linewidth=2,
                zorder=1,
            )
            ax.add_patch(arrow)

    # Tools annotations
    tools_list = ["bash", "file_viewer", "write_file", "edit_file"]
    ax.text(
        center_x,
        center_y,
        "Tools:\n" + ", ".join(tools_list[:2]) + ",\n" + ", ".join(tools_list[2:]),
        ha="center",
        va="center",
        fontsize=fs(7),
        color=COLORS["text_muted"],
        style="italic",
    )

    # Key behaviors
    ax.text(
        6.5,
        2.3,
        "Key Behaviors:",
        fontsize=fs(9),
        color=COLORS["text"],
        fontweight="bold",
    )
    behaviors = [
        "• Self-correcting: Auto-fixes linter errors",
        "• Context-aware: Uses stubs, not full code",
        "• Reference learning: Explores similar code",
        "• Iterative: lint → fix → re-lint loop",
    ]
    y_beh = 1.9
    for beh in behaviors:
        ax.text(6.7, y_beh, beh, fontsize=fs(7), color=COLORS["text_muted"])
        y_beh -= 0.3

    # ========== RIGHT PANEL: OUTPUT (Generated Code) ==========
    right_bg = FancyBboxPatch(
        (12.2, 0.5),
        5.5,
        9.5,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["success"],
        alpha=0.9,
        linewidth=2.5,
        zorder=0,
    )
    ax.add_patch(right_bg)
    ax.text(
        12.6, 9.7, "OUTPUT", fontsize=fs(14), color=COLORS["success"], fontweight="bold"
    )

    ax.text(
        12.6,
        9.2,
        "Generated: trainer.py",
        fontsize=fs(10),
        color=COLORS["text"],
        fontweight="bold",
    )

    # Code snippet box (move bottom edge up so it doesn't overlap status indicators below)
    output_box = FancyBboxPatch(
        (12.5, 2.5),
        4.9,
        6.3,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#1a1f2e",
        edgecolor=COLORS["success"],
        alpha=0.95,
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(output_box)

    output_code = [
        "from model.network import Network",
        "from data.loader import DataLoader",
        "from typing import Dict",
        "import torch",
        "",
        "def train(model: Network,",
        "          data: DataLoader,",
        "          epochs: int,",
        "          config: Dict) -> None:",
        '    """Execute training loop."""',
        "    for epoch in range(epochs):",
        "        total_loss = 0.0",
        "        for batch in data:",
        "            # Forward pass",
        "            output = model.forward(batch)",
        "            loss = compute_loss(output)",
        "            ",
        "            # Backward pass",
        "            model.backward(loss)",
        "            total_loss += loss.item()",
        "        ",
        '        print(f"Epoch {epoch}: {total_loss}")',
    ]

    y_out = 8.5
    for line in output_code:
        if line.startswith("from") or line.startswith("import"):
            ax.text(
                12.7,
                y_out,
                line,
                fontsize=fs(7),
                color=COLORS["accent2"],
                family="monospace",
            )
        elif line.startswith("def"):
            ax.text(
                12.7,
                y_out,
                line,
                fontsize=fs(7),
                color=COLORS["code_layer_light"],
                family="monospace",
                weight="bold",
            )
        elif '"""' in line or "# " in line:
            ax.text(
                12.7,
                y_out,
                line,
                fontsize=fs(7),
                color=COLORS["success"],
                family="monospace",
                style="italic",
            )
        elif line == "":
            y_out -= 0.1
            continue
        else:
            ax.text(
                12.7,
                y_out,
                line,
                fontsize=fs(7),
                color=COLORS["text"],
                family="monospace",
            )
        y_out -= 0.28

    # Status indicators (leave extra bottom padding so scaled fonts don't overflow)
    status_y = 2.05
    ax.text(
        12.7,
        status_y,
        "✓ Syntax validated (ast.parse)",
        fontsize=fs(7.5),
        color=COLORS["success"],
        weight="bold",
    )
    ax.text(
        12.7,
        status_y - 0.4,
        "✓ Imports resolved",
        fontsize=fs(7.5),
        color=COLORS["success"],
        weight="bold",
    )
    ax.text(
        12.7,
        status_y - 0.8,
        "✓ Type hints complete",
        fontsize=fs(7.5),
        color=COLORS["success"],
        weight="bold",
    )
    ax.text(
        12.7,
        status_y - 1.2,
        "✓ Linter passed (no errors)",
        fontsize=fs(7.5),
        color=COLORS["success"],
        weight="bold",
    )

    # ========== CONNECTING ARROWS ==========
    # Input -> Worker
    create_arrow(
        ax,
        (5.8, 7.0),
        (6.6, 7.0),
        COLORS["code_layer"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )
    # Worker -> Output
    create_arrow(
        ax,
        (11.6, 7.0),
        (12.4, 7.0),
        COLORS["success"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "code_worker_workflow.png"),
        dpi=150,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.close()
    print("[OK] Generated: code_worker_workflow.png")


def generate_code_integrator_workflow():
    """Generate the Code Integrator diagram (verification pipeline + branching)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 11), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.text(
        9,
        10.5,
        "Code Integrator",
        ha="center",
        fontsize=fs(22),
        color=COLORS["text"],
        fontweight="bold",
    )
    ax.text(
        9,
        9.95,
        "Global verification: syntax → entry → compile → tests",
        ha="center",
        fontsize=fs(12),
        color=COLORS["text_muted"],
    )

    # Pipeline lane
    lane = FancyBboxPatch(
        (0.9, 5.8),
        16.2,
        3.6,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["code_layer"],
        alpha=0.85,
        linewidth=2,
    )
    ax.add_patch(lane)
    ax.text(
        1.3,
        9.1,
        "Verification Pipeline",
        fontsize=fs(14),
        color=COLORS["code_layer_light"],
        fontweight="bold",
    )

    stages = [
        (2.5, 7.6, "AST Parse", "ast.parse(all .py)", COLORS["code_layer"]),
        (6.0, 7.6, "Entry Test", "python entry --help", COLORS["code_layer"]),
        (9.5, 7.6, "Compile", "compileall", COLORS["code_layer"]),
        (13.0, 7.6, "Unit Tests", "pytest", COLORS["code_layer"]),
    ]

    for x, y, title, sub, c in stages:
        create_rounded_box(ax, x, y, 2.8, 1.0, c, title, fontsize=fs(11))
        ax.text(
            x,
            y - 0.75,
            sub,
            fontsize=fs(9),
            color=COLORS["text_muted"],
            ha="center",
            family="monospace",
        )

    for i in range(len(stages) - 1):
        create_arrow(
            ax,
            (stages[i][0] + 1.5, 7.6),
            (stages[i + 1][0] - 1.5, 7.6),
            COLORS["arrow"],
            connectionstyle="arc3,rad=0",
            linewidth=2.5,
        )

    # Decision split (big diamond)
    diamond = Polygon(
        [(15.7, 7.6), (16.8, 8.3), (17.9, 7.6), (16.8, 6.9)],
        facecolor=COLORS["warning"],
        edgecolor=COLORS["warning"],
        linewidth=2,
        zorder=2,
    )
    ax.add_patch(diamond)
    ax.text(
        16.8,
        7.6,
        "PASS?",
        ha="center",
        va="center",
        fontsize=fs(11),
        color="white",
        fontweight="bold",
    )
    create_arrow(
        ax,
        (14.6, 7.6),
        (15.6, 7.6),
        COLORS["arrow"],
        connectionstyle="arc3,rad=0",
        linewidth=2.5,
    )

    # Success path
    ok = FancyBboxPatch(
        (13.0, 1.2),
        4.0,
        3.8,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["success"],
        alpha=0.9,
        linewidth=2,
    )
    ax.add_patch(ok)
    ax.text(
        13.3,
        4.6,
        "SUCCESS",
        fontsize=fs(14),
        color=COLORS["success"],
        fontweight="bold",
    )
    ax.text(
        13.3,
        3.9,
        "- runnable entry point\n- tests green\n- ready for Science Layer",
        fontsize=fs(10),
        color=COLORS["text"],
        family="monospace",
    )

    # Fail path -> FixBlueprint
    fb = FancyBboxPatch(
        (0.9, 1.2),
        11.5,
        3.8,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#2b1e1e",
        edgecolor=COLORS["error"],
        alpha=0.9,
        linewidth=2,
    )
    ax.add_patch(fb)
    ax.text(
        1.3,
        4.6,
        "FAIL → FixBlueprint",
        fontsize=fs(14),
        color=COLORS["error"],
        fontweight="bold",
    )

    # Show structured failure extraction
    ax.text(
        1.3,
        3.85,
        "TracebackParser → IntegrationIssue/TestFailureInfo",
        fontsize=fs(10),
        color=COLORS["text"],
        family="monospace",
    )
    ax.text(
        1.3,
        3.2,
        "FixIssueSpec:",
        fontsize=fs(10),
        color=COLORS["warning"],
        family="monospace",
        fontweight="bold",
    )
    ax.text(
        1.8,
        2.6,
        "- file_path\n- issue_type\n- message\n- suggestion\n- error_trace/line",
        fontsize=fs(10),
        color=COLORS["text"],
        family="monospace",
    )

    # Arrows from decision
    create_arrow(
        ax,
        (16.8, 6.9),
        (15.0, 5.0),
        COLORS["success"],
        connectionstyle="arc3,rad=0.15",
        linewidth=3,
    )
    create_arrow(
        ax,
        (16.8, 8.3),
        (12.0, 5.0),
        COLORS["error"],
        connectionstyle="arc3,rad=-0.2",
        linewidth=3,
    )

    ax.text(
        15.4, 6.0, "Yes", fontsize=fs(11), color=COLORS["success"], fontweight="bold"
    )
    ax.text(12.6, 6.0, "No", fontsize=fs(11), color=COLORS["error"], fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "code_integrator_workflow.png"),
        dpi=160,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close()
    print("[OK] Generated: code_integrator_workflow.png")


def generate_science_architect_workflow():
    """Generate the Science Architect diagram (Code Architect Style: Inputs -> Kanban -> Output)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(
        9,
        11.4,
        "Science Architect",
        ha="center",
        va="center",
        fontsize=fs(22),
        color=COLORS["text"],
        fontweight="bold",
    )
    ax.text(
        9,
        10.85,
        "Experiment design from Context/Goals to Plan (no execution)",
        ha="center",
        va="center",
        fontsize=fs(12),
        color=COLORS["text_muted"],
    )

    # Left: Inputs board
    left = FancyBboxPatch(
        (0.6, 6.4),
        6.2,
        4.2,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["accent2"],
        alpha=0.9,
        linewidth=2,
    )
    ax.add_patch(left)
    ax.text(
        1.0, 10.2, "Inputs", fontsize=fs(14), color=COLORS["accent2"], fontweight="bold"
    )

    create_rounded_box(
        ax,
        3.7,
        9.35,
        5.5,
        0.9,
        COLORS["accent2"],
        "Research Goal\n(from Proposal)",
        fontsize=fs(10),
    )
    create_rounded_box(
        ax,
        3.7,
        8.15,
        5.5,
        0.9,
        COLORS["code_layer"],
        "Code Context\n(CodeManifest + Blueprints)",
        fontsize=fs(10),
    )
    create_rounded_box(
        ax,
        3.7,
        6.95,
        5.5,
        0.9,
        COLORS["warning"],
        "Constraints\n(Compute budget / Time)",
        fontsize=fs(10),
    )

    # Center: Architect Loop
    mid = FancyBboxPatch(
        (7.2, 3.1),
        6.2,
        7.5,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["science_layer"],
        alpha=0.9,
        linewidth=2.5,
    )
    ax.add_patch(mid)
    ax.text(
        7.6,
        10.2,
        "Architect Loop",
        fontsize=fs(14),
        color=COLORS["science_layer_light"],
        fontweight="bold",
    )

    # Columns
    col_x = [7.8, 10.3, 12.6]
    col_titles = ["Understand", "Hypothesize", "Plan"]
    for x, t in zip(col_x, col_titles):
        ax.text(
            x,
            9.6,
            t,
            fontsize=fs(11),
            color=COLORS["text"],
            fontweight="bold",
            ha="center",
        )

    # Cards
    cards = [
        (7.8, 8.6, "Read code\n(file_viewer)", COLORS["code_layer"]),
        (7.8, 7.5, "Check data\n(bash ls/find)", COLORS["warning"]),
        (10.3, 8.6, "Formulate\nHypothesis", COLORS["science_layer"]),
        (10.3, 7.5, "Design Vars\n(Indep/Dep)", COLORS["accent"]),
        (12.6, 8.6, "Define Metrics\n(Goal: max/min)", COLORS["success"]),
        (12.6, 7.5, "Structure Tasks\n(DAG: baseline->exp)", COLORS["success"]),
    ]
    for x, y, txt, c in cards:
        create_rounded_box(ax, x, y, 2.1, 0.95, c, txt, fontsize=fs(9))

    # Tools Callout (matches Code Architect position)
    tool_box = FancyBboxPatch(
        (9.0, 3.6),
        4.0,
        2.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="#1a1f2e",
        edgecolor=COLORS["accent"],
        alpha=0.95,
        linewidth=1.8,
    )
    ax.add_patch(tool_box)
    ax.text(
        9.2,
        5.8,
        "Allowed tools",
        fontsize=fs(12),
        color=COLORS["accent"],
        fontweight="bold",
    )
    ax.text(
        9.2,
        5.2,
        "- bash(command)",
        fontsize=fs(9),
        color=COLORS["text"],
        family="monospace",
    )
    ax.text(
        9.2,
        4.7,
        "- file_viewer(path, start, end)",
        fontsize=fs(9),
        color=COLORS["text"],
        family="monospace",
    )
    ax.text(
        9.2,
        4.2,
        "No writing code, only reading",
        fontsize=fs(9),
        color=COLORS["warning"],
        style="italic",
    )

    # Right: Output (ExperimentPlan)
    right = FancyBboxPatch(
        (13.8, 1.0),
        3.6,
        9.6,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["success"],
        alpha=0.9,
        linewidth=2,
    )
    ax.add_patch(right)
    ax.text(
        14.1,
        10.2,
        "Output",
        fontsize=fs(14),
        color=COLORS["success"],
        fontweight="bold",
    )
    ax.text(
        14.1,
        9.55,
        "ExperimentPlan",
        fontsize=fs(11),
        color=COLORS["text"],
        fontweight="bold",
    )

    plan_lines = [
        "tasks:",
        "  - baseline:",
        "      args: {lr: 0.01}",
        "  - experiment:",
        "      args: {lr: 0.05}",
        "      depends_on: [baseline]",
        "",
        "metrics:",
        "  - name: accuracy",
        "    goal: maximize",
        "  - name: runtime",
        "    goal: minimize",
    ]
    y = 9.1
    for line in plan_lines:
        if line == "":
            y -= 0.25
            continue
        c = COLORS["text"] if ":" in line else COLORS["success"]
        ax.text(14.1, y, line, fontsize=fs(8.5), color=c, family="monospace")
        y -= 0.38

    # Flows
    create_arrow(
        ax,
        (6.8, 8.8),
        (7.2, 8.8),
        COLORS["accent2"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )
    create_arrow(
        ax,
        (6.8, 7.9),
        (7.2, 7.9),
        COLORS["code_layer"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )
    create_arrow(
        ax,
        (6.8, 7.0),
        (7.2, 7.0),
        COLORS["warning"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )
    create_arrow(
        ax,
        (13.4, 6.8),
        (13.8, 6.8),
        COLORS["success"],
        connectionstyle="arc3,rad=0",
        linewidth=3,
    )

    # Footer note
    ax.text(
        9,
        0.4,
        "Architect delivers the STRATEGY. Manager delivers the EXECUTION.",
        ha="center",
        fontsize=fs(10),
        color=COLORS["text_muted"],
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "science_architect_workflow.png"),
        dpi=160,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close()
    print("[OK] Generated: science_architect_workflow.png")


def generate_science_manager_workflow():
    """Generate the Science Manager diagram (Cluster Scheduler Style)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(
        9,
        11.4,
        "Science Manager",
        ha="center",
        fontsize=fs(24),
        color=COLORS["text"],
        fontweight="bold",
    )
    ax.text(
        9,
        10.9,
        "Parallel Experiment Orchestration & Resource Management",
        ha="center",
        fontsize=fs(14),
        color=COLORS["text_muted"],
    )

    # 1. Top: The Plan Queue
    ax.text(
        3.0,
        9.8,
        "Input: Task DAG",
        ha="center",
        fontsize=fs(14),
        color=COLORS["science_layer_light"],
        fontweight="bold",
    )
    queue_bg = FancyBboxPatch(
        (1.0, 8.5),
        4.0,
        1.0,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["science_layer"],
        linewidth=1.5,
    )
    ax.add_patch(queue_bg)

    # Small task nodes in queue
    for i, (x, c) in enumerate(
        zip(
            [1.3, 2.3, 3.3, 4.3],
            [
                COLORS["science_layer"],
                COLORS["science_layer"],
                COLORS["warning"],
                COLORS["warning"],
            ],
        )
    ):
        circle = Circle((x, 9.0), 0.35, facecolor=c, edgecolor="white")
        ax.add_patch(circle)
        ax.text(
            x,
            9.0,
            f"T{i}",
            ha="center",
            va="center",
            color="white",
            fontsize=fs(8),
            fontweight="bold",
        )
    ax.text(
        2.8,
        8.2,
        "(Pending Tasks)",
        ha="center",
        fontsize=fs(9),
        color=COLORS["text_muted"],
    )

    # 2. Middle: The Scheduler Logic
    scheduler_poly = Polygon(
        [(7.0, 9.5), (11.0, 9.5), (10.0, 8.0), (8.0, 8.0)],
        facecolor="#2d3748",
        edgecolor=COLORS["accent2"],
        linewidth=2,
    )
    ax.add_patch(scheduler_poly)
    ax.text(
        9.0,
        9.0,
        "Scheduler\nLoop",
        ha="center",
        va="center",
        fontsize=fs(12),
        color="white",
        fontweight="bold",
    )

    # Connector Input -> Scheduler
    create_arrow(ax, (5.2, 9.0), (7.0, 9.0), COLORS["science_layer_light"], linewidth=3)

    # 3. Bottom: The Worker Cluster (Swimlanes)
    cluster_bg = FancyBboxPatch(
        (1.0, 1.5),
        16.0,
        6.0,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor="#0d1117",
        edgecolor=COLORS["accent"],
        linewidth=2,
        alpha=0.9,
    )
    ax.add_patch(cluster_bg)
    ax.text(
        9.0,
        7.7,
        "Execution Cluster (Worker Pool)",
        ha="center",
        fontsize=fs(14),
        color=COLORS["accent"],
        fontweight="bold",
    )

    # Lanes
    lanes_y = [6.2, 4.8, 3.4]
    labels = ["Worker-1 (GPU0)", "Worker-2 (GPU1)", "Worker-3 (CPU)"]

    for y, label in zip(lanes_y, labels):
        # Lane Divider
        # ax.plot([1.5, 16.5], [y - 0.7, y - 0.7], color="#30363d", linewidth=1, linestyle="--")

        # Worker Label
        ax.text(
            2.5,
            y,
            label,
            ha="right",
            va="center",
            fontsize=fs(11),
            color=COLORS["text"],
            fontweight="bold",
            family="monospace",
        )

        # Timeline Bar
        ax.plot([3.0, 16.0], [y, y], color="#30363d", linewidth=2)

        # Task Blocks
        if "GPU0" in label:
            # Running Task
            rect = FancyBboxPatch(
                (4.0, y - 0.3),
                3.0,
                0.6,
                boxstyle="round,pad=0.02",
                facecolor=COLORS["success"],
                alpha=0.9,
            )
            ax.add_patch(rect)
            ax.text(
                5.5,
                y,
                "Task: baseline",
                ha="center",
                va="center",
                fontsize=fs(9),
                color="black",
                fontweight="bold",
            )
            # Finished Task
            # rect2 = FancyBboxPatch((8.0, y - 0.3), 2.5, 0.6, boxstyle="round,pad=0.02", facecolor="#1f6feb", alpha=0.6)
            # ax.add_patch(rect2)
        elif "GPU1" in label:
            # Running Task
            rect = FancyBboxPatch(
                (5.0, y - 0.3),
                3.5,
                0.6,
                boxstyle="round,pad=0.02",
                facecolor=COLORS["warning"],
                alpha=0.9,
            )
            ax.add_patch(rect)
            ax.text(
                6.75,
                y,
                "Task: lr_sweep_01",
                ha="center",
                va="center",
                fontsize=fs(9),
                color="black",
                fontweight="bold",
            )
        else:
            # Idle
            ax.text(
                4.0,
                y,
                "(Idle)",
                ha="left",
                va="center",
                fontsize=fs(9),
                color=COLORS["text_muted"],
                style="italic",
            )

    # Arrows from Scheduler to Lanes
    create_arrow(
        ax, (9.0, 8.0), (5.5, 6.6), COLORS["accent2"], linewidth=2, linestyle="--"
    )
    create_arrow(
        ax, (9.0, 8.0), (6.75, 5.2), COLORS["accent2"], linewidth=2, linestyle="--"
    )

    # 4. Right: Result Aggregator
    agg_bg = FancyBboxPatch(
        (13.5, 8.5),
        3.5,
        2.5,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["success"],
        linewidth=1.5,
    )
    ax.add_patch(agg_bg)
    ax.text(
        15.25,
        10.6,
        "Results",
        ha="center",
        fontsize=fs(14),
        color=COLORS["success"],
        fontweight="bold",
    )

    # File icons
    ax.text(14.5, 9.8, "[JSON] metrics.json", fontsize=fs(10), color=COLORS["text"])
    ax.text(14.5, 9.3, "[TXT] logs.txt", fontsize=fs(10), color=COLORS["text"])
    ax.text(14.5, 8.8, "[IMG] plots.png", fontsize=fs(10), color=COLORS["text"])

    # Arrow from Cluster to Aggregator
    create_arrow(
        ax,
        (16.0, 4.5),
        (17.5, 4.5),
        COLORS["success"],
        linewidth=2,
        connectionstyle="angle,angleA=0,angleB=90,rad=10",
    )
    # This might need a custom path, let's just do a simpler one or imply it
    # Draw arrow from right side of cluster up to results
    ax.annotate(
        "",
        xy=(15.25, 8.4),
        xytext=(15.25, 7.5),
        arrowprops=dict(arrowstyle="->", color=COLORS["success"], lw=3),
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "science_manager_workflow.png"),
        dpi=160,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close()
    print("[OK] Generated: science_manager_workflow.png")


def generate_science_worker_workflow():
    """Generate the Science Worker diagram (Modern Pipeline Style)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(
        9,
        11.4,
        "Science Worker",
        ha="center",
        fontsize=fs(24),
        color=COLORS["text"],
        fontweight="bold",
    )
    ax.text(
        9,
        10.9,
        "Atomic Experiment Execution Pipeline",
        ha="center",
        fontsize=fs(14),
        color=COLORS["text_muted"],
    )

    # Main Pipeline Tube
    # Draw a large gray rounded rect as the "Machine"
    machine_bg = FancyBboxPatch(
        (1.0, 3.0),
        16.0,
        6.0,
        boxstyle="round,pad=0.2,rounding_size=0.3",
        facecolor="#161b22",
        edgecolor="#30363d",
        linewidth=2,
    )
    ax.add_patch(machine_bg)
    ax.text(
        2.0,
        9.3,
        "Worker Process (PID: 1234)",
        fontsize=fs(12),
        color="#8b949e",
        fontweight="bold",
        family="monospace",
    )

    # Stages inside the machine
    stages = [
        ("1. Setup", "mkdir -p dir\ncd result_dir", COLORS["code_layer"]),
        ("2. Execute", "python train.py\n(subprocess)", COLORS["accent"]),
        ("3. Monitor", "Stream stdout\nCheck NaN", COLORS["warning"]),
        ("4. Persist", "Save JSON\nSync Artifacts", COLORS["success"]),
    ]

    start_x = 2.5
    gap = 3.8

    for i, (title, sub, col) in enumerate(stages):
        x = start_x + i * gap
        # Stage Box
        box = FancyBboxPatch(
            (x, 4.5),
            3.0,
            3.5,
            boxstyle="round,pad=0.1",
            facecolor=COLORS["bg"],
            edgecolor=col,
            linewidth=2,
        )
        ax.add_patch(box)

        # Header
        ax.text(
            x + 1.5,
            7.5,
            title,
            ha="center",
            fontsize=fs(12),
            color=col,
            fontweight="bold",
        )

        # Content
        ax.text(
            x + 1.5,
            6.0,
            sub,
            ha="center",
            va="center",
            fontsize=fs(10),
            color=COLORS["text_muted"],
            family="monospace",
        )

        # Arrow to next (except last)
        if i < len(stages) - 1:
            create_arrow(
                ax, (x + 3.1, 6.25), (x + 3.7, 6.25), COLORS["text"], linewidth=2
            )

    # Input (Left)
    ax.text(
        1.5,
        10.0,
        "Input: Task Spec",
        fontsize=fs(12),
        color=COLORS["code_layer_light"],
        fontweight="bold",
    )
    create_rounded_box(
        ax,
        1.5,
        10.5,
        3.5,
        1.2,
        COLORS["code_layer"],
        "cmd: python train.py\ndir: runs/exp_01",
        fontsize=fs(10),
    )
    create_arrow(ax, (3.25, 10.5), (3.25, 8.2), COLORS["code_layer"], linewidth=2)

    # Output (Right)
    ax.text(
        14.5,
        2.0,
        "Output: Artifacts",
        fontsize=fs(12),
        color=COLORS["success"],
        fontweight="bold",
    )

    # File Stack Visualization
    files = [
        (14.0, 1.0, "metrics.json", COLORS["success"]),
        (14.2, 1.3, "model.pth", COLORS["accent2"]),
        (14.4, 1.6, "log.txt", COLORS["text"]),
    ]
    for fx, fy, ft, fc in files:
        fbox = FancyBboxPatch(
            (fx, fy),
            2.5,
            0.8,
            boxstyle="round,pad=0.05",
            facecolor=COLORS["bg_light"],
            edgecolor=fc,
            linewidth=1,
        )
        ax.add_patch(fbox)
        ax.text(
            fx + 1.25,
            fy + 0.4,
            ft,
            ha="center",
            va="center",
            fontsize=fs(9),
            color=fc,
            family="monospace",
        )

    create_arrow(ax, (15.5, 4.3), (15.5, 2.5), COLORS["success"], linewidth=2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "science_worker_workflow.png"),
        dpi=160,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close()
    print("[OK] Generated: science_worker_workflow.png")


def generate_science_integrator_workflow():
    """Generate the Science Integrator diagram (Dashboard & Decision Style)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(
        9,
        11.4,
        "Science Integrator",
        ha="center",
        fontsize=fs(24),
        color=COLORS["text"],
        fontweight="bold",
    )

    # 1. Left: The "Dashboard" Monitor
    monitor_frame = FancyBboxPatch(
        (1.0, 1.5),
        10.0,
        8.5,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        facecolor="#0d1117",
        edgecolor="#58a6ff",
        linewidth=3,
    )
    ax.add_patch(monitor_frame)
    ax.text(
        6.0,
        10.2,
        "Analysis Dashboard",
        ha="center",
        fontsize=fs(14),
        color="#58a6ff",
        fontweight="bold",
    )

    # Chart 1: Loss Curve
    c1_bg = Rectangle((1.5, 5.5), 4.0, 3.0, facecolor="#161b22", edgecolor="#30363d")
    ax.add_patch(c1_bg)
    ax.text(3.5, 8.7, "Loss vs Epoch", ha="center", fontsize=fs(9), color="white")
    ax.plot(
        [1.6, 2.5, 3.5, 4.5, 5.4],
        [8.0, 7.0, 6.5, 6.3, 6.2],
        color="#f78166",
        linewidth=2,
    )
    ax.plot(
        [1.6, 2.5, 3.5, 4.5, 5.4],
        [8.0, 7.5, 7.3, 7.2, 7.1],
        color="#3fb950",
        linewidth=2,
    )

    # Chart 2: Bar Chart (Metrics)
    c2_bg = Rectangle((6.0, 5.5), 4.0, 3.0, facecolor="#161b22", edgecolor="#30363d")
    ax.add_patch(c2_bg)
    ax.text(8.0, 8.7, "Accuracy by Exp", ha="center", fontsize=fs(9), color="white")
    for i, h in enumerate([1.5, 2.0, 2.2, 1.8]):
        rect = Rectangle((6.5 + i * 0.8, 5.7), 0.5, h, facecolor="#238636")
        ax.add_patch(rect)

    # Text Summary
    text_bg = Rectangle((1.5, 2.0), 8.5, 3.0, facecolor="#161b22", edgecolor="#30363d")
    ax.add_patch(text_bg)
    ax.text(
        1.8,
        4.7,
        "> Analysis Summary:",
        fontsize=fs(10),
        color="white",
        fontweight="bold",
        family="monospace",
    )
    ax.text(
        1.8,
        4.0,
        "- Experiment B achieved highest accuracy (92%)",
        fontsize=fs(9),
        color="#8b949e",
        family="monospace",
    )
    ax.text(
        1.8,
        3.3,
        "- Baseline failed to converge in 3 runs",
        fontsize=fs(9),
        color="#d29922",
        family="monospace",
    )
    ax.text(
        1.8,
        2.6,
        "- Recommendation: Adopt settings from Exp B",
        fontsize=fs(9),
        color="#3fb950",
        family="monospace",
    )

    # 2. Right Top: Output (OptimizationTicket) - ABOVE Decision Logic
    opt_box = FancyBboxPatch(
        (12.0, 8.5),
        5.0,
        1.8,
        boxstyle="round,pad=0.1",
        facecolor="#2b1e1e",
        edgecolor=COLORS["error"],
        linewidth=2,
    )
    ax.add_patch(opt_box)
    ax.text(
        14.5,
        9.6,
        "Output: OptimizationTicket",
        ha="center",
        fontsize=fs(12),
        color=COLORS["error"],
        fontweight="bold",
    )
    ax.text(
        14.5,
        9.0,
        "(Request Code Changes)",
        ha="center",
        fontsize=fs(10),
        color=COLORS["text"],
    )

    # 3. Right Middle: Decision Logic (title removed per request)

    # Diamond: Goal Met?
    diamond = Polygon(
        [(14.5, 6.5), (16.2, 5.0), (14.5, 3.5), (12.8, 5.0)],
        facecolor=COLORS["bg_light"],
        edgecolor=COLORS["warning"],
        linewidth=2,
    )
    ax.add_patch(diamond)
    ax.text(
        14.5,
        5.0,
        "Goal Met?",
        ha="center",
        va="center",
        fontsize=fs(12),
        color=COLORS["warning"],
        fontweight="bold",
    )

    # 4. Right Bottom: SUCCESS
    success_box = FancyBboxPatch(
        (12.5, 1.0),
        4.0,
        2.0,
        boxstyle="round,pad=0.1",
        facecolor="#1a2e26",
        edgecolor=COLORS["success"],
        linewidth=2,
    )
    ax.add_patch(success_box)
    ax.text(
        14.5,
        2.2,
        "SUCCESS",
        ha="center",
        fontsize=fs(14),
        color=COLORS["success"],
        fontweight="bold",
    )
    ax.text(
        14.5, 1.6, "Report Findings", ha="center", fontsize=fs(10), color=COLORS["text"]
    )

    # Arrows
    # YES -> Success (down)
    create_arrow(ax, (14.5, 3.5), (14.5, 3.0), COLORS["success"], linewidth=3)
    ax.text(
        14.8, 3.3, "Yes", fontsize=fs(11), color=COLORS["success"], fontweight="bold"
    )

    # NO -> Output (up)
    create_arrow(ax, (14.5, 6.5), (14.5, 8.5), COLORS["error"], linewidth=3)
    ax.text(14.8, 7.5, "No", fontsize=fs(11), color=COLORS["error"], fontweight="bold")

    # Connection from Dashboard to Decision
    create_arrow(ax, (11.0, 5.0), (12.8, 5.0), COLORS["arrow"], linewidth=3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "science_integrator_workflow.png"),
        dpi=160,
        facecolor=COLORS["bg"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close()
    print("[OK] Generated: science_integrator_workflow.png")


def main():
    """Generate all diagrams."""
    print("=" * 50)
    print("Generating SuperAgent Architecture Diagrams...")
    print("Output directory:", OUTPUT_DIR)
    print("=" * 50)
    print()

    generate_main_architecture()
    generate_information_flow()
    generate_dag_scheduling()
    generate_cache_workflow()
    generate_validation_pipeline()
    generate_experiment_agent_framework()
    generate_experiment_agent_main_flow()

    # Code Layer Agents
    generate_code_architect_workflow()
    generate_code_manager_workflow()
    generate_code_worker_workflow()
    generate_code_integrator_workflow()

    # Science Layer Agents
    generate_science_architect_workflow()
    generate_science_manager_workflow()
    generate_science_worker_workflow()
    generate_science_integrator_workflow()

    print()
    print("=" * 50)
    print("All diagrams generated successfully!")
    print("Find them in:", OUTPUT_DIR)
    print("=" * 50)


if __name__ == "__main__":
    main()
