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


def create_rounded_box(
    ax,
    x,
    y,
    width,
    height,
    color,
    text,
    fontsize=10,
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
    ax, start, end, color=None, style="->", connectionstyle="arc3,rad=0.1", linewidth=2
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
        ax.text(4.5, y, desc, fontsize=9, color=COLORS["text_muted"], va="center")

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

    print()
    print("=" * 50)
    print("All diagrams generated successfully!")
    print("Find them in:", OUTPUT_DIR)
    print("=" * 50)


if __name__ == "__main__":
    main()
