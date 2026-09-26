"""Script to generate publication-grade, high-resolution architectural diagrams for the Tactical MARL System."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, title, subtitle=None, bg_color="#FFFFFF", border_color="#1A365D",
             text_color="#1A365D", boxstyle="round,pad=0.02,rounding_size=0.03", lw=1.5, title_fontsize=10, sub_fontsize=8):
    """Draw a styled rectangular box with title and subtitle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=boxstyle,
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        mutation_scale=1.0,
        zorder=2
    )
    ax.add_patch(box)
    
    if title:
        ty = y + h - 0.025 if subtitle else y + h / 2.0
        va = "top" if subtitle else "center"
        ax.text(
            x + w / 2.0, ty, title,
            ha="center", va=va,
            fontsize=title_fontsize, fontweight="bold",
            color=text_color, zorder=3, family="sans-serif"
        )
    
    if subtitle:
        ax.text(
            x + w / 2.0, y + (h - 0.04) / 2.0, subtitle,
            ha="center", va="center",
            fontsize=sub_fontsize, color=text_color,
            zorder=3, family="sans-serif", multialignment="center"
        )
    return box


def generate_10_layer_architecture_diagram():
    """Generate the full 10-Layer System Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(14, 11), dpi=300)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # Header / Title Block
    ax.text(
        0.5, 0.975, "TACTICAL SCENARIO SIMULATOR (TSS) — 10-LAYER H-MARL SYSTEM ARCHITECTURE",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1A365D", family="sans-serif"
    )
    ax.text(
        0.5, 0.952, "Decoupled Clean Architecture for Real-Time, Non-Deterministic Multi-Domain Autonomous Combat Simulation",
        ha="center", va="center", fontsize=9.5, fontstyle="italic", color="#4A5568", family="sans-serif"
    )

    # 1. DRDO TSS Runtime (External System)
    draw_box(
        ax, 0.08, 0.875, 0.84, 0.055,
        "DRDO TACTICAL SCENARIO SIMULATOR (TSS) RUNTIME",
        "Multi-Domain Combat Simulation Environment (Air, Ground, Maritime) | 50 Hz Frame Budget (20 ms)",
        bg_color="#1A365D", border_color="#0F243E", text_color="#FFFFFF", title_fontsize=11, sub_fontsize=8.5
    )

    # Arrow TSS <-> Layer 8
    ax.annotate(
        "", xy=(0.32, 0.815), xytext=(0.32, 0.875),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#2B6CB0", shrinkA=2, shrinkB=2)
    )
    ax.text(0.325, 0.845, "Raw Telemetry\n(Pos, Vel, Heading)", fontsize=7.5, color="#2B6CB0", fontweight="bold", va="center")

    ax.annotate(
        "", xy=(0.68, 0.875), xytext=(0.68, 0.815),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#C53030", shrinkA=2, shrinkB=2)
    )
    ax.text(0.685, 0.845, "Control Commands\n(Flight/Ground/Naval)", fontsize=7.5, color="#C53030", fontweight="bold", va="center")

    # 2. LAYER 8: TSS Integration Wrapper
    draw_box(
        ax, 0.08, 0.735, 0.84, 0.08,
        "LAYER 8: TSS INTEGRATION WRAPPER  (src/integration/)",
        "• PettingZoo ParallelEnv Multi-Agent Wrapper   • Gymnasium Single-Agent Adapter\n"
        "• Direct In-Process Zero-Copy Memory Adapter (Latency: 0.58 ms)   • Field Mapper & Unit Converter (km/h <-> m/s, deg <-> rad)",
        bg_color="#EBF8FF", border_color="#3182CE", text_color="#2B6CB0", title_fontsize=10.5, sub_fontsize=8
    )

    # Arrow Layer 8 <-> Layer 7
    ax.annotate(
        "", xy=(0.32, 0.675), xytext=(0.32, 0.735),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#3182CE", shrinkA=2, shrinkB=2)
    )
    ax.text(0.325, 0.705, "Normalized Obs Dict", fontsize=7.5, color="#3182CE", va="center")

    ax.annotate(
        "", xy=(0.68, 0.735), xytext=(0.68, 0.675),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#3182CE", shrinkA=2, shrinkB=2)
    )
    ax.text(0.685, 0.705, "Decoded Action Dict", fontsize=7.5, color="#3182CE", va="center")

    # 3. LAYER 7: Inference API
    draw_box(
        ax, 0.08, 0.595, 0.84, 0.08,
        "LAYER 7: HIGH-PERFORMANCE INFERENCE API  (src/api/)",
        "• FastAPI Asynchronous Microservice (/act, /act/batch, /reset, /health, /load_checkpoint)\n"
        "• ModelRegistry with Zero-Downtime Hot-Swapping (< 15 ms)   • Strict ActionCodec & ObservationCodec Validation (Latency: 4.21 ms)",
        bg_color="#EBF4FF", border_color="#4C51BF", text_color="#3730A3", title_fontsize=10.5, sub_fontsize=8
    )

    # Arrow Layer 7 <-> Layer 3
    ax.annotate(
        "", xy=(0.5, 0.535), xytext=(0.5, 0.595),
        arrowprops=dict(arrowstyle="<->", lw=1.8, color="#4C51BF", shrinkA=2, shrinkB=2)
    )
    ax.text(0.51, 0.565, "Tensors (Batch, Device)  <-->  Sampled Actions / Distributions", fontsize=8, color="#4C51BF", fontweight="bold", va="center")

    # 4. LAYER 3: Hierarchical MARL Engine (Sub-box containing Commander + 3 Domains)
    box_l3 = FancyBboxPatch(
        (0.08, 0.285), 0.84, 0.25,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#F7FAFC", edgecolor="#4C51BF", linewidth=1.5, zorder=1
    )
    ax.add_patch(box_l3)
    ax.text(
        0.5, 0.518, "LAYER 3: HIERARCHICAL MULTI-AGENT REINFORCEMENT LEARNING ENGINE  (src/marl/)",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1A365D", zorder=3
    )

    # 4a. Strategic Theater Commander
    draw_box(
        ax, 0.12, 0.445, 0.76, 0.055,
        "THEATER COMMANDER POLICY (CommanderPolicy)",
        "Recurrent GRU Core (256 Hidden Units) | Evaluates 53-dim Operational Picture | Generates Multi-Domain Sub-Goals (g_tau)",
        bg_color="#FEFCBF", border_color="#D69E2E", text_color="#744210", title_fontsize=9.5, sub_fontsize=7.5
    )

    # Arrows Commander -> Domain Policies
    for xp, domain_lbl in [(0.24, "Air Sub-Goal"), (0.50, "Ground Sub-Goal"), (0.76, "Maritime Sub-Goal")]:
        ax.annotate(
            "", xy=(xp, 0.40), xytext=(xp, 0.445),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#B7791F", shrinkA=2, shrinkB=2)
        )
        ax.text(xp, 0.422, domain_lbl, fontsize=7, color="#744210", ha="center", va="center", fontweight="bold")

    # 4b. Three Low-Level Domain Policy Boxes
    # Air Domain
    draw_box(
        ax, 0.11, 0.30, 0.24, 0.10,
        "AIR DOMAIN POLICIES",
        "• AirFightPolicy (AC1 / AC2)\n• AirEscapePolicy (Evasive)\n• Self-Attention (Perm. Inv.)\n• Factorized PPO (468 actions)",
        bg_color="#FFFFFF", border_color="#3182CE", text_color="#2B6CB0", title_fontsize=9, sub_fontsize=7.5
    )
    # Ground Domain
    draw_box(
        ax, 0.38, 0.30, 0.24, 0.10,
        "GROUND DOMAIN POLICIES",
        "• GroundEngagePolicy\n• GroundDefendPolicy\n• Self-Attention + HHAPPO\n• Hybrid Gaussian + Categorical",
        bg_color="#FFFFFF", border_color="#38A169", text_color="#22543D", title_fontsize=9, sub_fontsize=7.5
    )
    # Maritime Domain
    draw_box(
        ax, 0.65, 0.30, 0.24, 0.10,
        "MARITIME DOMAIN POLICIES",
        "• SeaEngagePolicy\n• SeaDefendPolicy\n• Self-Attention + HHAPPO\n• Naval Standoff & Screening",
        bg_color="#FFFFFF", border_color="#319795", text_color="#234E52", title_fontsize=9, sub_fontsize=7.5
    )

    # Arrow Layer 3 <-> Layer 2
    ax.annotate(
        "", xy=(0.5, 0.225), xytext=(0.5, 0.285),
        arrowprops=dict(arrowstyle="<->", lw=1.8, color="#234E52", shrinkA=2, shrinkB=2)
    )
    ax.text(0.51, 0.255, "Actions (Heading, Speed, Weapons)  <-->  Local Obs, Rewards, Dones", fontsize=8, color="#234E52", fontweight="bold", va="center")

    # 5. LAYER 2: 2.5D Multi-Domain Tactical Simulator
    draw_box(
        ax, 0.08, 0.145, 0.84, 0.08,
        "LAYER 2: 2.5D MULTI-DOMAIN TACTICAL SIMULATOR  (src/simulator/)",
        "• Continuous 100 km x 100 km Battlespace with Digital Elevation Contours & Radar Line-of-Sight Masking\n"
        "• 3D Flight Aerodynamics (Dubins energy-maneuverability)   • Surface Ground & Hydrodynamic Naval Kinematics\n"
        "• Combat Envelopes (Probability of Hit, Aspect Angles)   • 5-Tier Procedural Scenario Generator (1v1 to Joint 16+ Units)",
        bg_color="#E6FFFA", border_color="#319795", text_color="#234E52", title_fontsize=10.5, sub_fontsize=8
    )

    # Arrow Layer 2 <-> Layer 1
    ax.annotate(
        "", xy=(0.5, 0.095), xytext=(0.5, 0.145),
        arrowprops=dict(arrowstyle="<->", lw=1.5, color="#2D3748", shrinkA=2, shrinkB=2)
    )
    ax.text(0.51, 0.120, "Implements Formal Domain Interfaces & Typed Dataclasses", fontsize=8, color="#2D3748", va="center")

    # 6. LAYER 1: Core Interfaces
    draw_box(
        ax, 0.08, 0.035, 0.84, 0.06,
        "LAYER 1: CORE INTERFACES & FOUNDATIONAL ABSTRACTIONS  (src/core/)",
        "• Abstract Base Classes: BaseEnvironment, BaseEntity, BasePolicy (Zero Circular Dependency Architecture)\n"
        "• Strongly Typed Dataclasses: AirAction, GroundAction, SeaAction, CommanderAction, AirObservation (13), GroundObservation (9), SeaObservation (9)",
        bg_color="#EDF2F7", border_color="#4A5568", text_color="#1A202C", title_fontsize=10, sub_fontsize=7.5
    )

    fig.tight_layout()
    out_path = FIG_DIR / "system_architecture_10_layer.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved 10-layer architecture diagram: {out_path}")


def generate_hierarchical_policy_diagram():
    """Generate the Hierarchical Policy Decision Flow Diagram (Commander -> Domain Policies)."""
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=300)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    ax.text(
        0.5, 0.97, "HIERARCHICAL MULTI-AGENT DECISION PROCESS & NEURAL ARCHITECTURE",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1A365D", family="sans-serif"
    )
    ax.text(
        0.5, 0.94, "Dec-POMDP Formulation: Recurrent GRU Macro-Intent Coordinator + Domain Self-Attention Tactical Controllers",
        ha="center", va="center", fontsize=9.5, fontstyle="italic", color="#4A5568", family="sans-serif"
    )

    # 1. Global State Box
    draw_box(
        ax, 0.25, 0.83, 0.50, 0.07,
        "GLOBAL THEATER OBSERVATION  (S_tau in R^53)",
        "Friendly & Hostile Asset Counts, Force Balance Ratios, Objective Control Status, Sensor Tracks",
        bg_color="#EDF2F7", border_color="#4A5568", text_color="#1A202C", title_fontsize=10, sub_fontsize=8
    )

    ax.annotate(
        "", xy=(0.5, 0.77), xytext=(0.5, 0.83),
        arrowprops=dict(arrowstyle="->", lw=2, color="#1A365D", shrinkA=2, shrinkB=2)
    )

    # 2. Theater Commander Neural Policy
    draw_box(
        ax, 0.15, 0.63, 0.70, 0.14,
        "THEATER COMMANDER POLICY  (pi_C)  [Low Frequency: tau = 5 * dt]",
        "MLP Observation Encoder (Linear 53 -> 128, LayerNorm, ReLU)\n"
        "Recurrent Memory: Gated Recurrent Unit (GRU, 256 Hidden Units)  h_tau = GRU(S_tau, h_{tau-1})\n"
        "Categorical Macro Policy Head: Samples Multi-Domain Directives  g_tau ~ pi_C(g_tau | h_tau)",
        bg_color="#FEFCBF", border_color="#D69E2E", text_color="#744210", title_fontsize=11, sub_fontsize=8.5
    )

    # Directive Flow Arrows
    domain_coords = [
        (0.20, "Air Sub-Goal g_tau^air\n(CAP, Sweep, Escort)", "#3182CE"),
        (0.50, "Ground Sub-Goal g_tau^gnd\n(Advance, Defend, Ambush)", "#38A169"),
        (0.80, "Naval Sub-Goal g_tau^sea\n(Standoff, Screen, Littoral)", "#319795")
    ]
    for xc, lbl, col in domain_coords:
        ax.annotate(
            "", xy=(xc, 0.52), xytext=(xc, 0.63),
            arrowprops=dict(arrowstyle="->", lw=2, color=col, shrinkA=2, shrinkB=2)
        )
        ax.text(xc, 0.575, lbl, ha="center", va="center", fontsize=8, color=col, fontweight="bold")

    # 3. Three Domain Tactical Policy Blocks
    # Air Policy Block
    draw_box(
        ax, 0.05, 0.18, 0.28, 0.34,
        "AIR DOMAIN TACTICAL POLICIES\n(AirFight / AirEscape — 10 Hz)",
        "Local Obs: o_t^air in R^13 (Kinematics, Radar Contacts)\n"
        "───────────────────────────────────\n"
        "Multi-Head Self-Attention Block\n"
        "Scaled Dot-Product Token Invariance:\n"
        "Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V\n"
        "───────────────────────────────────\n"
        "Factorized Multi-Discrete Action Heads:\n"
        "• Heading: 13 Discrete Bins [-180, 180 deg]\n"
        "• Speed: 9 Discrete Bins [150, 600 m/s]\n"
        "• Cannon: Binary [Hold / Fire]\n"
        "• Rocket: Binary [Hold / Fire]\n"
        "Loss: Factorized PPO (epsilon = 0.20)",
        bg_color="#EBF8FF", border_color="#3182CE", text_color="#2B6CB0", title_fontsize=9.5, sub_fontsize=7.5
    )

    # Ground Policy Block
    draw_box(
        ax, 0.36, 0.18, 0.28, 0.34,
        "GROUND DOMAIN TACTICAL POLICIES\n(GroundEngage / Defend — 10 Hz)",
        "Local Obs: o_t^gnd in R^9 (Heading, Speed, Elevation)\n"
        "───────────────────────────────────\n"
        "Multi-Head Self-Attention Block\n"
        "Radar / Visual Masking Analysis\n"
        "Terrain Slope & Cover Ratio Assessment\n"
        "───────────────────────────────────\n"
        "HHAPPO Hybrid Continuous-Discrete Heads:\n"
        "• Continuous Gaussian Head: mu_theta, sigma_theta\n"
        "  - Steering Delta [-30, 30 deg]\n"
        "  - Speed Command [0, 30 m/s]\n"
        "• Discrete Categorical Heads:\n"
        "  - Weapon Select [3 Classes]\n"
        "  - Fire Trigger [Hold / Fire]",
        bg_color="#F0FFF4", border_color="#38A169", text_color="#22543D", title_fontsize=9.5, sub_fontsize=7.5
    )

    # Maritime Policy Block
    draw_box(
        ax, 0.67, 0.18, 0.28, 0.34,
        "MARITIME TACTICAL POLICIES\n(SeaEngage / SeaDefend — 10 Hz)",
        "Local Obs: o_t^sea in R^9 (Heading, Speed, Radar)\n"
        "───────────────────────────────────\n"
        "Multi-Head Self-Attention Block\n"
        "Surface Radar Horizon & Littoral Geometry\n"
        "Fleet Escort Screening Evaluation\n"
        "───────────────────────────────────\n"
        "HHAPPO Hybrid Continuous-Discrete Heads:\n"
        "• Continuous Gaussian Head: mu_theta, sigma_theta\n"
        "  - Rudder Delta [-15, 15 deg]\n"
        "  - Speed Command [0, 20 m/s]\n"
        "• Discrete Categorical Heads:\n"
        "  - Standoff Missile Select [2 Classes]\n"
        "  - Fire Trigger [Hold / Fire]",
        bg_color="#E6FFFA", border_color="#319795", text_color="#234E52", title_fontsize=9.5, sub_fontsize=7.5
    )

    # Bottom Environment Closed Loop
    draw_box(
        ax, 0.05, 0.03, 0.90, 0.10,
        "2.5D MULTI-DOMAIN SIMULATION CLOSED-LOOP ENVIRONMENT (TacticalEnv)",
        "Executes Joint Multi-Agent Actions  a_t = {a_t^air, a_t^gnd, a_t^sea}  |  Advances Kinematics (dt = 0.1 s)\n"
        "Computes Team Advantage (GAE: gamma=0.99, lambda=0.95)  |  Returns Next Partial Observations o_{t+1}^i and Rewards r_t^i",
        bg_color="#1A365D", border_color="#0F243E", text_color="#FFFFFF", title_fontsize=10, sub_fontsize=8
    )

    # Closed loop arrows
    ax.annotate(
        "", xy=(0.19, 0.13), xytext=(0.19, 0.18),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#3182CE", shrinkA=2, shrinkB=2)
    )
    ax.annotate(
        "", xy=(0.50, 0.13), xytext=(0.50, 0.18),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#38A169", shrinkA=2, shrinkB=2)
    )
    ax.annotate(
        "", xy=(0.81, 0.13), xytext=(0.81, 0.18),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#319795", shrinkA=2, shrinkB=2)
    )

    fig.tight_layout()
    out_path = FIG_DIR / "hierarchical_policy_flow.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved hierarchical policy flow diagram: {out_path}")


def generate_latency_benchmark_diagram():
    """Generate Latency Benchmark and Execution Lifecycle Diagram."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    # Categories and latencies
    benchmarks = [
        ("DRDO 50 Hz Frame Budget (Step Period)", 20.0, "#A0AEC0", "Max Allowed Total Step Time"),
        ("DRDO Contractual Inference Ceiling", 2.0, "#E53E3E", "Contractual Requirement (M1 Target <= 2.0 ms)"),
        ("REST HTTP Microservice (FastAPI)", 4.21, "#DD6B20", "Over-Network API Service (HTTP + JSON)"),
        ("Proposed In-Process Direct Adapter", 0.58, "#38A169", "Production In-Process Mode A (71% Under Ceiling)"),
        ("TacticalEnv Kinematics Physics Step", 0.22, "#3182CE", "Dubins Curves, Radar Line-of-Sight, Elevation"),
        ("Complete In-Process Step Cycle", 0.80, "#2B6CB0", "Physics (0.22 ms) + MARL Forward Inference (0.58 ms)")
    ]

    names = [b[0] for b in reversed(benchmarks)]
    times = [b[1] for b in reversed(benchmarks)]
    colors = [b[2] for b in reversed(benchmarks)]
    notes = [b[3] for b in reversed(benchmarks)]

    y_pos = range(len(names))
    bars = ax.barh(y_pos, times, color=colors, height=0.55, edgecolor="#1A202C", linewidth=1.2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5, fontweight="bold", family="sans-serif")
    ax.set_xlabel("Execution Time per Agent / Step (Milliseconds - Logarithmic Scale)", fontsize=10.5, fontweight="bold", family="sans-serif", color="#1A365D")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 35)

    # Add reference dashed line at 2.0 ms ceiling
    ax.axvline(2.0, color="#E53E3E", linestyle="--", linewidth=1.8, zorder=4)
    ax.text(2.05, 4.3, "DRDO Mandatory Ceiling (<= 2.0 ms)", color="#E53E3E", fontsize=9, fontweight="bold", rotation=90, va="top")

    # Add value annotations
    for bar, val, note in zip(bars, times, notes):
        w = bar.get_width()
        ax.text(
            w * 1.12, bar.get_y() + bar.get_height() / 2.0,
            f"{val:.2f} ms  ({note})",
            va="center", ha="left", fontsize=8.5, fontweight="bold", color="#2D3748"
        )

    ax.set_title("OPERATIONAL LATENCY BENCHMARK & EXECUTION SAFETY MARGINS", fontsize=12.5, fontweight="bold", color="#1A365D", pad=15)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#CBD5E0")
    ax.set_facecolor("#F7FAFC")
    fig.patch.set_facecolor("#FFFFFF")

    fig.tight_layout()
    out_path = FIG_DIR / "latency_execution_flow.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved latency benchmark diagram: {out_path}")


if __name__ == "__main__":
    generate_10_layer_architecture_diagram()
    generate_hierarchical_policy_diagram()
    generate_latency_benchmark_diagram()
