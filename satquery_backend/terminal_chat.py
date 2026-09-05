"""
SatQuery AI — God-Level Interactive Terminal Command-Centre
===========================================================
Live CLI interface executing:
  1. Task 6: God-Level Agent Orchestrator (Intent Analysis, Dual-Image Routing & Reasoning)
  2. Task 5: Optical-SAR Fusion & Satellite Model (RemoteCLIP + FusionAdapter / EuroSAT Adapter)
  3. Task 7: JSONL Audit Logger (Trace ID & Session Provenance)

Usage:
    python satquery_backend/terminal_chat.py
"""

from __future__ import annotations

import sys
import os
import time
import json
import warnings
from pathlib import Path

# Suppress warnings for clean UI
warnings.filterwarnings("ignore")

# Setup root path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from PIL import Image
import torch
import torch.nn.functional as F

from satquery_backend.agent.orchestrator import Orchestrator, ImageInfo
from satquery_backend.models.optical_sar_fusion import OpticalSARFusionModel
from satquery_backend.utils.logger import ExecutionTraceLogger
from satquery_backend.utils.raster_io import load_image

# ── ANSI Color Codes ────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    RED     = "\033[91m"
    GRAY    = "\033[90m"

_HERE = Path(__file__).resolve().parent
CLIP_WEIGHTS_PATH = str(_HERE / "weights" / "RemoteCLIP-ViT-B-32.pt")
ADAPTER_WEIGHTS   = str(_HERE / "weights" / "adapter_v1.pt")
SAT_BENCH_WEIGHTS = str(_HERE / "weights" / "satellite_benchmark_adapter.pt")
LOG_DIR           = str(_HERE / "logs")

# Built-in Satellite Presets for instant testing
PRESETS = {
    "1": ("Dense Forest (Sentinel-2 Optical)", "satquery_backend/data/eurosat/2750/Forest/Forest_1.jpg", "optical", 3),
    "2": ("Dam / Sea Lake Reservoir (Sentinel-2 Optical)", "satquery_backend/data/eurosat/2750/SeaLake/SeaLake_1.jpg", "optical", 3),
    "3": ("River & Water Channel (Sentinel-2 Optical)", "satquery_backend/data/eurosat/2750/River/River_1.jpg", "optical", 3),
    "4": ("Urban Residential Fabric (Sentinel-2 Optical)", "satquery_backend/data/eurosat/2750/Residential/Residential_1.jpg", "optical", 3),
    "5": ("USGS Landsat Optical Sample (GeoTIFF)", "satquery_backend/sample_data/optical/usgs_landsat_sample.tif", "optical", 3),
    "6": ("Sentinel-1 SAR Radar Sample (VV/VH GeoTIFF)", "satquery_backend/sample_data/sar/sentinel1_coreg_sample.tif", "sar", 2),
}

def print_header():
    print(f"\n{C.CYAN}{C.BOLD}{'='*76}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}   🛰️   SATQUERY AI  —  GOD-LEVEL AGENTIC COMMAND CENTRE & VQA   🛰️{C.RESET}")
    print(f"{C.GRAY}   ISRO SIH 2026 | PS 26167 | Multi-Sensor Orchestration • Fusion • Audit Logger{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{'='*76}{C.RESET}\n")

def make_bar(prob: float, length: int = 24) -> str:
    filled = int(round(prob * length))
    empty = length - filled
    bar = f"{C.GREEN}{'█' * filled}{C.GRAY}{'░' * empty}{C.RESET}"
    return bar

def resolve_image_choice(user_input: str) -> tuple[str, str, int] | None:
    """Returns (filepath, modality, bands) or None."""
    val = user_input.strip()
    if not val:
        return None
    if val in PRESETS:
        name, path, mod, bands = PRESETS[val]
        return path, mod, bands
    if Path(val).exists():
        ext = Path(val).suffix.lower()
        mod = "sar" if any(h in val.lower() for h in ["sar", "s1", "vv", "vh"]) else "optical"
        bands = 2 if mod == "sar" else 3
        return val, mod, bands
    return None

def main():
    print_header()

    # Step 0: Initialize System Components
    print(f"{C.BOLD}[System Initialization]{C.RESET}")
    print(f"  {C.YELLOW}►{C.RESET} Initializing God-Level Agent Orchestrator...")
    orch = Orchestrator(strict_dimension_check=False)
    print(f"    {C.GREEN}✔ Orchestrator Ready{C.RESET} (150+ Semantic Remote Sensing Rule & Intent Engine)")

    print(f"  {C.YELLOW}►{C.RESET} Loading Task 5 Optical-SAR Fusion Model (RemoteCLIP + FusionAdapter)...")
    adapter_path = ADAPTER_WEIGHTS if Path(ADAPTER_WEIGHTS).is_file() else None
    fusion = OpticalSARFusionModel(
        clip_weights_path=CLIP_WEIGHTS_PATH,
        adapter_weights_path=adapter_path,
    )
    fusion.eval()
    print(f"    {C.GREEN}✔ RemoteCLIP ViT-B/32 + FusionAdapter Ready{C.RESET}")

    print(f"  {C.YELLOW}►{C.RESET} Initializing Task 7 Execution Logger...")
    logger = ExecutionTraceLogger(log_dir=LOG_DIR)
    print(f"    {C.GREEN}✔ Audit Logger Ready{C.RESET} -> {C.GRAY}{logger.log_path}{C.RESET}\n")

    # Main Interactive Loop
    while True:
        print(f"\n{C.CYAN}{'─'*76}{C.RESET}")
        print(f"{C.BOLD}📥 STEP 1: Select Input Satellite Images{C.RESET}")
        print(f"{C.GRAY}Available Presets:{C.RESET}")
        for k, (name, path, mod, _) in PRESETS.items():
            exists = "✔" if Path(path).exists() else "✖"
            print(f"  {C.YELLOW}[{k}]{C.RESET} {name} {C.GRAY}({exists}){C.RESET}")
        print(f"  {C.YELLOW}[P]{C.RESET} Enter custom file path")
        print(f"  {C.YELLOW}[Q]{C.RESET} Quit terminal")

        # ── Primary Image Selection ──
        try:
            choice1 = input(f"\n{C.BOLD}► Select Primary Image 1 [1-6 or path] (or Q to exit): {C.RESET}").strip()
        except EOFError:
            break

        if choice1.lower() in ("q", "quit", "exit"):
            print(f"\n{C.CYAN}Exiting SatQuery Command Centre. Jai Hind! 🇮🇳{C.RESET}\n")
            break

        if choice1.lower() == "p":
            choice1 = input("Enter image 1 file path: ").strip()

        img1_info = resolve_image_choice(choice1)
        if not img1_info or not Path(img1_info[0]).exists():
            print(f"{C.RED}Error: Image 1 not found or invalid: '{choice1}'{C.RESET}")
            continue

        path1, mod1, bands1 = img1_info
        print(f"  {C.GREEN}✔ Image 1 Loaded:{C.RESET} {Path(path1).name} ({mod1.upper()}, {bands1} bands)")

        # ── Secondary Image Selection ──
        try:
            choice2 = input(f"{C.BOLD}► Select Secondary Image 2 (e.g. [6] for SAR Radar, [3] for River, or press Enter for Single Image): {C.RESET}").strip()
        except EOFError:
            choice2 = ""

        images_list: list[ImageInfo] = [
            ImageInfo(path=path1, width=256, height=256, bands=bands1, modality=mod1)
        ]

        if choice2:
            if choice2.lower() == "p":
                choice2 = input("Enter image 2 file path: ").strip()
            img2_info = resolve_image_choice(choice2)
            if img2_info and Path(img2_info[0]).exists():
                path2, mod2, bands2 = img2_info
                images_list.append(
                    ImageInfo(path=path2, width=256, height=256, bands=bands2, modality=mod2)
                )
                print(f"  {C.GREEN}✔ Image 2 Loaded:{C.RESET} {Path(path2).name} ({mod2.upper()}, {bands2} bands)")
            else:
                print(f"  {C.YELLOW}Notice: Image 2 '{choice2}' not found. Proceeding with single image.{C.RESET}")

        # ── Natural language query ──
        default_query = "Analyze land cover and detect water, forest, or urban structures in this region."
        try:
            query = input(f"\n{C.BOLD}💬 Enter your Satellite Query / Prompt:{C.RESET}\n{C.GRAY}(Default: '{default_query}'){C.RESET}\n> ").strip()
        except EOFError:
            query = default_query

        if not query:
            query = default_query

        # ─────────────────────────────────────────────────────────────────
        # TASK 6: AGENT ORCHESTRATION (GOD-LEVEL INTENT & ROUTE)
        # ─────────────────────────────────────────────────────────────────
        print(f"\n{C.MAGENTA}{C.BOLD}════════════════════════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.MAGENTA}{C.BOLD} 🤖 [TASK 6: GOD-LEVEL AGENT ORCHESTRATION & REASONING] {C.RESET}")
        print(f"{C.MAGENTA}{C.BOLD}════════════════════════════════════════════════════════════════════════════{C.RESET}")

        t_orch0 = time.perf_counter()
        decision = orch.route(query, images_list)
        t_orch = (time.perf_counter() - t_orch0) * 1000

        print(f"  {C.BOLD}Total Input Images   :{C.RESET} {C.YELLOW}{len(images_list)}{C.RESET} {C.GRAY}({[img.basename for img in images_list]}){C.RESET}")
        print(f"  {C.BOLD}Selected Route       :{C.RESET} {C.GREEN}{C.BOLD}{decision.task_type.value}{C.RESET}")
        print(f"  {C.BOLD}Confidence Score     :{C.RESET} {C.CYAN}{decision.confidence*100:.1f}%{C.RESET}")
        print(f"  {C.BOLD}Target Model Pipeline:{C.RESET} {C.YELLOW}{decision.target_model_pipeline}{C.RESET}")
        print(f"  {C.BOLD}Fired Routing Rules  :{C.RESET} {decision.routing_rules}")
        print(f"  {C.BOLD}Orchestrator Latency :{C.RESET} {t_orch:.2f} ms")

        if decision.reasoning_chain:
            print(f"\n  {C.BOLD}🧠 Orchestrator Reasoning Chain:{C.RESET}")
            for step in decision.reasoning_chain:
                print(f"    {C.GRAY}↳ {step}{C.RESET}")

        # ─────────────────────────────────────────────────────────────────
        # TASK 5: OPTICAL-SAR FUSION & SCENE ANALYSIS
        # ─────────────────────────────────────────────────────────────────
        print(f"\n{C.BLUE}{C.BOLD}════════════════════════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BLUE}{C.BOLD} 🛰️ [TASK 5: OPTICAL-SAR MODEL FUSION & REMOTE SENSING INFERENCE] {C.RESET}")
        print(f"{C.BLUE}{C.BOLD}════════════════════════════════════════════════════════════════════════════{C.RESET}")

        try:
            opt_path = images_list[0].path
            sar_path = images_list[1].path if len(images_list) > 1 else images_list[0].path

            opt_img, _ = load_image(opt_path)
            sar_img, _ = load_image(sar_path)

            result = fusion.analyze(opt_img, sar_img, prompt=query, top_k=5)

            print(f"  {C.BOLD}Insight Summary :{C.RESET} {result.insight}")
            print(f"  {C.BOLD}Primary Class   :{C.RESET} {C.YELLOW}{C.BOLD}{result.top_class}{C.RESET}")
            print(f"  {C.BOLD}Confidence      :{C.RESET} {C.GREEN}{result.confidence*100:.2f}%{C.RESET}")
            print(f"  {C.BOLD}Inference Time  :{C.RESET} {result.latency_ms:.1f} ms")
            print(f"  {C.BOLD}Active Model    :{C.RESET} {result.model_name}")

            print(f"\n  {C.BOLD}Top Predictions & Probabilities:{C.RESET}")
            for pred in result.top_k_predictions:
                p = pred["probability"]
                bar = make_bar(p, length=20)
                print(f"    • {pred['class']:<36} {bar}  {C.BOLD}{p*100:>5.1f}%{C.RESET}")

        except Exception as e:
            print(f"{C.RED}Fusion execution error: {e}{C.RESET}")
            continue

        # ─────────────────────────────────────────────────────────────────
        # TASK 7: JSONL AUDIT LOGGER
        # ─────────────────────────────────────────────────────────────────
        print(f"\n{C.YELLOW}{C.BOLD}════════════════════════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.YELLOW}{C.BOLD} 📜 [TASK 7: AUDIT LOGGING & SESSION PROVENANCE] {C.RESET}")
        print(f"{C.YELLOW}{C.BOLD}════════════════════════════════════════════════════════════════════════════{C.RESET}")

        trace = logger.build_trace(
            task_type     = decision.task_type.value,
            query         = query,
            input_files   = [img.path for img in images_list],
            model_name    = result.model_name,
            adapter_name  = "FusionAdapter_v1",
            parameters    = {"top_k": 5},
            routing_rules = decision.routing_rules,
            output        = result.insight,
            confidence    = result.confidence,
            latency_ms    = round(result.latency_ms + t_orch, 2),
        )
        logger.log(trace)

        print(f"  {C.BOLD}Trace ID        :{C.RESET} {C.CYAN}{trace['trace_id']}{C.RESET}")
        print(f"  {C.BOLD}Schema Version  :{C.RESET} {trace.get('schema_version', 'v1.0')}")
        print(f"  {C.BOLD}Logged Path     :{C.RESET} {logger.log_path}")
        print(f"  {C.BOLD}Log Status      :{C.RESET} {C.GREEN}✔ RECORDED TO AUDIT TRACE JSONL{C.RESET}")

        # Show pretty snippet
        print(f"\n  {C.GRAY}JSON Trace Preview:{C.RESET}")
        snippet = {
            "trace_id": trace["trace_id"],
            "task_type": trace["task_type"],
            "input_files": trace["input_files"],
            "top_class": result.top_class,
            "confidence": f"{result.confidence*100:.1f}%",
            "latency_ms": trace["latency_ms"],
            "timestamp": trace["timestamp"],
        }
        print(f"  {C.GRAY}{json.dumps(snippet, indent=4)}{C.RESET}")

        try:
            input(f"\n{C.BOLD}Press [Enter] to analyze another image...{C.RESET}")
        except EOFError:
            break

if __name__ == "__main__":
    main()
