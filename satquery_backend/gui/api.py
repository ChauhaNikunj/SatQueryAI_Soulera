"""
satquery_backend/gui/api.py
===========================
Production FastAPI backend for SatQuery AI (SIH PS 26167).
Endpoints:
  • POST /api/analyze          — Universal intent-routed multi-modal endpoint
  • POST /api/vqa              — Single-image VQA & Captioning (Qwen2.5-VL-3B 4-bit)
  • POST /api/change-detection — Bi-temporal change detection & change-VQA (Siamese ResNet18)
  • POST /api/fusion           — Optical-SAR cross-modal fusion (RemoteCLIP + FusionAdapter)
  • GET  /api/traces           — List execution traces
  • GET  /api/traces/{trace_id}— Get single execution trace
  • GET  /api/traces/download  — Download execution_trace.jsonl
  • GET  /api/evidence/{fname} — Fetch visual evidence artifact image
  • GET  /health               — System health and GPU VRAM telemetry
"""

from __future__ import annotations

import io
import os
import sys
import time
import uuid
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# Ensure project roots in sys.path
_HERE = Path(__file__).resolve().parent
_BACKEND_DIR = _HERE.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
for p in [str(_PROJECT_ROOT), str(_BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from satquery_backend.controller.agentic_router import route_query, TaskType
from satquery_backend.controller.logger import log_execution, ExecutionTraceLogger
from satquery_backend.models.unified_inference import qwen_engine, siamese_engine, fusion_engine

app = FastAPI(
    title="SatQuery AI Backend API",
    description="Vision-Language Assistant for Multimodal Remote Sensing Analysis (SIH 26167)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOADS = _PROJECT_ROOT / "scratch" / "uploads"
TEMP_UPLOADS.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR = _PROJECT_ROOT / "dashboard_results"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
TRACE_LOGGER = ExecutionTraceLogger(log_dir=str(_BACKEND_DIR / "logs"))


async def _save_upload(upload: UploadFile) -> str:
    """Save upload to scratch directory and return absolute path."""
    ext = Path(upload.filename or "file.png").suffix.lower()
    if not ext:
        ext = ".png"
    dest = TEMP_UPLOADS / f"upload_{uuid.uuid4().hex[:8]}_{upload.filename}"
    content = await upload.read()
    with open(dest, "wb") as f:
        f.write(content)
    return str(dest)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    import torch
    vram_gb = 0.0
    gpu_name = "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.memory_allocated() / 1e9, 2)

    return {
        "status": "healthy",
        "system": "SatQuery AI (SIH PS 26167)",
        "gpu": gpu_name,
        "vram_allocated_gb": vram_gb,
        "models": {
            "qwen2_5_vl": "Qwen2.5-VL-3B-Instruct (4-bit)",
            "siamese_resnet18": "Siamese ResNet18 + CDVQA Head",
            "optical_sar_fusion": "RemoteCLIP ViT-B/32 + FusionAdapter_v1",
            "agentic_router": "Task 6 Orchestration Controller",
            "trace_logger": "Task 7 Execution Logger",
        },
    }


@app.post("/api/vqa", tags=["Modality 1 & 2"])
async def single_image_vqa(
    file: UploadFile = File(..., description="Single satellite image"),
    query: str = Form(..., description="VQA question or caption steering prompt"),
    task_mode: str = Form("vqa", description="'vqa' or 'caption'"),
):
    """
    Single-image VQA and Captioning via Qwen2.5-VL-3B-Instruct (4-bit).
    """
    saved_path = await _save_upload(file)
    t0 = time.perf_counter()

    res = qwen_engine.predict(saved_path, query=query, task_mode=task_mode)
    latency = round((time.perf_counter() - t0) * 1000, 2)

    task_name = "SINGLE_IMAGE_CAPTIONING" if task_mode == "caption" else "SINGLE_VQA_GROUNDING"
    trace = log_execution(
        task=task_name,
        models_used=["Qwen2.5-VL-3B-Instruct"],
        input_images=[os.path.basename(saved_path)],
        parameters={"query": query, "mode": task_mode},
        outputs=[res["output"]],
        confidence=res["confidence"],
    )

    return {
        "trace_id": trace["trace_id"],
        "task": task_name,
        "model_used": res["model_name"],
        "output": res["output"],
        "confidence": res["confidence"],
        "latency_ms": latency,
        "trace": trace,
    }


@app.post("/api/change-detection", tags=["Modality 3 & 4"])
async def bi_temporal_change_detection(
    file_t1: UploadFile = File(..., description="Pre-event satellite image T1"),
    file_t2: UploadFile = File(..., description="Post-event satellite image T2"),
    query: Optional[str] = Form("", description="Optional change-VQA query"),
):
    """
    Bi-temporal Change Detection & Change-VQA via Siamese ResNet18.
    Returns: change description, VQA answer, and visual evidence change mask.
    """
    path_t1 = await _save_upload(file_t1)
    path_t2 = await _save_upload(file_t2)
    t0 = time.perf_counter()

    res = siamese_engine.predict(path_t1, path_t2, query=query or "")
    latency = round((time.perf_counter() - t0) * 1000, 2)

    trace = log_execution(
        task="BI_TEMPORAL_CHANGE",
        models_used=["SiameseResNet18_CDVQA"],
        input_images=[os.path.basename(path_t1), os.path.basename(path_t2)],
        parameters={"query": query, "method": res.get("parameters", {}).get("method")},
        outputs=[res["output"]],
        confidence=res["confidence"],
    )

    # Convert visual evidence to base64 for direct browser rendering
    viz_b64 = ""
    if res.get("visual_evidence_path") and os.path.exists(res["visual_evidence_path"]):
        with open(res["visual_evidence_path"], "rb") as img_f:
            viz_b64 = base64.b64encode(img_f.read()).decode("utf-8")

    return {
        "trace_id": trace["trace_id"],
        "task": "BI_TEMPORAL_CHANGE",
        "model_used": res["model_name"],
        "output": res["output"],
        "change_description": res["change_description"],
        "vqa_answer": res["vqa_answer"],
        "confidence": res["confidence"],
        "change_detected": res["change_detected"],
        "change_pct": res["change_pct"],
        "visual_evidence_url": f"/api/evidence/{os.path.basename(res['visual_evidence_path'])}",
        "visual_evidence_base64": viz_b64,
        "latency_ms": latency,
        "trace": trace,
    }


@app.post("/api/fusion", tags=["Modality 5"])
async def optical_sar_fusion(
    file_optical: UploadFile = File(..., description="Sentinel-2 Optical image"),
    file_sar: UploadFile = File(..., description="Sentinel-1 SAR image"),
    query: Optional[str] = Form("", description="Cross-modal analysis question"),
):
    """
    Optical-SAR Cross-Modal Fusion via RemoteCLIP + FusionAdapter.
    """
    opt_path = await _save_upload(file_optical)
    sar_path = await _save_upload(file_sar)
    t0 = time.perf_counter()

    res = fusion_engine.predict(opt_path, sar_path, query=query or "")
    latency = round((time.perf_counter() - t0) * 1000, 2)

    trace = log_execution(
        task="OPTICAL_SAR_CROSS_MODAL",
        models_used=["RemoteCLIP-ViT-B/32", "FusionAdapter_v1"],
        input_images=[os.path.basename(opt_path), os.path.basename(sar_path)],
        parameters={"query": query},
        outputs=[res["output"]],
        confidence=res["confidence"],
    )

    return {
        "trace_id": trace["trace_id"],
        "task": "OPTICAL_SAR_CROSS_MODAL",
        "model_used": res["model_name"],
        "output": res["output"],
        "top_class": res["top_class"],
        "top_k_predictions": res["top_k_predictions"],
        "confidence": res["confidence"],
        "latency_ms": latency,
        "trace": trace,
    }


@app.post("/api/analyze", tags=["Agentic Orchestrator"])
async def universal_analyze(
    files: List[UploadFile] = File(..., description="1 or 2 images"),
    query: str = Form(..., description="Natural language question"),
):
    """
    Universal Agentic Router Endpoint (Task 6).
    Automatically routes query + images to Qwen2.5-VL, Siamese ResNet18, or Optical-SAR Fusion.
    """
    saved_paths = []
    for f in files:
        saved_paths.append(await _save_upload(f))

    t0 = time.perf_counter()
    decision = route_query(query, saved_paths)

    output_text = ""
    confidence = 0.0
    model_name = ""
    extra: Dict[str, Any] = {}

    if decision.task_type == TaskType.OPTICAL_SAR_CROSS_MODAL:
        p_opt = saved_paths[0]
        p_sar = saved_paths[1] if len(saved_paths) > 1 else saved_paths[0]
        res = fusion_engine.predict(p_opt, p_sar, query=query)
        output_text = res["output"]
        confidence = res["confidence"]
        model_name = res["model_name"]
        extra = {"top_class": res["top_class"], "top_k": res["top_k_predictions"]}

    elif decision.task_type == TaskType.BI_TEMPORAL_CHANGE:
        p_t1 = saved_paths[0]
        p_t2 = saved_paths[1] if len(saved_paths) > 1 else saved_paths[0]
        res = siamese_engine.predict(p_t1, p_t2, query=query)
        output_text = res["output"]
        confidence = res["confidence"]
        model_name = res["model_name"]
        extra = {
            "change_description": res["change_description"],
            "vqa_answer": res["vqa_answer"],
            "visual_evidence_url": f"/api/evidence/{os.path.basename(res['visual_evidence_path'])}",
        }

    else:
        is_caption = any(k in query.lower() for k in ["caption", "describe", "summary", "overview"])
        res = qwen_engine.predict(saved_paths[0], query=query, task_mode="caption" if is_caption else "vqa")
        output_text = res["output"]
        confidence = res["confidence"]
        model_name = res["model_name"]

    latency = round((time.perf_counter() - t0) * 1000, 2)

    trace = log_execution(
        task=decision.task_type.value,
        models_used=[model_name],
        input_images=[os.path.basename(p) for p in saved_paths],
        parameters={"query": query, "routing_rules": decision.routing_rules},
        outputs=[output_text],
        confidence=confidence,
    )

    return {
        "trace_id": trace["trace_id"],
        "task_type": decision.task_type.value,
        "routing_rules": decision.routing_rules,
        "confidence": confidence,
        "model_name": model_name,
        "output": output_text,
        "latency_ms": latency,
        "trace": trace,
        "details": extra,
    }


@app.get("/api/evidence/{filename}", tags=["Evidence"])
async def get_evidence_image(filename: str):
    """Fetch visual evidence change mask or comparison image."""
    fpath = EVIDENCE_DIR / filename
    if not fpath.exists():
        raise HTTPException(404, detail="Evidence artifact not found.")
    return FileResponse(str(fpath), media_type="image/png")


@app.get("/api/traces", tags=["Audit"])
async def list_traces():
    """Retrieve all execution traces."""
    traces = TRACE_LOGGER.read_all()
    return {"total": len(traces), "traces": traces}


@app.get("/api/traces/{trace_id}", tags=["Audit"])
async def get_trace(trace_id: str):
    """Retrieve single execution trace by ID."""
    traces = TRACE_LOGGER.read_all()
    for t in traces:
        if t.get("trace_id") == trace_id:
            return t
    raise HTTPException(404, detail=f"Trace {trace_id} not found.")


@app.get("/api/traces/download", tags=["Audit"])
async def download_traces():
    """Download full execution_trace.jsonl file."""
    root_trace = _PROJECT_ROOT / "execution_trace.jsonl"
    backend_trace = Path(TRACE_LOGGER.log_path)
    target = root_trace if root_trace.exists() else backend_trace
    if not target.exists():
        raise HTTPException(404, detail="No execution traces logged yet.")
    return FileResponse(str(target), filename="execution_trace.jsonl", media_type="application/jsonlines")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("satquery_backend.gui.api:app", host="0.0.0.0", port=8001, reload=True)
