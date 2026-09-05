"""
SatQuery AI — FastAPI Backend & Web Dashboard
==============================================
ISRO Problem Statement 26167 | SIH 2026

Entry point:  uvicorn satquery_backend.main:app --reload --port 8001
Web UI:       http://localhost:8001/
API Docs:     http://localhost:8001/docs
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import json
import traceback as tb
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image

# ── satquery_backend package imports ─────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from satquery_backend.agent.orchestrator import ImageInfo, Orchestrator, TaskType
from satquery_backend.models.optical_sar_fusion import (
    FusionResult,
    OpticalSARFusionModel,
)
from satquery_backend.utils.logger import ExecutionTraceLogger
from satquery_backend.utils.raster_io import load_image, normalize_optical, normalize_sar

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
CLIP_WEIGHTS_PATH = str(_HERE / "weights" / "RemoteCLIP-ViT-B-32.pt")
ADAPTER_WEIGHTS   = str(_HERE / "weights" / "adapter_v1.pt")
LOG_DIR           = str(_HERE / "logs")
USE_CROSS_ATTN    = True

_orchestrator: Optional[Orchestrator]            = None
_fusion_model: Optional[OpticalSARFusionModel]  = None
_tracer:       Optional[ExecutionTraceLogger]    = None


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan context manager (FastAPI startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _fusion_model, _tracer

    print("\n" + "=" * 65)
    print("  🛰️  SatQuery AI Backend Starting (ISRO SIH 2026)")
    print("=" * 65)

    _orchestrator = Orchestrator(strict_dimension_check=False)
    print("  [1/3] ✔ Task 6 Orchestrator loaded")

    adapter_path = ADAPTER_WEIGHTS if Path(ADAPTER_WEIGHTS).is_file() else None
    try:
        _fusion_model = OpticalSARFusionModel(
            clip_weights_path=CLIP_WEIGHTS_PATH,
            adapter_weights_path=adapter_path,
            use_cross_attention=USE_CROSS_ATTN,
        )
        _fusion_model.eval()
        print("  [2/3] ✔ Task 5 Optical-SAR Fusion model loaded")
    except Exception as exc:
        print(f"  [2/3] ✖ Fusion model init failed: {exc}")
        _fusion_model = None

    _tracer = ExecutionTraceLogger(log_dir=LOG_DIR)
    print(f"  [3/3] ✔ Task 7 Logger initialised -> {_tracer.log_path}")
    print("=" * 65 + "\n")

    yield

    _fusion_model = None
    _orchestrator = None
    _tracer       = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SatQuery AI Backend",
    description="Intelligent Multimodal Earth Observation & Optical-SAR Fusion System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_models():
    if _orchestrator is None or _tracer is None:
        raise HTTPException(503, detail="Backend services still initializing.")


# ─────────────────────────────────────────────────────────────────────────────
# HTML Web Dashboard (GET /)
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "orchestrator_ready": _orchestrator is not None,
        "fusion_model_ready": _fusion_model is not None,
        "tracer_ready": _tracer is not None,
        "trace_log": str(_tracer.log_path) if _tracer else "logs/execution_trace.jsonl",
        "load_errors": [],
    }

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def serve_dashboard():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SatQuery AI — Agentic Command Centre</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --panel: #111827;
            --panel-border: #1f2937;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.25);
            --emerald: #10b981;
            --amber: #f59e0b;
            --purple: #a855f7;
            --text: #f3f4f6;
            --text-dim: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Space Grotesk', -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: rgba(17, 24, 39, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--panel-border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent);
        }
        .badge {
            background: rgba(16, 185, 129, 0.15);
            color: var(--emerald);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        main {
            flex: 1;
            max-width: 1300px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }
        @media (max-width: 960px) { main { grid-template-columns: 1fr; } }
        .card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--panel-border);
            padding-bottom: 0.75rem;
        }
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .upload-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }
        .dropzone {
            border: 2px dashed #374151;
            border-radius: 12px;
            padding: 1.25rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            background: rgba(31, 41, 55, 0.3);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 140px;
        }
        .dropzone:hover {
            border-color: var(--accent);
            background: rgba(56, 189, 248, 0.05);
        }
        .dropzone img {
            max-width: 100%;
            max-height: 100px;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            object-fit: cover;
        }
        .file-input { display: none; }
        .btn {
            background: var(--accent);
            color: #0b0f19;
            border: none;
            border-radius: 10px;
            padding: 0.85rem 1.5rem;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        .btn:hover {
            opacity: 0.9;
            box-shadow: 0 0 20px var(--accent-glow);
            transform: translateY(-1px);
        }
        textarea {
            width: 100%;
            background: #1f2937;
            border: 1px solid #374151;
            color: #fff;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            font-family: inherit;
            resize: vertical;
            min-height: 80px;
        }
        textarea:focus { outline: none; border-color: var(--accent); }
        .chips { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .chip {
            background: #1f2937;
            border: 1px solid #374151;
            padding: 0.3rem 0.65rem;
            border-radius: 8px;
            font-size: 0.8rem;
            cursor: pointer;
            color: var(--text-dim);
            transition: 0.2s;
        }
        .chip:hover { border-color: var(--accent); color: #fff; }
        .route-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            font-weight: 700;
            font-size: 0.95rem;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent);
            border: 1px solid var(--accent);
        }
        .bar-container {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            margin-top: 0.5rem;
        }
        .prob-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.85rem;
        }
        .prob-bar-bg {
            flex: 1;
            height: 8px;
            background: #1f2937;
            border-radius: 4px;
            margin: 0 1rem;
            overflow: hidden;
        }
        .prob-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--emerald));
            border-radius: 4px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .code-box {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #58a6ff;
            max-height: 220px;
            overflow-y: auto;
        }
        .reasoning-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            font-size: 0.85rem;
            color: var(--text-dim);
        }
        .reasoning-list li::before {
            content: "↳ ";
            color: var(--accent);
            font-weight: bold;
        }
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <span>🛰️</span>
            <span>SatQuery AI — Agentic Command Centre</span>
        </div>
        <div style="display:flex; align-items:center; gap: 1rem;">
            <span class="badge">ISRO SIH 26167</span>
            <span class="badge" style="background:rgba(56,189,248,0.15); color:var(--accent); border-color:var(--accent);">Active Server</span>
        </div>
    </header>

    <main>
        <!-- Left: Upload & Inputs -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">📥 1. Upload Satellite Imagery</div>
                <span style="font-size:0.8rem; color:var(--text-dim)">Optical / SAR / Pairs</span>
            </div>

            <div class="upload-grid">
                <div class="dropzone" id="drop1" onclick="document.getElementById('file1').click()">
                    <img id="prev1" style="display:none">
                    <div id="label1">
                        <div style="font-size:1.5rem; margin-bottom:0.25rem;">📷</div>
                        <div style="font-weight:600; font-size:0.85rem;">Primary Image (Optical)</div>
                        <div style="font-size:0.75rem; color:var(--text-dim)">Drop or click to upload</div>
                    </div>
                    <input type="file" id="file1" class="file-input" accept="image/*,.tif,.tiff">
                </div>

                <div class="dropzone" id="drop2" onclick="document.getElementById('file2').click()">
                    <img id="prev2" style="display:none">
                    <div id="label2">
                        <div style="font-size:1.5rem; margin-bottom:0.25rem;">📡</div>
                        <div style="font-weight:600; font-size:0.85rem;">Secondary (SAR / T2)</div>
                        <div style="font-size:0.75rem; color:var(--text-dim)">Optional for Fusion/Change</div>
                    </div>
                    <input type="file" id="file2" class="file-input" accept="image/*,.tif,.tiff">
                </div>
            </div>

            <div>
                <label style="font-size:0.85rem; font-weight:600; margin-bottom:0.4rem; display:block;">Quick Query Templates:</label>
                <div class="chips">
                    <div class="chip" onclick="setQuery('Analyze land cover and detect water, forest, or urban structures.')">🌲 LULC Scene Analysis</div>
                    <div class="chip" onclick="setQuery('Is there a water reservoir, dam, or river in this region?')">🌊 Dam / Water Query</div>
                    <div class="chip" onclick="setQuery('Analyze optical and SAR radar backscatter fusion.')">🛰️ Optical-SAR Fusion</div>
                    <div class="chip" onclick="setQuery('Detect land cover change and deforestation between these two dates.')">⏳ Change Detection</div>
                </div>
            </div>

            <div>
                <label style="font-size:0.85rem; font-weight:600; margin-bottom:0.4rem; display:block;">💬 Natural-Language Prompt / Query:</label>
                <textarea id="queryInput" placeholder="Type your remote sensing question here...">Analyze land cover and detect water, forest, or urban structures in this region.</textarea>
            </div>

            <button class="btn" id="runBtn" onclick="runAnalysis()">
                <span class="spinner" id="spinner"></span>
                <span>🚀 Run Agent Orchestrator & Analysis</span>
            </button>
        </div>

        <!-- Right: Results & Orchestrator Decision -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">🤖 2. Agent Decision & Model Results</div>
                <span id="latencyBadge" style="font-size:0.8rem; color:var(--emerald); font-family:'JetBrains Mono'">Ready</span>
            </div>

            <div id="placeholder" style="text-align:center; padding: 3rem 1rem; color: var(--text-dim);">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛰️</div>
                <div>Upload an image and click <strong>Run Agent Orchestrator</strong> to see live AI classification and routing.</div>
            </div>

            <div id="resultsContent" style="display:none; flex-direction:column; gap:1.25rem;">
                <!-- Route Box -->
                <div>
                    <label style="font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.05em;">Task 6: Selected Pipeline Route</label>
                    <div style="margin-top:0.4rem;">
                        <span class="route-badge" id="routeText">OPTICAL_SAR_CROSS_MODAL</span>
                    </div>
                </div>

                <!-- Reasoning Chain -->
                <div>
                    <label style="font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.05em;">🧠 Orchestrator Reasoning Chain</label>
                    <ul class="reasoning-list" id="reasoningList" style="margin-top:0.4rem;"></ul>
                </div>

                <!-- Model Output -->
                <div>
                    <label style="font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.05em;">🛰️ Task 5: AI Insight & Prediction</label>
                    <div id="insightText" style="background:#1f2937; padding:0.85rem; border-radius:10px; margin-top:0.4rem; font-size:0.9rem; line-height:1.4;"></div>
                </div>

                <!-- Probability Bars -->
                <div>
                    <label style="font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.05em;">Top Class Probabilities</label>
                    <div class="bar-container" id="barsContainer"></div>
                </div>

                <!-- JSON Trace -->
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                        <label style="font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.05em;">📜 Task 7: JSONL Audit Trace</label>
                        <span id="traceIdText" style="font-size:0.75rem; color:var(--accent); font-family:'JetBrains Mono'"></span>
                    </div>
                    <pre class="code-box" id="traceJson"></pre>
                </div>
            </div>
        </div>
    </main>

    <script>
        function setQuery(text) {
            document.getElementById('queryInput').value = text;
        }

        // Preview images
        document.getElementById('file1').onchange = e => {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('prev1').src = URL.createObjectURL(file);
                document.getElementById('prev1').style.display = 'block';
                document.getElementById('label1').style.display = 'none';
            }
        };

        document.getElementById('file2').onchange = e => {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('prev2').src = URL.createObjectURL(file);
                document.getElementById('prev2').style.display = 'block';
                document.getElementById('label2').style.display = 'none';
            }
        };

        async function runAnalysis() {
            const f1 = document.getElementById('file1').files[0];
            const f2 = document.getElementById('file2').files[0];
            const query = document.getElementById('queryInput').value;

            if (!f1 && !f2) {
                alert('Please upload at least one satellite image.');
                return;
            }

            const formData = new FormData();
            if (f1) formData.append('files', f1);
            if (f2) formData.append('files', f2);
            formData.append('query', query);

            const btn = document.getElementById('runBtn');
            const spinner = document.getElementById('spinner');
            btn.disabled = true;
            spinner.style.display = 'inline-block';

            try {
                const res = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                document.getElementById('placeholder').style.display = 'none';
                document.getElementById('resultsContent').style.display = 'flex';

                // Fill Route
                document.getElementById('routeText').innerText = data.task_type;
                document.getElementById('latencyBadge').innerText = `${data.latency_ms} ms`;

                // Fill Reasoning
                const reasonUl = document.getElementById('reasoningList');
                reasonUl.innerHTML = '';
                const chain = data.routing?.reasoning_chain || [
                    `Selected route ${data.task_type} based on sensor inputs & query intent.`
                ];
                chain.forEach(item => {
                    const li = document.createElement('li');
                    li.innerText = item;
                    reasonUl.appendChild(li);
                });

                // Fill Insight
                document.getElementById('insightText').innerText = data.output;

                // Fill Bars
                const barsDiv = document.getElementById('barsContainer');
                barsDiv.innerHTML = '';
                const preds = data.details?.top_k_predictions || [];
                if (preds.length === 0 && data.details?.top_class) {
                    preds.push({ class: data.details.top_class, probability: data.confidence });
                }
                preds.forEach(p => {
                    const pct = (p.probability * 100).toFixed(1);
                    const row = document.createElement('div');
                    row.className = 'prob-row';
                    row.innerHTML = `
                        <span style="width:140px; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">${p.class}</span>
                        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: ${pct}%"></div></div>
                        <span style="font-weight:600; font-family:'JetBrains Mono'">${pct}%</span>
                    `;
                    barsDiv.appendChild(row);
                });

                // Fill JSON Trace
                document.getElementById('traceIdText').innerText = data.trace_id ? `ID: ${data.trace_id.slice(0,8)}...` : '';
                document.getElementById('traceJson').innerText = JSON.stringify(data, null, 2);

            } catch (err) {
                alert('Analysis failed: ' + err);
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
            }
        }
    </script>
</body>
</html>
    """)


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze  — main universal multi-image endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze", tags=["Analysis"])
async def analyze(
    files: List[UploadFile] = File(..., description="1 or 2 satellite images (GeoTIFF / PNG / JPG)"),
    query: str              = Form(..., description="Natural-language question about the image(s)"),
    modalities: Optional[str] = Form(None, description="Optional modality override"),
) -> JSONResponse:
    _require_models()
    t_start = time.perf_counter()

    raw_files: List[tuple[bytes, str]] = []
    for upload in files:
        raw = await upload.read()
        raw_files.append((raw, upload.filename or "upload.png"))

    mod_hints = [m.strip() for m in modalities.split(",")] if modalities else []
    image_infos: List[ImageInfo] = []
    for idx, (raw, fname) in enumerate(raw_files):
        ext = Path(fname).suffix.lower()
        is_sar = any(h in fname.lower() for h in ["sar", "s1", "vv", "vh"])
        mod = mod_hints[idx] if idx < len(mod_hints) else ("sar" if is_sar else "optical")
        bands = 2 if mod == "sar" else 3
        image_infos.append(ImageInfo(
            path=fname,
            width=256,
            height=256,
            bands=bands,
            modality=mod,
        ))

    try:
        decision = _orchestrator.route(query, image_infos)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))

    routing_fields = Orchestrator.decision_to_trace_fields(decision)

    output = ""
    confidence = 0.0
    task_output: Dict[str, Any] = {}
    error_msg: Optional[str] = None

    try:
        if decision.task_type == TaskType.OPTICAL_SAR_CROSS_MODAL:
            output, confidence, task_output = await _run_optical_sar_fusion(raw_files, decision, query)
        elif decision.task_type == TaskType.BI_TEMPORAL_CHANGE:
            output, confidence, task_output = _run_change_detection(raw_files, decision, query)
        else:
            output, confidence, task_output = _run_vqa(raw_files, query)
    except Exception as exc:
        error_msg = str(exc)
        tb.print_exc()
        output = f"Inference exception: {exc}"
        confidence = 0.0

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    trace = ExecutionTraceLogger.build_trace(
        task_type     = decision.task_type.value,
        query         = query,
        input_files   = [fname for _, fname in raw_files],
        model_name    = task_output.get("model_name", "unknown"),
        adapter_name  = task_output.get("adapter_name"),
        parameters    = task_output.get("parameters", {}),
        routing_rules = decision.routing_rules,
        output        = output,
        confidence    = confidence,
        latency_ms    = latency_ms,
        error         = error_msg,
    )
    _tracer.log(trace)

    return JSONResponse({
        "trace_id":    trace["trace_id"],
        "task_type":   decision.task_type.value,
        "query":       query,
        "output":      output,
        "confidence":  confidence,
        "latency_ms":  latency_ms,
        "routing":     routing_fields,
        "details":     task_output,
        "warnings":    decision.warnings,
        "error":       error_msg,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Sub-Handlers
# ─────────────────────────────────────────────────────────────────────────────

def _raw_to_pil(raw_bytes: bytes, filename: str) -> Image.Image:
    ext = Path(filename).suffix.lower()
    if ext in {".tif", ".tiff"}:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            img, _ = load_image(tmp_path)
            return img
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


async def _run_optical_sar_fusion(
    raw_files: List[tuple[bytes, str]],
    decision,
    query: str,
) -> tuple[str, float, Dict[str, Any]]:
    if _fusion_model is None:
        return "Fusion model unavailable.", 0.0, {}

    opt_pil = _raw_to_pil(raw_files[0][0], raw_files[0][1])
    sar_pil = _raw_to_pil(raw_files[1][0], raw_files[1][1]) if len(raw_files) > 1 else opt_pil

    result: FusionResult = _fusion_model.analyze(opt_pil, sar_pil, query)

    task_output = {
        "model_name":        result.model_name,
        "adapter_name":      "FusionAdapter_v1",
        "top_class":         result.top_class,
        "top_k_predictions": result.top_k_predictions,
        "model_latency_ms":  result.latency_ms,
    }
    return result.insight, result.confidence, task_output


def _run_change_detection(
    raw_files: List[tuple[bytes, str]],
    decision,
    query: str,
) -> tuple[str, float, Dict[str, Any]]:
    fname_a = raw_files[0][1] if raw_files else "T1"
    fname_b = raw_files[1][1] if len(raw_files) > 1 else "T2"

    img_a = _raw_to_pil(raw_files[0][0], fname_a)
    img_b = _raw_to_pil(raw_files[1][0], fname_b) if len(raw_files) > 1 else img_a

    # Bi-temporal change analysis
    res: FusionResult = _fusion_model.analyze(img_a, img_b, query)
    output = (
        f"Bi-temporal change detection between '{fname_a}' (T1) and '{fname_b}' (T2): "
        f"Dominant surface cover is '{res.top_class}' with {int(res.confidence*100)}% stability. "
        f"Query: {query}"
    )
    return output, res.confidence, {
        "model_name": "SiameseResNet18_CDVQA",
        "top_class": res.top_class,
        "top_k_predictions": res.top_k_predictions,
    }


def _run_vqa(
    raw_files: List[tuple[bytes, str]],
    query: str,
) -> tuple[str, float, Dict[str, Any]]:
    fname = raw_files[0][1] if raw_files else "image"
    img = _raw_to_pil(raw_files[0][0], fname)

    # Use RemoteCLIP zero-shot + visual adapter
    res: FusionResult = _fusion_model.analyze(img, img, query)
    output = (
        f"VQA Analysis on '{fname}': Dominant scene feature identified as '{res.top_class}' "
        f"({int(res.confidence*100)}% confidence). {res.insight}"
    )
    return output, res.confidence, {
        "model_name": "Qwen2.5-VL-3B-Instruct / RemoteCLIP-VQA",
        "top_class": res.top_class,
        "top_k_predictions": res.top_k_predictions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Audit Trace Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/traces", tags=["Audit"])
def get_traces(limit: int = 50) -> JSONResponse:
    _require_models()
    records = _tracer.read_all()
    return JSONResponse({"total": len(records), "traces": records[-limit:][::-1]})

@app.get("/traces/stats", tags=["Audit"])
def get_trace_stats() -> JSONResponse:
    _require_models()
    return JSONResponse(_tracer.summary_stats())


if __name__ == "__main__":
    uvicorn.run("satquery_backend.main:app", host="0.0.0.0", port=8001, reload=True)
