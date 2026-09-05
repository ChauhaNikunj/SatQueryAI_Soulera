# SatQuery AI — Backend

**ISRO Problem Statement 26167 | SIH 2026**  
Domain-adapted Vision-Language framework for multimodal remote-sensing image analysis.

## Structure

```
satquery_backend/
├── main.py                      # FastAPI entry point
├── requirements.txt
├── .env.example                 # Environment variable reference
├── weights/                     # Put RemoteCLIP-ViT-B-32.pt here
├── logs/                        # execution_trace.jsonl auto-generated here
├── models/
│   └── optical_sar_fusion.py    # Task 5 — RemoteCLIP + FusionAdapter
├── agent/
│   └── orchestrator.py          # Task 6 — Agentic Router
└── utils/
    ├── logger.py                # Task 7 — JSONL Trace Logger
    └── raster_io.py             # GeoTIFF / Image I/O
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r satquery_backend/requirements.txt

# 2. Download RemoteCLIP weights
wget -P satquery_backend/weights/ \
  https://github.com/ChenDelong1999/RemoteCLIP/releases/download/v1.0/RemoteCLIP-ViT-B-32.pt

# 3. Run the server (from repo root)
uvicorn satquery_backend.main:app --reload --port 8001

# OR from inside the satquery_backend/ folder
python main.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness + model-load status |
| `POST` | `/analyze` | Universal endpoint — auto-routes via orchestrator |
| `POST` | `/analyze/optical-sar` | Explicit cross-modal fusion (Task 5) |
| `GET`  | `/traces` | Paginated execution trace log |
| `GET`  | `/traces/stats` | Aggregate statistics |
| `GET`  | `/traces/{trace_id}` | Single trace lookup by UUID |
| `DELETE` | `/traces` | Wipe trace log (dev only) |

## Example `curl` calls

```bash
# Single-image VQA
curl -X POST http://localhost:8001/analyze \
  -F "files=@scene.tif" \
  -F "query=What land cover is visible?"

# Cross-modal optical + SAR
curl -X POST http://localhost:8001/analyze \
  -F "files=@s2_scene.tif" \
  -F "files=@s1_scene.tif" \
  -F "query=Are built-up areas confirmed in SAR backscatter?" \
  -F "modalities=optical,sar"

# Explicit cross-modal endpoint
curl -X POST http://localhost:8001/analyze/optical-sar \
  -F "optical_file=@s2_scene.tif" \
  -F "sar_file=@s1_scene.tif" \
  -F "query=Confirm urban extent via SAR"

# View last 10 traces
curl http://localhost:8001/traces?limit=10

# Trace statistics
curl http://localhost:8001/traces/stats
```

## Environment Variables

See `.env.example` for all configurable options.

## Wiring remaining pipelines

- **VQA (`SINGLE_VQA_GROUNDING`)** — replace `_run_vqa_stub()` in `main.py`
  with your `module1_vqa/run_vqa.py` `run_vqa()` call.
- **Change Detection (`BI_TEMPORAL_CHANGE`)** — replace `_run_change_detection_stub()`
  with your trained `SiameseResNet18` inference.
