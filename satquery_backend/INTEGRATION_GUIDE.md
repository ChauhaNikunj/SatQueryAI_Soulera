# 🛰️ SatQuery AI — Team Integration Guide (SIH PS 26167)

This package contains the core intelligence engine for SatQuery AI:
* **Task 5**: Optical–SAR Cross-Modal Fusion Model (`RemoteCLIP ViT-B/32` + `FusionAdapter`)
* **Task 6**: God-Level Agentic Orchestration Controller (150+ EO keyword intent graph & physical sensor router)
* **Task 7**: Execution Trace Logger (`execution_trace.jsonl` audit provenance)
* **Master Pipeline**: Unified single-call Python interface (`pipeline.py`)
* **FastAPI Backend & Web Dashboard**: Live on `http://localhost:8001/`

---

## ⚡ Quick 3-Line Integration for Teammates

Whoever is building the Web GUI, API, or calling models only needs:

```python
from satquery_backend.pipeline import master_pipeline

# Automatically routes, executes, and logs trace!
result = master_pipeline.run(
    query="Are there dense forests or water bodies in this region?",
    image_paths=["satellite_image.tif"]  # 1 image, 2 temporal, or optical+SAR
)

print(result["task_type"])      # e.g., 'OPTICAL_SAR_CROSS_MODAL' or 'SINGLE_VQA_GROUNDING'
print(result["output"])         # AI synthesized text insight
print(result["confidence"])     # Confidence score (0.0 - 1.0)
print(result["model_name"])     # Model identifier
```

---

## 📁 Repository Directory Layout

```text
satquery_backend/
├── agent/
│   └── orchestrator.py         # Task 6: God-Level Orchestration Engine
├── controller/
│   ├── agentic_router.py       # Task 6 Blueprint Alias
│   └── logger.py               # Task 7 Blueprint Alias
├── models/
│   ├── optical_sar_fusion.py   # Task 5: RemoteCLIP + FusionAdapter Model
│   └── vsr_adapter.py          # Spatial Reasoning Adapter
├── utils/
│   ├── logger.py               # Task 7: Execution Trace Logger
│   └── raster_io.py            # GeoTIFF & Optical/SAR normalization
├── gui/
│   └── api.py                  # FastAPI server wrapper
├── checks/                     # Verification test scripts (Task 5, 6, 7)
│   ├── check_task5_fusion.py
│   ├── check_task6_orchestrator.py
│   ├── check_task7_logger.py
│   ├── check_api_health.py
│   └── check_satellite_benchmark.py
├── weights/
│   ├── RemoteCLIP-ViT-B-32.pt  # Frozen RemoteCLIP vision backbone
│   ├── adapter_v1.pt           # Trained Optical-SAR FusionAdapter
│   └── satellite_benchmark_adapter.pt # Trained EuroSAT Adapter (96.8% Acc)
├── logs/
│   └── execution_trace.jsonl   # Append-only JSONL audit logs
├── pipeline.py                 # Master Unified Pipeline Dispatcher
├── main.py                     # FastAPI Web App & Live Dashboard
├── terminal_chat.py            # Interactive CLI Command Centre
├── verify_pipeline.py          # All-in-one Judge Verification Suite
└── requirements.txt            # Dependencies
```

---

## 🚀 How to Run & Verify

```bash
cd /Users/abdularkansidd/abdulsidd
export PYTHONPATH=/Users/abdularkansidd/abdulsidd

# 1. Run All-in-One Judge Verification Suite
python satquery_backend/verify_pipeline.py

# 2. Run Interactive Terminal Command Centre
python satquery_backend/terminal_chat.py

# 3. Run FastAPI Backend & Web Dashboard (Port 8001)
uvicorn satquery_backend.main:app --reload --port 8001
# Open in browser: http://localhost:8001/
```
