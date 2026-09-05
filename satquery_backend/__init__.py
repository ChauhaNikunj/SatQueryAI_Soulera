"""
satquery_backend
================
SatQuery AI — Production Backend (SIH 2026, ISRO Problem Statement 26167)

Package structure
-----------------
    satquery_backend/
    ├── models/
    │   └── optical_sar_fusion.py   # Task 5: RemoteCLIP + Fusion Adapter
    ├── agent/
    │   └── orchestrator.py         # Task 6: Agentic Router & Dispatcher
    └── utils/
        ├── logger.py               # Task 7: JSONL Execution Trace Logger
        └── raster_io.py            # GeoTIFF / Image loading utilities
"""

from .models.optical_sar_fusion import OpticalSARFusionModel
from .agent.orchestrator import Orchestrator, RoutingDecision, TaskType
from .utils.logger import ExecutionTraceLogger, TraceRecord
from .utils.raster_io import load_image, load_geotiff, normalize_optical, normalize_sar

__all__ = [
    # Task 5
    "OpticalSARFusionModel",
    # Task 6
    "Orchestrator",
    "RoutingDecision",
    "TaskType",
    # Task 7
    "ExecutionTraceLogger",
    "TraceRecord",
    # Utilities
    "load_image",
    "load_geotiff",
    "normalize_optical",
    "normalize_sar",
]

__version__ = "0.1.0"
__author__ = "SatQuery AI Team — Soluera / SIH 2026"
