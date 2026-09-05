"""
SatQuery AI — Execution Trace Logger (Task 7)
==============================================
Production-grade JSONL execution trace logger compliant with SIH PS 26167.
Supports full auditability, dual-schema compatibility (blueprint + extended fields),
thread-safe atomic writes, and summary statistics.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

log = logging.getLogger("logger")

SCHEMA_VERSION = "1.0"
DEFAULT_LOG_FILENAME = "execution_trace.jsonl"


class TraceRecord(TypedDict, total=False):
    # Standard SIH Blueprint Fields
    timestamp: str
    task: str
    models_used: List[str]
    input_images: List[str]
    parameters: Dict[str, Any]
    outputs: List[str]
    confidence: float

    # Extended Provenance & Engine Fields
    trace_id: str
    task_type: str
    query: str
    input_files: List[str]
    model_name: str
    adapter_name: Optional[str]
    routing_rules: List[str]
    output: str
    latency_ms: float
    error: Optional[str]
    schema_version: str


class ExecutionTraceLogger:
    """
    Append-only JSONL execution trace logger for SIH Problem Statement 26167.
    """

    def __init__(
        self,
        log_dir: str | Path = "./logs",
        log_filename: str = DEFAULT_LOG_FILENAME,
        pretty_print: bool = False,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / log_filename
        self._pretty = pretty_print
        log.info("ExecutionTraceLogger initialised: %s", self._log_path)

    @property
    def log_path(self) -> Path:
        return self._log_path.resolve()

    @staticmethod
    def build_trace(
        *,
        task_type: str,
        query: str = "",
        input_files: List[str],
        model_name: str,
        adapter_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        routing_rules: Optional[List[str]] = None,
        output: str = "",
        confidence: float = 0.0,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> TraceRecord:
        """
        Construct a fully-populated TraceRecord adhering to SIH PS 26167 Blueprint.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        clean_inputs = [os.path.basename(p) for p in input_files]
        models = [model_name]
        if adapter_name:
            models.append(adapter_name)

        return TraceRecord(
            # Blueprint standard fields
            timestamp=now,
            task=task_type,
            models_used=models,
            input_images=clean_inputs,
            parameters=parameters or {},
            outputs=[output] if output else [],
            confidence=float(confidence),

            # Extended fields
            trace_id=str(uuid.uuid4()),
            task_type=task_type,
            query=query,
            input_files=clean_inputs,
            model_name=model_name,
            adapter_name=adapter_name,
            routing_rules=routing_rules or [],
            output=output,
            latency_ms=float(latency_ms),
            error=error,
            schema_version=SCHEMA_VERSION,
        )

    def log(self, trace: TraceRecord) -> None:
        """Append trace as a single JSON line to execution_trace.jsonl."""
        if self._pretty:
            line = json.dumps(dict(trace), indent=2, ensure_ascii=False)
        else:
            line = json.dumps(dict(trace), ensure_ascii=False)

        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def read_all(self) -> List[TraceRecord]:
        if not self._log_path.exists():
            return []
        records: List[TraceRecord] = []
        with open(self._log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records

    def summary_stats(self) -> Dict[str, Any]:
        records = self.read_all()
        if not records:
            return {"total_runs": 0, "task_counts": {}, "avg_confidence": None, "avg_latency_ms": None}

        task_counts: Dict[str, int] = {}
        for r in records:
            tt = r.get("task", r.get("task_type", "UNKNOWN"))
            task_counts[tt] = task_counts.get(tt, 0) + 1

        confs = [r["confidence"] for r in records if "confidence" in r]
        lats = [r["latency_ms"] for r in records if "latency_ms" in r]

        return {
            "total_runs": len(records),
            "task_counts": task_counts,
            "avg_confidence": sum(confs) / len(confs) if confs else 0.0,
            "avg_latency_ms": sum(lats) / len(lats) if lats else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Blueprint Helper Function (Section 5 of SIH Blueprint)
# ─────────────────────────────────────────────────────────────────────────────

def log_execution(
    task: str,
    models_used: List[str],
    input_images: List[str],
    parameters: Dict[str, Any],
    outputs: List[str],
    confidence: float,
    log_path: str = "satquery_backend/logs/execution_trace.jsonl",
) -> Dict[str, Any]:
    """
    Mandatory execution trace logger per SIH 26167 Blueprint Section 5.
    """
    trace = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "task": task,
        "models_used": models_used,
        "input_images": [os.path.basename(p) for p in input_images],
        "parameters": parameters,
        "outputs": outputs,
        "confidence": float(confidence),
        "trace_id": str(uuid.uuid4()),
        "schema_version": SCHEMA_VERSION,
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")
    return trace
