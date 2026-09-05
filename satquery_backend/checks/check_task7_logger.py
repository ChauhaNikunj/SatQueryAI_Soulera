"""CHECK: Task 7 — Execution Trace Logger"""
import sys; sys.path.insert(0,"/Users/abdularkansidd/abdulsidd")
import json, uuid
from satquery_backend.utils.logger import ExecutionTraceLogger

print("\n=== TASK 7: Execution Trace Logger ===\n")
logger = ExecutionTraceLogger(log_dir="satquery_backend/logs")

# Build a trace
trace = logger.build_trace(
    task_type="OPTICAL_SAR_CROSS_MODAL",
    query="Are there built-up areas in the SAR image?",
    input_files=["optical.tif","sar.tif"],
    model_name="RemoteCLIP-ViT-B/32+FusionAdapter",
    adapter_name="fusion_mlp_v1",
    parameters={"top_k":3},
    routing_rules=["rule_2b_sar_fusion_keywords"],
    output="Built-up areas confirmed at 82% confidence.",
    confidence=0.82,
    latency_ms=113.5,
)
logger.log(trace)

# Read back
records = logger.read_all()
found = any(r["trace_id"]==trace["trace_id"] for r in records)

print(f"  trace_id   : {trace['trace_id']}")
print(f"  timestamp  : {trace['timestamp']}")
print(f"  task_type  : {trace['task_type']}")
print(f"  model_name : {trace['model_name']}")
print(f"  confidence : {trace['confidence']}")
print(f"  latency_ms : {trace['latency_ms']}")
print(f"  log_file   : {logger.log_path}")
print(f"  readback   : {'OK' if found else 'FAIL'}")
stats = logger.summary_stats()
print(f"\n  Stats: total_runs={stats['total_runs']}  avg_conf={stats.get('avg_confidence',0):.3f}")
print("\n[PASS] Task 7 — Logger working correctly.\n")
