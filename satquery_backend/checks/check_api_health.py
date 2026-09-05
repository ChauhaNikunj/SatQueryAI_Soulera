"""CHECK: FastAPI Server — start it first with:
  cd /Users/abdularkansidd/abdulsidd
  uvicorn satquery_backend.main:app --reload --port 8001
"""
import sys, urllib.request, json

print("\n=== API: FastAPI Health Check ===\n")
try:
    with urllib.request.urlopen("http://localhost:8001/health", timeout=5) as r:
        data = json.loads(r.read())
    print(f"  status              : {data['status']}")
    print(f"  orchestrator_ready  : {data['orchestrator_ready']}")
    print(f"  fusion_model_ready  : {data['fusion_model_ready']}")
    print(f"  tracer_ready        : {data['tracer_ready']}")
    print(f"  trace_log           : {data['trace_log']}")
    print(f"  load_errors         : {data.get('load_errors',[])}")
    print(f"\n[PASS] FastAPI server is running at http://localhost:8001/docs\n")
except Exception as e:
    print(f"  [FAIL] Server not reachable: {e}")
    print("  Start server first:")
    print("    cd /Users/abdularkansidd/abdulsidd")
    print("    uvicorn satquery_backend.main:app --reload --port 8001\n")
