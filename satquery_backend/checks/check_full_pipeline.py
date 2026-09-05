"""CHECK: Full end-to-end pipeline verification"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/abdularkansidd/abdulsidd")
import numpy as np, rasterio
from PIL import Image
from satquery_backend.agent.orchestrator import Orchestrator, ImageInfo
from satquery_backend.models.optical_sar_fusion import OpticalSARFusionModel
from satquery_backend.utils.logger import ExecutionTraceLogger

print("\n=== FULL PIPELINE: Orchestrator → Fusion → Logger ===\n")

orch   = Orchestrator(strict_dimension_check=False)
model  = OpticalSARFusionModel(
    clip_weights_path="satquery_backend/weights/RemoteCLIP-ViT-B-32.pt",
    adapter_weights_path="satquery_backend/weights/adapter_v1.pt")
model.eval()
logger = ExecutionTraceLogger(log_dir="satquery_backend/logs")

OPT = "satquery_backend/sample_data/optical/usgs_landsat_sample.tif"
SAR = "satquery_backend/sample_data/sar/sentinel1_coreg_sample.tif"

with rasterio.open(OPT) as ds: ow,oh,ob=ds.width,ds.height,ds.count
with rasterio.open(SAR) as ds: sw,sh,sb=ds.width,ds.height,ds.count

query = "Are built-up areas visible in the SAR backscatter over this region?"

decision = orch.route(query,[ImageInfo(OPT,ow,oh,ob),ImageInfo(SAR,sw,sh,sb)])
print(f"  Orchestrator  → {decision.task_type.value}  (conf={decision.confidence})")
print(f"  Rules fired   → {decision.routing_rules}")

def load_opt(p):
    with rasterio.open(p) as ds: arr=ds.read().astype(np.float32)
    if arr.shape[0]==1: arr=np.stack([arr[0]]*3)
    arr=arr[:3,:224,:224]
    u8=((arr-arr.min())/(arr.max()-arr.min()+1e-8)*255).astype(np.uint8)
    return Image.fromarray(u8.transpose(1,2,0))
def load_sar(p):
    with rasterio.open(p) as ds: arr=ds.read().astype(np.float32)
    vv,vh=arr[0,:224,:224],arr[1,:224,:224]
    def n(x): return ((x-x.min())/(x.max()-x.min()+1e-8)*255).astype(np.uint8)
    return Image.fromarray(np.stack([n(vv),n(vh),n(vv/(vh+1e-6))],axis=-1))

result = model.analyze(load_opt(OPT), load_sar(SAR), query)
print(f"\n  Fusion output → {result.insight[:80]}...")
print(f"  Confidence    → {result.confidence:.4f}  ({result.confidence*100:.1f}%)")
print(f"  Top class     → {result.top_class}")
print(f"  Latency       → {result.latency_ms:.1f} ms")
print(f"\n  Top-3:")
for p in result.top_k_predictions:
    bar="#"*int(p["probability"]*35)
    print(f"    {p['class']:<40} {p['probability']:.4f}  {bar}")

trace = logger.build_trace(
    task_type=decision.task_type.value, query=query,
    input_files=[OPT,SAR], model_name=result.model_name,
    adapter_name="fusion_mlp_v1", parameters={"top_k":3},
    routing_rules=decision.routing_rules,
    output=result.insight, confidence=result.confidence,
    latency_ms=result.latency_ms)
logger.log(trace)
print(f"\n  Trace logged  → {trace['trace_id']}")
print(f"  JSONL file    → {logger.log_path}")
stats=logger.summary_stats()
print(f"  Total runs    → {stats['total_runs']}  avg_conf={stats.get('avg_confidence',0):.4f}")
print(f"\n[PASS] Full pipeline end-to-end working.\n")
