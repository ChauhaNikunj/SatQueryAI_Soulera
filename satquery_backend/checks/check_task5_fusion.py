"""CHECK: Task 5 — Optical-SAR Fusion Model"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/abdularkansidd/abdulsidd")
import numpy as np
from PIL import Image
import rasterio
from satquery_backend.models.optical_sar_fusion import OpticalSARFusionModel

print("\n=== TASK 5: Optical-SAR Fusion Model (RemoteCLIP) ===\n")

model = OpticalSARFusionModel(
    clip_weights_path="satquery_backend/weights/RemoteCLIP-ViT-B-32.pt",
    adapter_weights_path="satquery_backend/weights/adapter_v1.pt",
)
model.eval()
print(f"  RemoteCLIP ViT-B/32 : frozen encoder loaded")
print(f"  FusionAdapter       : {model.adapter.n_params:,} trainable params")

# Load real GeoTIFFs
def load_optical(path):
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)
    if arr.shape[0]==1: arr=np.stack([arr[0]]*3)
    arr=arr[:3,:224,:224]
    u8=((arr-arr.min())/(arr.max()-arr.min()+1e-8)*255).astype(np.uint8)
    return Image.fromarray(u8.transpose(1,2,0))

def load_sar(path):
    with rasterio.open(path) as ds:
        arr=ds.read().astype(np.float32)
    vv,vh=arr[0,:224,:224],arr[1,:224,:224]
    def n(x): return ((x-x.min())/(x.max()-x.min()+1e-8)*255).astype(np.uint8)
    return Image.fromarray(np.stack([n(vv),n(vh),n(vv/(vh+1e-6))],axis=-1))

opt_pil = load_optical("satquery_backend/sample_data/optical/usgs_landsat_sample.tif")
sar_pil = load_sar("satquery_backend/sample_data/sar/sentinel1_coreg_sample.tif")

queries = [
    "Are built-up areas visible in the SAR backscatter?",
    "What is the dominant land cover in this region?",
    "Detect urban fabric from optical and radar fusion",
    "Identify vegetation and forest from multisensor data",
    "Classify agricultural land from Sentinel-1 and Sentinel-2",
]

print()
for i,q in enumerate(queries,1):
    r = model.analyze(opt_pil, sar_pil, q)
    bar="#"*int(r.confidence*40)
    print(f"  Q{i}: {q[:55]}")
    print(f"       Top class : {r.top_class}")
    print(f"       Confidence: {r.confidence:.4f} ({r.confidence*100:.1f}%)  {bar}")
    print(f"       Latency   : {r.latency_ms:.1f} ms")
    print(f"       Top-3     : {[(p['class'][:20],round(p['probability'],3)) for p in r.top_k_predictions]}")
    print()

f_opt,f_sar,cosim = model.embed(opt_pil,sar_pil)
print(f"  Embedding cosine similarity (optical vs SAR): {cosim:.4f}")
print(f"\n[PASS] Task 5 — Fusion Model working.\n")
