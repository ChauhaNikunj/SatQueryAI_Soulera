"""CHECK: Task 6 — Agentic Orchestration Controller (uses real GeoTIFFs)"""
import sys; sys.path.insert(0,"/Users/abdularkansidd/abdulsidd")
from satquery_backend.agent.orchestrator import Orchestrator, ImageInfo

print("\n=== TASK 6: Agentic Orchestration Controller ===\n")
orch = Orchestrator(strict_dimension_check=False)

# Real files that exist on disk
OPT  = "/Users/abdularkansidd/abdulsidd/satquery_backend/sample_data/optical/usgs_landsat_sample.tif"
S2   = "/Users/abdularkansidd/abdulsidd/satquery_backend/sample_data/optical/sentinel2_B4B3B2_bengaluru.tif"
S1   = "/Users/abdularkansidd/abdulsidd/satquery_backend/sample_data/sar/sentinel1_VV_VH_bengaluru.tif"
SAR  = "/Users/abdularkansidd/abdulsidd/satquery_backend/sample_data/sar/sentinel1_coreg_sample.tif"

tests = [
    # (description, query, images, expected_task)
    ("1 image → SINGLE_VQA_GROUNDING",
     "What land cover is present in this satellite image?",
     [ImageInfo(S2, 256, 256, 3)],                                       # 1 image
     "SINGLE_VQA_GROUNDING"),

    ("2 images + 'SAR backscatter' → OPTICAL_SAR_CROSS_MODAL",
     "Analyze SAR backscatter and radar amplitude over this region",
     [ImageInfo(S2, 256, 256, 3), ImageInfo(S1, 256, 256, 2)],           # 2 images, diff bands
     "OPTICAL_SAR_CROSS_MODAL"),

    ("2 images + 'deforestation change' → BI_TEMPORAL_CHANGE",
     "Show deforestation and land change between these two images",
     [ImageInfo(S2, 256, 256, 3), ImageInfo(S2, 256, 256, 3)],           # 2 optical images, same bands
     "BI_TEMPORAL_CHANGE"),

    ("2 images + 'SAR fusion' → OPTICAL_SAR_CROSS_MODAL",
     "Compare optical with SAR VV VH backscatter fusion",
     [ImageInfo(S2, 256, 256, 3), ImageInfo(SAR, 256, 256, 2)],          # explicit SAR bands=2
     "OPTICAL_SAR_CROSS_MODAL"),

    ("2 images + 'before after difference' → BI_TEMPORAL_CHANGE",
     "Urban growth difference before and after construction",
     [ImageInfo(S2, 256, 256, 3), ImageInfo(S2, 256, 256, 3)],           # same modality = temporal
     "BI_TEMPORAL_CHANGE"),
]

all_pass = True
for i, (desc, query, imgs, expected) in enumerate(tests, 1):
    d = orch.route(query, imgs)
    ok = d.task_type.value == expected
    status = "PASS" if ok else "FAIL"
    if not ok: all_pass = False
    print(f"  Test {i}: [{status}]  {desc}")
    print(f"           Routed to : {d.task_type.value}")
    print(f"           Confidence: {d.confidence:.2f}")
    print(f"           Rules     : {d.routing_rules}")
    print()

# ── Error validation tests ─────────────────────────────────────────────
try:
    orch.route("test", [])
    print("  [FAIL] Should raise ValueError for 0 images"); all_pass=False
except ValueError as e:
    print(f"  [PASS] 0 images  → ValueError raised correctly")

try:
    orch.route("test", [ImageInfo("document.pdf", 0, 0, 0)])
    print("  [FAIL] Should raise ValueError for .pdf"); all_pass=False
except ValueError:
    print(f"  [PASS] .pdf ext  → ValueError raised correctly")

try:
    orch.route("test", [ImageInfo("video.mp4", 0, 0, 0)])
    print("  [FAIL] Should raise ValueError for .mp4"); all_pass=False
except ValueError:
    print(f"  [PASS] .mp4 ext  → ValueError raised correctly")

print(f"\n[{'PASS' if all_pass else 'FAIL'}] Task 6 — Orchestrator: all tests {'passed' if all_pass else 'FAILED'}.\n")
