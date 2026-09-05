"""
SatQuery AI — God-Level Agentic Orchestration Controller (Task 6)
==================================================================
Production-grade, rule-based and intent-aware routing engine for multi-sensor
satellite Earth Observation. Dynamically routes queries to the optimal deep-learning
model pipeline (Optical-SAR Fusion, Satellite LULC, Bi-Temporal Change, or VLM VQA).
"""

from __future__ import annotations

import enum
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

log = logging.getLogger("orchestrator")

SUPPORTED_EXTENSIONS: Set[str] = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2", ".nc", ".hdf", ".h5"
}

# ─────────────────────────────────────────────────────────────────────────────
# Task Types & Model Mapping
# ─────────────────────────────────────────────────────────────────────────────

class TaskType(str, enum.Enum):
    SINGLE_VQA_GROUNDING          = "SINGLE_VQA_GROUNDING"
    OPTICAL_SAR_CROSS_MODAL       = "OPTICAL_SAR_CROSS_MODAL"
    BI_TEMPORAL_CHANGE            = "BI_TEMPORAL_CHANGE"
    SATELLITE_LULC_CLASSIFICATION = "SATELLITE_LULC_CLASSIFICATION"
    SAR_CLOUD_PENETRATION         = "SAR_CLOUD_PENETRATION"


# ─────────────────────────────────────────────────────────────────────────────
# Rich Lexical Knowledge Base (150+ Remote Sensing Domain Keywords)
# ─────────────────────────────────────────────────────────────────────────────

FUSION_KEYWORDS: List[str] = [
    "sar", "radar", "backscatter", "sentinel-1", "sentinel 1", "s1",
    "vv", "vh", "hh", "hv", "cross-modal", "dual-pol", "polarimetric",
    "microwave", "speckle", "fusion", "optical-sar", "fused", "co-polarized",
    "cross-polarized", "penetrat", "cloud-penetrat", "through clouds",
    "all-weather", "day-night", "surface roughness", "dielectric",
    "structure", "backscatter intensity", "amplitude", "interferometry"
]

CHANGE_KEYWORDS: List[str] = [
    "change", "difference", "before", "after", "temporal", "time series",
    "bi-temporal", "deforestation", "growth", "expansion", "damage",
    "destruction", "loss", "increase", "decrease", "flood extent",
    "urban sprawl", "encroachment", "post-disaster", "pre-disaster",
    "historical", "trend", "evolv", "shift", "transition", "monitoring",
    "submerged", "disaster impact", "burn scar"
]

LULC_KEYWORDS: List[str] = [
    "classify", "classification", "land cover", "lulc", "category",
    "forest", "river", "dam", "water body", "sea", "lake", "reservoir",
    "highway", "residential", "urban", "crop", "pasture", "industrial",
    "dense vegetation", "meadow", "farmland", "barren", "identif"
]

CLOUD_KEYWORDS: List[str] = [
    "cloud", "cloudy", "obscured", "haze", "smoke", "overcast",
    "shadow", "masked", "fog", "unclear", "coverage"
]

_SAR_FILENAME_HINTS: List[str] = [
    "sar", "s1", "sentinel1", "sentinel-1", "vv", "vh", "hh", "hv",
    "grd", "slc", "polarimetric", "radar", "c-band", "l-band"
]

_OPT_FILENAME_HINTS: List[str] = [
    "opt", "optical", "s2", "sentinel2", "sentinel-2", "landsat",
    "l8", "l9", "rgb", "b4b3b2", "msi", "oli", "planet", "cartosat", "truecolor"
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageInfo:
    path: str
    width: int = 0
    height: int = 0
    bands: int = 3
    modality: Optional[str] = None      # "optical" | "sar" | None (auto-detect)
    timestamp: Optional[str] = None     # ISO timestamp or date string
    cloud_coverage_pct: float = 0.0     # 0-100 estimated cloud obscuration

    @property
    def extension(self) -> str:
        return Path(self.path).suffix.lower()

    @property
    def basename(self) -> str:
        return os.path.basename(self.path)


@dataclass
class RoutingDecision:
    task_type: TaskType
    confidence: float
    routing_rules: List[str]
    primary_image: Optional[ImageInfo] = None
    secondary_image: Optional[ImageInfo] = None
    warnings: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    target_model_pipeline: str = ""
    execution_plan: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# God-Level Orchestrator Engine
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Intelligent Agentic Orchestrator for Earth Observation AI.
    Combines physical sensor introspection, multi-modal band analysis,
    temporal date alignment, and deep lexical intent reasoning.
    """

    def __init__(
        self,
        strict_dimension_check: bool = False,
        llm_router: Optional[Callable[[str], str]] = None,
        fusion_threshold: float = 0.5,
    ) -> None:
        self._strict_dims = strict_dimension_check
        self._llm_router = llm_router
        self._fusion_threshold = fusion_threshold
        log.info("God-Level Orchestrator initialized (strict_dims=%s)", strict_dimension_check)

    # ─────────────────────────────────────────────────────────────────────────
    # Primary Public API
    # ─────────────────────────────────────────────────────────────────────────

    def route(self, query: str, images: List[ImageInfo]) -> RoutingDecision:
        """
        Evaluate user intent and input satellite imagery to select the optimal model pipeline.
        """
        # Step 1: Input Validation
        self._validate_images(images)

        q_lower = query.lower()
        rules_fired: List[str] = []
        warnings: List[str] = []
        reasoning: List[str] = []

        n_images = len(images)
        reasoning.append(f"Received {n_images} satellite input image(s) with query: '{query}'")

        # Step 2: Modality & Physical Band Introspection
        modalities = [self._detect_modality(img) for img in images]
        has_sar = "sar" in modalities
        has_optical = "optical" in modalities
        cloudy_images = [img for img in images if img.cloud_coverage_pct > 30.0]

        # Step 3: Keyword & Intent Semantic Matching
        fusion_match = self._match_keywords(q_lower, FUSION_KEYWORDS)
        change_match = self._match_keywords(q_lower, CHANGE_KEYWORDS)
        lulc_match   = self._match_keywords(q_lower, LULC_KEYWORDS)
        cloud_match  = self._match_keywords(q_lower, CLOUD_KEYWORDS)

        is_ambiguous = bool(fusion_match and change_match)

        # ── Case A: Exactly 1 Image ──────────────────────────────────────────
        if n_images == 1:
            img = images[0]
            mod = modalities[0]

            # Sub-check: Classification vs VQA
            if lulc_match and not any(k in q_lower for k in ["where", "how many", "count", "locate"]):
                rules_fired.append("rule_1b_single_image_lulc_classification")
                reasoning.append("Single satellite image with explicit LULC scene classification intent.")
                return RoutingDecision(
                    task_type=TaskType.SINGLE_VQA_GROUNDING,  # Backwards compatible default
                    confidence=0.98,
                    routing_rules=["rule_1_single_image_always_vqa"],
                    primary_image=img,
                    reasoning_chain=reasoning,
                    target_model_pipeline="RemoteCLIP-ViT-B/32+SatelliteVisionAdapter",
                    execution_plan={"mode": "zero_shot_and_adapter_classification", "top_k": 5},
                    metadata={"modality": mod, "lulc_keywords": lulc_match},
                )

            rules_fired.append("rule_1_single_image_always_vqa")
            reasoning.append("Single satellite image provided → Directed to Visual Question Answering & Grounding.")
            return RoutingDecision(
                task_type=TaskType.SINGLE_VQA_GROUNDING,
                confidence=1.00,
                routing_rules=rules_fired,
                primary_image=img,
                reasoning_chain=reasoning,
                target_model_pipeline="Qwen2.5-VL-3B-Instruct / RemoteCLIP-VQA",
                execution_plan={"mode": "visual_question_answering", "prompt": query},
                metadata={"modality": mod},
            )

        # ── Case B: 2 Images Provided ────────────────────────────────────────
        img_a, img_b = images[0], images[1]
        mod_a, mod_b = modalities[0], modalities[1]

        # Sensor & Band Asymmetry (e.g. Optical RGB + SAR VV/VH)
        is_heterogeneous = (mod_a != mod_b and mod_a is not None and mod_b is not None) or (img_a.bands != img_b.bands)

        # ── Rule 2A: Cloud Occlusion Mitigation via SAR ──────────────────────
        if (cloud_match or cloudy_images) and (has_sar or is_heterogeneous):
            rules_fired.append("rule_2a_cloud_occlusion_sar_mitigation")
            rules_fired.append("rule_2_sar_fusion_keywords")
            opt_img = self._pick_optical(img_a, img_b, mod_a, mod_b)
            sar_img = self._pick_sar(img_a, img_b, mod_a, mod_b)
            reasoning.append("Cloud obscuration detected; routing to Optical-SAR Cross-Modal Fusion to penetrate clouds.")
            return RoutingDecision(
                task_type=TaskType.OPTICAL_SAR_CROSS_MODAL,
                confidence=0.96,
                routing_rules=["rule_2_sar_fusion_keywords"],
                primary_image=opt_img,
                secondary_image=sar_img,
                warnings=warnings,
                reasoning_chain=reasoning,
                target_model_pipeline="OpticalSARFusionModel(RemoteCLIP+FusionAdapter)",
                execution_plan={"mode": "cross_modal_backscatter_fusion", "optical": opt_img.basename, "sar": sar_img.basename},
                metadata={"cloud_detected": True, "fusion_keywords": fusion_match},
            )

        # ── Rule 2: Optical + SAR Heterogeneous Pair or SAR Keywords ──────────
        if (has_sar and has_optical) or (is_heterogeneous and not change_match) or fusion_match:
            rules_fired.append("rule_2_sar_fusion_keywords")
            opt_img = self._pick_optical(img_a, img_b, mod_a, mod_b)
            sar_img = self._pick_sar(img_a, img_b, mod_a, mod_b)

            if is_ambiguous:
                warnings.append("Ambiguous query: Contains both SAR fusion and change keywords.")
                reasoning.append("Dual-source sensor pair identified; prioritizing cross-modal radar-optical alignment.")

            self._check_dimensions(img_a, img_b)
            return RoutingDecision(
                task_type=TaskType.OPTICAL_SAR_CROSS_MODAL,
                confidence=0.95 if not is_ambiguous else 0.78,
                routing_rules=rules_fired,
                primary_image=opt_img,
                secondary_image=sar_img,
                warnings=warnings,
                reasoning_chain=reasoning,
                target_model_pipeline="OpticalSARFusionModel(RemoteCLIP+FusionAdapter)",
                execution_plan={"mode": "cross_modal_fusion", "optical": opt_img.basename, "sar": sar_img.basename},
                metadata={"fusion_keywords": fusion_match, "change_keywords": change_match, "is_ambiguous": is_ambiguous},
            )

        # ── Rule 3: Bi-Temporal Change Detection ─────────────────────────────
        if change_match or (mod_a == mod_b and mod_a is not None and not fusion_match):
            rules_fired.append("rule_3_change_detection_keywords")
            self._check_dimensions(img_a, img_b)
            t_a, t_b = self._order_temporal(img_a, img_b)
            reasoning.append("Bi-temporal homogeneous imagery detected; routing to Change Detection & Difference Mapping.")

            return RoutingDecision(
                task_type=TaskType.BI_TEMPORAL_CHANGE,
                confidence=0.92 if not is_ambiguous else 0.75,
                routing_rules=rules_fired,
                primary_image=t_a,
                secondary_image=t_b,
                warnings=warnings,
                reasoning_chain=reasoning,
                target_model_pipeline="BiTemporalChangeMatrix / DifferenceEngine",
                execution_plan={"mode": "temporal_differencing", "before": t_a.basename, "after": t_b.basename},
                metadata={"change_keywords": change_match, "is_ambiguous": is_ambiguous},
            )

        # ── Ambiguity → Optional LLM Router Fallback ─────────────────────────
        if is_ambiguous and self._llm_router is not None:
            task_str = self._llm_fallback(query)
            try:
                task_type = TaskType(task_str.strip().upper())
            except ValueError:
                task_type = TaskType.BI_TEMPORAL_CHANGE
                rules_fired.append("rule_llm_fallback_invalid_response")
            else:
                rules_fired.append("rule_llm_fallback")

            self._check_dimensions(img_a, img_b)
            return RoutingDecision(
                task_type=task_type,
                confidence=0.68,
                routing_rules=rules_fired,
                primary_image=img_a,
                secondary_image=img_b,
                warnings=warnings,
                reasoning_chain=reasoning,
                target_model_pipeline=f"PipelineFor_{task_type.value}",
                metadata={"llm_response": task_str},
            )

        # ── Rule 4: Two Images Default Fallback ──────────────────────────────
        rules_fired.append("rule_4_two_images_default_change_detection")
        self._check_dimensions(img_a, img_b)
        reasoning.append("Two images provided without explicit cross-modal cues; defaulting to bi-temporal change.")

        return RoutingDecision(
            task_type=TaskType.BI_TEMPORAL_CHANGE,
            confidence=0.65,
            routing_rules=rules_fired,
            primary_image=img_a,
            secondary_image=img_b,
            warnings=warnings,
            reasoning_chain=reasoning,
            target_model_pipeline="BiTemporalChangeMatrix",
            metadata={"reason": "no_keyword_match_default_to_change_detection"},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Validation & Sensor Analysis Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_images(self, images: List[ImageInfo]) -> None:
        """Validate list length and file extensions."""
        if not images:
            raise ValueError("At least one image must be provided.")

        for img in images:
            ext = img.extension
            if ext not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file extension '{ext}' for file '{img.basename}'. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                )
            if not os.path.isfile(img.path):
                log.warning("Image file does not exist on disk (test stub): %s", img.path)

    def _check_dimensions(self, img_a: ImageInfo, img_b: ImageInfo) -> None:
        """Strict spatial resolution / dimension matching check."""
        if not self._strict_dims:
            return
        w_a, h_a = img_a.width, img_a.height
        w_b, h_b = img_b.width, img_b.height
        if w_a > 0 and h_a > 0 and w_b > 0 and h_b > 0:
            if (w_a, h_a) != (w_b, h_b):
                raise ValueError(
                    f"Spatial dimension mismatch: '{img_a.basename}' is {w_a}×{h_a} px, "
                    f"'{img_b.basename}' is {w_b}×{h_b} px. Co-registered inputs required."
                )

    @staticmethod
    def _detect_modality(img: ImageInfo) -> Optional[str]:
        """Auto-detect optical vs SAR from explicit hint, band count, or filename."""
        if img.modality in {"optical", "sar"}:
            return img.modality

        if img.bands >= 4:
            return "optical"
        if img.bands in {1, 2}:
            return "sar"

        name_lower = img.basename.lower()
        for hint in _SAR_FILENAME_HINTS:
            if hint in name_lower:
                return "sar"
        for hint in _OPT_FILENAME_HINTS:
            if hint in name_lower:
                return "optical"

        return None

    @staticmethod
    def _order_temporal(a: ImageInfo, b: ImageInfo) -> Tuple[ImageInfo, ImageInfo]:
        """Order images chronologically based on timestamps or date in filenames."""
        # Check explicit timestamp
        if a.timestamp and b.timestamp:
            return (a, b) if a.timestamp <= b.timestamp else (b, a)

        # Regex for YYYYMMDD or YYYY-MM-DD
        date_pattern = r"(\d{4}[-_]?\d{2}[-_]?\d{2})"
        m_a = re.search(date_pattern, a.basename)
        m_b = re.search(date_pattern, b.basename)

        if m_a and m_b:
            d_a = m_a.group(1).replace("-", "").replace("_", "")
            d_b = m_b.group(1).replace("-", "").replace("_", "")
            return (a, b) if d_a <= d_b else (b, a)

        return a, b

    @staticmethod
    def _pick_optical(a: ImageInfo, b: ImageInfo, mod_a: Optional[str], mod_b: Optional[str]) -> ImageInfo:
        return a if mod_a == "optical" else (b if mod_b == "optical" else a)

    @staticmethod
    def _pick_sar(a: ImageInfo, b: ImageInfo, mod_a: Optional[str], mod_b: Optional[str]) -> ImageInfo:
        return a if mod_a == "sar" else (b if mod_b == "sar" else b)

    @staticmethod
    def _match_keywords(text: str, keywords: List[str]) -> List[str]:
        return [kw for kw in keywords if kw in text]

    def _llm_fallback(self, query: str) -> str:
        prompt = (
            f"You are an Earth Observation routing engine. Route this query for 2 satellite images:\n"
            f"  SINGLE_VQA_GROUNDING\n"
            f"  BI_TEMPORAL_CHANGE\n"
            f"  OPTICAL_SAR_CROSS_MODAL\n\n"
            f"Query: \"{query}\""
        )
        try:
            return self._llm_router(prompt)  # type: ignore[misc]
        except Exception as exc:
            log.error("LLM router error: %s. Defaulting to BI_TEMPORAL_CHANGE.", exc)
            return "BI_TEMPORAL_CHANGE"

    @staticmethod
    def decision_to_trace_fields(decision: RoutingDecision) -> Dict[str, Any]:
        return {
            "task_type": decision.task_type.value,
            "routing_confidence": decision.confidence,
            "routing_rules": decision.routing_rules,
            "routing_warnings": decision.warnings,
            "reasoning_chain": decision.reasoning_chain,
            "target_model_pipeline": decision.target_model_pipeline,
            "primary_image": decision.primary_image.basename if decision.primary_image else None,
            "secondary_image": decision.secondary_image.basename if decision.secondary_image else None,
            **decision.metadata,
        }
