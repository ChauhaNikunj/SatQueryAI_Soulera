"""
SatQuery AI — Agentic Router (Task 6 Blueprint Alias)
=====================================================
Direct callable classification and routing module compliant with Blueprint Section 1 (#6).
Exposes Orchestrator, ImageInfo, RoutingDecision, and route_query().
"""

from __future__ import annotations

import os
from typing import List, Optional

from satquery_backend.agent.orchestrator import (
    Orchestrator,
    ImageInfo,
    RoutingDecision,
    TaskType,
    FUSION_KEYWORDS,
    CHANGE_KEYWORDS,
    LULC_KEYWORDS,
    CLOUD_KEYWORDS,
)

# Global router instance
_router = Orchestrator(strict_dimension_check=False)


def route_query(
    query: str,
    image_paths: List[str],
    modalities: Optional[List[str]] = None,
) -> RoutingDecision:
    """
    Classifies user intent and uploaded image(s) to route to the optimal model pipeline.

    Args:
        query: Natural language query from user.
        image_paths: List of 1 or 2 image file paths.
        modalities: Optional list specifying 'optical' or 'sar' per image.

    Returns:
        RoutingDecision with task_type, confidence, routing_rules, target_model_pipeline, etc.
    """
    image_infos = []
    for idx, p in enumerate(image_paths):
        mod = None
        if modalities and idx < len(modalities):
            mod = modalities[idx]
        else:
            is_sar = any(h in p.lower() for h in ["sar", "s1", "vv", "vh", "radar"])
            mod = "sar" if is_sar else "optical"

        bands = 2 if mod == "sar" else 3
        image_infos.append(
            ImageInfo(
                path=p,
                width=256,
                height=256,
                bands=bands,
                modality=mod,
            )
        )

    return _router.route(query, image_infos)
