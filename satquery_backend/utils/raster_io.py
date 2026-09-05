"""
satquery_backend/utils/raster_io.py
=====================================
Geospatial raster I/O helpers for SatQuery AI.

Responsibilities
----------------
* Load GeoTIFF files via rasterio (primary) or GDAL (fallback).
* Load standard images (PNG / JPEG) via PIL.
* Produce model-ready uint8 PIL.Image objects or float32 NumPy arrays.
* Provide SAR amplitude-to-pseudo-RGB and Sentinel-2-to-RGB pipelines.
* Expose spatial-metadata dicts consumed by the orchestrator for validation.

Public API
----------
    load_image(path)            -> (PIL.Image.Image, dict)
    load_geotiff(path)          -> (np.ndarray, dict)
    normalize_optical(arr)      -> PIL.Image.Image
    normalize_sar(arr)          -> PIL.Image.Image
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _percentile_stretch(
    arr: np.ndarray, lo: float = 2.0, hi: float = 98.0
) -> np.ndarray:
    """Linear percentile stretch to uint8 [0, 255]."""
    p_lo = np.percentile(arr, lo)
    p_hi = np.percentile(arr, hi)
    stretched = (arr - p_lo) / (p_hi - p_lo + 1e-6) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def _load_with_rasterio(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a GeoTIFF with rasterio. Returns (arr[bands,H,W], meta)."""
    try:
        import rasterio  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for GeoTIFF loading. "
            "Install with: pip install rasterio"
        ) from exc

    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (bands, H, W)
        meta: Dict[str, Any] = {
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "driver": src.driver,
            "nodata": src.nodata,
            "filepath": str(path),
        }
    logger.debug("rasterio: loaded %s  shape=%s", path, arr.shape)
    return arr, meta


def _load_with_gdal(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fallback GeoTIFF reader using GDAL."""
    try:
        from osgeo import gdal  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Neither rasterio nor GDAL is available. "
            "pip install rasterio   OR   pip install gdal"
        ) from exc

    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"GDAL could not open: {path}")

    n_bands = ds.RasterCount
    data = np.stack(
        [ds.GetRasterBand(b + 1).ReadAsArray() for b in range(n_bands)], axis=0
    ).astype(np.float32)
    gt = ds.GetGeoTransform()
    meta: Dict[str, Any] = {
        "crs": ds.GetProjection(),
        "transform": list(gt),
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "count": n_bands,
        "dtype": str(data.dtype),
        "driver": "GTiff",
        "nodata": None,
        "filepath": str(path),
    }
    ds = None  # close
    logger.debug("GDAL: loaded %s  shape=%s", path, data.shape)
    return data, meta


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_geotiff(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a GeoTIFF file. Attempts rasterio first; falls back to GDAL.

    Parameters
    ----------
    path : str
        Path to a .tif / .tiff file.

    Returns
    -------
    arr : np.ndarray, shape (bands, H, W), dtype float32
    meta : dict
        Keys: crs, transform, width, height, count, dtype, driver,
              nodata, filepath.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    IOError / ImportError
        If no suitable raster library is available or the file is corrupt.
    """
def _load_with_pil_or_cv2(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fallback GeoTIFF/TIFF reader using PIL or OpenCV."""
    try:
        pil_img = Image.open(path)
        arr = np.array(pil_img).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)
        meta = {
            "crs": None, "transform": None, "width": arr.shape[2], "height": arr.shape[1],
            "count": arr.shape[0], "dtype": str(arr.dtype), "driver": "PIL", "nodata": None, "filepath": str(path)
        }
        return arr, meta
    except Exception:
        import cv2
        mat = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mat is None:
            raise IOError(f"Failed to read raster at {path}")
        arr = mat.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)
        meta = {
            "crs": None, "transform": None, "width": arr.shape[2], "height": arr.shape[1],
            "count": arr.shape[0], "dtype": str(arr.dtype), "driver": "OpenCV", "nodata": None, "filepath": str(path)
        }
        return arr, meta


def load_geotiff(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a GeoTIFF file as a float32 NumPy array (bands, H, W).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GeoTIFF not found: {path}")
    try:
        return _load_with_rasterio(path)
    except ImportError:
        try:
            return _load_with_gdal(path)
        except (ImportError, Exception):
            logger.info("rasterio/GDAL unavailable; using PIL/OpenCV fallback for %s", path)
            return _load_with_pil_or_cv2(path)


def load_image(path: str) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Universal image loader — dispatches on file extension.

    * ``.tif`` / ``.tiff``  ->  load_geotiff + normalize_optical (or SAR heuristic)
    * ``.png`` / ``.jpg`` / ``.jpeg`` ->  PIL.Image.open

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.

    Returns
    -------
    pil_img : PIL.Image.Image (mode='RGB', uint8)
    meta : dict
        For GeoTIFF: full spatial metadata dict.
        For standard images: {width, height, count, filepath, crs, transform, mode}.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    ext = Path(path).suffix.lower()

    if ext in {".tif", ".tiff"}:
        arr, meta = load_geotiff(path)
        bands = arr.shape[0]

        if bands >= 10:
            # Sentinel-2 multispectral: select B04=Red, B03=Green, B02=Blue
            # (0-indexed positions 3, 2, 1 in the band stack)
            rgb_arr = arr[[3, 2, 1], ...]
            logger.debug("Sentinel-2 assumed; using band indices 3,2,1 for RGB")
        elif bands >= 3:
            rgb_arr = arr[:3, ...]
        else:
            # Single-band (SAR amplitude or panchromatic)
            gray = arr[0]
            rgb_arr = np.stack([gray, gray, gray], axis=0)

        pil_img = normalize_optical(rgb_arr)
        meta["mode"] = "geotiff"
        return pil_img, meta

    # Standard format
    pil_img = Image.open(path).convert("RGB")
    meta: Dict[str, Any] = {
        "width": pil_img.width,
        "height": pil_img.height,
        "count": 3,
        "filepath": str(path),
        "crs": None,
        "transform": None,
        "mode": "standard",
    }
    return pil_img, meta


def normalize_optical(arr: np.ndarray) -> Image.Image:
    """
    Convert a (3, H, W) float32 optical array to a uint8 RGB PIL Image.

    Applies a 2-98 percentile linear stretch per the Sentinel-2 display
    convention. Handles any positive-value scale (DN, TOA reflectance, etc.).

    Parameters
    ----------
    arr : np.ndarray  shape (3, H, W)

    Returns
    -------
    PIL.Image.Image (mode='RGB', uint8)
    """
    if arr.ndim != 3 or arr.shape[0] < 3:
        raise ValueError(
            f"normalize_optical expects shape (3, H, W); got {arr.shape}"
        )
    rgb = arr[:3].transpose(1, 2, 0)          # (H, W, 3)
    return Image.fromarray(_percentile_stretch(rgb), mode="RGB")


def normalize_sar(arr: np.ndarray) -> Image.Image:
    """
    Convert a SAR amplitude array to a pseudo-RGB uint8 PIL Image.

    Pseudo-RGB composite (Sentinel-1 dual-pol VV/VH):
        Channel R = VV amplitude (log-scale dB)
        Channel G = VH amplitude (log-scale dB)
        Channel B = VV - VH   (polarisation ratio; highlights urban/water)

    For single-polarisation input all three channels are set equal.

    Parameters
    ----------
    arr : np.ndarray  shape (bands, H, W)
        Linear amplitude or power values.

    Returns
    -------
    PIL.Image.Image (mode='RGB', uint8)
    """
    if arr.ndim != 3:
        raise ValueError(
            f"normalize_sar expects shape (bands, H, W); got {arr.shape}"
        )

    # log-scale for better visual dynamic range
    arr_db = 10.0 * np.log10(np.abs(arr) + 1e-10)

    if arr_db.shape[0] >= 2:
        vv   = arr_db[0]
        vh   = arr_db[1]
        diff = vv - vh
    else:
        vv = vh = arr_db[0]
        diff = np.zeros_like(vv)

    pseudo = np.stack([vv, vh, diff], axis=-1)   # (H, W, 3)
    return Image.fromarray(_percentile_stretch(pseudo), mode="RGB")
