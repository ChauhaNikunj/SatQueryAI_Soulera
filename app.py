import os
import sys
import json
import time
import uuid
import base64
from pathlib import Path

import streamlit as st
from PIL import Image

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from satquery_backend.controller.agentic_router import route_query, TaskType
from satquery_backend.controller.logger import log_execution, ExecutionTraceLogger
from satquery_backend.models.unified_inference import qwen_engine, siamese_engine, fusion_engine

# ============================================================
# SatQuery AI — Streamlit Frontend (SIH PS 26167)
# Clean native-Streamlit implementation
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Landing animation — plays once per browser session, then
# hands off to the Home page. Clip goes in ./assets/ or root
# ------------------------------------------------------------

INTRO_DIR = Path(__file__).parent / "assets"


def _find_intro_video() -> Path:
    """Case-insensitive lookup in ./assets/ and root directory."""
    candidates = [
        INTRO_DIR / "earth_scan.mp4",
        Path(__file__).parent / "earth_scan.mp4",
    ]
    for c in candidates:
        if c.is_file():
            return c
    if INTRO_DIR.is_dir():
        for p in INTRO_DIR.iterdir():
            if p.name.lower().startswith("earth_scan") and p.name.lower().endswith(".mp4"):
                return p
    return candidates[0]


INTRO_VIDEO = _find_intro_video()
INTRO_HOLD_SECONDS = 6  # auto-advance to Home if nobody clicks "Enter"

if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:

    st.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@600;700&display=swap');

    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    section[data-testid="stSidebar"] { display: none !important; }
    div[data-testid="collapsedControl"] {
        display: flex !important;
        z-index: 999999 !important;
    }
    div[data-testid="stDecoration"]:not(:has([data-testid="collapsedControl"])) {
        display: none !important;
    }
    div[data-testid="stToolbar"],
    div[data-testid="stStatusWidget"] { display: none !important; }

    .stApp { background: #050505; }

    .block-container {
        padding: 0 !important;
        max-width: none !important;
    }

    /* Full-bleed, borderless video behind everything */
    div[data-testid="stVideo"] {
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        z-index: 1;
        border-radius: 0 !important;
    }
    div[data-testid="stVideo"] video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: brightness(0.85) saturate(1.05);
    }

    /* Dark vignette so HUD text stays legible over the globe */
    .intro-vignette {
        position: fixed;
        inset: 0;
        z-index: 2;
        pointer-events: none;
        background:
            radial-gradient(circle at 50% 50%, transparent 35%, rgba(0,0,0,.55) 100%),
            linear-gradient(180deg, rgba(0,0,0,.55) 0%, transparent 22%, transparent 72%, rgba(0,0,0,.75) 100%);
    }

    /* Moving scanline sweep */
    .intro-scanline {
        position: fixed;
        left: 0; right: 0;
        height: 2px;
        z-index: 3;
        background: linear-gradient(90deg, transparent, rgba(250,204,21,.85), transparent);
        box-shadow: 0 0 18px 2px rgba(250,204,21,.5);
        animation: satquery-intro-sweep 3s ease-in-out infinite;
    }
    @keyframes satquery-intro-sweep {
        0%   { top: 8%;  opacity: 0; }
        8%   { opacity: 1; }
        50%  { top: 92%; opacity: 1; }
        92%  { opacity: 0; }
        100% { top: 92%; opacity: 0; }
    }

    /* Corner brackets — targeting-reticle framing */
    .intro-bracket {
        position: fixed;
        width: 34px; height: 34px;
        border: 2px solid rgba(250,204,21,.55);
        z-index: 4;
        opacity: .8;
    }
    .intro-bracket.tl { top: 26px;    left: 26px;   border-right: none; border-bottom: none; }
    .intro-bracket.tr { top: 26px;    right: 26px;  border-left: none;  border-bottom: none; }
    .intro-bracket.bl { bottom: 26px; left: 26px;   border-right: none; border-top: none; }
    .intro-bracket.br { bottom: 26px; right: 26px;  border-left: none;  border-top: none; }

    .intro-hud-top {
        position: fixed;
        top: 44px; left: 50%;
        transform: translateX(-50%);
        z-index: 5;
        text-align: center;
        color: #e6e9ef;
        font-family: "IBM Plex Mono", monospace;
        font-size: 11px;
        letter-spacing: 4px;
        text-transform: uppercase;
    }
    .intro-hud-top span { color: #facc15; }

    .intro-title {
        position: fixed;
        top: 50%; left: 50%;
        transform: translate(-50%, -78%);
        z-index: 5;
        text-align: center;
        color: #f5f6f8;
        font-family: "Inter", sans-serif;
        font-weight: 700;
        font-size: clamp(30px, 5vw, 54px);
        letter-spacing: .5px;
        text-shadow: 0 0 40px rgba(0,0,0,.7);
    }
    .intro-subtitle {
        position: fixed;
        top: 50%; left: 50%;
        transform: translate(-50%, 18px);
        z-index: 5;
        text-align: center;
        width: min(560px, 88vw);
        color: #aeb6c4;
        font-family: "IBM Plex Mono", monospace;
        font-size: 12px;
        letter-spacing: 1.5px;
    }

    .intro-caption {
        position: fixed;
        bottom: 148px; left: 50%;
        transform: translateX(-50%);
        z-index: 5;
        color: #7d8799;
        font-family: "IBM Plex Mono", monospace;
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .intro-caption .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #facc15;
        box-shadow: 0 0 0 0 rgba(250,204,21,.35);
        animation: satquery-pulse 1.6s infinite;
    }
    @keyframes satquery-pulse {
        0% { box-shadow: 0 0 0 0 rgba(250,204,21,.35); }
        70% { box-shadow: 0 0 0 7px rgba(250,204,21,0); }
        100% { box-shadow: 0 0 0 0 rgba(250,204,21,0); }
    }

    .intro-progress {
        position: fixed;
        bottom: 122px; left: 50%;
        transform: translateX(-50%);
        z-index: 5;
        width: min(340px, 70vw);
        height: 2px;
        background: rgba(255,255,255,.12);
        border-radius: 2px;
        overflow: hidden;
    }
    .intro-progress > div {
        height: 100%;
        background: #facc15;
        width: 0%;
        animation: satquery-intro-progress __HOLD__s linear forwards;
    }
    @keyframes satquery-intro-progress {
        from { width: 0%; }
        to   { width: 100%; }
    }

    /* Native Streamlit button, repositioned as the CTA */
    .stApp .block-container .stButton {
        position: fixed;
        bottom: 56px; left: 50%;
        transform: translateX(-50%);
        z-index: 10;
        width: min(300px, 82vw);
    }
    .stApp .block-container .stButton > button {
        min-height: 50px;
        width: 100%;
        border-radius: 999px;
        border: none;
        background: #facc15;
        color: #090909;
        font-weight: 700;
        font-family: "IBM Plex Mono", monospace;
        font-size: 12px;
        letter-spacing: 2px;
        text-transform: uppercase;
        box-shadow: 0 12px 34px rgba(250,204,21,.22);
        transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
    }
    .stApp .block-container .stButton > button:hover {
        background: #fde047;
        transform: translateY(-2px);
        box-shadow: 0 16px 40px rgba(250,204,21,.3);
    }
    </style>
    """.replace("__HOLD__", str(INTRO_HOLD_SECONDS)))

    st.html("""
    <div class="intro-vignette"></div>
    <div class="intro-scanline"></div>
    <div class="intro-bracket tl"></div>
    <div class="intro-bracket tr"></div>
    <div class="intro-bracket bl"></div>
    <div class="intro-bracket br"></div>
    <div class="intro-hud-top">SIH 2026 &nbsp;·&nbsp; PS <span>26167</span> &nbsp;·&nbsp; Space Technology</div>
    <div class="intro-title">SatQuery AI</div>
    <div class="intro-subtitle">An interactive vision-language assistant for multimodal remote-sensing image analysis through text queries.</div>
    <div class="intro-caption"><span class="dot"></span> Acquiring orbital feed &amp; initializing agentic controller…</div>
    <div class="intro-progress"><div></div></div>
    """)

    if INTRO_VIDEO.exists():
        try:
            st.video(str(INTRO_VIDEO), autoplay=True, muted=True, loop=True)
        except TypeError:
            st.video(str(INTRO_VIDEO))
    else:
        st.html(f"""
        <div style="position:fixed; inset:0; z-index:0; background:#050505;
                     display:flex; flex-direction:column; align-items:center; justify-content:center;
                     gap:8px; color:#4a5262; font-family:'IBM Plex Mono',monospace; font-size:11px;
                     letter-spacing:2px; text-align:center; padding:0 24px;">
            <div>MISSING VIDEO — ADD YOUR EARTH-SCAN CLIP HERE</div>
            <div style="color:#333c48; font-size:10px; letter-spacing:1px;">
                Expected at: {INTRO_VIDEO.resolve()}
            </div>
        </div>
        """)

    if st.button("Enter SatQuery AI  →", key="intro_enter"):
        st.session_state.entered = True
        st.session_state.page = "Home"
        st.rerun()

    time.sleep(INTRO_HOLD_SECONDS)
    if not st.session_state.entered:
        st.session_state.entered = True
        st.session_state.page = "Home"
        st.rerun()

    st.stop()

# ------------------------------------------------------------
# Global styling
# ------------------------------------------------------------

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #090909;
    --panel: #121212;
    --panel-2: #0e0e0e;
    --border: #282828;
    --border-soft: #202020;
    --text: #f2f2f2;
    --muted: #7d8799;
    --muted-2: #626c7c;
    --yellow: #facc15;
    --yellow-hover: #fde047;
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

/* === SIDEBAR ARROW FIX ===
   Hide decorative chrome — but never a container holding the
   collapsed-sidebar reopen control. */
div[data-testid="stDecoration"]:not(:has([data-testid*="ollapsed" i], [data-testid*="xpand" i])),
div[data-testid="stToolbar"]:not(:has([data-testid*="ollapsed" i], [data-testid*="xpand" i])),
div[data-testid="stStatusWidget"]:not(:has([data-testid*="ollapsed" i], [data-testid*="xpand" i])) {
    display: none !important;
}

/* === PATCH 2: never let the sidebar reopen arrow die === */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedContainer"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
div[data-testid="collapsedControl"] button,
div[data-testid="stSidebarCollapsedContainer"] button {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 999999 !important;
}

.block-container {
    max-width: none !important;
    padding: 54px 34px 50px 34px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 18% 8%, rgba(255,255,255,.035), transparent 28%),
        linear-gradient(180deg, #0d0f12 0%, #090a0c 100%) !important;
    border-right: 1px solid rgba(255,255,255,.075) !important;
    box-shadow: 10px 0 35px rgba(0,0,0,.16) !important;
}

section[data-testid="stSidebar"] > div {
    padding: 18px 12px !important;
}

section[data-testid="stSidebar"] ::-webkit-scrollbar {
    width: 5px;
}
section[data-testid="stSidebar"] ::-webkit-scrollbar-track {
    background: transparent;
}
section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: #252a31;
    border-radius: 10px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 5px 30px 5px;
}

.brand-icon {
    width: 40px;
    height: 40px;
    border-radius: 11px;
    background: var(--yellow);
    color: #090909;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: 700;
    box-shadow: 0 0 22px rgba(250,204,21,.08);
}

.brand-name {
    font-size: 15px;
    font-weight: 700;
    line-height: 1.15;
    color: #f4f5f7;
}

.brand-subtitle {
    margin-top: 4px;
    color: #788294;
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    letter-spacing: 2px;
}

.nav-label {
    margin: 0 12px 8px;
    color: #687080;
    font-family: "IBM Plex Mono", monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] .nav-label {
    color: #596373;
    margin-left: 11px;
}

/* Sidebar navigation */
section[data-testid="stSidebar"] .stButton {
    margin: 0 0 4px 0;
}

section[data-testid="stSidebar"] .stButton > button {
    position: relative !important;
    isolation: isolate !important;
    min-height: 40px;
    width: 100%;
    border: 1px solid transparent;
    border-radius: 11px;
    background: transparent;
    color: #8992a4;
    text-align: left;
    font-size: 14px;
    font-weight: 500;
    padding: 7px 12px;
    box-shadow: none;
    overflow: hidden;
    isolation: isolate;
    transition:
        color .22s ease,
        background .22s ease,
        border-color .22s ease,
        box-shadow .22s ease,
        transform .22s cubic-bezier(.2,.8,.2,1);
}

section[data-testid="stSidebar"] .stButton > button::before {
    content: "";
    position: absolute;
    inset: 1px;
    border-radius: 11px;
    background:
        radial-gradient(circle at 18% 15%, rgba(255,255,255,.10), transparent 35%),
        linear-gradient(105deg, rgba(255,255,255,.075), rgba(255,255,255,.018));
    opacity: 0;
    transform: scale(.96);
    transition: opacity .22s ease, transform .28s cubic-bezier(.2,.8,.2,1);
    pointer-events: none;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,.025);
    color: #f1f2f4;
    border-color: rgba(255,255,255,.10);
    transform: translateX(3px);
    backdrop-filter: blur(13px) saturate(125%);
    -webkit-backdrop-filter: blur(13px) saturate(125%);
    box-shadow:
        0 8px 24px rgba(0,0,0,.20),
        inset 0 1px 0 rgba(255,255,255,.055),
        inset 0 -1px 0 rgba(255,255,255,.018);
}

section[data-testid="stSidebar"] .stButton > button:hover::before {
    opacity: 1;
    transform: scale(1);
}

section[data-testid="stSidebar"] .stButton > button:focus:not(:focus-visible) {
    outline: none !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] .stButton > button:focus-visible {
    outline: none !important;
    border-color: rgba(250,204,21,.34) !important;
    box-shadow:
        0 0 0 1px rgba(250,204,21,.10),
        0 8px 24px rgba(0,0,0,.22) !important;
}

/* Sidebar scrollbar end */

/* Dedicated collapsed-sidebar rail */
header[data-testid="stHeader"] {
    height: 54px !important;
    background: rgba(9,9,9,.82) !important;
    backdrop-filter: blur(16px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(120%) !important;
    border-bottom: 1px solid rgba(255,255,255,.045) !important;
}

/* Sidebar toggle button (Open / Close arrow) */
button[data-testid="stSidebarCollapseButton"],
div[data-testid="collapsedControl"] button {
    position: fixed !important;
    left: 12px !important;
    top: 8px !important;
    width: 38px !important;
    height: 38px !important;
    margin: 0 !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 12px !important;
    background: rgba(25, 27, 30, 0.88) !important;
    backdrop-filter: blur(14px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(14px) saturate(120%) !important;
    color: #9ca3af !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32) !important;
    transition: all 0.18s ease !important;
    z-index: 999999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

button[data-testid="stSidebarCollapseButton"]:hover,
div[data-testid="collapsedControl"] button:hover {
    color: #facc15 !important;
    background: rgba(35, 37, 40, 0.95) !important;
    border-color: rgba(250, 204, 21, 0.32) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45), 0 0 18px rgba(250, 204, 21, 0.1) !important;
}

button[data-testid="stSidebarCollapseButton"]::after {
    content: "Open sidebar";
    position: absolute;
    left: 50px;
    top: 50%;
    transform: translateY(-50%) translateX(-4px);
    padding: 6px 9px;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 8px;
    background: rgba(22,24,27,.88);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: #d1d5db;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: .2px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity .16s ease, transform .16s ease;
    box-shadow: 0 8px 24px rgba(0,0,0,.28);
}

button[data-testid="stSidebarCollapseButton"]:hover::after {
    opacity: 1;
    transform: translateY(-50%) translateX(0);
}

.page-head {
    padding-left: 78px !important;
}

.page-head > div:first-child {
    margin-left: 0 !important;
}

/* Top heading */
.page-head {
    min-height: 114px;
    margin: 0 -34px 25px -34px;
    padding: 25px 34px;
    border-bottom: 1px solid var(--border-soft);
    background: #0a0a0a;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
}

.page-title {
    margin: 0;
    color: var(--text);
    font-size: 34px;
    font-weight: 700;
    letter-spacing: -1.3px;
    line-height: 1;
}

.page-subtitle {
    margin-top: 9px;
    color: #738097;
    font-family: "IBM Plex Mono", monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.status {
    margin-top: 0;
    border: 1px solid #292929;
    border-radius: 20px;
    background: #151515;
    color: #e5e7eb;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 600;
}

.status-dot {
    color: var(--yellow);
}

/* Cards */
.card-title {
    font-size: 17px;
    font-weight: 700;
}

.card-header-row {
    min-height: 62px;
    padding: 0 20px;
    border-bottom: 1px solid #252525;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.function-tag {
    border: 1px solid #292929;
    border-radius: 9px;
    background: #1a1a1a;
    color: var(--yellow);
    padding: 6px 10px;
    font-family: "IBM Plex Mono", monospace;
    font-size: 11px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important;
    border-radius: 17px !important;
    background: var(--panel) !important;
    transition: border-color .18s ease, background .18s ease, box-shadow .18s ease, transform .18s ease !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(255,255,255,.11) !important;
    box-shadow: 0 12px 34px rgba(0,0,0,.16), inset 0 1px 0 rgba(255,255,255,.025) !important;
}

.description {
    color: #7f8ba0;
    font-size: 14px;
    margin-bottom: 16px;
}

.mono-label {
    color: #738097;
    font-family: "IBM Plex Mono", monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Compact attachment / Browse files controls */
section[data-testid="stFileUploaderDropzone"] {
    position: relative !important;
    min-height: 48px !important;
    height: 48px !important;
    padding: 7px 10px !important;
    background: #0d0d0d !important;
    border: 1px dashed #303030 !important;
    border-radius: 11px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: visible !important;
    transition: border-color .18s ease, background .18s ease, box-shadow .18s ease !important;
}

section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
}

section[data-testid="stFileUploaderDropzone"] > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    padding: 0 !important;
}

section[data-testid="stFileUploaderDropzone"] button {
    min-height: 30px !important;
    height: 30px !important;
    padding: 3px 10px !important;
    margin: 0 auto !important;
    position: relative !important;
    background: #191919 !important;
    border-color: #303030 !important;
    color: #d7dbe2 !important;
    border-radius: 8px !important;
    transition: transform .15s ease, background .15s ease, border-color .15s ease !important;
}

section[data-testid="stFileUploaderDropzone"] button:hover {
    transform: translateY(-1px) !important;
    background: #222 !important;
    border-color: #facc15 !important;
}

section[data-testid="stFileUploaderDropzone"]:hover::after {
    content: "Max file size: 200 MB";
    position: absolute;
    left: 50%;
    top: -34px;
    transform: translateX(-50%);
    white-space: nowrap;
    padding: 6px 9px;
    border: 1px solid #303030;
    border-radius: 7px;
    background: #171717;
    color: #d7dbe2;
    font-size: 11px;
    font-weight: 500;
    line-height: 1.2;
    pointer-events: none;
    z-index: 999999;
    box-shadow: 0 5px 18px rgba(0,0,0,.35);
}

/* Text areas */
.stTextArea textarea {
    background: #0d0d0d !important;
    color: #e5e7eb !important;
    border: 1px solid #292929 !important;
    border-radius: 13px !important;
    font-size: 14px !important;
    transition: border-color .18s ease, box-shadow .18s ease, background .18s ease !important;
}

.stTextArea textarea:focus {
    border-color: var(--yellow) !important;
    box-shadow: 0 0 0 1px var(--yellow) !important;
}

.stTextArea textarea:focus::placeholder {
    opacity: 1 !important;
    color: #687489 !important;
}

.stTextArea textarea:hover {
    border-color: #3a3d43 !important;
    background: #101010 !important;
}

/* Evidence panel hover */
.evidence-empty {
    transition: background .2s ease, color .2s ease !important;
}

.evidence-empty:hover {
    background: #0f0f0f !important;
    color: #687489 !important;
}

/* Status / function pills */
.function-tag,
.status,
.analysis-badge {
    transition: border-color .16s ease, background .16s ease, color .16s ease, box-shadow .16s ease !important;
}

.function-tag:hover,
.status:hover,
.analysis-badge:hover {
    border-color: rgba(250,204,21,.28) !important;
    background: #1d1d1a !important;
    box-shadow: 0 0 16px rgba(250,204,21,.045) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 11px;
    border: 1px solid #303030;
    background: #171717;
    color: #dce1e8;
}

.stButton > button:hover {
    border-color: #454545;
    background: #202020;
    color: #fff;
}

.run-button .stButton > button {
    min-height: 44px;
    position: relative;
    background: var(--yellow) !important;
    color: #090909 !important;
    border-color: var(--yellow) !important;
    font-weight: 700 !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
}

.run-button .stButton > button:hover {
    background: var(--yellow-hover) !important;
    border-color: var(--yellow-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 7px 20px rgba(250,204,21,.10) !important;
}

.run-button .stButton > button:active {
    transform: translateY(0) scale(.985) !important;
}

/* Evidence image framing */
.evidence-image-frame {
    position: relative;
    overflow: hidden;
    border-radius: 12px;
    background: #0c0c0c;
}

.evidence-image-frame img {
    display: block;
}

/* Result/status badges */
.analysis-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 9px;
    border: 1px solid #292929;
    border-radius: 999px;
    background: #151515;
    color: #aeb6c4;
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    letter-spacing: 1px;
}

.analysis-pulse {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--yellow);
    box-shadow: 0 0 0 0 rgba(250,204,21,.35);
    animation: satquery-pulse 1.6s infinite;
}

div[data-testid="stAlert"] {
    border-radius: 11px !important;
}

/* Evidence empty area */
.evidence-empty {
    min-height: 360px;
    background: #0c0c0c;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #596273;
    font-family: "IBM Plex Mono", monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    transition: background .2s ease, color .2s ease !important;
}

.evidence-empty:hover {
    background: #0f0f0f !important;
    color: #687489 !important;
}

.trace {
    min-height: 210px;
    padding: 20px;
    border-top: 1px solid #252525;
}

/* Metrics */
.metric-icon {
    width: 35px;
    height: 35px;
    border-radius: 9px;
    background: #1b1b1b;
    color: var(--yellow);
    display: flex;
    align-items: center;
    justify-content: center;
}

.metric-value {
    margin-top: 16px;
    color: #d2d6dd;
    font-size: 28px;
    font-weight: 700;
}

.metric-value.empty {
    color: #687489;
}

.metric-label {
    margin-top: 5px;
    color: #687489;
    font-family: "IBM Plex Mono", monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
}

/* Map */
.map-empty {
    min-height: 575px;
    border: 1px solid #292929;
    border-radius: 16px;
    background:
        linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px),
        #202020;
    background-size: 32px 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #5f6671;
    font-family: "IBM Plex Mono", monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
}

/* Alerts */
.empty-notice {
    border: 1px solid #4a3d00;
    border-radius: 14px;
    background: #17150a;
    color: #8c8370;
    padding: 14px 17px;
    font-size: 13px;
}

/* Responsive */
@media (max-width: 900px) {
    .block-container {
        padding: 54px 18px 40px 18px !important;
    }

    .page-head {
        margin-left: -18px;
        margin-right: -18px;
        padding-left: 62px !important;
        padding-right: 18px;
    }

    .page-title {
        font-size: 28px;
    }
}

/* ============================================================
   SIDEBAR TARGET ACQUISITION
   ============================================================ */

/* ACTIVE / LOCKED STATE */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #090a0c !important;
    color: #facc15 !important;
    border: 1px solid #292b30 !important;
    box-shadow:
        inset 3px 0 0 #facc15,
        inset -3px 0 0 #facc15 !important;
    transform: none !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] * {
    color: #facc15 !important;
}

/* SCAN BEAM */
section[data-testid="stSidebar"] .stButton > button:focus::before,
section[data-testid="stSidebar"] .stButton > button:focus-visible::before {
    content: "";
    position: absolute;
    top: 3px;
    bottom: 3px;
    left: -28px;
    width: 2px;
    border-radius: 50%;
    background: #fff8bd;
    box-shadow:
        0 0 5px #facc15,
        0 0 13px rgba(250,204,21,.95),
        0 0 30px rgba(250,204,21,.55);
    opacity: 0;
    z-index: 5;
    pointer-events: none;
    animation: satquery-target-scan 1.55s cubic-bezier(.18,.72,.16,1) 1;
}

@keyframes satquery-target-scan {
    0%   { left: -28px; opacity: 0; }
    7%   { opacity: 1; }
    52%  { opacity: .9; }
    88%  { opacity: .55; }
    100% { left: calc(100% + 28px); opacity: 0; }
}

/* LOCKED-ON STATE */
section[data-testid="stSidebar"] .stButton > button[kind="primary"]::after {
    content: "";
    position: absolute;
    inset: 5px;
    border: 1px solid rgba(250,204,21,.22);
    border-radius: 7px;
    opacity: .55;
    pointer-events: none;
    z-index: 3;
    animation: satquery-lock 2.6s ease-in-out infinite;
}

@keyframes satquery-lock {
    0%, 100% { opacity: .32; inset: 5px; }
    50%      { opacity: .72; inset: 3px; }
}

@media (prefers-reduced-motion: reduce) {
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]::after {
        animation: none !important;
        opacity: .5 !important;
    }
}

/* Prevent accidental text highlighting on UI chrome. */
.stApp,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp span, .stApp label, .stApp button,
.stApp .nav-label, .stApp .brand, .stApp .brand-name, .stApp .brand-subtitle,
.stApp .page-title, .stApp .page-subtitle, .stApp .card-title,
.stApp .function-tag, .stApp .status, .stApp .analysis-badge,
.stApp .mono-label, .stApp .description, .stApp .metric-label,
.stApp .metric-value, .stApp .sidebar-collapse {
    user-select: none !important;
    -webkit-user-select: none !important;
}

.stApp textarea,
.stApp input,
.stApp [contenteditable="true"],
.stApp pre,
.stApp code {
    user-select: text !important;
    -webkit-user-select: text !important;
}

.stApp img,
.stApp svg {
    user-select: none !important;
    -webkit-user-drag: none !important;
}

</style>
""")

# ------------------------------------------------------------
# State
# ------------------------------------------------------------

PAGES = [
    "Home",
    "Visual Q&A",
    "Captioning",
    "Change Detection",
    "Optical-SAR Fusion",
    "Explore map",
    "Insights",
]

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.html("""
    <div class="brand">
        <div class="brand-icon">🛰</div>
        <div>
            <div class="brand-name">SatQuery AI</div>
            <div class="brand-subtitle">GEOSPATIAL VLM</div>
        </div>
    </div>
    """)

    home_active = st.session_state.page == "Home"
    if st.button(
        "🏠  Home",
        key="nav_Home",
        use_container_width=True,
        type="primary" if home_active else "secondary",
    ):
        st.session_state.page = "Home"
        st.rerun()

    st.html('<div class="nav-label" style="margin-top:18px;">Tasks</div>')

    for page in PAGES[1:5]:
        active = st.session_state.page == page
        if st.button(
            page,
            key=f"nav_{page}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.page = page
            st.rerun()

    st.html('<div class="nav-label" style="margin-top:22px;">Workspace</div>')

    for page in PAGES[5:]:
        active = st.session_state.page == page
        if st.button(
            page,
            key=f"nav_{page}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.page = page
            st.rerun()

    st.html("""
    <div style="height: calc(100vh - 510px); min-height: 120px;"></div>
    <div class="sidebar-collapse">
        ◧ &nbsp; Workspace
    </div>
    """)

# ------------------------------------------------------------
# Page data
# ------------------------------------------------------------

PAGE_INFO = {
    "Visual Q&A": (
        "Evidence-grounded satellite analysis · prototype",
        "answer_vqa()",
        "Ask about land cover, objects or scene content in a single tile.",
        "DEMO",
    ),
    "Captioning": (
        "Evidence-grounded satellite analysis · prototype",
        "describe_scene()",
        "Generate a grounded natural-language description of the tile.",
        "DEMO",
    ),
    "Change Detection": (
        "Evidence-grounded satellite analysis · prototype",
        "detect_change()",
        "Compare two co-registered tiles and localize what changed.",
        "DEMO",
    ),
    "Optical-SAR Fusion": (
        "Evidence-grounded satellite analysis · prototype",
        "fuse_optical_sar()",
        "Combine optical and SAR returns for all-weather interpretation.",
        "DEMO",
    ),
    "Explore map": (
        "Interactive basemap · illustrative",
        "",
        "",
        "MOCK",
    ),
    "Insights": (
        "Usage analytics · synthetic",
        "",
        "",
        "MOCK",
    ),
    "Home": (
        "SIH 2026 · PS 26167 · Space Technology (ISRO-SAC) · Team Soluera",
        "",
        "",
        "OVERVIEW",
    ),
}

subtitle, function_name, description, status = PAGE_INFO[st.session_state.page]

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

if st.session_state.page != "Home":
    st.html(f"""
    <div class="page-head">
        <div>
            <div class="page-title">{st.session_state.page}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        <div class="status"><span class="status-dot">●</span>&nbsp; {status}</div>
    </div>
    """)

# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.html("""
    <style>
    .home-hero {
        padding: 46px 40px 40px 40px;
        margin: 0 -34px 28px -34px;
        border-bottom: 1px solid #202020;
        background:
            radial-gradient(circle at 12% 0%, rgba(250,204,21,.06), transparent 45%),
            linear-gradient(180deg, #0d0d0d 0%, #090909 100%);
    }
    .home-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #facc15;
        font-family: "IBM Plex Mono", monospace;
        font-size: 10px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        border: 1px solid rgba(250,204,21,.25);
        border-radius: 999px;
        padding: 6px 13px;
        background: rgba(250,204,21,.05);
    }
    .home-title { margin-top: 20px; font-size: 40px; font-weight: 700; color: #f5f6f8; line-height: 1.15; }
    .home-title span { color: #facc15; }
    .home-tagline { margin-top: 12px; max-width: 680px; color: #9aa2b1; font-size: 15px; line-height: 1.55; }
    .home-cta-row { display: flex; gap: 14px; margin-top: 26px; flex-wrap: wrap; }
    .st-key-home_launch button {
        background: var(--yellow) !important; color: #090909 !important; font-weight: 700 !important;
        border: none !important; border-radius: 11px !important; padding: 10px 22px !important;
        font-family: "IBM Plex Mono", monospace !important; font-size: 12.5px !important;
        letter-spacing: 1px !important; box-shadow: 0 10px 26px rgba(250,204,21,.18) !important;
        transition: transform .18s ease, box-shadow .18s ease, background .18s ease !important;
    }
    .st-key-home_launch button:hover {
        background: var(--yellow-hover) !important; transform: translateY(-2px);
        box-shadow: 0 14px 32px rgba(250,204,21,.25) !important;
    }
    .st-key-home_map button {
        background: #171717 !important; color: #dce1e8 !important; border: 1px solid #303030 !important;
        border-radius: 11px !important; padding: 10px 22px !important;
        font-family: "IBM Plex Mono", monospace !important; font-size: 12.5px !important;
        font-weight: 600 !important; letter-spacing: 1px !important;
        transition: transform .18s ease, background .18s ease, border-color .18s ease !important;
    }
    .st-key-home_map button:hover {
        border-color: #454545 !important; background: #202020 !important; color: #fff !important; transform: translateY(-2px);
    }
    .home-section-label {
        font-family: "IBM Plex Mono", monospace; font-size: 10px; letter-spacing: 2.5px;
        text-transform: uppercase; color: #687489; margin: 6px 0 14px 2px;
    }
    .home-card { border: 1px solid #232323; border-radius: 14px; background: #101010; padding: 20px 22px; height: 100%; }
    .home-card h4 { margin: 0 0 8px 0; color: #f0f1f4; font-size: 15px; font-weight: 600; }
    .home-card p { margin: 0; color: #8b93a3; font-size: 13px; line-height: 1.55; }
    .home-card .icon {
        width: 34px; height: 34px; border-radius: 9px; background: #1b1b1b; color: #facc15;
        display: flex; align-items: center; justify-content: center; margin-bottom: 12px; font-size: 16px;
    }
    .home-chip {
        display: inline-flex; align-items: center; border: 1px solid #292929; border-radius: 999px;
        background: #141414; color: #b7bec9; font-family: "IBM Plex Mono", monospace;
        font-size: 11px; letter-spacing: .5px; padding: 6px 12px; margin: 0 8px 8px 0;
    }
    .home-step { display: flex; gap: 14px; padding: 16px 0; border-bottom: 1px solid #1c1c1c; }
    .home-step:last-child { border-bottom: none; }
    .home-step .num {
        flex: none; width: 28px; height: 28px; border-radius: 8px; background: #1b1b1b; color: #facc15;
        font-family: "IBM Plex Mono", monospace; font-size: 12px;
        display: flex; align-items: center; justify-content: center;
    }
    .home-step .body h5 { margin: 2px 0 4px 0; color: #eceef1; font-size: 13.5px; font-weight: 600; }
    .home-step .body p { margin: 0; color: #8b93a3; font-size: 12.5px; line-height: 1.5; }
    .home-team-card {
        border: 1px solid #232323; border-radius: 14px;
        background: radial-gradient(circle at 100% 0%, rgba(250,204,21,.05), transparent 55%), #101010;
        padding: 24px;
    }
    .home-team-badge {
        display: inline-block; color: #687489; font-family: "IBM Plex Mono", monospace;
        font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
    }
    </style>
    """)

    st.html("""
    <div class="home-hero">
        <span class="home-eyebrow">● Smart India Hackathon 2026 &nbsp;·&nbsp; PS 26167 &nbsp;·&nbsp; Remote Sensing / ISRO-SAC</span>
        <div class="home-title">Sat<span>Query</span> AI</div>
        <div class="home-tagline">
            An agentic, query-driven vision-language assistant: a user uploads satellite image(s)
            and asks a plain-English question; a controller figures out which specialist model to
            run — visual Q&A, captioning/grounding, bi-temporal change detection, or optical–SAR
            fusion — and returns an evidence-grounded answer with visual proof, a confidence
            score, and an auditable execution trace. Not "call a general VLM on a satellite
            image": real domain adaptation on BigEarthNet, evaluated on VRSBench, RSVQA and CDVQA.
        </div>
        <div class="home-cta-row">
            <span class="home-chip">Project type: Agentic VLM system (Optical + SAR + Multitemporal)</span>
            <span class="home-chip">Deliverable: Web app + fine-tuned models + agentic backend</span>
        </div>
    </div>
    """)

    cta_l, cta_r, _ = st.columns([0.16, 0.16, 0.68])
    with cta_l:
        if st.button("▶  Launch console", key="home_launch"):
            st.session_state.page = "Visual Q&A"
            st.rerun()
    with cta_r:
        if st.button("Explore map", key="home_map"):
            st.session_state.page = "Explore map"
            st.rerun()

    st.write("")
    st.write("")

    # --- The four tasks -------------------------------------
    st.html('<div class="home-section-label">What SatQuery AI does</div>')
    task_cols = st.columns(2, gap="medium")
    tasks = [
        ("❓", "Single-image VQA — baseline task",
         "Upload one optical, multispectral or SAR tile and ask a plain-English question about "
         "land cover, water bodies or scene content. Runs on a compact remote-sensing-adapted "
         "VLM; evaluated on VRSBench + RSVQA."),
        ("🖼", "Captioning / grounding",
         "A grounded natural-language description of the tile, with object referring expressions "
         "(bounding boxes from VRSBench) as the spatial-output stretch — geometry rendered "
         "directly on the evidence image."),
        ("⏱", "Change detection + change-VQA",
         "Bi-temporal pair analysis: what changed between two dates, where, and by how much — "
         "with a localized change mask. Built on the explicit multi-time fusion formulation "
         "of CDVQA."),
        ("📡", "Optical–SAR fusion (mandatory adaptation)",
         "Cross-modal joint analysis of co-registered Sentinel-2 (optical) and Sentinel-1 (SAR) "
         "— day/night, all-weather answers even when clouds blind the optical sensor. Adapted "
         "on BigEarthNet, the mandatory domain-adaptation dataset."),
    ]
    for col, (icon, title, body) in zip(task_cols, tasks):
        with col:
            st.html(f"""
            <div class="home-card">
                <div class="icon">{icon}</div>
                <h4>{title}</h4>
                <p>{body}</p>
            </div>
            """)

    st.html('<div class="home-section-label">Orchestration layer</div>')
    sol_cols = st.columns(3, gap="medium")
    solutions = [
        ("◆", "Agentic controller",
         "Rules-based query-intent classifier + input compatibility checker + model registry. "
         "Routes every request to the right specialist with zero manual tool switching, and "
         "emits a fully auditable execution trace."),
        ("📊", "Evidence-grounded output",
         "Every answer ships with visual overlays (bounding boxes, change masks), a separately "
         "scored evidence-quality indicator, and latency — language correctness and spatial "
         "correctness are validated as independent fields."),
        ("📄", "Downloadable execution report",
         "Every query appends to execution_trace.jsonl — task, models, parameters, outputs, "
         "confidence, timestamp — downloadable as a PDF/JSON report for judge review. "
         "Auditable agentic behaviour, not a black box."),
    ]
    for col, (icon, title, body) in zip(sol_cols, solutions):
        with col:
            st.html(f"""
            <div class="home-card">
                <div class="icon">{icon}</div>
                <h4>{title}</h4>
                <p>{body}</p>
            </div>
            """)

    st.write("")

    # --- Problem statement -----------------------------------
    st.html('<div class="home-section-label">The problem</div>')
    with st.container(border=True):
        st.html("""
        <p style="color:#c3c9d3; font-size:14px; line-height:1.7; margin:0;">
            Satellite imagery is abundant, but reading it still requires GIS specialists and
            slow, multi-tool workflows. Analysts must combine optical and SAR sources, compare
            scenes across time, and manually justify every conclusion — a process that doesn't
            scale for disaster response, defense, or environmental monitoring, where answers
            are needed in minutes, not hours. Generic VLMs don't help out of the box: they
            don't understand multispectral bands or SAR speckle, which is why the problem
            statement mandates real domain adaptation (fine-tuning on BigEarthNet) rather than
            a generic "call a big VLM" shortcut.
        </p>
        """)

    st.write("")

    # --- One query, start to finish --------------------------
    st.html('<div class="home-section-label">One query, start to finish</div>')
    with st.container(border=True):
        steps = [
            ("Ask", "Upload one or two tiles (optical, multispectral or SAR) and type a plain-English question."),
            ("Route", "The agentic controller classifies intent, validates input compatibility, and selects the specialist model(s) from the registry."),
            ("Analyse", "The specialist runs — visual Q&A, captioning/grounding, change detection, or optical–SAR fusion — grounded in the source imagery."),
            ("Verify", "The answer returns with visual evidence, calibrated scores, and a logged execution trace covering task, models, parameters and confidence."),
        ]
        for i, (title, body) in enumerate(steps, start=1):
            st.html(f"""
            <div class="home-step">
                <div class="num">{i:02d}</div>
                <div class="body"><h5>{title}</h5><p>{body}</p></div>
            </div>
            """)

    st.write("")

    # --- Requirement → model → dataset ------------------------
    st.html('<div class="home-section-label">Requirement → model → dataset</div>')
    req_cols = st.columns(2, gap="medium")
    reqs = [
        ("❓", "Single-image VQA — baseline",
         "Compact remote-sensing-adapted VLM (Qwen2.5-VL-3B 4-bit / Qwen3-VL-2B bake-off) evaluated on VRSBench / RSVQA subsets."),
        ("🖼", "Captioning / grounding",
         "Same backbone, multi-task; boxes and masks emitted as token sequences so the frontend renders evidence deterministically (GeoChat / GeoGround lessons)."),
        ("⏱", "Change detection + change-VQA",
         "CDVQA-style Siamese dual-image specialist with explicit temporal fusion — the paper shows the fusion mechanism matters more than backbone size."),
        ("📡", "Optical–SAR fusion (mandatory BigEarthNet adaptation)",
         "Paired Sentinel-1/Sentinel-2 encoders + small fusion head on a stratified, reproducible BigEarthNet subset; cross-sensor benefit measured against S1-only / S2-only baselines."),
    ]
    for col, (icon, title, body) in zip(req_cols, reqs):
        with col:
            st.html(f"""
            <div class="home-card">
                <div class="icon">{icon}</div>
                <h4>{title}</h4>
                <p>{body}</p>
            </div>
            """)

    st.write("")

    # --- Architecture + tech stack ----------------------------
    imp_col, tech_col = st.columns([1.1, 1], gap="large")

    with imp_col:
        st.html('<div class="home-section-label">Architecture — five layers</div>')
        with st.container(border=True):
            st.html("""
            <ul style="margin:0; padding-left:18px; color:#b7bec9; font-size:13px; line-height:1.85;">
                <li><b style="color:#eceef1;">Frontend / UI</b> — upload + compatibility panel, query console, evidence overlay, trace panel.</li>
                <li><b style="color:#eceef1;">Agentic controller</b> — query classifier → compatibility check → model registry → execution sequencer → confidence + trace log.</li>
                <li><b style="color:#eceef1;">Single-image models</b> — VQA + captioning/grounding.</li>
                <li><b style="color:#eceef1;">Temporal group</b> — change description + change-VQA.</li>
                <li><b style="color:#eceef1;">Optical–SAR fusion</b> — dual-encoder cross-modal analysis.</li>
                <li><b style="color:#eceef1;">Data / MLOps / infra</b> — datasets, preprocessing, GPU training, deployment, demo curation.</li>
            </ul>
            """)

    with tech_col:
        st.html('<div class="home-section-label">Tech stack</div>')
        with st.container(border=True):
            chips = [
                "Qwen2.5-VL-3B (4-bit)", "Siamese ResNet18", "RemoteCLIP ViT-B/32",
                "PyTorch", "HuggingFace", "rasterio / GDAL",
                "FastAPI", "Streamlit", "JSONL trace logger",
                "VRSBench", "RSVQA", "CDVQA", "BigEarthNet",
            ]
            st.html("".join(f'<span class="home-chip">{c}</span>' for c in chips))

    st.write("")

    # --- Build phases -----------------------------------------
    st.html('<div class="home-section-label">Build phases</div>')
    phase_cols = st.columns(3, gap="medium")
    phases = [
        ("◆", "Setup & data",
         "Contract-first JSON API schema locked before any training; mono-repo, shared GPU "
         "accounts, dataset downloads started early. Mock endpoints so nobody blocks."),
        ("▣", "Baselines → training",
         "Un-fine-tuned sanity passes prove plumbing; heaviest GPU job (BigEarthNet dual "
         "encoder) prioritized; real intent classifier + evidence overlays built against mocks in parallel."),
        ("▣", "Integrate & polish",
         "Swap mocks for real trained models, formal evaluation on held-out test splits, "
         "confidence calibration, curated demo imagery, two full dry-runs."),
    ]
    for col, (icon, title, body) in zip(phase_cols, phases):
        with col:
            st.html(f"""
            <div class="home-card">
                <div class="icon">{icon}</div>
                <h4>{title}</h4>
                <p>{body}</p>
            </div>
            """)

    st.write("")
    st.html("""
    <div class="empty-notice" style="margin-bottom:10px;">
        <strong style="color:#facc15;">ⓘ</strong>
        &nbsp; Honest constraints: zero-shot prompting for the conversational VLM, one real
        training module (temporal specialist), frozen encoders for fusion — working-but-imperfect
        beats nothing, and a pre-cached curated demo set protects the live demo from Wi-Fi/GPU risk.
    </div>
    """)

    st.write("")

    # --- Team -----------------------------------------------
    st.html('<div class="home-section-label">Team</div>')
    with st.container(border=True):
        st.html("""
        <div class="home-team-card">
            <div class="home-team-badge">SIH 2026 &nbsp;·&nbsp; PS 26167 &nbsp;·&nbsp; Team Soluera</div>
            <div style="margin-top:8px; font-size:22px; font-weight:700; color:#f5f6f8;">
                Team Soluera — 6 members
            </div>
            <p style="margin:10px 0 0 0; color:#8b93a3; font-size:13px; line-height:1.6; max-width:720px;">
                Three ML specialists (single-image VQA/captioning, change detection, optical–SAR
                fusion), one systems lead (agentic controller + FastAPI backend), one frontend
                lead (console, evidence overlay, trace panel), and one data/MLOps lead
                (datasets, preprocessing, GPU compute, timeline, demo curation).
                <i>Names go here once finalised.</i>
            </p>
        </div>
        """)

    st.write("")
    st.html("""
    <div class="empty-notice" style="margin-bottom:10px;">
        <strong style="color:#facc15;">ⓘ</strong>
        &nbsp; This build is a working prototype for SIH 2026 evaluation — inference backends
        connect the task consoles to live models ahead of the demo.
    </div>
    """)

# ============================================================
# TASK PAGES
# ============================================================

elif st.session_state.page in PAGES[1:5]:

    left, right = st.columns([1, 1], gap="large")

    # --------------------------------------------------------
    # LEFT — Query
    # --------------------------------------------------------

    with left:
        with st.container(border=True):

            st.html(f"""
            <div class="card-header-row" style="margin:-1px -1px 0 -1px;">
                <div class="card-title">Query</div>
                <div class="function-tag">{function_name}</div>
            </div>
            """)

            st.write("")

            st.html(f'<div class="description">{description}</div>')

            st.html("""
            <div style="display:flex; gap:7px; flex-wrap:wrap; margin:-6px 0 13px;">
                <span class="analysis-badge">SATELLITE ANALYSIS</span>
                <span class="analysis-badge">EVIDENCE GROUNDED</span>
            </div>
            """)

            # Presets dictionary for common remote sensing questions
            PRESETS = {
                "Visual Q&A": [
                    "What type of buildings and roads are visible in this tile?",
                    "Is there any water body or river visible?",
                    "What is the dominant land cover category?",
                    "Identify infrastructure and built-up structures.",
                ],
                "Captioning": [
                    "Provide a comprehensive scene caption with infrastructure and terrain.",
                    "Focus on urban density and transportation networks.",
                    "Focus on agricultural parcels and vegetation cover.",
                    "Describe the coastal / wetland features visible.",
                ],
                "Change Detection": [
                    "What changed between T1 and T2?",
                    "Did urban construction and built-up area increase?",
                    "Has vegetation or water body decreased?",
                    "Quantify the semantic land cover transition.",
                ],
                "Optical-SAR Fusion": [
                    "Cross-modal surface cover classification",
                    "What terrain is present under cloud cover?",
                    "Compare optical reflectance vs SAR backscatter returns",
                    "Identify urban structures obscured in the optical band",
                ],
            }

            cur_presets = PRESETS.get(st.session_state.page, [])
            st.html('<div class="mono-label" style="margin-bottom:6px;">QUERY PRESETS</div>')
            sel_preset = st.pills(
                "Select preset",
                cur_presets,
                key=f"pills_{st.session_state.page}",
                label_visibility="collapsed",
            )

            DEMO_DIR = _ROOT / "demo"
            TEMP_UPLOADS = _ROOT / "temp_uploads"
            TEMP_UPLOADS.mkdir(parents=True, exist_ok=True)

            if st.session_state.page in ["Visual Q&A", "Captioning"]:

                default_q = sel_preset or (
                    "What type of buildings and roads are visible in this tile?"
                    if st.session_state.page == "Visual Q&A"
                    else "Provide a comprehensive scene caption with infrastructure and terrain."
                )

                if st.session_state.page == "Visual Q&A":
                    question = st.text_area(
                        "Question",
                        value=default_q,
                        placeholder="Ask a plain-English question...",
                        height=90,
                        key=f"vqa_q_{st.session_state.page}",
                    )
                    sample_file = DEMO_DIR / "vqa_sample.png"
                else:
                    question = st.text_area(
                        "Caption guidance",
                        value=default_q,
                        placeholder='Optional: steer the caption (e.g. "focus on infrastructure")...',
                        height=90,
                        key=f"cap_q_{st.session_state.page}",
                    )
                    sample_file = DEMO_DIR / "caption_sample.png"

                uploaded = st.file_uploader(
                    "Browse files",
                    type=["png", "jpg", "jpeg", "tif", "tiff"],
                    key=f"upload_{st.session_state.page}",
                    label_visibility="collapsed",
                )

                if uploaded:
                    st.image(uploaded, use_container_width=True, caption=f"Uploaded: {uploaded.name}")
                else:
                    if sample_file.exists():
                        st.image(str(sample_file), use_container_width=True, caption=f"Demo scene active: {sample_file.name}")

                st.html(
                    '<div class="mono-label" style="font-style:italic; margin-top:5px; color:#687489;">'
                    'Leave image empty to run the pre-cached demo scene.'
                    '</div>'
                )

            else:
                if st.session_state.page == "Change Detection":
                    query_label = "Change query"
                    default_q = sel_preset or "What changed between T1 and T2?"
                    label_a = "IMAGE A (T1 PRE-CHANGE)"
                    label_b = "IMAGE B (T2 POST-CHANGE)"
                    sample_a = DEMO_DIR / "t1_delhi_2021.png"
                    sample_b = DEMO_DIR / "t2_delhi_2025.png"
                else:
                    query_label = "Fusion query"
                    default_q = sel_preset or "Cross-modal surface cover classification"
                    label_a = "OPTICAL (SENTINEL-2 RGB)"
                    label_b = "SAR (SENTINEL-1 C-BAND)"
                    sample_a = DEMO_DIR / "optical_sentinel2.png"
                    sample_b = DEMO_DIR / "sar_sentinel1.png"

                question = st.text_area(
                    query_label,
                    value=default_q,
                    placeholder="Enter query or leave default...",
                    height=90,
                    key=f"query_{st.session_state.page}",
                )

                attach_a, attach_b = st.columns(2, gap="small")

                with attach_a:
                    st.html(f'<div class="mono-label" style="margin-bottom:4px;">{label_a}</div>')
                    img_a = st.file_uploader(
                        "Browse file A",
                        type=["png", "jpg", "jpeg", "tif", "tiff"],
                        key=f"img_a_{st.session_state.page}",
                        label_visibility="collapsed",
                    )
                    if img_a:
                        st.image(img_a, use_container_width=True, caption=f"Uploaded: {img_a.name}")
                    elif sample_a.exists():
                        st.image(str(sample_a), use_container_width=True, caption=f"Demo: {sample_a.name}")

                with attach_b:
                    st.html(f'<div class="mono-label" style="margin-bottom:4px;">{label_b}</div>')
                    img_b = st.file_uploader(
                        "Browse file B",
                        type=["png", "jpg", "jpeg", "tif", "tiff"],
                        key=f"img_b_{st.session_state.page}",
                        label_visibility="collapsed",
                    )
                    if img_b:
                        st.image(img_b, use_container_width=True, caption=f"Uploaded: {img_b.name}")
                    elif sample_b.exists():
                        st.image(str(sample_b), use_container_width=True, caption=f"Demo: {sample_b.name}")

                st.html(
                    '<div class="mono-label" style="font-style:italic; margin-top:5px; color:#687489;">'
                    'Leave images empty to run the pre-cached demo pair.'
                    '</div>'
                )

            st.write("")

            st.markdown('<div class="run-button">', unsafe_allow_html=True)
            run_clicked = st.button(
                "▶  Analyse scene",
                key=f"run_{st.session_state.page}",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if run_clicked:
                active_page = st.session_state.page
                q_text = question.strip() or default_q

                with st.status(f"🛰️ SatQuery Agentic Pipeline: {active_page}...", expanded=True) as status_box:
                    t_overall_start = time.perf_counter()

                    # Stage 1: Ingestion
                    status_box.write("⚙️ Stage 1: Ingesting tile tensors & normalizing bands...")
                    t_ingest_start = time.perf_counter()

                    if active_page in ["Visual Q&A", "Captioning"]:
                        if uploaded:
                            in_path = TEMP_UPLOADS / f"upload_{int(time.time())}_{uploaded.name}"
                            with open(in_path, "wb") as f:
                                f.write(uploaded.getbuffer())
                            image_paths = [str(in_path)]
                            raw_input = Image.open(in_path)
                        else:
                            image_paths = [str(sample_file)]
                            raw_input = str(sample_file)
                        modalities = ["optical"]
                    else:
                        if img_a:
                            p_a = TEMP_UPLOADS / f"upload_a_{int(time.time())}_{img_a.name}"
                            with open(p_a, "wb") as f:
                                f.write(img_a.getbuffer())
                            raw_input_a = Image.open(p_a)
                            path_a = str(p_a)
                        else:
                            path_a = str(sample_a)
                            raw_input_a = str(sample_a)

                        if img_b:
                            p_b = TEMP_UPLOADS / f"upload_b_{int(time.time())}_{img_b.name}"
                            with open(p_b, "wb") as f:
                                f.write(img_b.getbuffer())
                            raw_input_b = Image.open(p_b)
                            path_b = str(p_b)
                        else:
                            path_b = str(sample_b)
                            raw_input_b = str(sample_b)

                        image_paths = [path_a, path_b]
                        modalities = ["optical", "sar"] if active_page == "Optical-SAR Fusion" else ["optical", "optical"]

                    lat_ingestion = round((time.perf_counter() - t_ingest_start) * 1000, 2)

                    # Stage 2: Agentic Routing
                    status_box.write("🧠 Stage 2: Agentic query intent & sensor compatibility routing...")
                    t_route_start = time.perf_counter()
                    routing_decision = route_query(q_text, image_paths, modalities=modalities)
                    lat_routing = round((time.perf_counter() - t_route_start) * 1000, 2)

                    # Stage 3: Neural Model Execution
                    status_box.write(f"⚡ Stage 3: Executing specialist backbone on CUDA...")
                    t_infer_start = time.perf_counter()

                    if active_page == "Visual Q&A":
                        result_dict = qwen_engine.predict(raw_input, query=q_text, task_mode="vqa")
                    elif active_page == "Captioning":
                        result_dict = qwen_engine.predict(raw_input, query=q_text, task_mode="caption")
                    elif active_page == "Change Detection":
                        result_dict = siamese_engine.predict(raw_input_a, raw_input_b, query=q_text)
                    elif active_page == "Optical-SAR Fusion":
                        result_dict = fusion_engine.predict(raw_input_a, raw_input_b, query=q_text)
                    else:
                        result_dict = {"output": "Unsupported page", "confidence": 0.0, "latency_ms": 0.0}

                    lat_infer = result_dict.get("latency_ms", round((time.perf_counter() - t_infer_start) * 1000, 2))

                    # Stage 4: Grounded Synthesis & Logging
                    status_box.write("📄 Stage 4: Grounding spatial evidence & appending to execution_trace.jsonl...")
                    t_log_start = time.perf_counter()

                    models_list = [result_dict.get("model_name", "UNKNOWN")]
                    if active_page == "Optical-SAR Fusion":
                        models_list.append("FusionAdapter_v1")

                    params_logged = dict(result_dict.get("parameters", {}))
                    params_logged["query"] = q_text

                    try:
                        trace_record = log_execution(
                            task=result_dict.get("task_type", active_page.upper().replace(" ", "_")),
                            models_used=models_list,
                            input_images=image_paths,
                            parameters=params_logged,
                            outputs=[result_dict.get("output", "")],
                            confidence=float(result_dict.get("confidence", 0.90)),
                        )
                    except Exception:
                        trace_record = {
                            "trace_id": str(uuid.uuid4()),
                            "task": active_page,
                        }

                    lat_logging = round((time.perf_counter() - t_log_start) * 1000, 2)

                    result_dict["routing_rules"] = routing_decision.routing_rules
                    result_dict["trace_id"] = trace_record.get("trace_id", "tr-live")
                    result_dict["stage_latencies"] = {
                        "ingestion": lat_ingestion,
                        "routing": lat_routing,
                        "inference": lat_infer,
                        "logging": lat_logging,
                    }
                    result_dict["query_used"] = q_text
                    result_dict["active_page"] = active_page

                    st.session_state[f"result_{active_page}"] = result_dict
                    status_box.update(label="✅ Analysis Complete!", state="complete", expanded=False)

                st.rerun()

    # --------------------------------------------------------
    # RIGHT — Evidence & Trace
    # --------------------------------------------------------

    with right:
        with st.container(border=True):

            res = st.session_state.get(f"result_{st.session_state.page}")

            st.html(f"""
            <div class="card-header-row" style="margin:-1px -1px 0 -1px;">
                <div class="card-title">Evidence &amp; trace</div>
                <div class="mono-label">{st.session_state.page.upper()}</div>
            </div>
            """)

            if not res:
                # No result yet — show readiness placeholder
                DEMO_DIR = _ROOT / "demo"
                if st.session_state.page == "Visual Q&A":
                    sample_prev = DEMO_DIR / "vqa_sample.png"
                elif st.session_state.page == "Captioning":
                    sample_prev = DEMO_DIR / "caption_sample.png"
                elif st.session_state.page == "Change Detection":
                    sample_prev = DEMO_DIR / "t2_delhi_2025.png"
                else:
                    sample_prev = DEMO_DIR / "optical_sentinel2.png"

                if sample_prev.exists():
                    st.image(str(sample_prev), use_container_width=True, caption="Sample tile preview — click 'Analyse scene' to run")

                st.html("""
                <div class="trace" style="min-height:190px;">
                    <div style="display:flex; align-items:center; justify-content:space-between;">
                        <div class="mono-label">&nbsp; INFERENCE TRACE</div>
                        <div class="analysis-badge"><span class="analysis-pulse"></span> READY</div>
                    </div>

                    <div style="margin-top:18px; display:grid; gap:9px;">
                        <div style="display:flex; justify-content:space-between; color:#687489; font-size:12px;">
                            <span>Image ingestion</span><span style="color:#22c55e;">READY</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; color:#687489; font-size:12px;">
                            <span>Agentic routing</span><span style="color:#22c55e;">READY</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; color:#687489; font-size:12px;">
                            <span>Neural backbone</span><span style="color:#22c55e;">READY</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; color:#687489; font-size:12px;">
                            <span>Evidence extraction</span><span style="color:#22c55e;">READY</span>
                        </div>
                    </div>
                </div>
                """)

            else:
                # Result exists — show Evidence & Structured Output
                st.write("")

                # 1. VISUAL EVIDENCE OVERLAY
                if st.session_state.page == "Change Detection":
                    viz_p = res.get("visual_evidence_path")
                    if viz_p and os.path.exists(viz_p):
                        st.image(
                            viz_p,
                            use_container_width=True,
                            caption="5-Panel Bi-Temporal Evidence: [ T1 Pre-Change | T2 Post-Change | SECOND Mask T1 | SECOND Mask T2 | Binary Difference ]",
                        )
                    mask_p = res.get("change_mask_path")
                    if mask_p and os.path.exists(mask_p):
                        with st.expander("🔎 View Standalone Binary Change Mask"):
                            st.image(mask_p, use_container_width=True, caption="Binary Change Segmentation (White = Changed Pixels)")

                elif st.session_state.page == "Optical-SAR Fusion":
                    f_col1, f_col2 = st.columns(2)
                    DEMO_DIR = _ROOT / "demo"
                    with f_col1:
                        st.image(str(DEMO_DIR / "optical_sentinel2.png"), use_container_width=True, caption="Sentinel-2 Optical (RGB)")
                    with f_col2:
                        st.image(str(DEMO_DIR / "sar_sentinel1.png"), use_container_width=True, caption="Sentinel-1 SAR (C-Band)")

                else:
                    # Visual Q&A / Captioning
                    DEMO_DIR = _ROOT / "demo"
                    img_f = DEMO_DIR / ("vqa_sample.png" if st.session_state.page == "Visual Q&A" else "caption_sample.png")
                    if img_f.exists():
                        st.image(str(img_f), use_container_width=True, caption=f"Evidence Tile · Grounded: {res.get('input_image', img_f.name)}")

                st.write("")

                # 2. RESULT BADGES & OUTPUT TEXT
                badge_color = "#22c55e" if res.get("confidence", 0) >= 0.85 else "#facc15"
                st.html(f"""
                <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:12px;">
                    <span class="analysis-badge" style="color:#facc15; border-color:rgba(250,204,21,.35);">
                        <span class="analysis-pulse"></span> INFERENCE COMPLETE
                    </span>
                    <span class="function-tag" style="color:#eceef1; background:#1e1e1e;">{res.get('model_name', 'Model')}</span>
                    <span class="mono-label" style="color:{badge_color}; font-weight:600;">CONF: {int(res.get('confidence', 0.9) * 100)}%</span>
                    <span class="mono-label" style="color:#8b93a3;">{res.get('latency_ms', 0)} ms</span>
                </div>
                """)

                output_text = res.get("output", "")
                st.html(f"""
                <div style="
                    border: 1px solid #282828;
                    border-left: 3px solid #facc15;
                    border-radius: 10px;
                    background: #0f0f10;
                    padding: 16px 18px;
                    color: #eceef1;
                    font-size: 14px;
                    line-height: 1.6;
                    font-family: 'Inter', -apple-system, sans-serif;
                    margin-bottom: 16px;
                ">
                    {output_text}
                </div>
                """)

                # 3. TASK-SPECIFIC METRIC CALLOUTS
                if st.session_state.page == "Change Detection":
                    c_col1, c_col2, c_col3 = st.columns(3)
                    with c_col1:
                        chg_flag = res.get("change_detected", False)
                        color = "#22c55e" if chg_flag else "#687489"
                        st.metric("Change Detected", "YES" if chg_flag else "NO")
                    with c_col2:
                        st.metric("Changed Area", f"{res.get('change_pct', 0.0)}%")
                    with c_col3:
                        st.metric("VQA Answer", str(res.get("vqa_answer", "N/A")))

                elif st.session_state.page == "Optical-SAR Fusion":
                    top_cls = res.get("top_class", "N/A")
                    top_k = res.get("top_k_predictions", [])
                    st.html(f"""
                    <div style="background:#141414; border:1px solid #232323; border-radius:10px; padding:12px 14px; margin-bottom:14px;">
                        <div class="mono-label" style="color:#facc15; margin-bottom:6px;">TOP PREDICTED LAND COVER: <b>{top_cls.upper()}</b></div>
                        <div style="color:#8b93a3; font-size:12px; font-family:'IBM Plex Mono',monospace;">
                            {' · '.join(f"{item.get('class', 'Class')}: {int(float(item.get('confidence', item.get('probability', 0.0)))*100)}%" for item in top_k[:3]) if top_k else ''}
                        </div>
                    </div>
                    """)

                # 4. LIVE INFERENCE TRACE BREAKDOWN
                stages = res.get("stage_latencies", {})
                lat_ing = stages.get("ingestion", 14.2)
                lat_rot = stages.get("routing", 8.5)
                lat_inf = stages.get("inference", res.get("latency_ms", 120.0))
                lat_log = stages.get("logging", 12.1)
                rules_str = ", ".join(res.get("routing_rules", ["agentic_router"]))

                st.html(f"""
                <div class="trace" style="min-height:160px; margin-top:10px;">
                    <div style="display:flex; align-items:center; justify-content:space-between;">
                        <div class="mono-label">&nbsp; INFERENCE TRACE BREAKDOWN</div>
                        <div class="analysis-badge" style="color:#facc15; border-color:#333;">TRACE ID: {res.get('trace_id', 'tr-live')[:8]}</div>
                    </div>

                    <div style="margin-top:16px; display:grid; gap:9px;">
                        <div style="display:flex; justify-content:space-between; color:#8b93a3; font-size:12px;">
                            <span>⚙️ Image Ingestion &amp; Tensor Prep</span><span style="font-family:'IBM Plex Mono'; color:#eceef1;">{lat_ing} ms</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; color:#8b93a3; font-size:12px;">
                            <span>🧠 Agentic Router [{rules_str}]</span><span style="font-family:'IBM Plex Mono'; color:#eceef1;">{lat_rot} ms</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; color:#8b93a3; font-size:12px;">
                            <span>⚡ Specialist Backbone ({res.get('model_name', 'Backbone')})</span><span style="font-family:'IBM Plex Mono'; color:#facc15;">{lat_inf} ms</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; color:#8b93a3; font-size:12px;">
                            <span>📄 Grounded Synthesis &amp; Logger (execution_trace.jsonl)</span><span style="font-family:'IBM Plex Mono'; color:#eceef1;">{lat_log} ms</span>
                        </div>
                    </div>
                </div>
                """)

                # 5. ACTION TOOLBAR
                btn_col1, btn_col2 = st.columns([0.65, 0.35])
                with btn_col1:
                    trace_json_str = json.dumps(res, indent=2, ensure_ascii=False)
                    st.download_button(
                        "📥 Download Trace (JSON)",
                        data=trace_json_str,
                        file_name=f"trace_{res.get('trace_id', 'satquery')[:8]}.json",
                        mime="application/json",
                        key=f"dl_trace_{st.session_state.page}",
                    )
                with btn_col2:
                    if st.button("🔄 Reset Analysis", key=f"clear_res_{st.session_state.page}"):
                        st.session_state[f"result_{st.session_state.page}"] = None
                        st.rerun()

# ============================================================
# EXPLORE MAP
# ============================================================

elif st.session_state.page == "Explore map":

    DEMO_DIR = _ROOT / "demo"

    LOCATIONS = [
        {
            "id": "delhi",
            "name": "Delhi NCR Urban Corridor",
            "coords": "28.6139° N, 77.2090° E",
            "sensor": "Sentinel-2 MSI (Bi-Temporal Multi-Year)",
            "gsd": "10 m / pixel",
            "task": "Change Detection",
            "date": "2021-04-12 vs 2025-03-28",
            "cloud": "1.2%",
            "desc": "Peri-urban development, conversion of agricultural lands into commercial logistics clusters and residential developments.",
            "img_path": str(DEMO_DIR / "t2_delhi_2025.png"),
            "preset_query": "What changed between T1 and T2?",
        },
        {
            "id": "bengaluru",
            "name": "Bengaluru Electronic City",
            "coords": "12.9716° N, 77.5946° E",
            "sensor": "Sentinel-2 Optical + Sentinel-1 SAR GRD",
            "gsd": "10 m / pixel",
            "task": "Optical-SAR Fusion",
            "date": "2025-08-14 (Monsoon Cloud Cover)",
            "cloud": "84.7% (Penetrated by SAR)",
            "desc": "Severe monsoon cloud cover blinding optical sensors; C-band SAR backscatter reveals dense tech parks and arterial infrastructure.",
            "img_path": str(DEMO_DIR / "optical_sentinel2.png"),
            "preset_query": "What terrain is present under cloud cover?",
        },
        {
            "id": "mumbai",
            "name": "Mumbai Port & Coastal Zone",
            "coords": "18.9438° N, 72.8427° E",
            "sensor": "Sentinel-2 High-Resolution RGB",
            "gsd": "10 m / pixel",
            "task": "Visual Q&A",
            "date": "2025-01-18",
            "cloud": "0.0%",
            "desc": "High-density maritime port infrastructure, container berths, breakwaters, and shipping fairways.",
            "img_path": str(DEMO_DIR / "vqa_sample.png"),
            "preset_query": "What type of buildings and roads are visible in this tile?",
        },
        {
            "id": "sundarbans",
            "name": "Sundarbans Biosphere Reserve",
            "coords": "21.9497° N, 89.1833° E",
            "sensor": "Sentinel-2 Multispectral (SWIR/NIR)",
            "gsd": "10 m / pixel",
            "task": "Captioning",
            "date": "2024-11-05",
            "cloud": "3.1%",
            "desc": "Dense tidal mangrove canopy, interconnected estuarine channels, and mudflat sediment dynamics.",
            "img_path": str(DEMO_DIR / "caption_sample.png"),
            "preset_query": "Describe the coastal / wetland features visible.",
        },
    ]

    left, right = st.columns([0.38, 1], gap="large")

    if "selected_map_loc" not in st.session_state:
        st.session_state.selected_map_loc = 0

    with left:
        with st.container(border=True):
            st.html('<div class="card-title" style="margin-bottom:12px;">Curated Satellite Scenes</div>')

            loc_names = [f"{loc['name']} ({loc['task']})" for loc in LOCATIONS]
            sel_idx = st.radio(
                "Select scene",
                range(len(LOCATIONS)),
                format_func=lambda i: loc_names[i],
                key="map_loc_radio",
                label_visibility="collapsed",
            )
            st.session_state.selected_map_loc = sel_idx
            loc = LOCATIONS[sel_idx]

            st.html(f"""
            <div style="background:#0f0f10; border:1px solid #232323; border-radius:10px; padding:14px; margin-top:14px;">
                <div class="mono-label" style="color:#facc15; margin-bottom:6px;">SCENE TELEMETRY</div>
                <div style="display:grid; gap:6px; font-size:12px; font-family:'IBM Plex Mono',monospace; color:#8b93a3;">
                    <div><b>TARGET:</b> <span style="color:#eceef1;">{loc['name']}</span></div>
                    <div><b>COORDINATES:</b> <span style="color:#eceef1;">{loc['coords']}</span></div>
                    <div><b>SENSOR:</b> <span style="color:#eceef1;">{loc['sensor']}</span></div>
                    <div><b>RESOLUTION:</b> <span style="color:#eceef1;">{loc['gsd']}</span></div>
                    <div><b>ACQUISITION:</b> <span style="color:#eceef1;">{loc['date']}</span></div>
                    <div><b>CLOUD COVER:</b> <span style="color:#eceef1;">{loc['cloud']}</span></div>
                </div>
            </div>
            """)

            st.write("")
            st.markdown('<div class="run-button">', unsafe_allow_html=True)
            if st.button(f"🚀  Load in {loc['task']} Console", use_container_width=True):
                st.session_state.page = loc["task"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            loc = LOCATIONS[st.session_state.selected_map_loc]
            st.html(f"""
            <div class="card-header-row" style="margin:-1px -1px 12px -1px;">
                <div class="card-title">Tactical Satellite HUD · {loc['name']}</div>
                <div class="mono-label">{loc['coords']}</div>
            </div>
            """)

            if os.path.exists(loc["img_path"]):
                st.image(loc["img_path"], use_container_width=True, caption=f"Sensor footprint: {loc['sensor']} · {loc['coords']}")

            st.html(f"""
            <div style="background:#121213; border:1px solid #242424; border-left:3px solid #facc15; border-radius:8px; padding:12px 16px; margin-top:10px;">
                <div style="color:#eceef1; font-size:13px; line-height:1.6;">
                    <b>Mission Briefing:</b> {loc['desc']}
                </div>
            </div>
            """)

# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "Insights":

    # Load traces from both execution_trace.jsonl and ./logs/execution_trace.jsonl
    traces: List[Dict[str, Any]] = []
    log_paths = [
        _ROOT / "execution_trace.jsonl",
        _ROOT / "satquery_backend" / "logs" / "execution_trace.jsonl",
        _ROOT / "logs" / "execution_trace.jsonl",
    ]

    seen_ids = set()
    for lp in log_paths:
        if lp.exists():
            try:
                with open(lp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                r = json.loads(line)
                                tid = r.get("trace_id", str(uuid.uuid4()))
                                if tid not in seen_ids:
                                    seen_ids.add(tid)
                                    traces.append(r)
                            except Exception:
                                pass
            except Exception:
                pass

    total_runs = len(traces)
    avg_lat = round(sum(float(t.get("latency_ms", 0)) for t in traces) / total_runs, 1) if total_runs else 0.0
    avg_conf = round(sum(float(t.get("confidence", 0)) for t in traces) / total_runs * 100, 1) if total_runs else 0.0
    task_types_seen = set(t.get("task", t.get("task_type", "UNKNOWN")) for t in traces)
    modality_count = max(len(task_types_seen), 1) if total_runs else 4

    metric_labels = [
        "TRACES RUN",
        "AVG SIM. LATENCY",
        "AVG CONFIDENCE",
        "ACTIVE MODALITIES",
    ]

    metric_values = [
        f"{total_runs}" if total_runs else "—",
        f"{avg_lat} ms" if total_runs else "—",
        f"{avg_conf}%" if total_runs else "—",
        f"{modality_count} / 4",
    ]

    metric_icons = ["⌁", "◷", "▣", "🛰️"]

    cols = st.columns(4)

    for col, label, icon, val in zip(cols, metric_labels, metric_icons, metric_values):
        with col:
            with st.container(border=True):
                st.html(f"""
                <div class="metric-icon">{icon}</div>
                <div class="metric-value {'empty' if val == '—' else ''}">{val}</div>
                <div class="metric-label">{label}</div>
                """)

    st.write("")

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.html('<div class="card-title" style="margin-bottom:12px;">Task Mix Distribution</div>')
            if total_runs > 0:
                task_counts: Dict[str, int] = {}
                for t in traces:
                    tt = t.get("task", t.get("task_type", "UNKNOWN"))
                    task_counts[tt] = task_counts.get(tt, 0) + 1

                for task_name, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = round((count / total_runs) * 100, 1)
                    st.html(f"""
                    <div style="margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; font-size:12px; font-family:'IBM Plex Mono',monospace; color:#eceef1; margin-bottom:4px;">
                            <span>{task_name}</span>
                            <span style="color:#facc15;">{count} runs ({pct}%)</span>
                        </div>
                        <div style="height:6px; background:#222; border-radius:3px; overflow:hidden;">
                            <div style="width:{pct}%; height:100%; background:#facc15; border-radius:3px;"></div>
                        </div>
                    </div>
                    """)
            else:
                st.html("""
                <div class="empty-dashboard" style="
                    min-height:220px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:#596273;
                    font-family:'IBM Plex Mono',monospace;
                    font-size:10px;
                    letter-spacing:1.5px;">
                    RUN QUERIES TO POPULATE TASK MIX
                </div>
                """)

    with right:
        with st.container(border=True):
            st.html('<div class="card-title" style="margin-bottom:12px;">Compute &amp; Hardware Telemetry</div>')
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU Execution"
            vram_mb = round(torch.cuda.memory_allocated() / 1e6, 1) if gpu_available else 0.0

            st.html(f"""
            <div style="display:grid; gap:12px; font-family:'IBM Plex Mono',monospace; font-size:12px; color:#8b93a3;">
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid #1f1f1f; padding-bottom:6px;">
                    <span>ACCELERATOR:</span><span style="color:#22c55e;">{gpu_name}</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid #1f1f1f; padding-bottom:6px;">
                    <span>VRAM ALLOCATED:</span><span style="color:#facc15;">{vram_mb} MB / 6144 MB</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid #1f1f1f; padding-bottom:6px;">
                    <span>QWEN QUANTIZATION:</span><span style="color:#eceef1;">4-Bit NF4 (bitsandbytes)</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid #1f1f1f; padding-bottom:6px;">
                    <span>FUSION ADAPTER:</span><span style="color:#eceef1;">adapter_v1.pt (TorchScript)</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span>CHANGE HEAD:</span><span style="color:#eceef1;">best_model.pth (Siamese ResNet18)</span>
                </div>
            </div>
            """)

    st.write("")

    # Full execution trace log table
    with st.container(border=True):
        st.html("""
        <div class="card-header-row" style="margin:-1px -1px 12px -1px;">
            <div class="card-title">Live Execution Trace Audit Log (execution_trace.jsonl)</div>
            <div class="mono-label">AUDITABLE MLOPS</div>
        </div>
        """)

        if traces:
            trace_table_data = []
            for t in reversed(traces[-15:]):
                trace_table_data.append({
                    "Timestamp": t.get("timestamp", "")[:19].replace("T", " "),
                    "Trace ID": t.get("trace_id", "")[:8],
                    "Task": t.get("task", t.get("task_type", "")),
                    "Model": t.get("model_name", ""),
                    "Latency (ms)": t.get("latency_ms", 0),
                    "Confidence": f"{int(float(t.get('confidence', 0))*100)}%",
                    "Output": (t.get("output", "") or "")[:90] + "...",
                })
            st.dataframe(trace_table_data, use_container_width=True)

            log_file_root = _ROOT / "execution_trace.jsonl"
            if log_file_root.exists():
                with open(log_file_root, "r", encoding="utf-8") as f:
                    content_jsonl = f.read()
                st.download_button(
                    "📥 Download Complete execution_trace.jsonl",
                    data=content_jsonl,
                    file_name="execution_trace.jsonl",
                    mime="text/plain",
                    key="dl_full_traces",
                )
        else:
            st.info("No traces logged yet. Head to any task console and click 'Analyse scene'!")

# EOF

