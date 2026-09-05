import time
from pathlib import Path

import streamlit as st

# ============================================================
# SatQuery AI — Streamlit Frontend
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
# hands off to the Home page. Clip goes in ./assets/
# ------------------------------------------------------------

INTRO_DIR = Path(__file__).parent / "assets"


def _find_intro_video() -> Path:
    """Case-insensitive lookup, tolerates double extensions."""
    fallback = INTRO_DIR / "earth_scan.mp4"
    if not INTRO_DIR.is_dir():
        return fallback
    files = {p.name.lower(): p for p in INTRO_DIR.iterdir() if p.is_file()}
    if "earth_scan.mp4" in files:
        return files["earth_scan.mp4"]
    return next(
        (p for name, p in sorted(files.items())
         if name.startswith("earth_scan") and name.endswith(".mp4")),
        fallback,
    )


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

            if st.session_state.page in ["Visual Q&A", "Captioning"]:

                if st.session_state.page == "Visual Q&A":
                    question = st.text_area(
                        "Question",
                        placeholder="Ask a plain-English question...",
                        height=105,
                        key="vqa_question",
                        label_visibility="collapsed",
                    )
                else:
                    question = st.text_area(
                        "Caption guidance",
                        placeholder='Optional: steer the caption (e.g. "focus on infrastructure")...',
                        height=105,
                        key="caption_question",
                        label_visibility="collapsed",
                    )

                uploaded = st.file_uploader(
                    "Browse files",
                    type=["png", "jpg", "jpeg", "tif", "tiff"],
                    key=f"upload_{st.session_state.page}",
                    label_visibility="collapsed",
                )

                if uploaded:
                    st.image(uploaded, use_container_width=True)

                st.html(
                    '<div class="mono-label" style="font-style:italic; margin-top:5px; color:#687489;">'
                    'Leave image empty to use the sample scene on the right.'
                    '</div>'
                )

            else:
                if st.session_state.page == "Change Detection":
                    query_label = "Change query"
                    placeholder = 'Optional: constrain the change type (e.g. "new construction")...'
                    question_key = "change_question"
                    label_a = "IMAGE A"
                    label_b = "IMAGE B"
                else:
                    query_label = "Fusion query"
                    placeholder = 'Ask across both modalities (e.g. "what is obscured by cloud?")...'
                    question_key = "fusion_question"
                    label_a = "OPTICAL"
                    label_b = "SAR"

                question = st.text_area(
                    query_label,
                    placeholder=placeholder,
                    height=105,
                    key=question_key,
                    label_visibility="collapsed",
                )

                attach_a, attach_b = st.columns(2, gap="small")

                with attach_a:
                    st.html(f'<div class="mono-label" style="margin-bottom:4px;">{label_a}</div>')
                    img_a = st.file_uploader(
                        "Browse files",
                        type=["png", "jpg", "jpeg", "tif", "tiff"],
                        key=f"img_a_{st.session_state.page}",
                        label_visibility="collapsed",
                    )
                    if img_a:
                        st.image(img_a, use_container_width=True)

                with attach_b:
                    st.html(f'<div class="mono-label" style="margin-bottom:4px;">{label_b}</div>')
                    img_b = st.file_uploader(
                        "Browse files",
                        type=["png", "jpg", "jpeg", "tif", "tiff"],
                        key=f"img_b_{st.session_state.page}",
                        label_visibility="collapsed",
                    )
                    if img_b:
                        st.image(img_b, use_container_width=True)

                st.html(
                    '<div class="mono-label" style="font-style:italic; margin-top:5px; color:#687489;">'
                    'Leave images empty to use the sample scene on the right.'
                    '</div>'
                )

            st.write("")

            st.markdown('<div class="run-button">', unsafe_allow_html=True)
            if st.button(
                "▶  Analyse scene",
                key=f"run_{st.session_state.page}",
                use_container_width=True,
            ):
                st.info("Inference backend is not connected yet.")
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # RIGHT — Evidence & Trace
    # --------------------------------------------------------

    with right:
        with st.container(border=True):

            st.html(f"""
            <div class="card-header-row" style="margin:-1px -1px 0 -1px;">
                <div class="card-title">Evidence &amp; trace</div>
                <div class="mono-label">{st.session_state.page.upper()}</div>
            </div>
            """)

            st.html("""
            <div class="evidence-empty" style="min-height:360px;">
                <div style="text-align:center;">
                    <div class="analysis-badge"><span class="analysis-pulse"></span> READY FOR ANALYSIS</div>
                    <div style="margin-top:14px;">NO EVIDENCE IMAGE</div>
                    <div style="margin-top:7px; color:#414958; font-size:9px; letter-spacing:1px;">
                        RESULTS WILL APPEAR HERE
                    </div>
                </div>
            </div>

            <div class="trace" style="min-height:190px;">
                <div style="display:flex; align-items:center; justify-content:space-between;">
                    <div class="mono-label">&nbsp; INFERENCE TRACE</div>
                    <div class="analysis-badge">WAITING</div>
                </div>

                <div style="margin-top:18px; display:grid; gap:9px;">
                    <div style="display:flex; justify-content:space-between; color:#687489; font-size:12px;">
                        <span>Image ingestion</span><span>—</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; color:#687489; font-size:12px;">
                        <span>Scene analysis</span><span>—</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; color:#687489; font-size:12px;">
                        <span>Evidence extraction</span><span>—</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; color:#687489; font-size:12px;">
                        <span>Response generation</span><span>—</span>
                    </div>
                </div>
            </div>
            """)

# ============================================================
# EXPLORE MAP
# ============================================================

elif st.session_state.page == "Explore map":

    left, right = st.columns([0.32, 1], gap="large")

    with left:
        with st.container(border=True):
            st.html('<div class="mono-label">SAMPLE TILES</div>')
            st.write("")
            st.html("""
            <div style="
                min-height:520px;
                display:flex;
                align-items:center;
                justify-content:center;
                color:#596273;
                font-family:'IBM Plex Mono',monospace;
                font-size:10px;
                letter-spacing:1.5px;">
                NO SAMPLE TILES
            </div>
            """)

    with right:
        st.html('<div class="map-empty">MAP DATA NOT CONNECTED</div>')

# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "Insights":

    metric_labels = [
        "TRACES RUN",
        "AVG SIM. LATENCY",
        "AVG REGIONS / TILE",
        "PEAK GFLOPS",
    ]

    metric_icons = ["⌁", "◷", "▱", "▣"]

    cols = st.columns(4)

    for col, label, icon in zip(cols, metric_labels, metric_icons):
        with col:
            with st.container(border=True):
                st.html(f"""
                <div class="metric-icon">{icon}</div>
                <div class="metric-value empty">—</div>
                <div class="metric-label">{label}</div>
                """)

    st.write("")

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.html("""
            <div class="card-title">Task mix</div>
            <div class="empty-dashboard" style="
                min-height:240px;
                display:flex;
                align-items:center;
                justify-content:center;
                color:#596273;
                font-family:'IBM Plex Mono',monospace;
                font-size:10px;
                letter-spacing:1.5px;">
                NO DATA AVAILABLE
            </div>
            """)

    with right:
        with st.container(border=True):
            st.html("""
            <div class="card-title">Throughput</div>
            <div class="empty-dashboard" style="
                min-height:240px;
                display:flex;
                align-items:center;
                justify-content:center;
                color:#596273;
                font-family:'IBM Plex Mono',monospace;
                font-size:10px;
                letter-spacing:1.5px;">
                NO DATA AVAILABLE
            </div>
            """)

    st.write("")

    st.html("""
    <div class="empty-notice">
        <strong style="color:#facc15;">⚠</strong>
        &nbsp; No telemetry or model metrics are connected yet.
    </div>
    """)

# EOF
