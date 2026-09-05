"""
Interactive Web Application for Bi-Temporal Change Detection & Change-VQA
Zero external web dependencies (uses Python standard library http.server + PyTorch).
Access via browser at: http://localhost:8080
"""

import os
import io
import json
import base64
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import cv2
import numpy as np
import torch

from infer import (
    load_and_preprocess_pair,
    mask_to_rgb,
    apply_morphological_filtering,
    probs_to_mask,
    resolve_grounded_answer,
    CLASS_TO_RGB,
    GROUNDED_CLASS_STRINGS
)
from dataset import (
    CLASS_NAMES,
    ANSWER_VOCAB,
    IDX_TO_ANS,
    QuestionTokenizer
)
from model import BiTemporalChangeModel, generate_change_description

MODEL = None
TOKENIZER = None
DEVICE = None
CHECKPOINT_PATH = "C:/satquery/checkpoints/best_model.pth"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SatQuery: Bi-Temporal Change Detection & Change-VQA</title>
    <style>
        :root {
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --accent: #58a6ff;
            --accent-hover: #1f6feb;
            --text: #c9d1d9;
            --heading: #f0f6fc;
            --success: #2ea043;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 24px;
            line-height: 1.6;
        }
        header {
            max-width: 1280px;
            margin: 0 auto 24px auto;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 { color: var(--heading); font-size: 24px; }
        .badge {
            background: #238636;
            color: #fff;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .container {
            max-width: 1280px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 24px;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        h2 { font-size: 16px; color: var(--heading); margin-bottom: 12px; }
        label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--heading); }
        select, input[type="text"], input[type="file"] {
            width: 100%;
            padding: 9px 12px;
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            font-size: 13px;
            margin-bottom: 12px;
        }
        select:focus, input[type="text"]:focus {
            outline: none;
            border-color: var(--accent);
        }
        button {
            width: 100%;
            padding: 12px 16px;
            background: var(--accent-hover);
            color: #fff;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            margin-top: 6px;
        }
        button:hover { background: var(--accent); }
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }
        .gallery-item {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px;
            text-align: center;
        }
        .gallery-item img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            display: block;
            min-height: 140px;
            background: #161b22;
        }
        .gallery-item span {
            display: block;
            margin-top: 6px;
            font-size: 12px;
            font-weight: 600;
            color: var(--heading);
        }
        .summary-box {
            background: #090d13;
            border-left: 4px solid var(--accent);
            padding: 14px 16px;
            border-radius: 4px;
            font-size: 14px;
            color: var(--heading);
            margin-bottom: 20px;
        }
        .qa-list { list-style: none; }
        .qa-item {
            border-bottom: 1px solid var(--border);
            padding: 12px 0;
        }
        .qa-item:last-child { border-bottom: none; }
        .qa-q { font-weight: 600; color: var(--heading); margin-bottom: 4px; }
        .qa-a {
            display: inline-block;
            background: #1f242c;
            color: #58a6ff;
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 4px;
            margin-right: 8px;
        }
        .qa-method { font-size: 12px; color: #8b949e; }
        .loader {
            display: none;
            text-align: center;
            padding: 14px;
            color: var(--accent);
            font-weight: 600;
        }
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
            font-size: 11px;
        }
        .legend-item { display: flex; align-items: center; gap: 4px; }
        .color-dot { width: 10px; height: 10px; border-radius: 2px; }
        .tab-bar {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }
        .tab-btn {
            flex: 1;
            padding: 6px 10px;
            background: #21262d;
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text);
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
        }
        .tab-btn.active {
            background: var(--accent-hover);
            color: #fff;
            border-color: var(--accent);
        }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>SatQuery Multimodal System</h1>
            <p style="font-size: 13px; color: #8b949e;">Bi-Temporal Remote Sensing Change Detection & Change-VQA</p>
        </div>
        <div>
            <span class="badge">Model Active (RTX 4050 GPU)</span>
        </div>
    </header>

    <div class="container">
        <!-- Control Sidebar -->
        <div class="sidebar">
            <div class="card">
                <h2>Select or Upload Satellite Pair</h2>
                
                <div class="tab-bar">
                    <button type="button" class="tab-btn active" id="tabPreset" onclick="switchInputTab('preset')">Preset Pairs</button>
                    <button type="button" class="tab-btn" id="tabCustom" onclick="switchInputTab('custom')">Paste / Upload</button>
                </div>

                <!-- Tab 1: Presets & Filename Input -->
                <div id="sectionPreset">
                    <label for="pairInput">Enter Filename (e.g. 02180.png):</label>
                    <input type="text" id="pairInput" value="02180.png" placeholder="Enter image filename">

                    <label for="pairSelect">Or Select Preset Pair:</label>
                    <select id="pairSelect" onchange="document.getElementById('pairInput').value = this.value;">
                        <option value="02180.png" selected>02180.png (Urban Expansion)</option>
                        <option value="00011.png">00011.png (Vegetation Transition)</option>
                        <option value="00013.png">00013.png (Building Reconstruction)</option>
                        <option value="00025.png">00025.png (Land Clearing)</option>
                        <option value="00003.png">00003.png</option>
                        <option value="00015.png">00015.png</option>
                        <option value="00018.png">00018.png</option>
                        <option value="00020.png">00020.png</option>
                    </select>
                </div>

                <!-- Tab 2: Direct File Upload -->
                <div id="sectionCustom" style="display: none;">
                    <label for="fileT1">Upload T1 (Pre-event Image):</label>
                    <input type="file" id="fileT1" accept="image/*">

                    <label for="fileT2">Upload T2 (Post-event Image):</label>
                    <input type="file" id="fileT2" accept="image/*">
                </div>

                <h2 style="margin-top: 14px;">Change-VQA Question</h2>
                <label for="vqaQuestion">Query the Change:</label>
                <input type="text" id="vqaQuestion" value="Have the regions of buildings changed?">
                
                <label for="quickQuestions">Quick Question Presets:</label>
                <select id="quickQuestions" onchange="document.getElementById('vqaQuestion').value = this.value;">
                    <option value="Have the regions of buildings changed?">Have the regions of buildings changed?</option>
                    <option value="Did the areas of trees change?">Did the areas of trees change?</option>
                    <option value="how much vegetation decreased">how much vegetation decreased</option>
                    <option value="What is the percentage of changed areas?">What is the percentage of changed areas?</option>
                    <option value="How much of the area has not changed?">How much of the area has not changed?</option>
                    <option value="What is the change ratio of low vegetation in the pre-event image?">What is the change ratio of low vegetation in the pre-event image?</option>
                    <option value="What is the change percentage of buildings in the second image?">What is the change percentage of buildings in the second image?</option>
                    <option value="What type of change is the largest?">What type of change is the largest?</option>
                </select>

                <div style="margin-top: 12px; margin-bottom: 14px; background: #0d1117; padding: 12px; border: 1px solid var(--border); border-radius: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <label for="sensSlider" style="margin-bottom: 0; font-size: 13px;">Detection Sensitivity:</label>
                        <span id="sensVal" style="font-size: 12px; font-weight: 600; color: var(--accent);">0.35 — Balanced SOTA</span>
                    </div>
                    <input type="range" id="sensSlider" min="0.20" max="0.50" step="0.05" value="0.35" style="width: 100%; cursor: pointer;" oninput="updateSensitivityLabel(this.value)">
                    <div style="display: flex; justify-content: space-between; font-size: 11px; color: #8b949e; margin-top: 4px;">
                        <span>0.20 (High Recall)</span>
                        <span>0.35 (Balanced SOTA)</span>
                        <span>0.50 (Strict Argmax)</span>
                    </div>
                </div>

                <button id="runBtn" onclick="executeInference()">Run Multi-Task Inference</button>
                <div class="loader" id="loader">Processing Bi-Temporal Features...</div>
                <div id="saveNotification" style="display: none; background: #23863622; border: 1px solid #238636; border-radius: 6px; padding: 10px; margin-top: 12px; font-size: 12px; color: #3fb950; text-align: center;"></div>
            </div>

            <div class="card">
                <h2>Class Color Legend</h2>
                <div class="legend">
                    <div class="legend-item"><div class="color-dot" style="background: #ffffff; border: 1px solid #444;"></div> No Change</div>
                    <div class="legend-item"><div class="color-dot" style="background: rgb(128,128,128);"></div> NVG Ground</div>
                    <div class="legend-item"><div class="color-dot" style="background: rgb(0,255,0);"></div> Tree</div>
                    <div class="legend-item"><div class="color-dot" style="background: rgb(0,128,0);"></div> Low Veg</div>
                    <div class="legend-item"><div class="color-dot" style="background: rgb(0,0,255);"></div> Water</div>
                    <div class="legend-item"><div class="color-dot" style="background: rgb(128,0,0);"></div> Buildings</div>
                    <div class="legend-item"><div class="color-dot" style="background: rgb(255,0,0);"></div> Playgrounds</div>
                </div>
            </div>
        </div>

        <!-- Main Output Canvas -->
        <div class="main-content">
            <div class="card">
                <h2>Task 3: Bi-Temporal Segmentation & Change Mask</h2>
                <div class="gallery-grid">
                    <div class="gallery-item">
                        <img id="imgT1" src="/api/image?pair=02180.png&type=t1" alt="T1 Satellite">
                        <span>T1 Image (Pre)</span>
                    </div>
                    <div class="gallery-item">
                        <img id="imgT2" src="/api/image?pair=02180.png&type=t2" alt="T2 Satellite">
                        <span>T2 Image (Post)</span>
                    </div>
                    <div class="gallery-item">
                        <img id="imgMask1" src="/api/image?pair=02180.png&type=mask1" alt="Pred T1 Mask">
                        <span>Predicted T1 Mask</span>
                    </div>
                    <div class="gallery-item">
                        <img id="imgMask2" src="/api/image?pair=02180.png&type=mask2" alt="Pred T2 Mask">
                        <span>Predicted T2 Mask</span>
                    </div>
                    <div class="gallery-item">
                        <img id="imgBinChange" src="/api/image?pair=02180.png&type=bin" alt="Binary Change">
                        <span>Binary Change Mask</span>
                    </div>
                </div>

                <h2>Factual Change Description (Area Delta Grounding)</h2>
                <div class="summary-box" id="descBox">
                    Between the two images, Non-vegetated ground surface decreased by 5.0%; Tree cover increased by 4.4%; Low vegetation decreased by 21.7%; Built-up area increased by 22.2% (total changed area: 34.7%).
                </div>
                <div style="margin-top: 8px; font-size: 11px; color: #8b949e; line-height: 1.4;">
                    ℹ️ <strong>How to read:</strong> <em>Total changed area</em> is the union of all pixels where land-cover transitioned between T1 and T2. Individual percentages (e.g. Tree cover decreased by X%) represent the net surface area difference between T1 and T2.
                </div>
            </div>

            <div class="card">
                <h2>Task 4: Change-Based Visual Question Answering (Change-VQA)</h2>
                <ul class="qa-list" id="qaResults">
                    <li class="qa-item">
                        <div class="qa-q">Have the regions of buildings changed?</div>
                        <div><span class="qa-a">yes</span> <span class="qa-method">Neural Cross-Attention (Confidence: 94.4%)</span></div>
                    </li>
                    <li class="qa-item">
                        <div class="qa-q">What is the percentage of changed areas?</div>
                        <div><span class="qa-a">30_to_40</span> <span class="qa-method">Grounded Mask Verification (Calculated: 34.7%)</span></div>
                    </li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        let currentMode = 'preset';

        function updateSensitivityLabel(val) {
            const v = parseFloat(val);
            let label = v.toFixed(2);
            if (v >= 0.48) label += " — Strict Argmax";
            else if (v >= 0.32 && v <= 0.38) label += " — Balanced SOTA";
            else label += " — High Recall";
            document.getElementById('sensVal').innerText = label;
        }

        function switchInputTab(mode) {
            currentMode = mode;
            if (mode === 'preset') {
                document.getElementById('sectionPreset').style.display = 'block';
                document.getElementById('sectionCustom').style.display = 'none';
                document.getElementById('tabPreset').classList.add('active');
                document.getElementById('tabCustom').classList.remove('active');
            } else {
                document.getElementById('sectionPreset').style.display = 'none';
                document.getElementById('sectionCustom').style.display = 'block';
                document.getElementById('tabPreset').classList.remove('active');
                document.getElementById('tabCustom').classList.add('active');
            }
        }

        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => resolve(reader.result.split(',')[1]);
                reader.onerror = error => reject(error);
            });
        }

        async function executeInference() {
            const question = document.getElementById('vqaQuestion').value;
            const threshold = parseFloat(document.getElementById('sensSlider').value);
            const loader = document.getElementById('loader');
            const runBtn = document.getElementById('runBtn');

            loader.style.display = 'block';
            runBtn.disabled = true;

            try {
                let payload = { question: question, threshold: threshold };

                if (currentMode === 'preset') {
                    const pair = document.getElementById('pairInput').value.trim() || '02180.png';
                    payload.pair = pair;
                } else {
                    const f1 = document.getElementById('fileT1').files[0];
                    const f2 = document.getElementById('fileT2').files[0];
                    if (!f1 || !f2) {
                        alert('Please select both T1 and T2 images to upload.');
                        loader.style.display = 'none';
                        runBtn.disabled = false;
                        return;
                    }
                    payload.t1_b64 = await fileToBase64(f1);
                    payload.t2_b64 = await fileToBase64(f2);
                }

                const response = await fetch('/api/infer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Update visual images directly from base64 response
                document.getElementById('imgT1').src = 'data:image/png;base64,' + data.img_t1;
                document.getElementById('imgT2').src = 'data:image/png;base64,' + data.img_t2;
                document.getElementById('imgMask1').src = 'data:image/png;base64,' + data.img_mask1;
                document.getElementById('imgMask2').src = 'data:image/png;base64,' + data.img_mask2;
                document.getElementById('imgBinChange').src = 'data:image/png;base64,' + data.img_bin;

                // Update text description
                document.getElementById('descBox').innerText = data.change_description;

                // Update QA list
                const qaList = document.getElementById('qaResults');
                qaList.innerHTML = '';
                data.vqa_results.forEach(item => {
                    const li = document.createElement('li');
                    li.className = 'qa-item';
                    let methodText = '';
                    if (item.method === 'grounded') {
                        methodText = item.details 
                            ? `Grounded Mask Verification (${item.details})` 
                            : 'Grounded Mask Verification (100% Deterministic)';
                    } else {
                        methodText = `Neural Cross-Attention (Confidence: ${item.confidence.toFixed(1)}%)`;
                    }
                    li.innerHTML = `
                        <div class="qa-q">${item.question}</div>
                        <div><span class="qa-a">${item.answer}</span> <span class="qa-method">${methodText}</span></div>
                    `;
                    qaList.appendChild(li);
                });

                // Update save notification
                const note = document.getElementById('saveNotification');
                if (data.saved_record) {
                    note.style.display = 'block';
                    note.innerHTML = '📁 <strong>Saved:</strong> ' + data.saved_record + '<br><span style="color: #8b949e; font-size: 11px;">Image & JSON report saved in dashboard_results/</span>';
                }
            } catch (err) {
                alert('Inference failed: ' + err.message);
            } finally {
                loader.style.display = 'none';
                runBtn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

CACHE_MASKS = {}


class SatQueryHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            return

        if path == '/api/image':
            pair = params.get('pair', ['02180.png'])[0]
            img_type = params.get('type', ['t1'])[0]
            self.serve_preset_image(pair, img_type)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/infer':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode('utf-8'))

            try:
                results = run_app_inference(data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(results).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return

        self.send_response(404)
        self.end_headers()

    def serve_preset_image(self, pair: str, img_type: str):
        im1_path = os.path.join("C:/satquery/im1", pair)
        im2_path = os.path.join("C:/satquery/im2", pair)

        if pair not in CACHE_MASKS:
            run_app_inference({'pair': pair, 'question': 'Have the regions of buildings changed?'})

        cache = CACHE_MASKS.get(pair, {})
        if img_type == 't1':
            img = cv2.imread(im1_path)
        elif img_type == 't2':
            img = cv2.imread(im2_path)
        elif img_type == 'mask1':
            img = cv2.cvtColor(mask_to_rgb(cache['mask1']), cv2.COLOR_RGB2BGR)
        elif img_type == 'mask2':
            img = cv2.cvtColor(mask_to_rgb(cache['mask2']), cv2.COLOR_RGB2BGR)
        elif img_type == 'bin':
            bin_m = ((cache['mask1'] > 0) | (cache['mask2'] > 0)).astype(np.uint8) * 255
            img = cv2.cvtColor(bin_m, cv2.COLOR_GRAY2BGR)
        else:
            img = np.zeros((256, 256, 3), dtype=np.uint8)

        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)

        img_resized = cv2.resize(img, (256, 256))
        _, encoded = cv2.imencode('.png', img_resized)

        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.end_headers()
        self.wfile.write(encoded.tobytes())


def b64_to_cv2(b64_str: str) -> np.ndarray:
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def cv2_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')


def run_app_inference(data: dict) -> dict:
    global MODEL, TOKENIZER, DEVICE, CACHE_MASKS

    custom_q = data.get('question', 'Have the regions of buildings changed?')
    threshold = float(data.get('threshold', 0.35))

    if 't1_b64' in data and 't2_b64' in data:
        im1 = b64_to_cv2(data['t1_b64'])
        im2 = b64_to_cv2(data['t2_b64'])
        pair_key = 'custom_upload'
    else:
        pair = data.get('pair', '02180.png')
        pair_key = pair
        im1_path = os.path.join("C:/satquery/im1", pair)
        im2_path = os.path.join("C:/satquery/im2", pair)
        im1 = cv2.imread(im1_path)
        im2 = cv2.imread(im2_path)

    if im1 is None or im2 is None:
        raise ValueError(f"Could not load image pair.")

    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    im1_res = cv2.resize(im1_rgb, (256, 256))
    im2_res = cv2.resize(im2_rgb, (256, 256))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    t1_t = torch.from_numpy(im1_res.transpose(2, 0, 1)).float() / 255.0
    t2_t = torch.from_numpy(im2_res.transpose(2, 0, 1)).float() / 255.0

    t1_norm = ((t1_t - mean) / std).unsqueeze(0).to(DEVICE)
    t2_norm = ((t2_t - mean) / std).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = MODEL(t1_norm, t2_norm)
        p1 = torch.softmax(out['logits_mask1'], dim=1)
        p2 = torch.softmax(out['logits_mask2'], dim=1)

    m1_pred = probs_to_mask(p1, threshold=threshold).squeeze(0).cpu().numpy()
    m2_pred = probs_to_mask(p2, threshold=threshold).squeeze(0).cpu().numpy()
    m1_np = apply_morphological_filtering(m1_pred)
    m2_np = apply_morphological_filtering(m2_pred)

    CACHE_MASKS[pair_key] = {'mask1': m1_np, 'mask2': m2_np}

    change_desc = generate_change_description(torch.from_numpy(m1_np), torch.from_numpy(m2_np))

    # Evaluate Questions
    questions_to_eval = [
        custom_q,
        "What is the percentage of changed areas?",
        "Did the areas of trees change?",
        "What type of change is the largest?"
    ]
    seen = set()
    questions_to_eval = [q for q in questions_to_eval if not (q.lower() in seen or seen.add(q.lower()))]

    vqa_results = []
    for q_text in questions_to_eval:
        grounded_ans, grounded_exp = resolve_grounded_answer(q_text, m1_np, m2_np)
        if grounded_ans is not None:
            vqa_results.append({
                'question': q_text,
                'answer': grounded_ans,
                'confidence': 100.0,
                'method': 'grounded',
                'details': grounded_exp
            })
        else:
            q_tok = TOKENIZER.encode(q_text).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out_q = MODEL(t1_norm, t2_norm, q_tok)
                probs = torch.softmax(out_q['logits_vqa'], dim=-1)
                conf, idx = probs.max(dim=-1)
                ans_str = IDX_TO_ANS[idx.item()]
                vqa_results.append({
                    'question': q_text,
                    'answer': ans_str,
                    'confidence': conf.item() * 100.0,
                    'method': 'neural'
                })

    mask1_rgb = mask_to_rgb(m1_np)
    mask2_rgb = mask_to_rgb(m2_np)
    bin_change = ((m1_np > 0) | (m2_np > 0)).astype(np.uint8) * 255
    bin_change_rgb = cv2.cvtColor(bin_change, cv2.COLOR_GRAY2RGB)

    # Auto-save results to dashboard_results folder
    output_dir = "C:/satquery/dashboard_results"
    os.makedirs(output_dir, exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_pair = pair_key.replace('.png', '').replace('/', '_').replace('\\', '_')

    # 1. Composite 5-panel visualization image
    combined_viz = np.concatenate([
        cv2.cvtColor(im1_res, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(im2_res, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(mask1_rgb, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(mask2_rgb, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(bin_change_rgb, cv2.COLOR_RGB2BGR)
    ], axis=1)
    viz_filename = f"run_{ts}_{safe_pair}.png"
    cv2.imwrite(os.path.join(output_dir, viz_filename), combined_viz)

    # 2. Structured JSON record
    json_filename = f"run_{ts}_{safe_pair}.json"
    record = {
        'timestamp': ts,
        'pair': pair_key,
        'change_description': change_desc,
        'vqa_results': vqa_results,
        'saved_visualization': viz_filename
    }
    with open(os.path.join(output_dir, json_filename), 'w') as f:
        json.dump(record, f, indent=2)

    return {
        'change_description': change_desc,
        'vqa_results': vqa_results,
        'saved_record': f"{viz_filename}",
        'img_t1': cv2_to_b64(cv2.cvtColor(im1_res, cv2.COLOR_RGB2BGR)),
        'img_t2': cv2_to_b64(cv2.cvtColor(im2_res, cv2.COLOR_RGB2BGR)),
        'img_mask1': cv2_to_b64(cv2.cvtColor(mask1_rgb, cv2.COLOR_RGB2BGR)),
        'img_mask2': cv2_to_b64(cv2.cvtColor(mask2_rgb, cv2.COLOR_RGB2BGR)),
        'img_bin': cv2_to_b64(cv2.cvtColor(bin_change_rgb, cv2.COLOR_RGB2BGR))
    }


def init_app(checkpoint_path: str, port: int = 8080):
    global MODEL, TOKENIZER, DEVICE

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing SatQuery Web Server on {DEVICE}...")

    TOKENIZER = QuestionTokenizer()
    MODEL = BiTemporalChangeModel(
        vocab_size=TOKENIZER.vocab_size + 10,
        num_classes=7,
        num_answers=len(ANSWER_VOCAB),
        pretrained=False
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        MODEL.load_state_dict(checkpoint['model_state_dict'])
    else:
        MODEL.load_state_dict(checkpoint)
    MODEL.eval()
    print("Checkpoint loaded successfully.")

    server = HTTPServer(('0.0.0.0', port), SatQueryHandler)
    print(f"\n========================================================")
    print(f"  SatQuery Interactive Web UI live at: http://localhost:{port}")
    print(f"========================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="C:/satquery/checkpoints/best_model.pth")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    init_app(args.checkpoint, args.port)
