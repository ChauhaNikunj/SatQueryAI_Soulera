"""
Generate comprehensive dashboard visual results for 10 diverse validation pairs.
Saves all outputs to C:/satquery/results/ including:
  1. Full dashboard composite images (showing all masks, descriptions, and VQA cards)
  2. Structured JSON metadata records
  3. Executive summary markdown report
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from dataset import (
    CLASS_NAMES,
    ANSWER_VOCAB,
    IDX_TO_ANS,
    QuestionTokenizer
)
from model import BiTemporalChangeModel, generate_change_description
from infer import (
    load_and_preprocess_pair,
    mask_to_rgb,
    apply_morphological_filtering,
    probs_to_mask,
    resolve_grounded_answer,
    CLASS_TO_RGB
)

OUTPUT_DIR = "C:/satquery/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "C:/satquery/checkpoints/best_model.pth"

# Load Fonts
FONT_TITLE = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 20)
FONT_SUBTITLE = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 12)
FONT_CARD_TITLE = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 14)
FONT_BODY = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 13)
FONT_BODY_BOLD = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 13)
FONT_SMALL = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 11)
FONT_SMALL_BOLD = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 11)

# Color Palette (Matches Dark GitHub Theme of Dashboard)
BG_COLOR = (13, 17, 23)          # #0d1117
CARD_BG = (22, 27, 34)           # #161b22
CARD_BORDER = (48, 54, 61)       # #30363d
TEXT_WHITE = (240, 246, 252)     # #f0f6fc
TEXT_MUTED = (139, 148, 158)     # #8b949e
TEXT_BODY = (201, 209, 217)      # #c9d1d9
ACCENT_BLUE = (88, 166, 255)     # #58a6ff
PILL_BLUE = (31, 111, 235)       # #1f6feb
BADGE_GREEN = (35, 134, 54)      # #238636

SAMPLE_PAIRS = [
    '01908.png',  # 18.6% change (Balanced urban & vegetation shift)
    '04002.png',  # 23.7% change (Substantial building construction)
    '03182.png',  #  2.5% change (Subtle low vegetation delta)
    '06565.png',  #  0.0% change (Zero-change control pair)
    '02239.png',  #  5.8% change (Playground appearance & tree growth)
    '02692.png',  #  7.5% change (Bare ground to building conversion)
    '05570.png',  # 32.5% change (Major land clearance & water delta)
    '11054.png',  # 33.7% change (Urban expansion & water presence)
    '00446.png',  # 19.7% change (Vegetation restoration)
    '08871.png'   # 28.3% change (Ground excavation & vegetation loss)
]


def render_dashboard_image(pair_name, im1_rgb, im2_rgb, mask1_np, mask2_np, change_desc, vqa_items, threshold=0.35):
    """
    Renders an exact high-fidelity reproduction of what is shown on screen in the dashboard.
    """
    W, H = 1280, 950
    canvas = Image.new('RGB', (W, H), color=BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # 1. Header
    draw.text((24, 16), "SatQuery Multimodal System", fill=TEXT_WHITE, font=FONT_TITLE)
    draw.text((24, 42), "Bi-Temporal Remote Sensing Semantic Change Detection & Change-VQA", fill=TEXT_MUTED, font=FONT_SUBTITLE)

    # Header right pill: Pair & Sensitivity
    pill_text = f"Pair: {pair_name}  |  Threshold: {threshold:.2f} (Balanced SOTA)  |  Device: {DEVICE}"
    pill_w = int(draw.textlength(pill_text, font=FONT_SMALL_BOLD)) + 20
    pill_x = W - 24 - pill_w
    draw.rounded_rectangle([pill_x, 22, pill_x + pill_w, 46], radius=12, fill=BADGE_GREEN)
    draw.text((pill_x + 10, 27), pill_text, fill=TEXT_WHITE, font=FONT_SMALL_BOLD)

    # Divider
    draw.line([(24, 66), (W - 24, 66)], fill=CARD_BORDER, width=1)

    # 2. Task 3 Card: Segmentation & Masks
    c1_y1, c1_y2 = 78, 480
    draw.rounded_rectangle([20, c1_y1, W - 20, c1_y2], radius=8, fill=CARD_BG, outline=CARD_BORDER, width=1)
    draw.text((36, c1_y1 + 12), "Task 3: Bi-Temporal Segmentation & Change Mask", fill=TEXT_WHITE, font=FONT_CARD_TITLE)

    # 5 Thumbnails
    thumb_sz = 224
    spacing = ( (W - 40) - 32 - (5 * thumb_sz) ) // 4
    start_x = 36
    img_y = c1_y1 + 42

    mask1_rgb = mask_to_rgb(mask1_np)
    mask2_rgb = mask_to_rgb(mask2_np)
    bin_np = ((mask1_np > 0) | (mask2_np > 0)).astype(np.uint8) * 255
    bin_rgb = cv2.cvtColor(bin_np, cv2.COLOR_GRAY2RGB)

    panels = [
        (Image.fromarray(cv2.resize(im1_rgb, (thumb_sz, thumb_sz))), "T1 Image (Pre)"),
        (Image.fromarray(cv2.resize(im2_rgb, (thumb_sz, thumb_sz))), "T2 Image (Post)"),
        (Image.fromarray(cv2.resize(mask1_rgb, (thumb_sz, thumb_sz), interpolation=cv2.INTER_NEAREST)), "Predicted T1 Mask"),
        (Image.fromarray(cv2.resize(mask2_rgb, (thumb_sz, thumb_sz), interpolation=cv2.INTER_NEAREST)), "Predicted T2 Mask"),
        (Image.fromarray(cv2.resize(bin_rgb, (thumb_sz, thumb_sz), interpolation=cv2.INTER_NEAREST)), "Binary Change Mask")
    ]

    for i, (panel_img, label) in enumerate(panels):
        px = start_x + i * (thumb_sz + spacing)
        # Inner thumb border
        draw.rectangle([px - 1, img_y - 1, px + thumb_sz, img_y + thumb_sz], outline=CARD_BORDER, width=1)
        canvas.paste(panel_img, (px, img_y))

        # Thumbnail Label
        lbl_w = int(draw.textlength(label, font=FONT_SMALL_BOLD))
        lbl_x = px + (thumb_sz - lbl_w) // 2
        draw.text((lbl_x, img_y + thumb_sz + 8), label, fill=TEXT_BODY, font=FONT_SMALL_BOLD)

    # Class Color Legend Bar
    legend_y = img_y + thumb_sz + 34
    draw.line([(36, legend_y - 6), (W - 36, legend_y - 6)], fill=(30, 36, 44), width=1)
    
    legend_items = [
        ("No Change", (255, 255, 255)),
        ("NVG Ground", (128, 128, 128)),
        ("Tree", (0, 255, 0)),
        ("Low Veg", (0, 128, 0)),
        ("Water", (0, 0, 255)),
        ("Buildings", (128, 0, 0)),
        ("Playgrounds", (255, 0, 0))
    ]
    leg_total_w = sum(int(draw.textlength(name, font=FONT_SMALL)) + 28 for name, _ in legend_items)
    leg_x = (W - leg_total_w) // 2

    for name, col in legend_items:
        draw.rectangle([leg_x, legend_y + 2, leg_x + 12, legend_y + 14], fill=col, outline=(80, 80, 80))
        draw.text((leg_x + 18, legend_y), name, fill=TEXT_BODY, font=FONT_SMALL)
        leg_x += int(draw.textlength(name, font=FONT_SMALL)) + 28

    # 3. Factual Change Description Card
    c2_y1, c2_y2 = 492, 608
    draw.rounded_rectangle([20, c2_y1, W - 20, c2_y2], radius=8, fill=(9, 13, 19), outline=CARD_BORDER, width=1)
    # Accent cyan/blue left stripe
    draw.rectangle([20, c2_y1 + 4, 25, c2_y2 - 4], fill=ACCENT_BLUE)

    draw.text((38, c2_y1 + 10), "Factual Change Description (Area Delta Grounding)", fill=ACCENT_BLUE, font=FONT_CARD_TITLE)
    
    # Text wrapping for description
    words = change_desc.split(' ')
    lines = []
    curr = []
    for w in words:
        test_line = " ".join(curr + [w])
        if draw.textlength(test_line, font=FONT_BODY) > 1180:
            lines.append(" ".join(curr))
            curr = [w]
        else:
            curr.append(w)
    if curr:
        lines.append(" ".join(curr))

    y_desc = c2_y1 + 34
    for l in lines:
        draw.text((38, y_desc), l, fill=TEXT_WHITE, font=FONT_BODY)
        y_desc += 20

    draw.text((38, c2_y1 + 82), "Note: Total changed area is the union of all modified pixels. Category percentages indicate net land-cover shifts.", fill=TEXT_MUTED, font=FONT_SMALL)

    # 4. Task 4 Change-VQA Card
    c3_y1, c3_y2 = 620, 915
    draw.rounded_rectangle([20, c3_y1, W - 20, c3_y2], radius=8, fill=CARD_BG, outline=CARD_BORDER, width=1)
    draw.text((36, c3_y1 + 12), "Task 4: Change-Based Visual Question Answering (Change-VQA)", fill=TEXT_WHITE, font=FONT_CARD_TITLE)

    qa_start_y = c3_y1 + 44
    for idx, item in enumerate(vqa_items):
        cur_y = qa_start_y + idx * 56
        # Question
        q_str = f"Q{idx+1}: {item['question']}"
        draw.text((38, cur_y), q_str, fill=TEXT_BODY, font=FONT_BODY)

        # Answer pill
        ans_text = str(item['answer'])
        ans_w = int(draw.textlength(ans_text, font=FONT_BODY_BOLD)) + 18
        ans_x = 38
        ans_y = cur_y + 22
        draw.rounded_rectangle([ans_x, ans_y, ans_x + ans_w, ans_y + 24], radius=6, fill=PILL_BLUE)
        draw.text((ans_x + 9, ans_y + 4), ans_text, fill=TEXT_WHITE, font=FONT_BODY_BOLD)

        # Method badge
        if item['method'] == 'grounded':
            detail_str = item.get('details', '100% Deterministic Grounding')
            method_badge = f"Grounded Mask Verification ({detail_str})"
            badge_color = (63, 185, 80) # light green
        else:
            conf = item.get('confidence', 95.0)
            method_badge = f"Neural Cross-Attention (Confidence: {conf:.1f}%)"
            badge_color = TEXT_MUTED

        draw.text((ans_x + ans_w + 14, ans_y + 4), method_badge, fill=badge_color, font=FONT_SMALL)

    # 5. Bottom Status Footer
    draw.text((24, 925), f"SatQuery Multi-Task Evaluation | SECOND Dataset Benchmark | Saved at C:/satquery/results/ | Model: best_model.pth", fill=TEXT_MUTED, font=FONT_SMALL)

    return canvas


def run_all_samples():
    print("=" * 60)
    print("Generating Dashboard Visual Results for 10 Random Pairs")
    print(f"Destination Folder: {OUTPUT_DIR}")
    print(f"Device: {DEVICE} | Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)

    tok = QuestionTokenizer()
    model = BiTemporalChangeModel(
        vocab_size=tok.vocab_size + 10,
        num_classes=7,
        num_answers=len(ANSWER_VOCAB),
        pretrained=False
    ).to(DEVICE)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    summary_records = []

    for i, pair in enumerate(SAMPLE_PAIRS, 1):
        print(f"\n[{i:02d}/10] Processing pair: {pair} ...")
        p1_path = f"C:/satquery/im1/{pair}"
        p2_path = f"C:/satquery/im2/{pair}"

        t1_norm, t2_norm, im1_rgb, im2_rgb = load_and_preprocess_pair(p1_path, p2_path, image_size=256)
        t1_norm = t1_norm.to(DEVICE)
        t2_norm = t2_norm.to(DEVICE)

        with torch.no_grad():
            out = model(t1_norm, t2_norm)
            p1 = F.softmax(out['logits_mask1'], dim=1)
            p2 = F.softmax(out['logits_mask2'], dim=1)

        m1_pred = probs_to_mask(p1, threshold=0.35).squeeze(0).cpu().numpy()
        m2_pred = probs_to_mask(p2, threshold=0.35).squeeze(0).cpu().numpy()
        m1_np = apply_morphological_filtering(m1_pred)
        m2_np = apply_morphological_filtering(m2_pred)

        desc = generate_change_description(torch.from_numpy(m1_np), torch.from_numpy(m2_np))

        # Standard set of CDVQA questions covering all key query types
        questions = [
            "Have the regions of buildings changed?",
            "What is the percentage of changed areas?",
            "Did the areas of trees change?",
            "What type of change is the largest?"
        ]

        vqa_items = []
        for q_text in questions:
            grounded_ans, grounded_exp = resolve_grounded_answer(q_text, m1_np, m2_np)
            if grounded_ans is not None:
                vqa_items.append({
                    'question': q_text,
                    'answer': grounded_ans,
                    'confidence': 100.0,
                    'method': 'grounded',
                    'details': grounded_exp
                })
            else:
                q_tok = tok.encode(q_text).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out_q = model(t1_norm, t2_norm, q_tok)
                    prob_q = F.softmax(out_q['logits_vqa'], dim=-1)
                    conf, idx = prob_q.max(dim=-1)
                    ans_str = IDX_TO_ANS[idx.item()]
                vqa_items.append({
                    'question': q_text,
                    'answer': ans_str,
                    'confidence': float(conf.item() * 100.0),
                    'method': 'neural'
                })

        # Render full dashboard composite image
        dash_img = render_dashboard_image(pair, im1_rgb, im2_rgb, m1_np, m2_np, desc, vqa_items, threshold=0.35)
        
        img_out_name = f"dashboard_{i:02d}_{pair.replace('.png', '')}.png"
        img_out_path = os.path.join(OUTPUT_DIR, img_out_name)
        dash_img.save(img_out_path)

        # Save structured JSON
        json_out_name = f"result_{i:02d}_{pair.replace('.png', '')}.json"
        json_out_path = os.path.join(OUTPUT_DIR, json_out_name)
        total_ch = float(((m1_np > 0) | (m2_np > 0)).mean() * 100.0)

        record = {
            'index': i,
            'pair': pair,
            'total_changed_percent': round(total_ch, 2),
            'change_description': desc,
            'vqa_results': vqa_items,
            'dashboard_image': img_out_name
        }
        with open(json_out_path, 'w') as f:
            json.dump(record, f, indent=2)

        summary_records.append(record)
        print(f"  -> Saved {img_out_name} (Total Change: {total_ch:.1f}%)")

    # Generate Summary Markdown Report
    summary_md_path = os.path.join(OUTPUT_DIR, "summary_report.md")
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write("# SatQuery Multi-Task Evaluation — 10 Sampled Validation Pairs\n\n")
        f.write(f"**Date:** 2026-09-05  \n")
        f.write(f"**Output Directory:** `C:/satquery/results/`  \n")
        f.write(f"**Model Checkpoint:** `C:/satquery/checkpoints/best_model.pth`  \n")
        f.write(f"**Detection Sensitivity:** `0.35 (Balanced SOTA)`  \n\n")
        f.write("---\n\n")
        f.write("## Overview of Results\n\n")
        f.write("| # | Pair Name | Total Change (%) | Dominant Change / Description | Buildings Changed? | Largest Change Type |\n")
        f.write("|---|---|---|---|---|---|\n")
        for rec in summary_records:
            q_bldg = next((x['answer'] for x in rec['vqa_results'] if 'buildings' in x['question']), 'N/A')
            q_largest = next((x['answer'] for x in rec['vqa_results'] if 'largest' in x['question']), 'N/A')
            # shorten desc for table
            short_desc = rec['change_description'].replace("Between the two images, ", "").split("(")[0].strip()
            if not short_desc:
                short_desc = "No significant change"
            f.write(f"| {rec['index']:02d} | `{rec['pair']}` | **{rec['total_changed_percent']:.1f}%** | {short_desc} | `{q_bldg}` | `{q_largest}` |\n")

        f.write("\n\n---\n\n")
        f.write("## Detailed Pair Breakdowns\n\n")
        for rec in summary_records:
            f.write(f"### {rec['index']:02d}. Pair `{rec['pair']}`\n")
            f.write(f"- **Total Changed Area:** {rec['total_changed_percent']:.1f}%\n")
            f.write(f"- **Factual Description:** {rec['change_description']}\n")
            f.write(f"- **Visual Dashboard Artifact:** [`{rec['dashboard_image']}`]({rec['dashboard_image']})\n")
            f.write(f"- **Change-VQA Query Outputs:**\n")
            for item in rec['vqa_results']:
                m_info = item.get('details', f"Confidence: {item.get('confidence', 0):.1f}%")
                f.write(f"  - *{item['question']}* &rarr; **`{item['answer']}`** ({m_info})\n")
            f.write("\n")

    print(f"\nSuccessfully generated 10 dashboard visual outputs and summary report in {OUTPUT_DIR}")


if __name__ == '__main__':
    run_all_samples()
