import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from dataset import (
    CLASS_NAMES,
    RGB_TO_CLASS,
    ANSWER_VOCAB,
    ANS_TO_IDX,
    IDX_TO_ANS,
    QuestionTokenizer,
    rgb_mask_to_class_indices
)
from model import BiTemporalChangeModel, generate_change_description

CLASS_TO_RGB = {
    0: (255, 255, 255), # No change (White)
    1: (128, 128, 128), # NVG surface (Gray)
    2: (0, 255, 0),     # Tree (Light Green)
    3: (0, 128, 0),     # Low veg (Dark Green)
    4: (0, 0, 255),     # Water (Blue)
    5: (128, 0, 0),     # Buildings (Maroon)
    6: (255, 0, 0)      # Playgrounds (Red)
}

GROUNDED_CLASS_KW = [
    (1, ['non-vegetated', 'nvg', 'ground surface']),
    (2, ['tree', 'trees']),
    (3, ['low vegetation']),
    (4, ['water']),
    (5, ['building', 'buildings', 'built-up']),
    (6, ['playground', 'playgrounds'])
]

GROUNDED_CLASS_STRINGS = {
    1: 'NVG_surface',
    2: 'trees',
    3: 'low_vegetation',
    4: 'water',
    5: 'buildings',
    6: 'playgrounds'
}


def pct_to_bin_str(pct: float) -> str:
    """Maps continuous percentage [0..100] to the 11 discrete CDVQA ratio bins."""
    if pct < 0.5:
        return '0'
    elif pct <= 10.0:
        return '0_to_10'
    elif pct <= 20.0:
        return '10_to_20'
    elif pct <= 30.0:
        return '20_to_30'
    elif pct <= 40.0:
        return '30_to_40'
    elif pct <= 50.0:
        return '40_to_50'
    elif pct <= 60.0:
        return '50_to_60'
    elif pct <= 70.0:
        return '60_to_70'
    elif pct <= 80.0:
        return '70_to_80'
    elif pct <= 90.0:
        return '80_to_90'
    else:
        return '90_to_100'


def probs_to_mask(probs: torch.Tensor, threshold: float = 0.35) -> torch.Tensor:
    """
    Converts 7-class probability tensor (B, 7, H, W) or (7, H, W) to class mask (0..6).
    If threshold < 0.5, uses adaptive foreground thresholding where a pixel is classified
    as foreground class c if max(P_fg) >= threshold. Otherwise, falls back to argmax.
    """
    if probs.dim() == 4:
        probs = probs.squeeze(0)

    if threshold >= 0.50:
        return probs.argmax(dim=0)

    # Foreground classes are 1..6
    fg_probs = probs[1:]  # (6, H, W)
    max_fg, argmax_fg = fg_probs.max(dim=0)
    argmax_fg = argmax_fg + 1  # shift index to 1..6

    mask = torch.where(max_fg >= threshold, argmax_fg, torch.zeros_like(argmax_fg))
    return mask


def resolve_grounded_answer(q_text: str, mask1, mask2):
    """
    Neuro-symbolic resolver: computes factually grounded answers directly from
    the predicted semantic change masks for ratio, transition, and extremum queries.
    Returns (answer_str, explanation) or (None, None) for binary presence queries.
    """
    q = q_text.lower()
    m1 = mask1.numpy() if isinstance(mask1, torch.Tensor) else mask1
    m2 = mask2.numpy() if isinstance(mask2, torch.Tensor) else mask2
    total_px = m1.size
    total_change_pct = ((m1 > 0) | (m2 > 0)).sum() / total_px * 100.0

    # 0. Binary Presence queries ("Did the areas of...", "Have the regions of...")
    if any(w in q for w in ['did the areas of', 'have the regions of', 'have the areas of', 'has the area of', 'did the region of']):
        if total_change_pct < 0.5:
            return 'no', f"Calculated total change < 0.5% ({total_change_pct:.1f}%)"
        for c_idx, kw_list in GROUNDED_CLASS_KW:
            if any(kw in q for kw in kw_list):
                c_pct = ((m1 == c_idx) | (m2 == c_idx)).sum() / total_px * 100.0
                if c_pct < 0.5:
                    return 'no', f"Detected {CLASS_NAMES[c_idx]} changed area is < 0.5% ({c_pct:.1f}%)"
                else:
                    return 'yes', f"Detected {CLASS_NAMES[c_idx]} changed area: {c_pct:.1f}%"

    # 1. Ratio / Percentage queries
    if any(w in q for w in ['ratio', 'percentage', 'proportion', 'how much']):
        if any(w in q for w in ['unchanged', 'not changed', 'non-change']):
            pct = 100.0 - total_change_pct
            return pct_to_bin_str(pct), f"Calculated unchanged area: {pct:.1f}%"
        if any(w in q for w in ['changed areas', 'changed regions', 'area has changed']):
            pct = total_change_pct
            return pct_to_bin_str(pct), f"Calculated total changed area: {pct:.1f}%"

        # General vegetation (trees + low veg) decrease/increase/ratio
        if 'vegetation' in q and not ('low vegetation' in q):
            pct1 = ((m1 == 2) | (m1 == 3)).sum() / total_px * 100.0
            pct2 = ((m2 == 2) | (m2 == 3)).sum() / total_px * 100.0
            if any(w in q for w in ['decreas', 'reduc', 'drop', 'loss']):
                dec = max(0.0, pct1 - pct2)
                return pct_to_bin_str(dec), f"Calculated vegetation decrease: {dec:.1f}%"
            elif any(w in q for w in ['increas', 'grow', 'gain']):
                inc = max(0.0, pct2 - pct1)
                return pct_to_bin_str(inc), f"Calculated vegetation increase: {inc:.1f}%"
            else:
                pct = pct1 if any(w in q for w in ['first', 'pre-']) else pct2
                return pct_to_bin_str(pct), f"Calculated vegetation area: {pct:.1f}%"

        for c_idx, kw_list in GROUNDED_CLASS_KW:
            if any(kw in q for kw in kw_list):
                if any(w in q for w in ['decreas', 'reduc', 'drop', 'loss']):
                    p1 = (m1 == c_idx).sum() / total_px * 100.0
                    p2 = (m2 == c_idx).sum() / total_px * 100.0
                    dec = max(0.0, p1 - p2)
                    return pct_to_bin_str(dec), f"Calculated {CLASS_NAMES[c_idx]} decrease: {dec:.1f}%"
                elif any(w in q for w in ['increas', 'grow', 'gain']):
                    p1 = (m1 == c_idx).sum() / total_px * 100.0
                    p2 = (m2 == c_idx).sum() / total_px * 100.0
                    inc = max(0.0, p2 - p1)
                    return pct_to_bin_str(inc), f"Calculated {CLASS_NAMES[c_idx]} increase: {inc:.1f}%"
                elif any(w in q for w in ['first', 'pre-']):
                    pct = (m1 == c_idx).sum() / total_px * 100.0
                    return pct_to_bin_str(pct), f"Calculated T1 {CLASS_NAMES[c_idx]} changed area: {pct:.1f}%"
                else:
                    pct = (m2 == c_idx).sum() / total_px * 100.0
                    return pct_to_bin_str(pct), f"Calculated T2 {CLASS_NAMES[c_idx]} changed area: {pct:.1f}%"

    # 2. Transition queries: "changed to what"
    if 'changed to' in q:
        for c_idx, kw_list in GROUNDED_CLASS_KW:
            if any(kw in q for kw in kw_list):
                changed_pixels = m2[(m1 == c_idx) & (m2 != c_idx) & (m2 > 0)]
                if len(changed_pixels) > 0:
                    counts = np.bincount(changed_pixels)
                    target_to = np.argmax(counts)
                    return GROUNDED_CLASS_STRINGS[target_to], f"Dominant transition for {CLASS_NAMES[c_idx]}"

    # 3. Extremum queries: "smallest change" or "largest change"
    if 'smallest' in q:
        m = m1 if any(w in q for w in ['first', 'pre-']) else (m2 if any(w in q for w in ['second', 'post-']) else None)
        scores = {}
        for c_idx in range(1, 7):
            cnt = (m == c_idx).sum() if m is not None else ((m1 == c_idx).sum() + (m2 == c_idx).sum())
            if cnt > 0:
                scores[c_idx] = cnt
        if scores:
            best_c = min(scores, key=scores.get)
            return GROUNDED_CLASS_STRINGS[best_c], f"Smallest detected change area: {CLASS_NAMES[best_c]}"

    if 'largest' in q:
        m = m1 if any(w in q for w in ['first', 'pre-']) else (m2 if any(w in q for w in ['second', 'post-']) else None)
        scores = {}
        for c_idx in range(1, 7):
            cnt = (m == c_idx).sum() if m is not None else ((m1 == c_idx).sum() + (m2 == c_idx).sum())
            scores[c_idx] = cnt
        if scores:
            best_c = max(scores, key=scores.get)
            return GROUNDED_CLASS_STRINGS[best_c], f"Largest detected change area: {CLASS_NAMES[best_c]}"

    return None, None


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Converts a 2D integer class mask (0..6) back to an RGB image for visualization."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_TO_RGB.items():
        rgb[mask == cls_idx] = color
    return rgb


def apply_morphological_filtering(mask_np: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Applies morphological opening + closing to remove salt-and-pepper noise and sharpen edges."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Opening removes isolated noise speckles
    opened = cv2.morphologyEx(mask_np.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    # Closing fills small pinholes inside detected buildings/land covers
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def load_and_preprocess_pair(im1_path, im2_path, image_size=256):
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)
    if im1 is None or im2 is None:
        raise FileNotFoundError(f"Could not load images from {im1_path} or {im2_path}")

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    im1_resized = cv2.resize(im1, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    im2_resized = cv2.resize(im2, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    t1_tensor = torch.from_numpy(im1_resized.transpose(2, 0, 1)).float() / 255.0
    t2_tensor = torch.from_numpy(im2_resized.transpose(2, 0, 1)).float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t1_norm = ((t1_tensor - mean) / std).unsqueeze(0)
    t2_norm = ((t2_tensor - mean) / std).unsqueeze(0)

    return t1_norm, t2_norm, im1_resized, im2_resized


def run_inference(
    checkpoint_path: str,
    im1_path: str,
    im2_path: str,
    questions: list,
    output_viz_path: str = None,
    image_size: int = 256,
    use_tta: bool = True,
    morph_clean: bool = True,
    threshold: float = 0.35,
    device: str = None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print(f"\n==========================================")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device} | Test-Time Augmentation (TTA): {use_tta}")
    print(f"==========================================")

    tokenizer = QuestionTokenizer()
    model = BiTemporalChangeModel(
        vocab_size=tokenizer.vocab_size + 10,
        num_classes=7,
        num_answers=len(ANSWER_VOCAB),
        pretrained=False  # weights are restored from checkpoint
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    t1_tensor, t2_tensor, im1_rgb, im2_rgb = load_and_preprocess_pair(im1_path, im2_path, image_size)
    t1_tensor = t1_tensor.to(device)
    t2_tensor = t2_tensor.to(device)

    # 8-fold TTA transforms
    transforms = [
        (0, False), (1, False), (2, False), (3, False),
        (0, True),  (1, True),  (2, True),  (3, True)
    ] if use_tta else [(0, False)]

    # 1. Task 3: Change Mask Prediction with TTA
    accum_mask1_probs = None
    accum_mask2_probs = None

    with torch.no_grad():
        for rot_k, do_flip in transforms:
                curr_t1 = t1_tensor
                curr_t2 = t2_tensor

                if do_flip:
                    curr_t1 = TF.hflip(curr_t1)
                    curr_t2 = TF.hflip(curr_t2)
                if rot_k > 0:
                    curr_t1 = torch.rot90(curr_t1, rot_k, [2, 3])
                    curr_t2 = torch.rot90(curr_t2, rot_k, [2, 3])

                out = model(curr_t1, curr_t2)
                p1 = F.softmax(out['logits_mask1'], dim=1)
                p2 = F.softmax(out['logits_mask2'], dim=1)

                # Invert transform
                if rot_k > 0:
                    p1 = torch.rot90(p1, -rot_k, [2, 3])
                    p2 = torch.rot90(p2, -rot_k, [2, 3])
                if do_flip:
                    p1 = TF.hflip(p1)
                    p2 = TF.hflip(p2)

                if accum_mask1_probs is None:
                    accum_mask1_probs = p1
                    accum_mask2_probs = p2
                else:
                    accum_mask1_probs += p1
                    accum_mask2_probs += p2

    accum_mask1_probs /= len(transforms)
    accum_mask2_probs /= len(transforms)

    pred_mask1 = probs_to_mask(accum_mask1_probs, threshold=threshold).cpu()
    pred_mask2 = probs_to_mask(accum_mask2_probs, threshold=threshold).cpu()

    # Morphological boundary cleanup (removes single-pixel noise, smooths boundaries)
    if morph_clean:
        pred_mask1 = torch.from_numpy(apply_morphological_filtering(pred_mask1.numpy()))
        pred_mask2 = torch.from_numpy(apply_morphological_filtering(pred_mask2.numpy()))

    change_desc = generate_change_description(pred_mask1, pred_mask2)

    print("\n------------------------------------------------------------")
    print("TASK 3 — BI-TEMPORAL CHANGE DETECTION RESULTS")
    print("------------------------------------------------------------")
    print(f"T1 Image: {im1_path}")
    print(f"T2 Image: {im2_path}")
    print(f"\nGenerated Change Description:\n>> \"{change_desc}\"")

    total_px = pred_mask1.numel()
    print("\nPredicted Land-Cover Distribution:")
    for cls_idx in range(1, 7):
        pct1 = ((pred_mask1 == cls_idx).sum().item() / total_px) * 100.0
        pct2 = ((pred_mask2 == cls_idx).sum().item() / total_px) * 100.0
        if pct1 > 0.1 or pct2 > 0.1:
            print(f"  - {CLASS_NAMES[cls_idx]:<30}: T1={pct1:5.1f}%  -->  T2={pct2:5.1f}%  (diff: {pct2-pct1:+5.1f}%)")

    # 2. Task 4: Change-based VQA with TTA
    print("\n------------------------------------------------------------")
    print("TASK 4 — CHANGE-BASED VQA RESULTS (WITH CROSS-ATTENTION & TTA)")
    print("------------------------------------------------------------")

    vqa_results = []
    for q_text in questions:
        q_tokens = tokenizer.encode(q_text).unsqueeze(0).to(device)
        accum_vqa_probs = None

        with torch.no_grad():
            for rot_k, do_flip in transforms:
                    curr_t1 = t1_tensor
                    curr_t2 = t2_tensor
                    if do_flip:
                        curr_t1 = TF.hflip(curr_t1)
                        curr_t2 = TF.hflip(curr_t2)
                    if rot_k > 0:
                        curr_t1 = torch.rot90(curr_t1, rot_k, [2, 3])
                        curr_t2 = torch.rot90(curr_t2, rot_k, [2, 3])

                    out_q = model(curr_t1, curr_t2, q_tokens)
                    p_vqa = F.softmax(out_q['logits_vqa'], dim=-1)

                    if accum_vqa_probs is None:
                        accum_vqa_probs = p_vqa
                    else:
                        accum_vqa_probs += p_vqa

        accum_vqa_probs /= len(transforms)
        top_prob, top_idx = accum_vqa_probs.max(dim=-1)
        ans_str = IDX_TO_ANS[top_idx.item()]
        conf = top_prob.item() * 100.0

        # Check neuro-symbolic mask grounding for ratio, transition, and extremum queries
        grounded_ans, grounded_exp = resolve_grounded_answer(q_text, pred_mask1, pred_mask2)

        if grounded_ans is not None:
            final_ans = grounded_ans
            note = f"Grounded Mask Verification: {grounded_exp}"
            vqa_results.append({'question': q_text, 'answer': final_ans, 'confidence': 100.0, 'method': 'grounded'})
            print(f"Question: \"{q_text}\"")
            print(f"  --> Predicted Answer: >> {final_ans} << ({note})")
            print(f"      Neural Head Output: {ans_str} (Confidence: {conf:.1f}%)\n")
        else:
            final_ans = ans_str
            vqa_results.append({'question': q_text, 'answer': final_ans, 'confidence': conf, 'method': 'neural'})
            print(f"Question: \"{q_text}\"")
            print(f"  --> Predicted Answer: >> {final_ans} << (Confidence: {conf:.1f}%)\n")

    # 3. Optional Visualization
    if output_viz_path:
        mask1_rgb = mask_to_rgb(pred_mask1.numpy())
        mask2_rgb = mask_to_rgb(pred_mask2.numpy())
        bin_change = ((pred_mask1 > 0) | (pred_mask2 > 0)).numpy().astype(np.uint8) * 255
        bin_change_rgb = cv2.cvtColor(bin_change, cv2.COLOR_GRAY2RGB)

        combined = np.concatenate([im1_rgb, im2_rgb, mask1_rgb, mask2_rgb, bin_change_rgb], axis=1)
        cv2.imwrite(output_viz_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Saved visual comparison artifact to: {output_viz_path}")

    return {
        'change_description': change_desc,
        'vqa_results': vqa_results,
        'pred_mask1': pred_mask1,
        'pred_mask2': pred_mask2
    }


def main():
    parser = argparse.ArgumentParser(description="State-of-the-Art Inference for Change Detection + Change VQA")
    parser.add_argument("--checkpoint", type=str, default="C:/satquery/checkpoints/best_model.pth")
    parser.add_argument("--im1", type=str, default="C:/satquery/im1/02180.png")
    parser.add_argument("--im2", type=str, default="C:/satquery/im2/02180.png")
    parser.add_argument("--questions", nargs='+', default=None, help="List of custom questions to ask")
    parser.add_argument("--output_viz", type=str, default="C:/satquery/output_inference.png")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--no_morph", action="store_true", help="Disable Morphological noise cleanup")
    args = parser.parse_args()

    if args.questions:
        eval_questions = args.questions
    else:
        eval_questions = [
            "Have the regions of buildings changed?",
            "Did the areas of trees change?",
            "What is the percentage of changed areas?",
            "How much of the area has not changed?",
            "What is the change ratio of low vegetation in the pre-event image?",
            "What is the change percentage of buildings in the second image?",
            "What type of change is the largest?"
        ]

    run_inference(
        checkpoint_path=args.checkpoint,
        im1_path=args.im1,
        im2_path=args.im2,
        questions=eval_questions,
        output_viz_path=args.output_viz,
        image_size=args.image_size,
        use_tta=not args.no_tta,
        morph_clean=not args.no_morph
    )


if __name__ == '__main__':
    main()
