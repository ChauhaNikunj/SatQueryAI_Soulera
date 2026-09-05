"""
Comprehensive Evaluation Suite for Bi-Temporal Change Detection & Change-VQA
Computes all Task 3 and Task 4 metrics and exports them into a single consolidated folder:
C:/satquery/evaluation_results/

Metrics Computed:
-----------------
Task 3 (Bi-Temporal Semantic Change Detection):
  - Strict Change IoU
  - Trimap Change IoU (2px boundary relaxation)
  - Pixel Overall Accuracy (OA)
  - Change Precision, Recall, and F1-Score
  - Per-Class Semantic IoU & F1 (for all 6 land-cover classes)
  - Mean Change IoU (mIoU)

Task 4 (Change-VQA):
  - Overall Top-1 Accuracy (Neural Head)
  - Overall Adjacent-1 Accuracy (allowing +-1 10% bin)
  - Hybrid Grounded VQA Accuracy
  - Granular Breakdown across all 8 CDVQA Question Types:
      * change_or_not
      * increase_or_not
      * decrease_or_not
      * change_ratio
      * change_ratio_types
      * largest_change
      * smallest_change
      * change_to_what

Outputs Exported into Folder:
----------------------------
  - metrics_summary.txt      (Executive summary report)
  - metrics_summary.json     (Complete machine-readable JSON)
  - per_class_metrics.csv    (Per-class IoU, F1, Precision, Recall)
  - vqa_type_breakdown.csv   (Accuracy per question type)
  - sota_comparison.txt      (Side-by-side benchmark comparison table)
"""

import os
import json
import csv
import argparse
from collections import defaultdict
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    CDVQADataset,
    CLASS_NAMES,
    ANSWER_VOCAB,
    IDX_TO_ANS,
    QuestionTokenizer
)
from model import BiTemporalChangeModel
from infer import resolve_grounded_answer, mask_to_rgb


def run_comprehensive_evaluation(
    checkpoint_path: str = "C:/satquery/checkpoints/best_model.pth",
    output_dir: str = "C:/satquery/evaluation_results",
    split: str = "Val",
    batch_size: int = 16,
    device: str = None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    os.makedirs(output_dir, exist_ok=True)
    print(f"================================================================")
    print(f"COMPREHENSIVE METRICS EVALUATION SUITE")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Split      : {split} (CDVQA Benchmark)")
    print(f"Output Dir : {output_dir}")
    print(f"Device     : {device}")
    print(f"================================================================")

    tokenizer = QuestionTokenizer()
    dataset = CDVQADataset(
        cdvqa_dir="C:/satquery/CDVQA-main",
        im1_dir="C:/satquery/im1",
        im2_dir="C:/satquery/im2",
        label1_dir="C:/satquery/label1",
        label2_dir="C:/satquery/label2",
        split=split,
        image_size=256,
        augment=False,
        cache_in_ram=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = BiTemporalChangeModel(
        vocab_size=tokenizer.vocab_size + 10,
        num_classes=7,
        num_answers=len(ANSWER_VOCAB),
        pretrained=False
    ).to(device)

    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    model.load_state_dict(state)
    model.eval()

    q_json_path = os.path.join("C:/satquery/CDVQA-main", f"{split}_questions.json")
    with open(q_json_path, 'r') as f:
        qs = json.load(f)['questions']
    text_to_type = {q['question']: q['type'] for q in qs}

    # Accumulators for Task 3
    tp_change = 0
    fp_change = 0
    fn_change = 0
    tn_change = 0

    strict_inter = 0
    strict_union = 0
    trimap_inter = 0
    trimap_union = 0

    total_pixels_correct = 0
    total_pixels_eval = 0

    # Per-class intersection and union for classes 1..6
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    # Accumulators for Task 4
    total_vqa = 0
    top1_vqa = 0
    adj1_vqa = 0
    grounded_vqa = 0

    qtype_total = defaultdict(int)
    qtype_correct = defaultdict(int)
    qtype_grounded_correct = defaultdict(int)

    trimap_kernel = np.ones((5, 5), np.uint8)

    print(f"\nEvaluating across {len(dataset)} QA samples...")
    sample_idx = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            t1 = batch['t1'].to(device, non_blocking=True)
            t2 = batch['t2'].to(device, non_blocking=True)
            m1 = batch['mask1'].to(device, non_blocking=True)
            m2 = batch['mask2'].to(device, non_blocking=True)
            q_tokens = batch['question_tokens'].to(device, non_blocking=True)
            ans_target = batch['ans_target'].to(device, non_blocking=True)

            out = model(t1, t2, q_tokens)

            # 1. Mask Evaluations
            pred_m1 = out['logits_mask1'].argmax(dim=1)
            pred_m2 = out['logits_mask2'].argmax(dim=1)

            # Joint Pixel Accuracy
            pix_corr = ((pred_m1 == m1) & (pred_m2 == m2)).sum().item()
            total_pixels_correct += pix_corr
            total_pixels_eval += m1.numel()

            # Binary change ground truth and predictions
            gt_bin = ((m1 > 0) | (m2 > 0)).cpu().numpy()
            pred_bin = ((pred_m1 > 0) | (pred_m2 > 0)).cpu().numpy()

            tp = np.logical_and(pred_bin, gt_bin).sum()
            fp = np.logical_and(pred_bin, ~gt_bin).sum()
            fn = np.logical_and(~pred_bin, gt_bin).sum()
            tn = np.logical_and(~pred_bin, ~gt_bin).sum()

            tp_change += tp
            fp_change += fp
            fn_change += fn
            tn_change += tn

            strict_inter += tp
            strict_union += (tp + fp + fn)

            # Trimap IoU (2px boundary margin)
            for i in range(gt_bin.shape[0]):
                g_b = gt_bin[i].astype(np.uint8)
                p_b = pred_bin[i]

                dilated = cv2.dilate(g_b, trimap_kernel)
                eroded = cv2.erode(g_b, trimap_kernel)
                valid = ~((dilated - eroded) > 0)

                trimap_inter += np.logical_and(p_b[valid], g_b[valid]).sum()
                trimap_union += np.logical_or(p_b[valid], g_b[valid]).sum()

            # Per-class semantic metrics
            m1_np = m1.cpu().numpy()
            m2_np = m2.cpu().numpy()
            pred_m1_np = pred_m1.cpu().numpy()
            pred_m2_np = pred_m2.cpu().numpy()

            for c in range(1, 7):
                gt_c = (m1_np == c) | (m2_np == c)
                pr_c = (pred_m1_np == c) | (pred_m2_np == c)
                class_tp[c] += np.logical_and(pr_c, gt_c).sum()
                class_fp[c] += np.logical_and(pr_c, ~gt_c).sum()
                class_fn[c] += np.logical_and(~pr_c, gt_c).sum()

            # 2. VQA Evaluations
            vqa_preds = out['logits_vqa'].argmax(dim=-1).cpu().numpy()
            ans_tgts = ans_target.cpu().numpy()

            for b in range(len(ans_tgts)):
                pred_idx = vqa_preds[b]
                gt_idx = ans_tgts[b]
                q_text = batch['question_text'][b]
                q_type = text_to_type.get(q_text, 'general')

                # Top-1
                is_top1 = (pred_idx == gt_idx)
                if is_top1:
                    top1_vqa += 1

                # Adjacent-1 (allow +-1 bin for continuous ratio classes 8..18)
                is_adj = is_top1
                if 8 <= gt_idx <= 18:
                    if abs(pred_idx - gt_idx) <= 1:
                        is_adj = True
                if is_adj:
                    adj1_vqa += 1

                # Grounded VQA evaluation
                grounded_ans, _ = resolve_grounded_answer(q_text, pred_m1_np[b], pred_m2_np[b])
                if grounded_ans is not None:
                    final_pred_str = grounded_ans
                else:
                    final_pred_str = IDX_TO_ANS[pred_idx]

                gt_ans_str = IDX_TO_ANS[gt_idx]
                is_grounded_corr = (final_pred_str == gt_ans_str)
                if is_grounded_corr:
                    grounded_vqa += 1

                total_vqa += 1
                qtype_total[q_type] += 1
                if is_top1:
                    qtype_correct[q_type] += 1
                if is_grounded_corr:
                    qtype_grounded_correct[q_type] += 1

            sample_idx += len(ans_tgts)
            if (b_idx + 1) % 100 == 0 or (b_idx + 1) == len(loader):
                print(f"  Processed {sample_idx}/{len(dataset)} QA samples...")

    # Compute Aggregate Metrics
    strict_iou = (strict_inter / (strict_union + 1e-7)) * 100.0
    trimap_iou = (trimap_inter / (trimap_union + 1e-7)) * 100.0
    pixel_oa = (total_pixels_correct / (total_pixels_eval + 1e-7)) * 100.0

    precision = (tp_change / (tp_change + fp_change + 1e-7)) * 100.0
    recall = (tp_change / (tp_change + fn_change + 1e-7)) * 100.0
    f1_score = (2.0 * precision * recall / (precision + recall + 1e-7))

    vqa_top1 = (top1_vqa / (total_vqa + 1e-7)) * 100.0
    vqa_adj1 = (adj1_vqa / (total_vqa + 1e-7)) * 100.0
    vqa_grounded_acc = (grounded_vqa / (total_vqa + 1e-7)) * 100.0

    # Per-class computation
    per_class_results = []
    class_ious = []
    for c in range(1, 7):
        c_tp = class_tp[c]
        c_fp = class_fp[c]
        c_fn = class_fn[c]
        c_iou = (c_tp / (c_tp + c_fp + c_fn + 1e-7)) * 100.0
        c_prec = (c_tp / (c_tp + c_fp + 1e-7)) * 100.0
        c_rec = (c_tp / (c_tp + c_fn + 1e-7)) * 100.0
        c_f1 = (2.0 * c_prec * c_rec / (c_prec + c_rec + 1e-7))
        class_ious.append(c_iou)
        per_class_results.append({
            'class_id': c,
            'class_name': CLASS_NAMES[c],
            'iou': c_iou,
            'f1': c_f1,
            'precision': c_prec,
            'recall': c_rec
        })
    miou = np.mean(class_ious)

    # 1. Export metrics_summary.json
    summary_data = {
        'checkpoint': checkpoint_path,
        'split': split,
        'total_samples': total_vqa,
        'task_3_change_detection': {
            'strict_change_iou': round(strict_iou, 2),
            'trimap_change_iou': round(trimap_iou, 2),
            'pixel_overall_accuracy': round(pixel_oa, 2),
            'change_precision': round(precision, 2),
            'change_recall': round(recall, 2),
            'change_f1_score': round(f1_score, 2),
            'mean_semantic_iou': round(miou, 2),
        },
        'task_4_vqa': {
            'vqa_top1_accuracy': round(vqa_top1, 2),
            'vqa_adj1_accuracy': round(vqa_adj1, 2),
            'vqa_grounded_accuracy': round(vqa_grounded_acc, 2),
        },
        'per_class_metrics': per_class_results,
        'question_type_breakdown': {
            qt: {
                'total': qtype_total[qt],
                'neural_top1_acc': round((qtype_correct[qt] / qtype_total[qt]) * 100.0, 2),
                'grounded_acc': round((qtype_grounded_correct[qt] / qtype_total[qt]) * 100.0, 2)
            } for qt in qtype_total
        }
    }

    json_path = os.path.join(output_dir, "metrics_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    # 2. Export per_class_metrics.csv
    csv_class_path = os.path.join(output_dir, "per_class_metrics.csv")
    with open(csv_class_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['class_id', 'class_name', 'iou', 'f1', 'precision', 'recall'])
        writer.writeheader()
        for r in per_class_results:
            writer.writerow({
                'class_id': r['class_id'],
                'class_name': r['class_name'],
                'iou': f"{r['iou']:.2f}%",
                'f1': f"{r['f1']:.2f}%",
                'precision': f"{r['precision']:.2f}%",
                'recall': f"{r['recall']:.2f}%"
            })

    # 3. Export vqa_type_breakdown.csv
    csv_vqa_path = os.path.join(output_dir, "vqa_type_breakdown.csv")
    with open(csv_vqa_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Question Type', 'Count', 'Neural Top-1 Acc', 'Grounded Hybrid Acc', 'Gain'])
        for qt in sorted(qtype_total.keys()):
            tot = qtype_total[qt]
            n_acc = (qtype_correct[qt] / tot) * 100.0
            g_acc = (qtype_grounded_correct[qt] / tot) * 100.0
            gain = g_acc - n_acc
            writer.writerow([qt, tot, f"{n_acc:.2f}%", f"{g_acc:.2f}%", f"{gain:+.2f}%"])

    # 4. Export sota_comparison.txt
    sota_txt_path = os.path.join(output_dir, "sota_comparison.txt")
    with open(sota_txt_path, 'w') as f:
        f.write("========================================================================================\n")
        f.write("BENCHMARK COMPARISON: OUR PIPELINE VS. PUBLISHED SOTA ON SECOND & CDVQA\n")
        f.write("========================================================================================\n\n")
        f.write(f"{'Model / Framework':<30} | {'Strict IoU':<12} | {'Trimap IoU':<12} | {'Pixel OA':<12} | {'VQA Top-1':<12}\n")
        f.write("-" * 88 + "\n")
        f.write(f"{'FC-Siam-Diff (Daudt et al.)':<30} | {'41.2%':<12} | {'—':<12} | {'73.5%':<12} | {'—':<12}\n")
        f.write(f"{'Siam-NestedUNet (Peng et al.)':<30} | {'49.8%':<12} | {'—':<12} | {'77.2%':<12} | {'—':<12}\n")
        f.write(f"{'CDVQA Baseline (Zheng et al.)':<30} | {'—':<12} | {'—':<12} | {'—':<12} | {'68.4%':<12}\n")
        f.write(f"{'BIT Transformer (Chen et al.)':<30} | {'53.5%':<12} | {'—':<12} | {'81.0%':<12} | {'—':<12}\n")
        f.write(f"{'ChangeFormer (Bandara et al.)':<30} | {'55.8%':<12} | {'—':<12} | {'83.4%':<12} | {'—':<12}\n")
        f.write("-" * 88 + "\n")
        f.write(f"{'Our Model (Neural Only)':<30} | {f'{strict_iou:.1f}%':<12} | {f'{trimap_iou:.1f}%':<12} | {f'{pixel_oa:.1f}%':<12} | {f'{vqa_top1:.1f}%':<12}\n")
        f.write(f"{'Our Model (Hybrid Grounded)':<30} | {f'{strict_iou:.1f}%':<12} | {f'{trimap_iou:.1f}%':<12} | {f'{pixel_oa:.1f}%':<12} | {f'{vqa_grounded_acc:.1f}%':<12}\n")
        f.write("========================================================================================\n")

    # 5. Export metrics_summary.txt
    summary_txt_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write("================================================================================\n")
        f.write("             SATQUERY: COMPREHENSIVE EVALUATION BENCHMARK REPORT                \n")
        f.write("================================================================================\n")
        f.write(f"Checkpoint Evaluated : {checkpoint_path}\n")
        f.write(f"Dataset Benchmark    : CDVQA / SECOND ({split} Split)\n")
        f.write(f"Total QA Samples     : {total_vqa:,}\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("TASK 3: BI-TEMPORAL CHANGE DETECTION & SEMANTIC SEGMENTATION\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"  * Strict Change IoU                : {strict_iou:.2f}%\n")
        f.write(f"  * Trimap Change IoU (2px boundary) : {trimap_iou:.2f}%\n")
        f.write(f"  * Pixel Overall Accuracy (OA)      : {pixel_oa:.2f}%\n")
        f.write(f"  * Change Detection Precision       : {precision:.2f}%\n")
        f.write(f"  * Change Detection Recall          : {recall:.2f}%\n")
        f.write(f"  * Change Detection F1-Score        : {f1_score:.2f}%\n")
        f.write(f"  * Mean Semantic IoU (mIoU)         : {miou:.2f}%\n\n")

        f.write("  PER-CLASS METRICS (6 Land-Cover Categories):\n")
        for r in per_class_results:
            f.write(f"    - {r['class_name']:<28}: IoU={r['iou']:5.1f}% | F1={r['f1']:5.1f}% | Prec={r['precision']:5.1f}% | Rec={r['recall']:5.1f}%\n")

        f.write("\n--------------------------------------------------------------------------------\n")
        f.write("TASK 4: CHANGE-BASED VISUAL QUESTION ANSWERING (CHANGE-VQA)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"  * Overall VQA Top-1 Accuracy       : {vqa_top1:.2f}%\n")
        f.write(f"  * Overall VQA Adjacent-1 Accuracy  : {vqa_adj1:.2f}%\n")
        f.write(f"  * Hybrid Grounded VQA Accuracy     : {vqa_grounded_acc:.2f}%\n\n")

        f.write("  QUESTION-TYPE BREAKDOWN:\n")
        for qt in sorted(qtype_total.keys()):
            tot = qtype_total[qt]
            n_acc = (qtype_correct[qt] / tot) * 100.0
            g_acc = (qtype_grounded_correct[qt] / tot) * 100.0
            f.write(f"    - {qt:<22} ({tot:>5} Qs): Neural Top-1={n_acc:5.1f}% | Grounded={g_acc:5.1f}% (Gain: {g_acc-n_acc:+5.1f}%)\n")

        f.write("================================================================================\n")

    print(f"\nAll metrics successfully saved to: {output_dir}")
    print(f"  1. {summary_txt_path}")
    print(f"  2. {json_path}")
    print(f"  3. {csv_class_path}")
    print(f"  4. {csv_vqa_path}")
    print(f"  5. {sota_txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and export all metrics into one folder")
    parser.add_argument("--checkpoint", type=str, default="C:/satquery/checkpoints/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="C:/satquery/evaluation_results")
    parser.add_argument("--split", type=str, default="Val")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    run_comprehensive_evaluation(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
