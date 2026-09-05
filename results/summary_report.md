# SatQuery Multi-Task Evaluation — 10 Sampled Validation Pairs

**Date:** 2026-09-05  
**Output Directory:** `C:/satquery/results/`  
**Model Checkpoint:** `C:/satquery/checkpoints/best_model.pth`  
**Detection Sensitivity:** `0.35 (Balanced SOTA)`  

---

## Overview of Results

| # | Pair Name | Total Change (%) | Dominant Change / Description | Buildings Changed? | Largest Change Type |
|---|---|---|---|---|---|
| 01 | `01908.png` | **18.6%** | Non-vegetated ground surface decreased by 9.5%; Tree cover increased by 3.9%; Low vegetation increased by 1.2%; Built-up area increased by 4.2% | `yes` | `NVG_surface` |
| 02 | `04002.png` | **23.7%** | Non-vegetated ground surface decreased by 19.1%; Tree cover increased by 4.6%; Low vegetation decreased by 1.3%; Built-up area increased by 15.2% | `yes` | `NVG_surface` |
| 03 | `03182.png` | **2.5%** | Non-vegetated ground surface decreased by 1.3%; Low vegetation increased by 0.9% | `yes` | `buildings` |
| 04 | `06565.png` | **0.4%** | No significant land-cover change detected between the two images. | `no` | `buildings` |
| 05 | `02239.png` | **5.8%** | Non-vegetated ground surface decreased by 1.1%; Tree cover increased by 2.2%; Low vegetation decreased by 2.8%; Playground area increased by 1.3% | `yes` | `NVG_surface` |
| 06 | `02692.png` | **7.5%** | Non-vegetated ground surface decreased by 7.5%; Built-up area increased by 7.4% | `yes` | `NVG_surface` |
| 07 | `05570.png` | **32.5%** | Non-vegetated ground surface increased by 15.9%; Tree cover decreased by 3.4%; Low vegetation decreased by 16.3%; Water body decreased by 2.4%; Built-up area increased by 5.9% | `yes` | `NVG_surface` |
| 08 | `11054.png` | **33.7%** | Non-vegetated ground surface decreased by 20.4%; Low vegetation increased by 5.4%; Water body increased by 1.3%; Built-up area increased by 13.6% | `yes` | `NVG_surface` |
| 09 | `00446.png` | **19.7%** | Non-vegetated ground surface decreased by 13.5%; Low vegetation increased by 10.3%; Built-up area increased by 3.1% | `yes` | `NVG_surface` |
| 10 | `08871.png` | **28.3%** | Non-vegetated ground surface increased by 22.5%; Tree cover decreased by 2.8%; Low vegetation decreased by 17.1%; Built-up area decreased by 2.0% | `yes` | `NVG_surface` |


---

## Detailed Pair Breakdowns

### 01. Pair `01908.png`
- **Total Changed Area:** 18.6%
- **Factual Description:** Between the two images, Non-vegetated ground surface decreased by 9.5%; Tree cover increased by 3.9%; Low vegetation increased by 1.2%; Built-up area increased by 4.2% (total changed area: 18.6%).
- **Visual Dashboard Artifact:** [`dashboard_01_01908.png`](dashboard_01_01908.png)
- **Change-VQA Query Outputs:**
  - *Have the regions of buildings changed?* &rarr; **`yes`** (Detected buildings changed area: 8.4%)
  - *What is the percentage of changed areas?* &rarr; **`10_to_20`** (Calculated total changed area: 18.6%)
  - *Did the areas of trees change?* &rarr; **`yes`** (Detected tree changed area: 4.0%)
  - *What type of change is the largest?* &rarr; **`NVG_surface`** (Largest detected change area: non-vegetated ground surface)

### 02. Pair `04002.png`
- **Total Changed Area:** 23.7%
- **Factual Description:** Between the two images, Non-vegetated ground surface decreased by 19.1%; Tree cover increased by 4.6%; Low vegetation decreased by 1.3%; Built-up area increased by 15.2% (total changed area: 23.7%).
- **Visual Dashboard Artifact:** [`dashboard_02_04002.png`](dashboard_02_04002.png)
- **Change-VQA Query Outputs:**
  - *Have the regions of buildings changed?* &rarr; **`yes`** (Detected buildings changed area: 17.0%)
  - *What is the percentage of changed areas?* &rarr; **`20_to_30`** (Calculated total changed area: 23.7%)
  - *Did the areas of trees change?* &rarr; **`yes`** (Detected tree changed area: 4.8%)
  - *What type of change is the largest?* &rarr; **`NVG_surface`** (Largest detected change area: non-vegetated ground surface)

### 03. Pair `03182.png`
- **Total Changed Area:** 2.5%
- **Factual Description:** Between the two images, Non-vegetated ground surface decreased by 1.3%; Low vegetation increased by 0.9% (total changed area: 2.5%).
- **Visual Dashboard Artifact:** [`dashboard_03_03182.png`](dashboard_03_03182.png)
- **Change-VQA Query Outputs:**
  - *Have the regions of buildings changed?* &rarr; **`yes`** (Detected buildings changed area: 2.1%)
  - *What is the percentage of changed areas?* &rarr; **`0_to_10`** (Calculated total changed area: 2.5%)
  - *Did the areas of trees change?* &rarr; **`no`** (Detected tree changed area is < 0.5% (0.0%))
  - *What type of change is the largest?* &rarr; **`buildings`** (Largest detected change area: buildings)

### 04. Pair `06565.png`
- **Total Changed Area:** 0.4%
- **Factual Description:** No significant land-cover change detected between the two images.
- **Visual Dashboard Artifact:** [`dashboard_04_06565.png`](dashboard_04_06565.png)
- **Change-VQA Query Outputs:**
  - *Have the regions of buildings changed?* &rarr; **`no`** (Calculated total change < 0.5% (0.4%))
  - *What is the percentage of changed areas?* &rarr; **`0`** (Calculated total changed area: 0.4%)
  - *Did the areas of trees change?* &rarr; **`no`** (Calculated total change < 0.5% (0.4%))
  - *What type of change is the largest?* &rarr; **`buildings`** (Largest detected change area: buildings)

### 05. Pair `02239.png`
- **Total Changed Area:** 5.8%
- **Factual Description:** Between the two images, Non-vegetated ground surface decreased by 1.1%; Tree cover increased by 2.2%; Low vegetation decreased by 2.8%; Playground area increased by 1.3% (total changed area: 5.8%).
- **Visual Dashboard Artifact:** [`dashboard_05_02239.png`](dashboard_05_02239.png)
- **Change-VQA Query Outputs:**
  - *Have the regions of buildings changed?* &rarr; **`yes`** (Detected buildings changed area: 0.9%)
  - *What is the percentage of changed areas?* &rarr; **`0_to_10`** (Calculated total changed area: 5.8%)
  - *Did the areas of trees change?* &rarr; **`yes`** (Detected tree changed area: 2.2%)
  - *What type of change is the largest?* &rarr; **`NVG_surface`** (Largest detected change area: non-vegetated ground surface)

### 06. Pair `02692.png`
- **Total Changed Area:** 7.5%
- **Factual Description:** Between the two images, Non-vegetated ground surface decreased by 7.5%; Built-up area increased by 7.4% (total changed area: 7.5%).
- **Visual Dashboard Artifact:** [`dashboard_06_02692.png`](dashboard_06_02692.png)
- **Change-VQA Query Outputs:**
  - *Have the regions of buildings changed?* &rarr; **`yes`** (Detected buildings changed area: 7.4%)
  - *What is the percentage of changed areas?* &rarr; **`0_to_10`** (Calculated total changed area: 7.5%)
  - *Did the areas of trees change?* &rarr; **`no`** (Detected tree changed area is < 0.5% (0.0%))
  - *What type of change is the largest?* &rarr; **`NVG_surface`** (Largest detected change area: non-vegetated ground surface)

### 07. Pair `05570.png`
- **Total Changed Area:** 32.5%
- **Factual Description:** Between the two images, Non-vegetated ground surface increased by 15.9%; Tree cover decreased by 3.4%; Low vegetation decreased by 16.3%; Water body decreased by 2.4%; Built-up area increased by 5.9% (total changed area: 32.5%).
- **Visual Dashboard Artifact:** [`dashboard_07_05570.png`](dashboard_07_05570.png)
- **Change-VQA Query Outputs:**
  - *Have the regions of buildings changed?* &rarr; **`yes`** (Detected buildings changed area: 5.9%)
  - *What is the percentage of changed areas?* &rarr; **`30_to_40`** (Calculated total changed area: 32.5%)
  - *Did the areas of trees change?* &rarr; **`yes`** (Detected tree changed area: 5.6%)
  - *What type of change is the largest?* &rarr; **`NVG_surface`** (Largest detected change area: non-vegetated ground surface)

### 08. Pair `11054.png`
- **Total Changed Area:** 33.7%
- **Factual Description:** Between the two images, Non-vegetated ground surface decreased by 20.4%; Low vegetation increased by 5.4%; Water body increased by 1.3%; Built-up area increased by 13.6% (total changed area: 33.7%).
- **Visual Dashboard Artifact:** [`dashboard_08_11054.png`](dashboard_08_11054.png)
- **Change-VQA Query Outputs:**
  - *Have the regions of buildings changed?* &rarr; **`yes`** (Detected buildings changed area: 13.8%)
  - *What is the percentage of changed areas?* &rarr; **`30_to_40`** (Calculated total changed area: 33.7%)
  - *Did the areas of trees change?* &rarr; **`no`** (Detected tree changed area is < 0.5% (0.1%))
  - *What type of change is the largest?* &rarr; **`NVG_surface`** (Largest detected change area: non-vegetated ground surface)

### 09. Pair `00446.png`
- **Total Changed Area:** 19.7%
- **Factual Description:** Between the two images, Non-vegetated ground surface decreased by 13.5%; Low vegetation increased by 10.3%; Built-up area increased by 3.1% (total changed area: 19.7%).
- **Visual Dashboard Artifact:** [`dashboard_09_00446.png`](dashboard_09_00446.png)
- **Change-VQA Query Outputs:**
  - *Have the regions of buildings changed?* &rarr; **`yes`** (Detected buildings changed area: 6.2%)
  - *What is the percentage of changed areas?* &rarr; **`10_to_20`** (Calculated total changed area: 19.7%)
  - *Did the areas of trees change?* &rarr; **`no`** (Detected tree changed area is < 0.5% (0.2%))
  - *What type of change is the largest?* &rarr; **`NVG_surface`** (Largest detected change area: non-vegetated ground surface)

### 10. Pair `08871.png`
- **Total Changed Area:** 28.3%
- **Factual Description:** Between the two images, Non-vegetated ground surface increased by 22.5%; Tree cover decreased by 2.8%; Low vegetation decreased by 17.1%; Built-up area decreased by 2.0% (total changed area: 28.3%).
- **Visual Dashboard Artifact:** [`dashboard_10_08871.png`](dashboard_10_08871.png)
- **Change-VQA Query Outputs:**
  - *Have the regions of buildings changed?* &rarr; **`yes`** (Detected buildings changed area: 7.6%)
  - *What is the percentage of changed areas?* &rarr; **`20_to_30`** (Calculated total changed area: 28.3%)
  - *Did the areas of trees change?* &rarr; **`yes`** (Detected tree changed area: 2.8%)
  - *What type of change is the largest?* &rarr; **`NVG_surface`** (Largest detected change area: non-vegetated ground surface)

