@echo off
echo Starting Training: FULL DATASET (1600 image pairs, 65,967 QA samples)
echo Resolution: 256x256, Batch Size: 16, Spatial Cross-Attention + SOTA Combined Loss...
python train.py
pause
