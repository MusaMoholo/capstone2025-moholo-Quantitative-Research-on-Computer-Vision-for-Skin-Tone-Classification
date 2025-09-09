# Fairness-Aware Fitzpatrick Classification Under Low Compute

This repository accompanies the MSc thesis **“Fairness-Aware Fitzpatrick Classification Under Low Compute.”**  
It investigates 6-class Fitzpatrick skin-tone classification using a MobileNetV3-Large backbone with calibration and simple preprocessing. The repo includes training/evaluation code, figure-generation scripts, and a small Streamlit demo to illustrate how a product **could** embed the model.

> **Research use only.** This is **not** a clinical device and must not be used for diagnosis or triage.

## What’s here
- **`src/`** — Training/evaluation pipelines, metrics, calibration, figure scripts, and data utilities.
- **`streamlit_app/`** — Minimal demo app (illustrative only).
- **`outputs/`** — Run artifacts (logs, metrics, checkpoints) created locally.
- **`exports/champion/`** — Inference bundle (e.g., `model.keras`, `T.txt`, optional maps/configs).

## At a glance
- Task: 6-class (Fitzpatrick I–VI) image classification
- Model: MobileNetV3-Large + classifier head
- Reporting: macro-F1 and reliability (temperature scaling)
- Repro: scripts to regenerate figures/tables from thesis

## Citation
Moholo, M. (2025). *QFairness-Aware Fitzpatrick Classification Under Low Compute*. MSc Thesis.  
Repository: https://github.com/MusaMoholo/capstone2025-moholo-Quantitative-Research-on-Computer-Vision-for-Skin-Tone-Classification

## License
Specify a license (e.g., MIT/Apache-2.0) or “All rights reserved.”
