# VIP-HTD Hockey Dataset Repository

This repository provides tools and pretrained models for detection, tracking, and feature extraction on the VIP-HTD Hockey Dataset. Follow the instructions below to set up your environment, organize files, and run experiments.

---

## 1. Environment Setup

Create the required conda environment using the provided `environment.yml` file:

`conda env create -f environment.yml`

`python3 setup.py develop`

---
---

## 3. Pretrained Model Weights

All required weights for the VIP-HTD dataset are stored in the `./pretrained` directory:

- **Detection model (VIP-HTD):** `yolox_vip-htd.pth.tar`
- **Tracking model (trained on SportMOT, validated on VIP-HTD):** `sportsmot_best_model.pth`
- **Feature extraction (ReID) model (fine-tuned on VIP-HTD):** `vip-htd-features.pth`

---

## 4. Data Preparation

Convert your data into COCO format before running experiments:

`python tools/convert_vip_to_coco.py`


---

## 5. Running Validation and Test Scripts

- **Validation on VIP-HTD Dataset:**


`python buffer_validation_vip.py --expn <experiment_name>`


- **Testing on VIP-HTD Dataset:**


`python buffer_validation_vip.py --expn <experiment_name> --test`


Replace `<experiment_name>` with your desired experiment identifier.

---

## 6. Notes

- Ensure all model weights are correctly placed in the `pretrained` directory before running scripts.
- Data **must** be converted to COCO format prior to validation or testing.