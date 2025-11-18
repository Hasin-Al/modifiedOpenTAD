# Modified OpenTAD for ADATAD

This repository contains modifications to **OpenTAD** to make it compatible with **Python 3.12**, **CUDA 12.6**, and **PyTorch 2.8.0**. The original OpenTAD codebase is not compatible with this environment due to changes in `setuptools` and some library APIs. The modifications allow ADATAD to run smoothly on **Google Colab** and similar environments.

---

## ⚡ Changes Made

1. **Setuptools Compatibility:** Downgraded `setuptools` to `69.5.1` to avoid installation/build issues.
2. **MMCV and MMACTION2:** Installed `mmcv-lite==2.0.0` and `mmaction2==1.1.0` with `mim` for PyTorch 2.8.0 + CUDA 12.6 compatibility.
3. **Custom NMS and ROI modules:** Added `pyproject.toml` files in the following directories for proper installation:
   - `opentad/models/utils/post_processing/nms`
   - `opentad/models/roi_heads/roi_extractors/align1d`
   - `opentad/models/roi_heads/roi_extractors/boundary_pooling`
4. **CUDA Kernels:** Replaced `type()` with `scalar_type()` in `Align1D_cuda_kernal.cu` at lines 230 and 272.
5. **Numpy Compatibility:** Downgraded `numpy` to `<2.0` to avoid breaking changes.

---

## 📝 Setup Instructions

### 1. Clone the repository

```bash
git clone <your-forked-repo-url>
cd OpenTAD
```

### 2. Prepare Data and Upload Pretrained Weight 

- Add your videos in:  
  `OpenTAD/data/thumos-14/raw_data/videos/`
- Add your THUMOS annotations in:  
  `OpenTAD/data/thumos-14/annotations/`
- Or Chnage your dataset paths @:
  `OpenTAD/configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py [line 1,2,3]`
- Add your Pretrained weight in:
  `OpneTAD/pretrained/`
- Add your videomae weight in:
  `openTAD/pretrained/videomae/`
-Or Chnage your pretrained path @:
  `OpenTAD/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py [line 93]`

  `
  

---

### 3. Install Dependencies

```bash
# Install OpenMIM
!pip install openmim
```
```bash
# Downgrade setuptools
!pip install "setuptools==69.5.1" --force-reinstall
```
```bash
# Install MMCV and MMACTION2
!pip install mmcv-lite==2.0.0
!mim install mmaction2==1.1.0
```
```bash
!pip install mmengine wandb scipy einops pandas tqdm ninja imgaug pytorchvideo
!pip install numpy gdown
```

---

### 4. Install Custom Modules

```bash
# NMS module
cd opentad/models/utils/post_processing/nms
pip install .

# align1d ROI extractor
cd ../../../../roi_heads/roi_extractors/align1d
pip install --no-build-isolation .

# boundary_pooling ROI extractor
cd ../boundary_pooling
pip install --no-build-isolation .
```

---

### 5. Numpy Compatibility

```bash
pip install "numpy<2.0" --force-reinstall
```

---

### 6. Adjust Configurations

- **RPN Head Classes:**  
  Open `OpenTAD/configs/_base_/models/actionformer.py`  
  Update `num_classes` in RPN head block (line 23) to `20` for THUMOS data.

- **Scheduler & Workflow for Adapters:**  
  Open `OpenTAD/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`  
  - Adjust `epochs` in scheduler block (line 136)  
  - Adjust workflow block (line 150)

> You can select any architecture and modify accordingly.

---

### 7. Training Command

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py
```

> This will run VideoMAE-Small with adapters.

---

## 📖 References

- Colab Notebook for setup and running:  
  [My colab notebook](https://colab.research.google.com/drive/1gp0wXWoyV0E3jACq47wzwwHvkae2Z3h6?usp=sharing)

---

## 🔧 Notes

- These modifications are specifically for **Python 3.12, CUDA 12.6, and PyTorch 2.8.0**.  
- Original OpenTAD may not work with these versions due to removed or changed libraries.
- The modified setup has been tested on Google Colab.

---

## 🎯 Summary

With these changes, you can run ADATAD on the latest Python and CUDA versions without encountering library or build issues, and train on THUMOS-14 data using VideoM
