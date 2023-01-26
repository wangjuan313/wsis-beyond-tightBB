# Weakly Supervised Image Segmentation Beyond Tight Bounding Box Annotations

This repository hosts the codes for the implementation of the paper **Weakly Supervised Image Segmentation Beyond Tight Bounding Box Annotations** (under review).

# Dataset preprocessing

Download [Promise12](https://promise12.grand-challenge.org/) dataset, and put it on the "data/prostate" folder.

Download [Atlas](http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html) dataset, and put it on the "data/atlas" folder.

Run the following codes for preprocessing:

```bash
# trainig and valid subsets for promise12 dataset
python preprocess/slice_promise_train_val.py
python preprocess/slice_promise_augment_train_val.py

# trainig and valid subsets for atlas dataset
python preprocess/slice_atlas.py
```

# Training

```bash
# The following experiments include MIL baseline (exp_no=0), 
# parallel transformation based MIL (exp_no=1)
# polar transformation based MIL (exp_no=2,3,4,5), 
# the proposed approach with weighted alpha-softmax approaximation (exp_no=6,8),
# the proposed approach with weighted alpha-quasimax approaximation (exp_no=7,9),
CUDA_VISIBLE_DEVICES=0 python tools/train_atlas_beyond_tightbb.py --n_exp exp_no
CUDA_VISIBLE_DEVICES=0 python tools/train_promise_beyond_tightbb.py --n_exp exp_no
```

```bash
# Dice validation results for promise12 dataset, exp_no=0,1,2,3,4,5
CUDA_VISIBLE_DEVICES=0 python tools/valid_promise_beyond_tightbb.py --n_exp exp_no
# Dice validation results for atlas dataset, exp_no=0,1,2,3,4,5
CUDA_VISIBLE_DEVICES=0 python tools/valid_atlasbeyond_tightbbx.py --n_exp exp_no
```

# Performance summary

```bash
python tools/report_promise_beyond_tightbb.py
python tools/report_atlas_beyond_tightbb.py
```

# Center visualization

```bash
# exp_no = 1,2,3,4
python tools/plot_promise_polar_center.py --n_exp exp_no
python tools/plot_atlas_polar_center.py --n_exp exp_no
```

## Citations

Please consider citing our paper in your publications if the project helps your research.

```
@inproceedings{wang2021bounding,
  title={Bounding Box Tightness Prior for Weakly Supervised Image Segmentation},
  author={Wang, Juan and Xia, Bin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={526--536},
  year={2021},
  organization={Springer}
}
@article{wang2022polar,
  title={Polar Transformation Based Multiple Instance Learning Assisting Weakly Supervised Image Segmentation With Loose Bounding Box Annotations},
  author={Wang, Juan and Xia, Bin},
  journal={arXiv preprint arXiv:2203.06000},
  year={2022}
}
```
