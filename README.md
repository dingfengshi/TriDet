# [CVPR2023] TriDet: Temporal Action Detection with Relative Boundary Modeling

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tridet-temporal-action-detection-with/temporal-action-localization-on-hacs)](https://paperswithcode.com/sota/temporal-action-localization-on-hacs?p=tridet-temporal-action-detection-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tridet-temporal-action-detection-with/temporal-action-localization-on-thumos14)](https://paperswithcode.com/sota/temporal-action-localization-on-thumos14?p=tridet-temporal-action-detection-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tridet-temporal-action-detection-with/temporal-action-localization-on-epic-kitchens)](https://paperswithcode.com/sota/temporal-action-localization-on-epic-kitchens?p=tridet-temporal-action-detection-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tridet-temporal-action-detection-with/temporal-action-localization-on-activitynet)](https://paperswithcode.com/sota/temporal-action-localization-on-activitynet?p=tridet-temporal-action-detection-with)

![Image Title](framework.jpg)

## Overview

This repository contains the code for _TriDet: Temporal Action Detection with Relative Boundary
Modeling_ [paper](https://arxiv.org/abs/2303.07347), which has been accepted for CVPR2023. Our code is built upon the
codebase from [ActionFormer](https://github.com/happyharrycn/actionformer_release)
and [Detectron2](https://github.com/facebookresearch/detectron2), and we would like to express our
gratitude for their outstanding work.

To quickly get start with the model architecture, you can focus mainly on the following files:

- `libs/modeling/blocks.py`
- `libs/modeling/backbones.py`
- `libs/modeling/meta_archs.py`

## Update Log
- 2023.9.14 An extended version is updated to [Arxiv](https://arxiv.org/abs/2309.05590) 
- 2023.3.16 We release the code and the
  pretrained [checkpoints](https://drive.google.com/drive/folders/1eVROG6z-vHtm4AnXsh4N8ruUKkAidLqZ?usp=sharing).
- 2023.3.14 The pre-print version of our [paper](https://arxiv.org/abs/2303.07347) is updated to Arxiv.
- 2023.2.28 Our paper has been accepted for CVPR2023.

## Installation

1. Please ensure that you have installed PyTorch and CUDA. **(This code requires PyTorch version >= 1.11. We use
   version=1.11.0 in our experiments)**

* We conduct all our experiments on a single A100 GPU and the training results may vary depending on the type of GPU used.

2. Install the required packages by running the following command:

```shell
pip install  -r requirements.txt
```

3. Install NMS

```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

4. Done! We are ready to get start!

## Data Preparation

- We adpot the feature for **THUMOS14**, **ActivityNet** and **Epic-Kitchen** datasets
  from ActionFormer repository ([see here](https://github.com/happyharrycn/actionformer_release)).
  To use these features, please download them from their link and unpack them into the `./data` folder.

- For the **HACS** dataset, we use the [official I3D feature](http://hacs.csail.mit.edu/hacs_segments_features.zip) of
  the RGB stream and the [SlowFast feautre](https://github.com/qinzhi-0110/Temporal-Context-Aggregation-Network-Pytorch)
  from TCANet in our experiments.
  Please unpack the I3D feature into `./data/hacs/i3d_feature` and the SlowFast feature
  into `./data/hacs/slowfast_feature`. We provide processed
  annotation json files for the I3D feature and the SlowFast feature in the `./data/hacs/annotations` folder.
- The folder structure for `./data/hacs/i3d_feature` should be as follows:
  ```
  ./data/hacs/i3d_feature
  |
  |───training/
  │    └───xxx.npy
  │    └───...
  └───validation/
  │    └───xxx.npy
  │    └───...
  └───testing/
  │    └───xxx.npy
  │    └───...
  ```
- The folder structure for `./data/hacs/slowfast_feature` should be as follows:
  ```
  ./data/hacs/slowfast_feature
  |
  |───training/
  │    └───xxx.pkl
  │    └───...
  └───validation/
  │    └───xxx.pkl
  │    └───...

  ```

## Quick Start

We provide a list of scripts that allow you to reproduce our results with just one click. These scripts are located in
the `./tools` folder and include:

- thumos_i3d_script.sh
- epic_noun_slowfast_script.sh
- epic_verb_slowfast_script.sh
- hacs_slowfast_script.sh
- ant_tsp_script.sh

To easily reproduce our results, simply run the following command:

```shell
bash SCRIPT_PATH GPU_NUM
```

For example, if you want to train and eval our model on THUMOS14 dataset using the first GPU on you machine, you can
run:

```shell
bash tools/thumos_i3d_script.sh 0
```

The mean average precision (mAP) results for each dataset are:

| Dataset  | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | Avg   |
|----------|-------|-------|-------|-------|-------|-------|
| THUMOS14 | 83.62 | 80.07 | 72.94 | 62.35 | 47.35 | 69.27 |

| Dataset           | 0.1   | 0.2   | 0.3   | 0.4   | 0.5   | Avg   |
|-------------------|-------|-------|-------|-------|-------|-------|
| EPIC-KITCHEN-noun | 27.38 | 26.28 | 24.60 | 22.23 | 18.28 | 23.76 |

| Dataset           | 0.1   | 0.2   | 0.3   | 0.4   | 0.5   | Avg   |
|-------------------|-------|-------|-------|-------|-------|-------|
| EPIC-KITCHEN-verb | 28.72 | 27.57 | 26.19 | 24.26 | 20.83 | 25.51 |

| Dataset | 0.5   | 0.75  | 0.95  | Avg   |
|---------|-------|-------|-------|-------|
| HACS    | 56.90 | 39.33 | 11.24 | 38.69 |

| Dataset     | 0.5   | 0.75  | 0.95 | Avg   |
|-------------|-------|-------|------|-------|
| ActivityNet | 54.71 | 38.01 | 8.35 | 36.77 |

*There has been a slight improvement in the results of some datasets compared to those reported in the paper.

## Test

We offer pre-trained models for each dataset, which you can download the chechpoints
from [Google Drive](https://drive.google.com/drive/folders/1eVROG6z-vHtm4AnXsh4N8ruUKkAidLqZ?usp=sharing). The command
for test is

```shell
python eval.py ./configs/CONFIG_FILE PATH_TO_CHECKPOINT
```

## Contact

If you have any questions about the code, feel free to contact shidingfeng at buaa dot edu dot cn.

## References

If you find this work helpful, please consider citing our paper

```
@inproceedings{shi2023tridet,
  title={TriDet: Temporal Action Detection with Relative Boundary Modeling},
  author={Shi, Dingfeng and Zhong, Yujie and Cao, Qiong and Ma, Lin and Li, Jia and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18857--18866},
  year={2023}
}
```
 
