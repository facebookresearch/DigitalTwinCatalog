# Large Reconstruction Models Implementation from Surreal DTC
This is the implementation of [Large Reconstruction Model](https://yiconghong.me/LRM/) (LRM) from Surreal DTC for feed-forward sparse-view reconstruction. We provide 2 variations of LRM, LRM-NeRF and LRM-VolSDF, which reconstruct 3D objects in the form of neural radiance fields and sign distance fields respectively. Please refer to our [CVPR paper](https://arxiv.org/abs/2504.08541) and [website](https://www.projectaria.com/datasets/dtc/) for more implementation details and example reconstruction results.

**Note**: The current release does not include model weights, which we are working on. Please stay tuned.

## Installation
We recommend running our code on a Linux machine with a full Anaconda environment ([link](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh)). Windows WSL is also likely to work when testing locally. The installation steps are as follows:
1. Create a new conda environment. We tested python 3.12.
   ```
   conda create -n lrm_dtc python=3.12
   ```
2. Install [PyTorch](https://pytorch.org/). We tested Pytorch 2.7.1 with cuda 12.8. CUDA version higher than 11.8 is likely to work.
3. Install other dependencies.
   ```
   pip install -r requirements.txt
   ```

## Local Training and Testing
We provide a simple example of training and inference scripts to help user testing our code locally. The scripts have been verified on A6000 and H200 GPUs. The training process requires close to 25GB GPU memory. Reducing `batch_size_per_gpu` in training scripts can help reduce the memory usage but will also make training less table.

To test LRM-NeRF
```
bash scripts/train_nerf_local.sh
bash scripts/test_nerf_local.sh
```

To test LRM-VolSDF
```
bash scripts/train_sdf_local.sh
bash scripts/test_sdf_local.sh
```

Intermediate training and inference results will be saved in `experiments` folder.

## Large-scale Training
We also prepare example SLURM scripts for launching a training job on a cluster. Current scripts only support overfitting on the one example included in `data` folder with only 1 node. Full training scripts will be released together with model weights.

For LRM-NeRF
```
sbatch scripts/train_nerf.sh
```

For LRM-VolSDF
```
sbatch scripts/train_sdf.sh
```
## Citations
If you use this implementaiton, please cite the following 2 papers:
```
@inproceedings{dong2025dtc,
title={Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset},
author={Dong, Zhao and Chen, Ka and Lv, Zhaoyang and Yu, Hong-Xing and Zhang, Yunzhi and Zhang, Cheng and Zhu, Yufeng and Tian, Stephen and Li, Zhengqin and Moffatt, Geordie and Christofferson, Sean and Fort, James and Pan, Xiaqing and Yan, Mingfei and Wu, Jiajun and Ren, Carl Yuheng and Newcombe, Richard},
journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month={June},
year={2025},
}
```
and
```
@inproceedings{li2025lirm,
title={LIRM: Large Inverse Rendering Model for Progressive Reconstruction of Shape, Materials and View-dependent Radiance Fields},
author={Zhengqin Li and Dilin Wang and Ka Chen and Zhaoyang Lv and Thu Nguyen-Phuoc and Milim Lee and Jia-bin Huang
  and Lei Xiao and Cheng Zhang and Yufeng Zhu and Carl S. Marshall and Yufeng Ren and Richard Newcombe and Zhao Dong},
journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2025},
}
```
