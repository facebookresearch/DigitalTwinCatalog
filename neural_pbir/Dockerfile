FROM nvcr.io/nvidia/pytorch:21.06-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y build-essential
RUN apt install -y libopencv-dev libopenmpi-dev
RUN apt install -y blender
RUN apt install -y locales
RUN locale-gen en_US.UTF-8

COPY . src

RUN pip install --upgrade ninja fvcore
RUN pip install --upgrade imageio imageio-ffmpeg scikit-image lpips \
                           mmcv==1.7.0 yapf==0.40.1 \
                           trimesh PyMCubes gpytoolbox

RUN pip install psdr-jit irtk largesteps

# Pytorch3D
RUN FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

RUN pip install src/neural_surface_recon/lib/cuda/

## TCNN_CUDA_ARCHITECTURES: CUDA architectures, required by tiny-cuda-nn
## H100 = 90
## 40X0 = 89
## 30X0 = 86
## A100 = 80
## 20X0 = 75
## TITAN V / V100 = 70
## 10X0 / TITAN Xp = 61
## 9X0 = 52
## K80 = 37
RUN TCNN_CUDA_ARCHITECTURES=70 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
