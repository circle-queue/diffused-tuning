# diffused-tuning
A gui for applying stable-diffusion

Currently supports generating- and inpaiting images

Example: (45s)
![](https://github.com/circle-queue/diffused-tuning/blob/main/example.gif?raw=true)

## Requirements
### System
An NVIDIA GPU with 8GB+ of VRam (E.g. RTX 3060) & ~30GB? storage

### Software
A matching version of Cuda and Pytorch 2+, e.g. [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) and [Pytorch 2](https://pytorch.org/get-started/locally/)

For memory issues, see the notes for [stable-diffusion](https://huggingface.co/stabilityai/stable-diffusion-2)

### Usage
See the [stable-diffusion license](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE)

## Installation
Ensure above requirements.

(Optionally create a virtual environment)

```
git clone https://github.com/circle-queue/diffused-tuning
pip install -e diffused-tuning
```