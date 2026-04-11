# Projected Coupled Diffusion (PCD)

[![Paper](https://img.shields.io/badge/arXiv-2508.10531-B31B1B)](https://arxiv.org/abs/2508.10531)
[![Paper](https://img.shields.io/badge/OpenReview-1FEm5JLpvg-FFA500)](https://openreview.net/forum?id=1FEm5JLpvg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Hao Luan](https://edmundluan.github.io)<sup>1</sup>, 
Yi Xian Goh<sup>2</sup>, 
[See-Kiong Ng](https://comp.nus.edu.sg/~ngsk)<sup>1,3</sup>,
[Chun Kai Ling](https://lingchunkai.github.io)<sup>1</sup>

<sup>1</sup>Department of Computer Science, School
of Computing, National University of Singapore;  
<sup>2</sup>Department of Artificial Intelligence, Faculty of Computer Science \& Information Technology, Universiti Malaya;  
<sup>3</sup>Institute of Data Science, National University of Singapore. 


This repository contains the implementation of *Projected Coupled Diffusion* (PCD). 


## 📢 Introduction
We propose *Projected Coupled Diffusion (PCD)*, a novel test-time framework for constrained joint generation. PCD introduces a coupled guidance term into the generative dynamics to encourage coordination between diffusion models and incorporates a projection step at each diffusion step to enforce hard constraints. 
Empirically, we demonstrate the effectiveness of PCD in application scenarios of image-pair generation, object manipulation, and multi-robot motion planning. 


## 🔨 Usage
Please see specific instructions in the folders [Toy-Example](./toy_example/README.md), [Multi-Robot](./multi_robot/README.md), [PushT](./pusht/README.md), and [Image-Pair](./image_pair/README.md), respectively. 


<!--
## 📄 Appendix
This repository also contains the extended version of our paper with [appendix](./appendix/pcd_appendix.pdf). 
-->


## 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.


## 🙏 Acknowledgement
- Our implementation of multi-robot experiments is based on [MMD](https://github.com/yoraish/mmd). 
- Our PushT implementation is based on [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and [LTLDoG](https://ieeexplore.ieee.org/document/10637680). 
- Our image generation experiments use [Diffusers](https://github.com/huggingface/diffusers), pretrained model weights of [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), and the [FFHQ-Aging Dataset](https://github.com/royorel/FFHQ-Aging-Dataset). 


## ✏️ Citation 
If you find this repo or the ideas presented in our paper useful for your research, please consider citing our paper.
```
@inproceedings{luan2026projected,
    title={Projected Coupled Diffusion for Test-Time Constrained Joint Generation},
    author={Hao Luan and Yi Xian Goh and See-Kiong Ng and Chun Kai Ling},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=1FEm5JLpvg}
}
```
