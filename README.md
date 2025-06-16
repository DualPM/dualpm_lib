# Dual Point Maps
This is a rendering utility for rasterizing a dual mesh to a dual point map.

A dual mesh is a mesh with canonically posed vertices in a canonical coordinate system and simultaneously posed vertices in world coordinates.

Usage:
```
import dualpm
import torch

with torch.inference_mode():
  dual_point_map = dualpm.render_dual_point_map(
    <canonical vertices / torch.Tensor>,
    <posed vertices / torch.Tensor>,
    <faces / torch.Tensor>,
    <view matrix / torch.Tensor>,
    <projection matrix / torch.Tensor>,
    **hparams 
  )
```
for necessary hyperparameters see `dualpm.py`


## Citation
If you used our rendering library please cite us!

**Paper:** [arXiv:2412.04464 [cs.CV]](https://arxiv.org/abs/2412.04464)  
**Project page:** https://dualpm.github.io/

bibtex
```
@InProceedings{Kaye2025,
  author    = {Ben Kaye and Tomas Jakab and Shangzhe Wu and Christian Rupprecht and Andrea Vedaldi},
  title     = {DualPM: Dual Posed‑Canonical Point Maps for 3D Shape and Pose Reconstruction},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {6425--6435}
}
```

## Installation
```
pip install [-e] .
```