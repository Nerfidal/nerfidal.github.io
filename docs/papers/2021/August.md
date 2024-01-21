---
layout: default
title: August
parent: 2021
nav_order: 8
---
<!---metadata--->

## Self-Calibrating Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-08-31 | Yoonwoo Jeong, Seokjun Ahn, Christopher Choy, Animashree Anandkumar, Minsu Cho, Jaesik Park | cs.CV | [PDF](http://arxiv.org/pdf/2108.13826v2){: .btn .btn-green } |

**Abstract**: In this work, we propose a camera self-calibration algorithm for generic
cameras with arbitrary non-linear distortions. We jointly learn the geometry of
the scene and the accurate camera parameters without any calibration objects.
Our camera model consists of a pinhole model, a fourth order radial distortion,
and a generic noise model that can learn arbitrary non-linear camera
distortions. While traditional self-calibration algorithms mostly rely on
geometric constraints, we additionally incorporate photometric consistency.
This requires learning the geometry of the scene, and we use Neural Radiance
Fields (NeRF). We also propose a new geometric loss function, viz., projected
ray distance loss, to incorporate geometric consistency for complex non-linear
camera models. We validate our approach on standard real image datasets and
demonstrate that our model can learn the camera intrinsics and extrinsics
(pose) from scratch without COLMAP initialization. Also, we show that learning
accurate camera models in a differentiable manner allows us to improve PSNR
over baselines. Our module is an easy-to-use plugin that can be applied to NeRF
variants to improve performance. The code and data are currently available at
https://github.com/POSTECH-CVLab/SCNeRF.

Comments:
- Accepted in ICCV21, Project Page:
  https://postech-cvlab.github.io/SCNeRF/

---

## iButter: Neural Interactive Bullet Time Generator for Human  Free-viewpoint Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-08-12 | Liao Wang, Ziyu Wang, Pei Lin, Yuheng Jiang, Xin Suo, Minye Wu, Lan Xu, Jingyi Yu | cs.CV | [PDF](http://arxiv.org/pdf/2108.05577v1){: .btn .btn-green } |

**Abstract**: Generating ``bullet-time'' effects of human free-viewpoint videos is critical
for immersive visual effects and VR/AR experience. Recent neural advances still
lack the controllable and interactive bullet-time design ability for human
free-viewpoint rendering, especially under the real-time, dynamic and general
setting for our trajectory-aware task. To fill this gap, in this paper we
propose a neural interactive bullet-time generator (iButter) for
photo-realistic human free-viewpoint rendering from dense RGB streams, which
enables flexible and interactive design for human bullet-time visual effects.
Our iButter approach consists of a real-time preview and design stage as well
as a trajectory-aware refinement stage. During preview, we propose an
interactive bullet-time design approach by extending the NeRF rendering to a
real-time and dynamic setting and getting rid of the tedious per-scene
training. To this end, our bullet-time design stage utilizes a hybrid training
set, light-weight network design and an efficient silhouette-based sampling
strategy. During refinement, we introduce an efficient trajectory-aware scheme
within 20 minutes, which jointly encodes the spatial, temporal consistency and
semantic cues along the designed trajectory, achieving photo-realistic
bullet-time viewing experience of human activities. Extensive experiments
demonstrate the effectiveness of our approach for convenient interactive
bullet-time design and photo-realistic human free-viewpoint video generation.

Comments:
- Accepted by ACM MM 2021

---

## FLAME-in-NeRF : Neural control of Radiance Fields for Free View Face  Animation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-08-10 | ShahRukh Athar, Zhixin Shu, Dimitris Samaras | cs.CV | [PDF](http://arxiv.org/pdf/2108.04913v1){: .btn .btn-green } |

**Abstract**: This paper presents a neural rendering method for controllable portrait video
synthesis. Recent advances in volumetric neural rendering, such as neural
radiance fields (NeRF), has enabled the photorealistic novel view synthesis of
static scenes with impressive results. However, modeling dynamic and
controllable objects as part of a scene with such scene representations is
still challenging. In this work, we design a system that enables both novel
view synthesis for portrait video, including the human subject and the scene
background, and explicit control of the facial expressions through a
low-dimensional expression representation. We leverage the expression space of
a 3D morphable face model (3DMM) to represent the distribution of human facial
expressions, and use it to condition the NeRF volumetric function. Furthermore,
we impose a spatial prior brought by 3DMM fitting to guide the network to learn
disentangled control for scene appearance and facial actions. We demonstrate
the effectiveness of our method on free view synthesis of portrait videos with
expression controls. To train a scene, our method only requires a short video
of a subject captured by a mobile device.

Comments:
- version 1.0.0

---

## Differentiable Surface Rendering via Non-Differentiable Sampling

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-08-10 | Forrester Cole, Kyle Genova, Avneesh Sud, Daniel Vlasic, Zhoutong Zhang | cs.GR | [PDF](http://arxiv.org/pdf/2108.04886v1){: .btn .btn-green } |

**Abstract**: We present a method for differentiable rendering of 3D surfaces that supports
both explicit and implicit representations, provides derivatives at occlusion
boundaries, and is fast and simple to implement. The method first samples the
surface using non-differentiable rasterization, then applies differentiable,
depth-aware point splatting to produce the final image. Our approach requires
no differentiable meshing or rasterization steps, making it efficient for large
3D models and applicable to isosurfaces extracted from implicit surface
definitions. We demonstrate the effectiveness of our method for implicit-,
mesh-, and parametric-surface-based inverse rendering and neural-network
training applications. In particular, we show for the first time efficient,
differentiable rendering of an isosurface extracted from a neural radiance
field (NeRF), and demonstrate surface-based, rather than volume-based,
rendering of a NeRF.

Comments:
- Accepted to ICCV 2021

---

## NeuralMVS: Bridging Multi-View Stereo and Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-08-09 | Radu Alexandru Rosu, Sven Behnke | cs.CV | [PDF](http://arxiv.org/pdf/2108.03880v2){: .btn .btn-green } |

**Abstract**: Multi-View Stereo (MVS) is a core task in 3D computer vision. With the surge
of novel deep learning methods, learned MVS has surpassed the accuracy of
classical approaches, but still relies on building a memory intensive dense
cost volume. Novel View Synthesis (NVS) is a parallel line of research and has
recently seen an increase in popularity with Neural Radiance Field (NeRF)
models, which optimize a per scene radiance field. However, NeRF methods do not
generalize to novel scenes and are slow to train and test. We propose to bridge
the gap between these two methodologies with a novel network that can recover
3D scene geometry as a distance function, together with high-resolution color
images. Our method uses only a sparse set of images as input and can generalize
well to novel scenes. Additionally, we propose a coarse-to-fine sphere tracing
approach in order to significantly increase speed. We show on various datasets
that our method reaches comparable accuracy to per-scene optimized methods
while being able to generalize and running significantly faster. We provide the
source code at https://github.com/AIS-Bonn/neural_mvs

Comments:
- Accepted for International Joint Conference on Neural Networks
  (IJCNN) 2022. Code available at https://github.com/AIS-Bonn/neural_mvs