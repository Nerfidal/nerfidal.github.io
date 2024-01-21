---
layout: default
title: July
parent: 2021
nav_order: 7
---
<!---metadata--->

## A Deep Signed Directional Distance Function for Object Shape  Representation



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-07-23 | Ehsan Zobeidi, Nikolay Atanasov | cs.CV | [PDF](http://arxiv.org/pdf/2107.11024v2){: .btn .btn-green } |

**Abstract**: Neural networks that map 3D coordinates to signed distance function (SDF) or
occupancy values have enabled high-fidelity implicit representations of object
shape. This paper develops a new shape model that allows synthesizing novel
distance views by optimizing a continuous signed directional distance function
(SDDF). Similar to deep SDF models, our SDDF formulation can represent whole
categories of shapes and complete or interpolate across shapes from partial
input data. Unlike an SDF, which measures distance to the nearest surface in
any direction, an SDDF measures distance in a given direction. This allows
training an SDDF model without 3D shape supervision, using only distance
measurements, readily available from depth camera or Lidar sensors. Our model
also removes post-processing steps like surface extraction or rendering by
directly predicting distance at arbitrary locations and viewing directions.
Unlike deep view-synthesis techniques, such as Neural Radiance Fields, which
train high-capacity black-box models, our model encodes by construction the
property that SDDF values decrease linearly along the viewing direction. This
structure constraint not only results in dimensionality reduction but also
provides analytical confidence about the accuracy of SDDF predictions,
regardless of the distance to the object surface.

---

## 3D Neural Scene Representations for Visuomotor Control

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-07-08 | Yunzhu Li, Shuang Li, Vincent Sitzmann, Pulkit Agrawal, Antonio Torralba | cs.RO | [PDF](http://arxiv.org/pdf/2107.04004v2){: .btn .btn-green } |

**Abstract**: Humans have a strong intuitive understanding of the 3D environment around us.
The mental model of the physics in our brain applies to objects of different
materials and enables us to perform a wide range of manipulation tasks that are
far beyond the reach of current robots. In this work, we desire to learn models
for dynamic 3D scenes purely from 2D visual observations. Our model combines
Neural Radiance Fields (NeRF) and time contrastive learning with an
autoencoding framework, which learns viewpoint-invariant 3D-aware scene
representations. We show that a dynamics model, constructed over the learned
representation space, enables visuomotor control for challenging manipulation
tasks involving both rigid bodies and fluids, where the target is specified in
a viewpoint different from what the robot operates on. When coupled with an
auto-decoding framework, it can even support goal specification from camera
viewpoints that are outside the training distribution. We further demonstrate
the richness of the learned 3D dynamics model by performing future prediction
and novel view synthesis. Finally, we provide detailed ablation studies
regarding different system designs and qualitative analysis of the learned
representations.

Comments:
- Accepted to Conference on Robot Learning (CoRL 2021) as Oral
  Presentation. The first two authors contributed equally. Project Page:
  https://3d-representation-learning.github.io/nerf-dy/

---

## Depth-supervised NeRF: Fewer Views and Faster Training for Free

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-07-06 | Kangle Deng, Andrew Liu, Jun-Yan Zhu, Deva Ramanan | cs.CV | [PDF](http://arxiv.org/pdf/2107.02791v2){: .btn .btn-green } |

**Abstract**: A commonly observed failure mode of Neural Radiance Field (NeRF) is fitting
incorrect geometries when given an insufficient number of input views. One
potential reason is that standard volumetric rendering does not enforce the
constraint that most of a scene's geometry consist of empty space and opaque
surfaces. We formalize the above assumption through DS-NeRF (Depth-supervised
Neural Radiance Fields), a loss for learning radiance fields that takes
advantage of readily-available depth supervision. We leverage the fact that
current NeRF pipelines require images with known camera poses that are
typically estimated by running structure-from-motion (SFM). Crucially, SFM also
produces sparse 3D points that can be used as "free" depth supervision during
training: we add a loss to encourage the distribution of a ray's terminating
depth matches a given 3D keypoint, incorporating depth uncertainty. DS-NeRF can
render better images given fewer training views while training 2-3x faster.
Further, we show that our loss is compatible with other recently proposed NeRF
methods, demonstrating that depth is a cheap and easily digestible supervisory
signal. And finally, we find that DS-NeRF can support other types of depth
supervision such as scanned depth sensors and RGB-D reconstruction outputs.

Comments:
- Project page: http://www.cs.cmu.edu/~dsnerf/ GitHub:
  https://github.com/dunbar12138/DSNeRF