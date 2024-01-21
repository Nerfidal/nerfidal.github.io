---
layout: default
title: February
parent: 2021
nav_order: 2
---
<!---metadata--->

## NeRF--: Neural Radiance Fields Without Known Camera Parameters

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-02-14 | Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, Victor Adrian Prisacariu | cs.CV | [PDF](http://arxiv.org/pdf/2102.07064v4){: .btn .btn-green } |

**Abstract**: Considering the problem of novel view synthesis (NVS) from only a set of 2D
images, we simplify the training process of Neural Radiance Field (NeRF) on
forward-facing scenes by removing the requirement of known or pre-computed
camera parameters, including both intrinsics and 6DoF poses. To this end, we
propose NeRF$--$, with three contributions: First, we show that the camera
parameters can be jointly optimised as learnable parameters with NeRF training,
through a photometric reconstruction; Second, to benchmark the camera parameter
estimation and the quality of novel view renderings, we introduce a new dataset
of path-traced synthetic scenes, termed as Blender Forward-Facing Dataset
(BLEFF); Third, we conduct extensive analyses to understand the training
behaviours under various camera motions, and show that in most scenarios, the
joint optimisation pipeline can recover accurate camera parameters and achieve
comparable novel view synthesis quality as those trained with COLMAP
pre-computed camera parameters. Our code and data are available at
https://nerfmm.active.vision.

Comments:
- Project page see https://nerfmm.active.vision. Add a break point
  analysis experiment and release a BLEFF dataset

---

## A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape,  Appearance, and Pose

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-02-11 | Shih-Yang Su, Frank Yu, Michael Zollhoefer, Helge Rhodin | cs.CV | [PDF](http://arxiv.org/pdf/2102.06199v3){: .btn .btn-green } |

**Abstract**: While deep learning reshaped the classical motion capture pipeline with
feed-forward networks, generative models are required to recover fine alignment
via iterative refinement. Unfortunately, the existing models are usually
hand-crafted or learned in controlled conditions, only applicable to limited
domains. We propose a method to learn a generative neural body model from
unlabelled monocular videos by extending Neural Radiance Fields (NeRFs). We
equip them with a skeleton to apply to time-varying and articulated motion. A
key insight is that implicit models require the inverse of the forward
kinematics used in explicit surface models. Our reparameterization defines
spatial latent variables relative to the pose of body parts and thereby
overcomes ill-posed inverse operations with an overparameterization. This
enables learning volumetric body shape and appearance from scratch while
jointly refining the articulated pose; all without ground truth labels for
appearance, pose, or 3D shape on the input videos. When used for
novel-view-synthesis and motion capture, our neural model improves accuracy on
diverse datasets. Project website: https://lemonatsu.github.io/anerf/ .

Comments:
- NeurIPS 2021. Project website: https://lemonatsu.github.io/anerf/

---

## Conditions de Kan sur les nerfs des $ω$-catégories

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-02-08 | Félix Loubaton | math.CT | [PDF](http://arxiv.org/pdf/2102.04281v3){: .btn .btn-green } |

**Abstract**: We show that the Street nerve of a strict $\omega$-category $C$ is a Kan
complex (respectively a quasi-category) if and only if the $n$-cells of $C$ for
$n\geq 1$ (respectively $n> 1$) are weakly invertible. Moreover, we equip
$\mathcal{N}(C)$ with a structure of saturated complicial set where the
$n$-simplices correspond to morphisms from the $n^{th}$ oriental to $C$ sending
the unique non-trivial $n$-cell of the domain to a weakly invertible cell of
$C$.

Comments:
- 52 pages, in French