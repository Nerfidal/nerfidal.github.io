---
layout: default
title: October
parent: 2021
nav_order: 10
---
<!---metadata--->

## Neural-PIL: Neural Pre-Integrated Lighting for Reflectance Decomposition

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-10-27 | Mark Boss, Varun Jampani, Raphael Braun, Ce Liu, Jonathan T. Barron, Hendrik P. A. Lensch | cs.CV | [PDF](http://arxiv.org/pdf/2110.14373v1){: .btn .btn-green } |

**Abstract**: Decomposing a scene into its shape, reflectance and illumination is a
fundamental problem in computer vision and graphics. Neural approaches such as
NeRF have achieved remarkable success in view synthesis, but do not explicitly
perform decomposition and instead operate exclusively on radiance (the product
of reflectance and illumination). Extensions to NeRF, such as NeRD, can perform
decomposition but struggle to accurately recover detailed illumination, thereby
significantly limiting realism. We propose a novel reflectance decomposition
network that can estimate shape, BRDF, and per-image illumination given a set
of object images captured under varying illumination. Our key technique is a
novel illumination integration network called Neural-PIL that replaces a costly
illumination integral operation in the rendering with a simple network query.
In addition, we also learn deep low-dimensional priors on BRDF and illumination
representations using novel smooth manifold auto-encoders. Our decompositions
can result in considerably better BRDF and light estimates enabling more
accurate novel view-synthesis and relighting compared to prior art. Project
page: https://markboss.me/publication/2021-neural-pil/

Comments:
- Project page: https://markboss.me/publication/2021-neural-pil/ Video:
  https://youtu.be/AsdAR5u3vQ8 - Accepted at NeurIPS 2021

---

## Dex-NeRF: Using a Neural Radiance Field to Grasp Transparent Objects

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-10-27 | Jeffrey Ichnowski, Yahav Avigal, Justin Kerr, Ken Goldberg | cs.RO | [PDF](http://arxiv.org/pdf/2110.14217v1){: .btn .btn-green } |

**Abstract**: The ability to grasp and manipulate transparent objects is a major challenge
for robots. Existing depth cameras have difficulty detecting, localizing, and
inferring the geometry of such objects. We propose using neural radiance fields
(NeRF) to detect, localize, and infer the geometry of transparent objects with
sufficient accuracy to find and grasp them securely. We leverage NeRF's
view-independent learned density, place lights to increase specular
reflections, and perform a transparency-aware depth-rendering that we feed into
the Dex-Net grasp planner. We show how additional lights create specular
reflections that improve the quality of the depth map, and test a setup for a
robot workcell equipped with an array of cameras to perform transparent object
manipulation. We also create synthetic and real datasets of transparent objects
in real-world settings, including singulated objects, cluttered tables, and the
top rack of a dishwasher. In each setting we show that NeRF and Dex-Net are
able to reliably compute robust grasps on transparent objects, achieving 90%
and 100% grasp success rates in physical experiments on an ABB YuMi, on objects
where baseline methods fail.

Comments:
- 11 pages, 9 figures, to be published in the Conference on Robot
  Learning (CoRL) 2021

---

## H-NeRF: Neural Radiance Fields for Rendering and Temporal Reconstruction  of Humans in Motion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-10-26 | Hongyi Xu, Thiemo Alldieck, Cristian Sminchisescu | cs.CV | [PDF](http://arxiv.org/pdf/2110.13746v2){: .btn .btn-green } |

**Abstract**: We present neural radiance fields for rendering and temporal (4D)
reconstruction of humans in motion (H-NeRF), as captured by a sparse set of
cameras or even from a monocular video. Our approach combines ideas from neural
scene representation, novel-view synthesis, and implicit statistical geometric
human representations, coupled using novel loss functions. Instead of learning
a radiance field with a uniform occupancy prior, we constrain it by a
structured implicit human body model, represented using signed distance
functions. This allows us to robustly fuse information from sparse views and
generalize well beyond the poses or views observed in training. Moreover, we
apply geometric constraints to co-learn the structure of the observed subject
-- including both body and clothing -- and to regularize the radiance field to
geometrically plausible solutions. Extensive experiments on multiple datasets
demonstrate the robustness and the accuracy of our approach, its generalization
capabilities significantly outside a small training set of poses and views, and
statistical extrapolation beyond the observed shape.

---

## Neural Relightable Participating Media Rendering



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-10-25 | Quan Zheng, Gurprit Singh, Hans-Peter Seidel | cs.CV | [PDF](http://arxiv.org/pdf/2110.12993v1){: .btn .btn-green } |

**Abstract**: Learning neural radiance fields of a scene has recently allowed realistic
novel view synthesis of the scene, but they are limited to synthesize images
under the original fixed lighting condition. Therefore, they are not flexible
for the eagerly desired tasks like relighting, scene editing and scene
composition. To tackle this problem, several recent methods propose to
disentangle reflectance and illumination from the radiance field. These methods
can cope with solid objects with opaque surfaces but participating media are
neglected. Also, they take into account only direct illumination or at most
one-bounce indirect illumination, thus suffer from energy loss due to ignoring
the high-order indirect illumination. We propose to learn neural
representations for participating media with a complete simulation of global
illumination. We estimate direct illumination via ray tracing and compute
indirect illumination with spherical harmonics. Our approach avoids computing
the lengthy indirect bounces and does not suffer from energy loss. Our
experiments on multiple scenes show that our approach achieves superior visual
quality and numerical performance compared to state-of-the-art methods, and it
can generalize to deal with solid objects with opaque surfaces as well.

Comments:
- Accepted to NeurIPS 2021

---

## CIPS-3D: A 3D-Aware Generator of GANs Based on Conditionally-Independent  Pixel Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-10-19 | Peng Zhou, Lingxi Xie, Bingbing Ni, Qi Tian | cs.CV | [PDF](http://arxiv.org/pdf/2110.09788v1){: .btn .btn-green } |

**Abstract**: The style-based GAN (StyleGAN) architecture achieved state-of-the-art results
for generating high-quality images, but it lacks explicit and precise control
over camera poses. The recently proposed NeRF-based GANs made great progress
towards 3D-aware generators, but they are unable to generate high-quality
images yet. This paper presents CIPS-3D, a style-based, 3D-aware generator that
is composed of a shallow NeRF network and a deep implicit neural representation
(INR) network. The generator synthesizes each pixel value independently without
any spatial convolution or upsampling operation. In addition, we diagnose the
problem of mirror symmetry that implies a suboptimal solution and solve it by
introducing an auxiliary discriminator. Trained on raw, single-view images,
CIPS-3D sets new records for 3D-aware image synthesis with an impressive FID of
6.97 for images at the $256\times256$ resolution on FFHQ. We also demonstrate
several interesting directions for CIPS-3D such as transfer learning and
3D-aware face stylization. The synthesis results are best viewed as videos, so
we recommend the readers to check our github project at
https://github.com/PeterouZh/CIPS-3D

Comments:
- 3D-aware GANs based on NeRF, https://github.com/PeterouZh/CIPS-3D

---

## StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image  Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-10-18 | Jiatao Gu, Lingjie Liu, Peng Wang, Christian Theobalt | cs.CV | [PDF](http://arxiv.org/pdf/2110.08985v1){: .btn .btn-green } |

**Abstract**: We propose StyleNeRF, a 3D-aware generative model for photo-realistic
high-resolution image synthesis with high multi-view consistency, which can be
trained on unstructured 2D images. Existing approaches either cannot synthesize
high-resolution images with fine details or yield noticeable 3D-inconsistent
artifacts. In addition, many of them lack control over style attributes and
explicit 3D camera poses. StyleNeRF integrates the neural radiance field (NeRF)
into a style-based generator to tackle the aforementioned challenges, i.e.,
improving rendering efficiency and 3D consistency for high-resolution image
generation. We perform volume rendering only to produce a low-resolution
feature map and progressively apply upsampling in 2D to address the first
issue. To mitigate the inconsistencies caused by 2D upsampling, we propose
multiple designs, including a better upsampler and a new regularization loss.
With these designs, StyleNeRF can synthesize high-resolution images at
interactive rates while preserving 3D consistency at high quality. StyleNeRF
also enables control of camera poses and different levels of styles, which can
generalize to unseen views. It also supports challenging tasks, including
zoom-in and-out, style mixing, inversion, and semantic editing.

Comments:
- 24 pages, 19 figures. Project page: http://jiataogu.me/style_nerf/

---

## NeRS: Neural Reflectance Surfaces for Sparse-view 3D Reconstruction in  the Wild

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-10-14 | Jason Y. Zhang, Gengshan Yang, Shubham Tulsiani, Deva Ramanan | cs.CV | [PDF](http://arxiv.org/pdf/2110.07604v3){: .btn .btn-green } |

**Abstract**: Recent history has seen a tremendous growth of work exploring implicit
representations of geometry and radiance, popularized through Neural Radiance
Fields (NeRF). Such works are fundamentally based on a (implicit) volumetric
representation of occupancy, allowing them to model diverse scene structure
including translucent objects and atmospheric obscurants. But because the vast
majority of real-world scenes are composed of well-defined surfaces, we
introduce a surface analog of such implicit models called Neural Reflectance
Surfaces (NeRS). NeRS learns a neural shape representation of a closed surface
that is diffeomorphic to a sphere, guaranteeing water-tight reconstructions.
Even more importantly, surface parameterizations allow NeRS to learn (neural)
bidirectional surface reflectance functions (BRDFs) that factorize
view-dependent appearance into environmental illumination, diffuse color
(albedo), and specular "shininess." Finally, rather than illustrating our
results on synthetic scenes or controlled in-the-lab capture, we assemble a
novel dataset of multi-view images from online marketplaces for selling goods.
Such "in-the-wild" multi-view image sets pose a number of challenges, including
a small number of views with unknown/rough camera estimates. We demonstrate
that surface-based neural reconstructions enable learning from such data,
outperforming volumetric neural rendering-based reconstructions. We hope that
NeRS serves as a first step toward building scalable, high-quality libraries of
real-world shape, materials, and illumination. The project page with code and
video visualizations can be found at https://jasonyzhang.com/ners.

Comments:
- In NeurIPS 2021. v2-3: Fixed minor typos

---

## LENS: Localization enhanced by NeRF synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-10-13 | Arthur Moreau, Nathan Piasco, Dzmitry Tsishkou, Bogdan Stanciulescu, Arnaud de La Fortelle | cs.CV | [PDF](http://arxiv.org/pdf/2110.06558v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have recently demonstrated photo-realistic
results for the task of novel view synthesis. In this paper, we propose to
apply novel view synthesis to the robot relocalization problem: we demonstrate
improvement of camera pose regression thanks to an additional synthetic dataset
rendered by the NeRF class of algorithm. To avoid spawning novel views in
irrelevant places we selected virtual camera locations from NeRF internal
representation of the 3D geometry of the scene. We further improved
localization accuracy of pose regressors using synthesized realistic and
geometry consistent images as data augmentation during training. At the time of
publication, our approach improved state of the art with a 60% lower error on
Cambridge Landmarks and 7-scenes datasets. Hence, the resulting accuracy
becomes comparable to structure-based methods, without any architecture
modification or domain adaptation constraints. Since our method allows almost
infinite generation of training data, we investigated limitations of camera
pose regression depending on size and distribution of data used for training on
public benchmarks. We concluded that pose regression accuracy is mostly bounded
by relatively small and biased datasets rather than capacity of the pose
regression model to solve the localization task.

Comments:
- Accepted at CoRL 2021

---

## Neural Radiance Fields Approach to Deep Multi-View Photometric Stereo



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-10-11 | Berk Kaya, Suryansh Kumar, Francesco Sarno, Vittorio Ferrari, Luc Van Gool | cs.CV | [PDF](http://arxiv.org/pdf/2110.05594v1){: .btn .btn-green } |

**Abstract**: We present a modern solution to the multi-view photometric stereo problem
(MVPS). Our work suitably exploits the image formation model in a MVPS
experimental setup to recover the dense 3D reconstruction of an object from
images. We procure the surface orientation using a photometric stereo (PS)
image formation model and blend it with a multi-view neural radiance field
representation to recover the object's surface geometry. Contrary to the
previous multi-staged framework to MVPS, where the position, iso-depth
contours, or orientation measurements are estimated independently and then
fused later, our method is simple to implement and realize. Our method performs
neural rendering of multi-view images while utilizing surface normals estimated
by a deep photometric stereo network. We render the MVPS images by considering
the object's surface normals for each 3D sample point along the viewing
direction rather than explicitly using the density gradient in the volume space
via 3D occupancy information. We optimize the proposed neural radiance field
representation for the MVPS setup efficiently using a fully connected deep
network to recover the 3D geometry of an object. Extensive evaluation on the
DiLiGenT-MV benchmark dataset shows that our method performs better than the
approaches that perform only PS or only multi-view stereo (MVS) and provides
comparable results against the state-of-the-art multi-stage fusion methods.

Comments:
- Accepted for publication at IEEE/CVF WACV 2022. 18 pages

---

## TyXe: Pyro-based Bayesian neural nets for Pytorch



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-10-01 | Hippolyt Ritter, Theofanis Karaletsos | stat.ML | [PDF](http://arxiv.org/pdf/2110.00276v1){: .btn .btn-green } |

**Abstract**: We introduce TyXe, a Bayesian neural network library built on top of Pytorch
and Pyro. Our leading design principle is to cleanly separate architecture,
prior, inference and likelihood specification, allowing for a flexible workflow
where users can quickly iterate over combinations of these components. In
contrast to existing packages TyXe does not implement any layer classes, and
instead relies on architectures defined in generic Pytorch code. TyXe then
provides modular choices for canonical priors, variational guides, inference
techniques, and layer selections for a Bayesian treatment of the specified
architecture. Sampling tricks for variance reduction, such as local
reparameterization or flipout, are implemented as effect handlers, which can be
applied independently of other specifications. We showcase the ease of use of
TyXe to explore Bayesian versions of popular models from various libraries: toy
regression with a pure Pytorch neural network; large-scale image classification
with torchvision ResNets; graph neural networks based on DGL; and Neural
Radiance Fields built on top of Pytorch3D. Finally, we provide convenient
abstractions for variational continual learning. In all cases the change from a
deterministic to a Bayesian neural network comes with minimal modifications to
existing code, offering a broad range of researchers and practitioners alike
practical access to uncertainty estimation techniques. The library is available
at https://github.com/TyXe-BDL/TyXe.

Comments:
- Previously presented at PROBPROG 2020

---

## Vision-Only Robot Navigation in a Neural Radiance World

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-10-01 | Michal Adamkiewicz, Timothy Chen, Adam Caccavale, Rachel Gardner, Preston Culbertson, Jeannette Bohg, Mac Schwager | cs.RO | [PDF](http://arxiv.org/pdf/2110.00168v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have recently emerged as a powerful paradigm
for the representation of natural, complex 3D scenes. NeRFs represent
continuous volumetric density and RGB values in a neural network, and generate
photo-realistic images from unseen camera viewpoints through ray tracing. We
propose an algorithm for navigating a robot through a 3D environment
represented as a NeRF using only an on-board RGB camera for localization. We
assume the NeRF for the scene has been pre-trained offline, and the robot's
objective is to navigate through unoccupied space in the NeRF to reach a goal
pose. We introduce a trajectory optimization algorithm that avoids collisions
with high-density regions in the NeRF based on a discrete time version of
differential flatness that is amenable to constraining the robot's full pose
and control inputs. We also introduce an optimization based filtering method to
estimate 6DoF pose and velocities for the robot in the NeRF given only an
onboard RGB camera. We combine the trajectory planner with the pose filter in
an online replanning loop to give a vision-based robot navigation pipeline. We
present simulation results with a quadrotor robot navigating through a jungle
gym environment, the inside of a church, and Stonehenge using only an RGB
camera. We also demonstrate an omnidirectional ground robot navigating through
the church, requiring it to reorient to fit through the narrow gap. Videos of
this work can be found at https://mikh3x4.github.io/nerf-navigation/ .