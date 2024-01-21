---
layout: default
title: February
parent: 2022
nav_order: 2
---
<!---metadata--->

## Pix2NeRF: Unsupervised Conditional $π$-GAN for Single Image to Neural  Radiance Fields Translation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-02-26 | Shengqu Cai, Anton Obukhov, Dengxin Dai, Luc Van Gool | cs.CV | [PDF](http://arxiv.org/pdf/2202.13162v1){: .btn .btn-green } |

**Abstract**: We propose a pipeline to generate Neural Radiance Fields~(NeRF) of an object
or a scene of a specific class, conditioned on a single input image. This is a
challenging task, as training NeRF requires multiple views of the same scene,
coupled with corresponding poses, which are hard to obtain. Our method is based
on $\pi$-GAN, a generative model for unconditional 3D-aware image synthesis,
which maps random latent codes to radiance fields of a class of objects. We
jointly optimize (1) the $\pi$-GAN objective to utilize its high-fidelity
3D-aware generation and (2) a carefully designed reconstruction objective. The
latter includes an encoder coupled with $\pi$-GAN generator to form an
auto-encoder. Unlike previous few-shot NeRF approaches, our pipeline is
unsupervised, capable of being trained with independent images without 3D,
multi-view, or pose supervision. Applications of our pipeline include 3d avatar
generation, object-centric novel view synthesis with a single input image, and
3d-aware super-resolution, to name a few.

Comments:
- 16 pages, 10 figures

---

## Learning Multi-Object Dynamics with Compositional Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-02-24 | Danny Driess, Zhiao Huang, Yunzhu Li, Russ Tedrake, Marc Toussaint | cs.CV | [PDF](http://arxiv.org/pdf/2202.11855v3){: .btn .btn-green } |

**Abstract**: We present a method to learn compositional multi-object dynamics models from
image observations based on implicit object encoders, Neural Radiance Fields
(NeRFs), and graph neural networks. NeRFs have become a popular choice for
representing scenes due to their strong 3D prior. However, most NeRF approaches
are trained on a single scene, representing the whole scene with a global
model, making generalization to novel scenes, containing different numbers of
objects, challenging. Instead, we present a compositional, object-centric
auto-encoder framework that maps multiple views of the scene to a set of latent
vectors representing each object separately. The latent vectors parameterize
individual NeRFs from which the scene can be reconstructed. Based on those
latent vectors, we train a graph neural network dynamics model in the latent
space to achieve compositionality for dynamics prediction. A key feature of our
approach is that the latent vectors are forced to encode 3D information through
the NeRF decoder, which enables us to incorporate structural priors in learning
the dynamics models, making long-term predictions more stable compared to
several baselines. Simulated and real world experiments show that our method
can model and learn the dynamics of compositional scenes including rigid and
deformable objects. Video: https://dannydriess.github.io/compnerfdyn/

Comments:
- v3: real robot exp

---

## Fourier PlenOctrees for Dynamic Radiance Field Rendering in Real-time

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-02-17 | Liao Wang, Jiakai Zhang, Xinhang Liu, Fuqiang Zhao, Yanshun Zhang, Yingliang Zhang, Minye Wu, Lan Xu, Jingyi Yu | cs.CV | [PDF](http://arxiv.org/pdf/2202.08614v2){: .btn .btn-green } |

**Abstract**: Implicit neural representations such as Neural Radiance Field (NeRF) have
focused mainly on modeling static objects captured under multi-view settings
where real-time rendering can be achieved with smart data structures, e.g.,
PlenOctree. In this paper, we present a novel Fourier PlenOctree (FPO)
technique to tackle efficient neural modeling and real-time rendering of
dynamic scenes captured under the free-view video (FVV) setting. The key idea
in our FPO is a novel combination of generalized NeRF, PlenOctree
representation, volumetric fusion and Fourier transform. To accelerate FPO
construction, we present a novel coarse-to-fine fusion scheme that leverages
the generalizable NeRF technique to generate the tree via spatial blending. To
tackle dynamic scenes, we tailor the implicit network to model the Fourier
coefficients of timevarying density and color attributes. Finally, we construct
the FPO and train the Fourier coefficients directly on the leaves of a union
PlenOctree structure of the dynamic sequence. We show that the resulting FPO
enables compact memory overload to handle dynamic objects and supports
efficient fine-tuning. Extensive experiments show that the proposed method is
3000 times faster than the original NeRF and achieves over an order of
magnitude acceleration over SOTA while preserving high visual quality for the
free-viewpoint rendering of unseen dynamic scenes.

Comments:
- Project page: https://aoliao12138.github.io/FPO/

---

## NeuVV: Neural Volumetric Videos with Immersive Rendering and Editing

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-02-12 | Jiakai Zhang, Liao Wang, Xinhang Liu, Fuqiang Zhao, Minzhang Li, Haizhao Dai, Boyuan Zhang, Wei Yang, Lan Xu, Jingyi Yu | cs.CV | [PDF](http://arxiv.org/pdf/2202.06088v1){: .btn .btn-green } |

**Abstract**: Some of the most exciting experiences that Metaverse promises to offer, for
instance, live interactions with virtual characters in virtual environments,
require real-time photo-realistic rendering. 3D reconstruction approaches to
rendering, active or passive, still require extensive cleanup work to fix the
meshes or point clouds. In this paper, we present a neural volumography
technique called neural volumetric video or NeuVV to support immersive,
interactive, and spatial-temporal rendering of volumetric video contents with
photo-realism and in real-time. The core of NeuVV is to efficiently encode a
dynamic neural radiance field (NeRF) into renderable and editable primitives.
We introduce two types of factorization schemes: a hyper-spherical harmonics
(HH) decomposition for modeling smooth color variations over space and time and
a learnable basis representation for modeling abrupt density and color changes
caused by motion. NeuVV factorization can be integrated into a Video Octree
(VOctree) analogous to PlenOctree to significantly accelerate training while
reducing memory overhead. Real-time NeuVV rendering further enables a class of
immersive content editing tools. Specifically, NeuVV treats each VOctree as a
primitive and implements volume-based depth ordering and alpha blending to
realize spatial-temporal compositions for content re-purposing. For example, we
demonstrate positioning varied manifestations of the same performance at
different 3D locations with different timing, adjusting color/texture of the
performer's clothing, casting spotlight shadows and synthesizing distance
falloff lighting, etc, all at an interactive speed. We further develop a hybrid
neural-rasterization rendering framework to support consumer-level VR headsets
so that the aforementioned volumetric video viewing and editing, for the first
time, can be conducted immersively in virtual 3D space.

---

## Block-NeRF: Scalable Large Scene Neural View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-02-10 | Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P. Srinivasan, Jonathan T. Barron, Henrik Kretzschmar | cs.CV | [PDF](http://arxiv.org/pdf/2202.05263v1){: .btn .btn-green } |

**Abstract**: We present Block-NeRF, a variant of Neural Radiance Fields that can represent
large-scale environments. Specifically, we demonstrate that when scaling NeRF
to render city-scale scenes spanning multiple blocks, it is vital to decompose
the scene into individually trained NeRFs. This decomposition decouples
rendering time from scene size, enables rendering to scale to arbitrarily large
environments, and allows per-block updates of the environment. We adopt several
architectural changes to make NeRF robust to data captured over months under
different environmental conditions. We add appearance embeddings, learned pose
refinement, and controllable exposure to each individual NeRF, and introduce a
procedure for aligning appearance between adjacent NeRFs so that they can be
seamlessly combined. We build a grid of Block-NeRFs from 2.8 million images to
create the largest neural scene representation to date, capable of rendering an
entire neighborhood of San Francisco.

Comments:
- Project page: https://waymo.com/research/block-nerf/

---

## PVSeRF: Joint Pixel-, Voxel- and Surface-Aligned Radiance Field for  Single-Image Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-02-10 | Xianggang Yu, Jiapeng Tang, Yipeng Qin, Chenghong Li, Linchao Bao, Xiaoguang Han, Shuguang Cui | cs.CV | [PDF](http://arxiv.org/pdf/2202.04879v1){: .btn .btn-green } |

**Abstract**: We present PVSeRF, a learning framework that reconstructs neural radiance
fields from single-view RGB images, for novel view synthesis. Previous
solutions, such as pixelNeRF, rely only on pixel-aligned features and suffer
from feature ambiguity issues. As a result, they struggle with the
disentanglement of geometry and appearance, leading to implausible geometries
and blurry results. To address this challenge, we propose to incorporate
explicit geometry reasoning and combine it with pixel-aligned features for
radiance field prediction. Specifically, in addition to pixel-aligned features,
we further constrain the radiance field learning to be conditioned on i)
voxel-aligned features learned from a coarse volumetric grid and ii) fine
surface-aligned features extracted from a regressed point cloud. We show that
the introduction of such geometry-aware features helps to achieve a better
disentanglement between appearance and geometry, i.e. recovering more accurate
geometries and synthesizing higher quality images of novel views. Extensive
experiments against state-of-the-art methods on ShapeNet benchmarks demonstrate
the superiority of our approach for single-image novel view synthesis.

---

## MedNeRF: Medical Neural Radiance Fields for Reconstructing 3D-aware  CT-Projections from a Single X-ray

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-02-02 | Abril Corona-Figueroa, Jonathan Frawley, Sam Bond-Taylor, Sarath Bethapudi, Hubert P. H. Shum, Chris G. Willcocks | eess.IV | [PDF](http://arxiv.org/pdf/2202.01020v3){: .btn .btn-green } |

**Abstract**: Computed tomography (CT) is an effective medical imaging modality, widely
used in the field of clinical medicine for the diagnosis of various
pathologies. Advances in Multidetector CT imaging technology have enabled
additional functionalities, including generation of thin slice multiplanar
cross-sectional body imaging and 3D reconstructions. However, this involves
patients being exposed to a considerable dose of ionising radiation. Excessive
ionising radiation can lead to deterministic and harmful effects on the body.
This paper proposes a Deep Learning model that learns to reconstruct CT
projections from a few or even a single-view X-ray. This is based on a novel
architecture that builds from neural radiance fields, which learns a continuous
representation of CT scans by disentangling the shape and volumetric depth of
surface and internal anatomical structures from 2D images. Our model is trained
on chest and knee datasets, and we demonstrate qualitative and quantitative
high-fidelity renderings and compare our approach to other recent radiance
field-based methods. Our code and link to our datasets are available at
https://github.com/abrilcf/mednerf

Comments:
- 6 pages, 4 figures, accepted at IEEE EMBC 2022

---

## CLA-NeRF: Category-Level Articulated Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-02-01 | Wei-Cheng Tseng, Hung-Ju Liao, Lin Yen-Chen, Min Sun | cs.CV | [PDF](http://arxiv.org/pdf/2202.00181v3){: .btn .btn-green } |

**Abstract**: We propose CLA-NeRF -- a Category-Level Articulated Neural Radiance Field
that can perform view synthesis, part segmentation, and articulated pose
estimation. CLA-NeRF is trained at the object category level using no CAD
models and no depth, but a set of RGB images with ground truth camera poses and
part segments. During inference, it only takes a few RGB views (i.e., few-shot)
of an unseen 3D object instance within the known category to infer the object
part segmentation and the neural radiance field. Given an articulated pose as
input, CLA-NeRF can perform articulation-aware volume rendering to generate the
corresponding RGB image at any camera pose. Moreover, the articulated pose of
an object can be estimated via inverse rendering. In our experiments, we
evaluate the framework across five categories on both synthetic and real-world
data. In all cases, our method shows realistic deformation results and accurate
articulated pose estimation. We believe that both few-shot articulated object
rendering and articulated pose estimation open doors for robots to perceive and
interact with unseen articulated objects.

Comments:
- accepted by ICRA 2022