---
layout: default
title: December
parent: 2021
nav_order: 12
---
<!---metadata--->

## InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-31 | Mijeong Kim, Seonguk Seo, Bohyung Han | cs.CV | [PDF](http://arxiv.org/pdf/2112.15399v2){: .btn .btn-green } |

**Abstract**: We present an information-theoretic regularization technique for few-shot
novel view synthesis based on neural implicit representation. The proposed
approach minimizes potential reconstruction inconsistency that happens due to
insufficient viewpoints by imposing the entropy constraint of the density in
each ray. In addition, to alleviate the potential degenerate issue when all
training images are acquired from almost redundant viewpoints, we further
incorporate the spatially smoothness constraint into the estimated images by
restricting information gains from a pair of rays with slightly different
viewpoints. The main idea of our algorithm is to make reconstructed scenes
compact along individual rays and consistent across rays in the neighborhood.
The proposed regularizers can be plugged into most of existing neural volume
rendering techniques based on NeRF in a straightforward way. Despite its
simplicity, we achieve consistently improved performance compared to existing
neural view synthesis methods by large margins on multiple standard benchmarks.

Comments:
- CVPR 2022, Website: http://cv.snu.ac.kr/research/InfoNeRF

---

## Learning Implicit Body Representations from Double Diffusion Based  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-23 | Guangming Yao, Hongzhi Wu, Yi Yuan, Lincheng Li, Kun Zhou, Xin Yu | cs.CV | [PDF](http://arxiv.org/pdf/2112.12390v2){: .btn .btn-green } |

**Abstract**: In this paper, we present a novel double diffusion based neural radiance
field, dubbed DD-NeRF, to reconstruct human body geometry and render the human
body appearance in novel views from a sparse set of images. We first propose a
double diffusion mechanism to achieve expressive representations of input
images by fully exploiting human body priors and image appearance details at
two levels. At the coarse level, we first model the coarse human body poses and
shapes via an unclothed 3D deformable vertex model as guidance. At the fine
level, we present a multi-view sampling network to capture subtle geometric
deformations and image detailed appearances, such as clothing and hair, from
multiple input views. Considering the sparsity of the two level features, we
diffuse them into feature volumes in the canonical space to construct neural
radiance fields. Then, we present a signed distance function (SDF) regression
network to construct body surfaces from the diffused features. Thanks to our
double diffused representations, our method can even synthesize novel views of
unseen subjects. Experiments on various datasets demonstrate that our approach
outperforms the state-of-the-art in both geometric reconstruction and novel
view synthesis.

Comments:
- 6 pages, 5 figures

---

## BANMo: Building Animatable 3D Neural Models from Many Casual Videos

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-23 | Gengshan Yang, Minh Vo, Natalia Neverova, Deva Ramanan, Andrea Vedaldi, Hanbyul Joo | cs.CV | [PDF](http://arxiv.org/pdf/2112.12761v3){: .btn .btn-green } |

**Abstract**: Prior work for articulated 3D shape reconstruction often relies on
specialized sensors (e.g., synchronized multi-camera systems), or pre-built 3D
deformable models (e.g., SMAL or SMPL). Such methods are not able to scale to
diverse sets of objects in the wild. We present BANMo, a method that requires
neither a specialized sensor nor a pre-defined template shape. BANMo builds
high-fidelity, articulated 3D models (including shape and animatable skinning
weights) from many monocular casual videos in a differentiable rendering
framework. While the use of many videos provides more coverage of camera views
and object articulations, they introduce significant challenges in establishing
correspondence across scenes with different backgrounds, illumination
conditions, etc. Our key insight is to merge three schools of thought; (1)
classic deformable shape models that make use of articulated bones and blend
skinning, (2) volumetric neural radiance fields (NeRFs) that are amenable to
gradient-based optimization, and (3) canonical embeddings that generate
correspondences between pixels and an articulated model. We introduce neural
blend skinning models that allow for differentiable and invertible articulated
deformations. When combined with canonical embeddings, such models allow us to
establish dense correspondences across videos that can be self-supervised with
cycle consistency. On real and synthetic datasets, BANMo shows higher-fidelity
3D reconstructions than prior works for humans and animals, with the ability to
render realistic images from novel viewpoints and poses. Project webpage:
banmo-www.github.io .

Comments:
- CVPR 2022 camera-ready version (last update: May 2022)

---

## 3D-aware Image Synthesis via Learning Structural and Textural  Representations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-20 | Yinghao Xu, Sida Peng, Ceyuan Yang, Yujun Shen, Bolei Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2112.10759v2){: .btn .btn-green } |

**Abstract**: Making generative models 3D-aware bridges the 2D image space and the 3D
physical world yet remains challenging. Recent attempts equip a Generative
Adversarial Network (GAN) with a Neural Radiance Field (NeRF), which maps 3D
coordinates to pixel values, as a 3D prior. However, the implicit function in
NeRF has a very local receptive field, making the generator hard to become
aware of the global structure. Meanwhile, NeRF is built on volume rendering
which can be too costly to produce high-resolution results, increasing the
optimization difficulty. To alleviate these two problems, we propose a novel
framework, termed as VolumeGAN, for high-fidelity 3D-aware image synthesis,
through explicitly learning a structural representation and a textural
representation. We first learn a feature volume to represent the underlying
structure, which is then converted to a feature field using a NeRF-like model.
The feature field is further accumulated into a 2D feature map as the textural
representation, followed by a neural renderer for appearance synthesis. Such a
design enables independent control of the shape and the appearance. Extensive
experiments on a wide range of datasets show that our approach achieves
sufficiently higher image quality and better 3D control than the previous
methods.

Comments:
- CVPR 2022 camera-ready, Project page:
  https://genforce.github.io/volumegan/

---

## Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual  Fly-Throughs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-20 | Haithem Turki, Deva Ramanan, Mahadev Satyanarayanan | cs.CV | [PDF](http://arxiv.org/pdf/2112.10703v2){: .btn .btn-green } |

**Abstract**: We use neural radiance fields (NeRFs) to build interactive 3D environments
from large-scale visual captures spanning buildings or even multiple city
blocks collected primarily from drones. In contrast to single object scenes (on
which NeRFs are traditionally evaluated), our scale poses multiple challenges
including (1) the need to model thousands of images with varying lighting
conditions, each of which capture only a small subset of the scene, (2)
prohibitively large model capacities that make it infeasible to train on a
single GPU, and (3) significant challenges for fast rendering that would enable
interactive fly-throughs.
  To address these challenges, we begin by analyzing visibility statistics for
large-scale scenes, motivating a sparse network structure where parameters are
specialized to different regions of the scene. We introduce a simple geometric
clustering algorithm for data parallelism that partitions training images (or
rather pixels) into different NeRF submodules that can be trained in parallel.
  We evaluate our approach on existing datasets (Quad 6k and UrbanScene3D) as
well as against our own drone footage, improving training speed by 3x and PSNR
by 12%. We also evaluate recent NeRF fast renderers on top of Mega-NeRF and
introduce a novel method that exploits temporal coherence. Our technique
achieves a 40x speedup over conventional NeRF rendering while remaining within
0.8 db in PSNR quality, exceeding the fidelity of existing fast renderers.

Comments:
- CVPR 2022 Project page: https://meganerf.cmusatyalab.org GitHub:
  https://github.com/cmusatyalab/mega-nerf

---

## HVTR: Hybrid Volumetric-Textural Rendering for Human Avatars

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-19 | Tao Hu, Tao Yu, Zerong Zheng, He Zhang, Yebin Liu, Matthias Zwicker | cs.CV | [PDF](http://arxiv.org/pdf/2112.10203v2){: .btn .btn-green } |

**Abstract**: We propose a novel neural rendering pipeline, Hybrid Volumetric-Textural
Rendering (HVTR), which synthesizes virtual human avatars from arbitrary poses
efficiently and at high quality. First, we learn to encode articulated human
motions on a dense UV manifold of the human body surface. To handle complicated
motions (e.g., self-occlusions), we then leverage the encoded information on
the UV manifold to construct a 3D volumetric representation based on a dynamic
pose-conditioned neural radiance field. While this allows us to represent 3D
geometry with changing topology, volumetric rendering is computationally heavy.
Hence we employ only a rough volumetric representation using a pose-conditioned
downsampled neural radiance field (PD-NeRF), which we can render efficiently at
low resolutions. In addition, we learn 2D textural features that are fused with
rendered volumetric features in image space. The key advantage of our approach
is that we can then convert the fused features into a high-resolution,
high-quality avatar by a fast GAN-based textural renderer. We demonstrate that
hybrid rendering enables HVTR to handle complicated motions, render
high-quality avatars under user-controlled poses/shapes and even loose
clothing, and most importantly, be efficient at inference time. Our
experimental results also demonstrate state-of-the-art quantitative results.

Comments:
- Accepted to 3DV 2022. See more results at
  https://www.cs.umd.edu/~taohu/hvtr/ Demo:
  https://www.youtube.com/watch?v=LE0-YpbLlkY

---

## Solving Inverse Problems with NerfGANs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-16 | Giannis Daras, Wen-Sheng Chu, Abhishek Kumar, Dmitry Lagun, Alexandros G. Dimakis | cs.CV | [PDF](http://arxiv.org/pdf/2112.09061v1){: .btn .btn-green } |

**Abstract**: We introduce a novel framework for solving inverse problems using NeRF-style
generative models. We are interested in the problem of 3-D scene reconstruction
given a single 2-D image and known camera parameters. We show that naively
optimizing the latent space leads to artifacts and poor novel view rendering.
We attribute this problem to volume obstructions that are clear in the 3-D
geometry and become visible in the renderings of novel views. We propose a
novel radiance field regularization method to obtain better 3-D surfaces and
improved novel views given single view observations. Our method naturally
extends to general inverse problems including inpainting where one observes
only partially a single view. We experimentally evaluate our method, achieving
visual improvements and performance boosts over the baselines in a wide range
of tasks. Our method achieves $30-40\%$ MSE reduction and $15-25\%$ reduction
in LPIPS loss compared to the previous state of the art.

Comments:
- 16 pages, 18 figures

---

## GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-16 | Yu Deng, Jiaolong Yang, Jianfeng Xiang, Xin Tong | cs.CV | [PDF](http://arxiv.org/pdf/2112.08867v3){: .btn .btn-green } |

**Abstract**: 3D-aware image generative modeling aims to generate 3D-consistent images with
explicitly controllable camera poses. Recent works have shown promising results
by training neural radiance field (NeRF) generators on unstructured 2D images,
but still can not generate highly-realistic images with fine details. A
critical reason is that the high memory and computation cost of volumetric
representation learning greatly restricts the number of point samples for
radiance integration during training. Deficient sampling not only limits the
expressive power of the generator to handle fine details but also impedes
effective GAN training due to the noise caused by unstable Monte Carlo
sampling. We propose a novel approach that regulates point sampling and
radiance field learning on 2D manifolds, embodied as a set of learned implicit
surfaces in the 3D volume. For each viewing ray, we calculate ray-surface
intersections and accumulate their radiance generated by the network. By
training and rendering such radiance manifolds, our generator can produce high
quality images with realistic fine details and strong visual 3D consistency.

Comments:
- CVPR2022 Oral. Project page: https://yudeng.github.io/GRAM/

---

## HeadNeRF: A Real-time NeRF-based Parametric Head Model

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-10 | Yang Hong, Bo Peng, Haiyao Xiao, Ligang Liu, Juyong Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2112.05637v3){: .btn .btn-green } |

**Abstract**: In this paper, we propose HeadNeRF, a novel NeRF-based parametric head model
that integrates the neural radiance field to the parametric representation of
the human head. It can render high fidelity head images in real-time on modern
GPUs, and supports directly controlling the generated images' rendering pose
and various semantic attributes. Different from existing related parametric
models, we use the neural radiance fields as a novel 3D proxy instead of the
traditional 3D textured mesh, which makes that HeadNeRF is able to generate
high fidelity images. However, the computationally expensive rendering process
of the original NeRF hinders the construction of the parametric NeRF model. To
address this issue, we adopt the strategy of integrating 2D neural rendering to
the rendering process of NeRF and design novel loss terms. As a result, the
rendering speed of HeadNeRF can be significantly accelerated, and the rendering
time of one frame is reduced from 5s to 25ms. The well designed loss terms also
improve the rendering accuracy, and the fine-level details of the human head,
such as the gaps between teeth, wrinkles, and beards, can be represented and
synthesized by HeadNeRF. Extensive experimental results and several
applications demonstrate its effectiveness. The trained parametric model is
available at https://github.com/CrisHY1995/headnerf.

Comments:
- Accepted by CVPR2022. Project page:
  https://crishy1995.github.io/HeadNeRF-Project/

---

## BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale  Scene Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-10 | Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao, Anyi Rao, Christian Theobalt, Bo Dai, Dahua Lin | cs.CV | [PDF](http://arxiv.org/pdf/2112.05504v4){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) has achieved outstanding performance in
modeling 3D objects and controlled scenes, usually under a single scale. In
this work, we focus on multi-scale cases where large changes in imagery are
observed at drastically different scales. This scenario vastly exists in
real-world 3D environments, such as city scenes, with views ranging from
satellite level that captures the overview of a city, to ground level imagery
showing complex details of an architecture; and can also be commonly identified
in landscape and delicate minecraft 3D models. The wide span of viewing
positions within these scenes yields multi-scale renderings with very different
levels of detail, which poses great challenges to neural radiance field and
biases it towards compromised results. To address these issues, we introduce
BungeeNeRF, a progressive neural radiance field that achieves level-of-detail
rendering across drastically varied scales. Starting from fitting distant views
with a shallow base block, as training progresses, new blocks are appended to
accommodate the emerging details in the increasingly closer views. The strategy
progressively activates high-frequency channels in NeRF's positional encoding
inputs and successively unfolds more complex details as the training proceeds.
We demonstrate the superiority of BungeeNeRF in modeling diverse multi-scale
scenes with drastically varying views on multiple data sources (city models,
synthetic, and drone captured data) and its support for high-quality rendering
in different levels of detail.

Comments:
- Accepted to ECCV22; Previous version: CityNeRF: Building NeRF at City
  Scale; Project page can be found in https://city-super.github.io/citynerf

---

## Plenoxels: Radiance Fields without Neural Networks



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-09 | Alex Yu, Sara Fridovich-Keil, Matthew Tancik, Qinhong Chen, Benjamin Recht, Angjoo Kanazawa | cs.CV | [PDF](http://arxiv.org/pdf/2112.05131v1){: .btn .btn-green } |

**Abstract**: We introduce Plenoxels (plenoptic voxels), a system for photorealistic view
synthesis. Plenoxels represent a scene as a sparse 3D grid with spherical
harmonics. This representation can be optimized from calibrated images via
gradient methods and regularization without any neural components. On standard,
benchmark tasks, Plenoxels are optimized two orders of magnitude faster than
Neural Radiance Fields with no loss in visual quality.

Comments:
- For video and code, please see https://alexyu.net/plenoxels

---

## Deep Visual Constraints: Neural Implicit Models for Manipulation  Planning from Visual Input

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-09 | Jung-Su Ha, Danny Driess, Marc Toussaint | cs.RO | [PDF](http://arxiv.org/pdf/2112.04812v3){: .btn .btn-green } |

**Abstract**: Manipulation planning is the problem of finding a sequence of robot
configurations that involves interactions with objects in the scene, e.g.,
grasping and placing an object, or more general tool-use. To achieve such
interactions, traditional approaches require hand-engineering of object
representations and interaction constraints, which easily becomes tedious when
complex objects/interactions are considered. Inspired by recent advances in 3D
modeling, e.g. NeRF, we propose a method to represent objects as continuous
functions upon which constraint features are defined and jointly trained. In
particular, the proposed pixel-aligned representation is directly inferred from
images with known camera geometry and naturally acts as a perception component
in the whole manipulation pipeline, thereby enabling long-horizon planning only
from visual input. Project page:
https://sites.google.com/view/deep-visual-constraints

Comments:
- IEEE Robotics and Automation Letters (RA-L) 2022

---

## CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-09 | Can Wang, Menglei Chai, Mingming He, Dongdong Chen, Jing Liao | cs.CV | [PDF](http://arxiv.org/pdf/2112.05139v3){: .btn .btn-green } |

**Abstract**: We present CLIP-NeRF, a multi-modal 3D object manipulation method for neural
radiance fields (NeRF). By leveraging the joint language-image embedding space
of the recent Contrastive Language-Image Pre-Training (CLIP) model, we propose
a unified framework that allows manipulating NeRF in a user-friendly way, using
either a short text prompt or an exemplar image. Specifically, to combine the
novel view synthesis capability of NeRF and the controllable manipulation
ability of latent representations from generative models, we introduce a
disentangled conditional NeRF architecture that allows individual control over
both shape and appearance. This is achieved by performing the shape
conditioning via applying a learned deformation field to the positional
encoding and deferring color conditioning to the volumetric rendering stage. To
bridge this disentangled latent representation to the CLIP embedding, we design
two code mappers that take a CLIP embedding as input and update the latent
codes to reflect the targeted editing. The mappers are trained with a
CLIP-based matching loss to ensure the manipulation accuracy. Furthermore, we
propose an inverse optimization method that accurately projects an input image
to the latent codes for manipulation to enable editing on real images. We
evaluate our approach by extensive experiments on a variety of text prompts and
exemplar images and also provide an intuitive interface for interactive
editing. Our implementation is available at
https://cassiepython.github.io/clipnerf/

Comments:
- To Appear at CVPR 2022

---

## NeRF for Outdoor Scene Relighting

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-09 | Viktor Rudnev, Mohamed Elgharib, William Smith, Lingjie Liu, Vladislav Golyanik, Christian Theobalt | cs.CV | [PDF](http://arxiv.org/pdf/2112.05140v2){: .btn .btn-green } |

**Abstract**: Photorealistic editing of outdoor scenes from photographs requires a profound
understanding of the image formation process and an accurate estimation of the
scene geometry, reflectance and illumination. A delicate manipulation of the
lighting can then be performed while keeping the scene albedo and geometry
unaltered. We present NeRF-OSR, i.e., the first approach for outdoor scene
relighting based on neural radiance fields. In contrast to the prior art, our
technique allows simultaneous editing of both scene illumination and camera
viewpoint using only a collection of outdoor photos shot in uncontrolled
settings. Moreover, it enables direct control over the scene illumination, as
defined through a spherical harmonics model. For evaluation, we collect a new
benchmark dataset of several outdoor sites photographed from multiple
viewpoints and at different times. For each time, a 360 degree environment map
is captured together with a colour-calibration chequerboard to allow accurate
numerical evaluations on real data against ground truth. Comparisons against
SoTA show that NeRF-OSR enables controllable lighting and viewpoint editing at
higher quality and with realistic self-shadowing reproduction. Our method and
the dataset are publicly available at https://4dqv.mpi-inf.mpg.de/NeRF-OSR/.

Comments:
- 22 pages, 10 figures, 2 tables; ECCV 2022; project web page:
  https://4dqv.mpi-inf.mpg.de/NeRF-OSR/

---

## Geometry-Guided Progressive NeRF for Generalizable and Efficient Neural  Human Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-08 | Mingfei Chen, Jianfeng Zhang, Xiangyu Xu, Lijuan Liu, Yujun Cai, Jiashi Feng, Shuicheng Yan | cs.CV | [PDF](http://arxiv.org/pdf/2112.04312v3){: .btn .btn-green } |

**Abstract**: In this work we develop a generalizable and efficient Neural Radiance Field
(NeRF) pipeline for high-fidelity free-viewpoint human body synthesis under
settings with sparse camera views. Though existing NeRF-based methods can
synthesize rather realistic details for human body, they tend to produce poor
results when the input has self-occlusion, especially for unseen humans under
sparse views. Moreover, these methods often require a large number of sampling
points for rendering, which leads to low efficiency and limits their real-world
applicability. To address these challenges, we propose a Geometry-guided
Progressive NeRF (GP-NeRF). In particular, to better tackle self-occlusion, we
devise a geometry-guided multi-view feature integration approach that utilizes
the estimated geometry prior to integrate the incomplete information from input
views and construct a complete geometry volume for the target human body.
Meanwhile, for achieving higher rendering efficiency, we introduce a
progressive rendering pipeline through geometry guidance, which leverages the
geometric feature volume and the predicted density values to progressively
reduce the number of sampling points and speed up the rendering process.
Experiments on the ZJU-MoCap and THUman datasets show that our method
outperforms the state-of-the-arts significantly across multiple generalization
settings, while the time cost is reduced > 70% via applying our efficient
progressive rendering pipeline.

---

## Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-07 | Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T. Barron, Pratul P. Srinivasan | cs.CV | [PDF](http://arxiv.org/pdf/2112.03907v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) is a popular view synthesis technique that
represents a scene as a continuous volumetric function, parameterized by
multilayer perceptrons that provide the volume density and view-dependent
emitted radiance at each location. While NeRF-based techniques excel at
representing fine geometric structures with smoothly varying view-dependent
appearance, they often fail to accurately capture and reproduce the appearance
of glossy surfaces. We address this limitation by introducing Ref-NeRF, which
replaces NeRF's parameterization of view-dependent outgoing radiance with a
representation of reflected radiance and structures this function using a
collection of spatially-varying scene properties. We show that together with a
regularizer on normal vectors, our model significantly improves the realism and
accuracy of specular reflections. Furthermore, we show that our model's
internal representation of outgoing radiance is interpretable and useful for
scene editing.

Comments:
- Project page: https://dorverbin.github.io/refnerf/

---

## CG-NeRF: Conditional Generative Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-07 | Kyungmin Jo, Gyumin Shim, Sanghun Jung, Soyoung Yang, Jaegul Choo | cs.CV | [PDF](http://arxiv.org/pdf/2112.03517v1){: .btn .btn-green } |

**Abstract**: While recent NeRF-based generative models achieve the generation of diverse
3D-aware images, these approaches have limitations when generating images that
contain user-specified characteristics. In this paper, we propose a novel
model, referred to as the conditional generative neural radiance fields
(CG-NeRF), which can generate multi-view images reflecting extra input
conditions such as images or texts. While preserving the common characteristics
of a given input condition, the proposed model generates diverse images in fine
detail. We propose: 1) a novel unified architecture which disentangles the
shape and appearance from a condition given in various forms and 2) the
pose-consistent diversity loss for generating multimodal outputs while
maintaining consistency of the view. Experimental results show that the
proposed method maintains consistent image quality on various condition types
and achieves superior fidelity and diversity compared to existing NeRF-based
generative models.

---

## Dense Depth Priors for Neural Radiance Fields from Sparse Input Views

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-06 | Barbara Roessle, Jonathan T. Barron, Ben Mildenhall, Pratul P. Srinivasan, Matthias Nießner | cs.CV | [PDF](http://arxiv.org/pdf/2112.03288v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) encode a scene into a neural representation
that enables photo-realistic rendering of novel views. However, a successful
reconstruction from RGB images requires a large number of input views taken
under static conditions - typically up to a few hundred images for room-size
scenes. Our method aims to synthesize novel views of whole rooms from an order
of magnitude fewer images. To this end, we leverage dense depth priors in order
to constrain the NeRF optimization. First, we take advantage of the sparse
depth data that is freely available from the structure from motion (SfM)
preprocessing step used to estimate camera poses. Second, we use depth
completion to convert these sparse points into dense depth maps and uncertainty
estimates, which are used to guide NeRF optimization. Our method enables
data-efficient novel view synthesis on challenging indoor scenes, using as few
as 18 images for an entire scene.

Comments:
- CVPR 2022, project page:
  https://barbararoessle.github.io/dense_depth_priors_nerf/ , video:
  https://youtu.be/zzkvvdcvksc

---

## HumanNeRF: Efficiently Generated Human Radiance Field from Sparse Inputs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-06 | Fuqiang Zhao, Wei Yang, Jiakai Zhang, Pei Lin, Yingliang Zhang, Jingyi Yu, Lan Xu | cs.CV | [PDF](http://arxiv.org/pdf/2112.02789v3){: .btn .btn-green } |

**Abstract**: Recent neural human representations can produce high-quality multi-view
rendering but require using dense multi-view inputs and costly training. They
are hence largely limited to static models as training each frame is
infeasible. We present HumanNeRF - a generalizable neural representation - for
high-fidelity free-view synthesis of dynamic humans. Analogous to how IBRNet
assists NeRF by avoiding per-scene training, HumanNeRF employs an aggregated
pixel-alignment feature across multi-view inputs along with a pose embedded
non-rigid deformation field for tackling dynamic motions. The raw HumanNeRF can
already produce reasonable rendering on sparse video inputs of unseen subjects
and camera settings. To further improve the rendering quality, we augment our
solution with an appearance blending module for combining the benefits of both
neural volumetric rendering and neural texture blending. Extensive experiments
on various multi-view dynamic human datasets demonstrate the generalizability
and effectiveness of our approach in synthesizing photo-realistic free-view
humans under challenging motions and with very sparse camera view inputs.

Comments:
- https://zhaofuq.github.io/humannerf/

---

## MoFaNeRF: Morphable Facial Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-04 | Yiyu Zhuang, Hao Zhu, Xusen Sun, Xun Cao | cs.CV | [PDF](http://arxiv.org/pdf/2112.02308v2){: .btn .btn-green } |

**Abstract**: We propose a parametric model that maps free-view images into a vector space
of coded facial shape, expression and appearance with a neural radiance field,
namely Morphable Facial NeRF. Specifically, MoFaNeRF takes the coded facial
shape, expression and appearance along with space coordinate and view direction
as input to an MLP, and outputs the radiance of the space point for
photo-realistic image synthesis. Compared with conventional 3D morphable models
(3DMM), MoFaNeRF shows superiority in directly synthesizing photo-realistic
facial details even for eyes, mouths, and beards. Also, continuous face
morphing can be easily achieved by interpolating the input shape, expression
and appearance codes. By introducing identity-specific modulation and texture
encoder, our model synthesizes accurate photometric details and shows strong
representation ability. Our model shows strong ability on multiple applications
including image-based fitting, random generation, face rigging, face editing,
and novel view synthesis. Experiments show that our method achieves higher
representation ability than previous parametric models, and achieves
competitive performance in several applications. To the best of our knowledge,
our work is the first facial parametric model built upon a neural radiance
field that can be used in fitting, generation and manipulation. The code and
data is available at https://github.com/zhuhao-nju/mofanerf.

Comments:
- accepted to ECCV2022; code available at
  http://github.com/zhuhao-nju/mofanerf

---

## NeRF-SR: High-Quality Neural Radiance Fields using Supersampling

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-03 | Chen Wang, Xian Wu, Yuan-Chen Guo, Song-Hai Zhang, Yu-Wing Tai, Shi-Min Hu | cs.CV | [PDF](http://arxiv.org/pdf/2112.01759v3){: .btn .btn-green } |

**Abstract**: We present NeRF-SR, a solution for high-resolution (HR) novel view synthesis
with mostly low-resolution (LR) inputs. Our method is built upon Neural
Radiance Fields (NeRF) that predicts per-point density and color with a
multi-layer perceptron. While producing images at arbitrary scales, NeRF
struggles with resolutions that go beyond observed images. Our key insight is
that NeRF benefits from 3D consistency, which means an observed pixel absorbs
information from nearby views. We first exploit it by a supersampling strategy
that shoots multiple rays at each image pixel, which further enforces
multi-view constraint at a sub-pixel level. Then, we show that NeRF-SR can
further boost the performance of supersampling by a refinement network that
leverages the estimated depth at hand to hallucinate details from related
patches on only one HR reference image. Experiment results demonstrate that
NeRF-SR generates high-quality results for novel view synthesis at HR on both
synthetic and real-world datasets without any external information.

Comments:
- Accepted to MM 2022. Project Page:
  https://cwchenwang.github.io/NeRF-SR

---

## CoNeRF: Controllable Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-03 | Kacper Kania, Kwang Moo Yi, Marek Kowalski, Tomasz Trzciński, Andrea Tagliasacchi | cs.CV | [PDF](http://arxiv.org/pdf/2112.01983v2){: .btn .btn-green } |

**Abstract**: We extend neural 3D representations to allow for intuitive and interpretable
user control beyond novel view rendering (i.e. camera control). We allow the
user to annotate which part of the scene one wishes to control with just a
small number of mask annotations in the training images. Our key idea is to
treat the attributes as latent variables that are regressed by the neural
network given the scene encoding. This leads to a few-shot learning framework,
where attributes are discovered automatically by the framework, when
annotations are not provided. We apply our method to various scenes with
different types of controllable attributes (e.g. expression control on human
faces, or state control in movement of inanimate objects). Overall, we
demonstrate, to the best of our knowledge, for the first time novel view and
novel attribute re-rendering of scenes from a single video.

Comments:
- Project page: https://conerf.github.io/

---

## Efficient Neural Radiance Fields for Interactive Free-viewpoint Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-02 | Haotong Lin, Sida Peng, Zhen Xu, Yunzhi Yan, Qing Shuai, Hujun Bao, Xiaowei Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2112.01517v3){: .btn .btn-green } |

**Abstract**: This paper aims to tackle the challenge of efficiently producing interactive
free-viewpoint videos. Some recent works equip neural radiance fields with
image encoders, enabling them to generalize across scenes. When processing
dynamic scenes, they can simply treat each video frame as an individual scene
and perform novel view synthesis to generate free-viewpoint videos. However,
their rendering process is slow and cannot support interactive applications. A
major factor is that they sample lots of points in empty space when inferring
radiance fields. We propose a novel scene representation, called ENeRF, for the
fast creation of interactive free-viewpoint videos. Specifically, given
multi-view images at one frame, we first build the cascade cost volume to
predict the coarse geometry of the scene. The coarse geometry allows us to
sample few points near the scene surface, thereby significantly improving the
rendering speed. This process is fully differentiable, enabling us to jointly
learn the depth prediction and radiance field networks from RGB images.
Experiments on multiple benchmarks show that our approach exhibits competitive
performance while being at least 60 times faster than previous generalizable
radiance field methods.

Comments:
- SIGGRAPH Asia 2022; Project page: https://zju3dv.github.io/enerf/

---

## Zero-Shot Text-Guided Object Generation with Dream Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-02 | Ajay Jain, Ben Mildenhall, Jonathan T. Barron, Pieter Abbeel, Ben Poole | cs.CV | [PDF](http://arxiv.org/pdf/2112.01455v2){: .btn .btn-green } |

**Abstract**: We combine neural rendering with multi-modal image and text representations
to synthesize diverse 3D objects solely from natural language descriptions. Our
method, Dream Fields, can generate the geometry and color of a wide range of
objects without 3D supervision. Due to the scarcity of diverse, captioned 3D
data, prior methods only generate objects from a handful of categories, such as
ShapeNet. Instead, we guide generation with image-text models pre-trained on
large datasets of captioned images from the web. Our method optimizes a Neural
Radiance Field from many camera views so that rendered images score highly with
a target caption according to a pre-trained CLIP model. To improve fidelity and
visual quality, we introduce simple geometric priors, including
sparsity-inducing transmittance regularization, scene bounds, and new MLP
architectures. In experiments, Dream Fields produce realistic, multi-view
consistent object geometry and color from a variety of natural language
captions.

Comments:
- CVPR 2022. 13 pages. Website: https://ajayj.com/dreamfields

---

## 3D-Aware Semantic-Guided Generative Model for Human Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-02 | Jichao Zhang, Enver Sangineto, Hao Tang, Aliaksandr Siarohin, Zhun Zhong, Nicu Sebe, Wei Wang | cs.CV | [PDF](http://arxiv.org/pdf/2112.01422v2){: .btn .btn-green } |

**Abstract**: Generative Neural Radiance Field (GNeRF) models, which extract implicit 3D
representations from 2D images, have recently been shown to produce realistic
images representing rigid/semi-rigid objects, such as human faces or cars.
However, they usually struggle to generate high-quality images representing
non-rigid objects, such as the human body, which is of a great interest for
many computer graphics applications. This paper proposes a 3D-aware
Semantic-Guided Generative Model (3D-SGAN) for human image synthesis, which
combines a GNeRF with a texture generator. The former learns an implicit 3D
representation of the human body and outputs a set of 2D semantic segmentation
masks. The latter transforms these semantic masks into a real image, adding a
realistic texture to the human appearance. Without requiring additional 3D
information, our model can learn 3D human representations with a
photo-realistic, controllable generation. Our experiments on the DeepFashion
dataset show that 3D-SGAN significantly outperforms the most recent baselines.
The code is available at https://github.com/zhangqianhui/3DSGAN

Comments:
- ECCV 2022. 29 pages

---

## Learning Neural Light Fields with Ray-Space Embedding Networks

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-02 | Benjamin Attal, Jia-Bin Huang, Michael Zollhoefer, Johannes Kopf, Changil Kim | cs.CV | [PDF](http://arxiv.org/pdf/2112.01523v3){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) produce state-of-the-art view synthesis
results. However, they are slow to render, requiring hundreds of network
evaluations per pixel to approximate a volume rendering integral. Baking NeRFs
into explicit data structures enables efficient rendering, but results in a
large increase in memory footprint and, in many cases, a quality reduction. In
this paper, we propose a novel neural light field representation that, in
contrast, is compact and directly predicts integrated radiance along rays. Our
method supports rendering with a single network evaluation per pixel for small
baseline light field datasets and can also be applied to larger baselines with
only a few evaluations per pixel. At the core of our approach is a ray-space
embedding network that maps the 4D ray-space manifold into an intermediate,
interpolable latent space. Our method achieves state-of-the-art quality on
dense forward-facing datasets such as the Stanford Light Field dataset. In
addition, for forward-facing scenes with sparser inputs we achieve results that
are competitive with NeRF-based approaches in terms of quality while providing
a better speed/quality/memory trade-off with far fewer network evaluations.

Comments:
- CVPR 2022 camera ready revision. Major changes include: 1. Additional
  comparison to NeX on Stanford, RealFF, Shiny datasets 2. Experiment on 360
  degree lego bulldozer scene in the appendix, using Pluecker parameterization
  3. Moving student-teacher results to the appendix 4. Clarity edits -- in
  particular, making it clear that our Stanford evaluation *does not* use
  subdivision

---

## RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from  Sparse Inputs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-12-01 | Michael Niemeyer, Jonathan T. Barron, Ben Mildenhall, Mehdi S. M. Sajjadi, Andreas Geiger, Noha Radwan | cs.CV | [PDF](http://arxiv.org/pdf/2112.00724v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have emerged as a powerful representation for
the task of novel view synthesis due to their simplicity and state-of-the-art
performance. Though NeRF can produce photorealistic renderings of unseen
viewpoints when many input views are available, its performance drops
significantly when this number is reduced. We observe that the majority of
artifacts in sparse input scenarios are caused by errors in the estimated scene
geometry, and by divergent behavior at the start of training. We address this
by regularizing the geometry and appearance of patches rendered from unobserved
viewpoints, and annealing the ray sampling space during training. We
additionally use a normalizing flow model to regularize the color of unobserved
viewpoints. Our model outperforms not only other methods that optimize over a
single scene, but in many cases also conditional models that are extensively
pre-trained on large multi-view datasets.

Comments:
- Project page available at
  https://m-niemeyer.github.io/regnerf/index.html