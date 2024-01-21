---
layout: default
title: July
parent: 2023
nav_order: 7
---
<!---metadata--->

## Dynamic PlenOctree for Adaptive Sampling Refinement in Explicit NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-28 | Haotian Bai, Yiqi Lin, Yize Chen, Lin Wang | cs.CV | [PDF](http://arxiv.org/pdf/2307.15333v1){: .btn .btn-green } |

**Abstract**: The explicit neural radiance field (NeRF) has gained considerable interest
for its efficient training and fast inference capabilities, making it a
promising direction such as virtual reality and gaming. In particular,
PlenOctree (POT)[1], an explicit hierarchical multi-scale octree
representation, has emerged as a structural and influential framework. However,
POT's fixed structure for direct optimization is sub-optimal as the scene
complexity evolves continuously with updates to cached color and density,
necessitating refining the sampling distribution to capture signal complexity
accordingly. To address this issue, we propose the dynamic PlenOctree DOT,
which adaptively refines the sample distribution to adjust to changing scene
complexity. Specifically, DOT proposes a concise yet novel hierarchical feature
fusion strategy during the iterative rendering process. Firstly, it identifies
the regions of interest through training signals to ensure adaptive and
efficient refinement. Next, rather than directly filtering out valueless nodes,
DOT introduces the sampling and pruning operations for octrees to aggregate
features, enabling rapid parameter learning. Compared with POT, our DOT
outperforms it by enhancing visual quality, reducing over $55.15$/$68.84\%$
parameters, and providing 1.7/1.9 times FPS for NeRF-synthetic and Tanks $\&$
Temples, respectively. Project homepage:https://vlislab22.github.io/DOT.
  [1] Yu, Alex, et al. "Plenoctrees for real-time rendering of neural radiance
fields." Proceedings of the IEEE/CVF International Conference on Computer
Vision. 2021.

Comments:
- Accepted by ICCV2023

---

## Improved Neural Radiance Fields Using Pseudo-depth and Fusion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-27 | Jingliang Li, Qiang Zhou, Chaohui Yu, Zhengda Lu, Jun Xiao, Zhibin Wang, Fan Wang | cs.CV | [PDF](http://arxiv.org/pdf/2308.03772v1){: .btn .btn-green } |

**Abstract**: Since the advent of Neural Radiance Fields, novel view synthesis has received
tremendous attention. The existing approach for the generalization of radiance
field reconstruction primarily constructs an encoding volume from nearby source
images as additional inputs. However, these approaches cannot efficiently
encode the geometric information of real scenes with various scale
objects/structures. In this work, we propose constructing multi-scale encoding
volumes and providing multi-scale geometry information to NeRF models. To make
the constructed volumes as close as possible to the surfaces of objects in the
scene and the rendered depth more accurate, we propose to perform depth
prediction and radiance field reconstruction simultaneously. The predicted
depth map will be used to supervise the rendered depth, narrow the depth range,
and guide points sampling. Finally, the geometric information contained in
point volume features may be inaccurate due to occlusion, lighting, etc. To
this end, we propose enhancing the point volume feature from depth-guided
neighbor feature fusion. Experiments demonstrate the superior performance of
our method in both novel view synthesis and dense geometry modeling without
per-scene optimization.

---

## MapNeRF: Incorporating Map Priors into Neural Radiance Fields for  Driving View Simulation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-27 | Chenming Wu, Jiadai Sun, Zhelun Shen, Liangjun Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2307.14981v2){: .btn .btn-green } |

**Abstract**: Simulating camera sensors is a crucial task in autonomous driving. Although
neural radiance fields are exceptional at synthesizing photorealistic views in
driving simulations, they still fail to generate extrapolated views. This paper
proposes to incorporate map priors into neural radiance fields to synthesize
out-of-trajectory driving views with semantic road consistency. The key insight
is that map information can be utilized as a prior to guiding the training of
the radiance fields with uncertainty. Specifically, we utilize the coarse
ground surface as uncertain information to supervise the density field and warp
depth with uncertainty from unknown camera poses to ensure multi-view
consistency. Experimental results demonstrate that our approach can produce
semantic consistency in deviated views for vehicle camera simulation. The
supplementary video can be viewed at https://youtu.be/jEQWr-Rfh3A.

Comments:
- Accepted by IEEE/RSJ International Conference on Intelligent Robots
  and Systems (IROS) 2023

---

## NeRF-Det: Learning Geometry-Aware Volumetric Representation for  Multi-View 3D Object Detection

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-27 | Chenfeng Xu, Bichen Wu, Ji Hou, Sam Tsai, Ruilong Li, Jialiang Wang, Wei Zhan, Zijian He, Peter Vajda, Kurt Keutzer, Masayoshi Tomizuka | cs.CV | [PDF](http://arxiv.org/pdf/2307.14620v1){: .btn .btn-green } |

**Abstract**: We present NeRF-Det, a novel method for indoor 3D detection with posed RGB
images as input. Unlike existing indoor 3D detection methods that struggle to
model scene geometry, our method makes novel use of NeRF in an end-to-end
manner to explicitly estimate 3D geometry, thereby improving 3D detection
performance. Specifically, to avoid the significant extra latency associated
with per-scene optimization of NeRF, we introduce sufficient geometry priors to
enhance the generalizability of NeRF-MLP. Furthermore, we subtly connect the
detection and NeRF branches through a shared MLP, enabling an efficient
adaptation of NeRF to detection and yielding geometry-aware volumetric
representations for 3D detection. Our method outperforms state-of-the-arts by
3.9 mAP and 3.1 mAP on the ScanNet and ARKITScenes benchmarks, respectively. We
provide extensive analysis to shed light on how NeRF-Det works. As a result of
our joint-training design, NeRF-Det is able to generalize well to unseen scenes
for object detection, view synthesis, and depth estimation tasks without
requiring per-scene optimization. Code is available at
\url{https://github.com/facebookresearch/NeRF-Det}.

Comments:
- Accepted by ICCV 2023

---

## MARS: An Instance-aware, Modular and Realistic Simulator for Autonomous  Driving

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-27 | Zirui Wu, Tianyu Liu, Liyi Luo, Zhide Zhong, Jianteng Chen, Hongmin Xiao, Chao Hou, Haozhe Lou, Yuantao Chen, Runyi Yang, Yuxin Huang, Xiaoyu Ye, Zike Yan, Yongliang Shi, Yiyi Liao, Hao Zhao | cs.CV | [PDF](http://arxiv.org/pdf/2307.15058v1){: .btn .btn-green } |

**Abstract**: Nowadays, autonomous cars can drive smoothly in ordinary cases, and it is
widely recognized that realistic sensor simulation will play a critical role in
solving remaining corner cases by simulating them. To this end, we propose an
autonomous driving simulator based upon neural radiance fields (NeRFs).
Compared with existing works, ours has three notable features: (1)
Instance-aware. Our simulator models the foreground instances and background
environments separately with independent networks so that the static (e.g.,
size and appearance) and dynamic (e.g., trajectory) properties of instances can
be controlled separately. (2) Modular. Our simulator allows flexible switching
between different modern NeRF-related backbones, sampling strategies, input
modalities, etc. We expect this modular design to boost academic progress and
industrial deployment of NeRF-based autonomous driving simulation. (3)
Realistic. Our simulator set new state-of-the-art photo-realism results given
the best module selection. Our simulator will be open-sourced while most of our
counterparts are not. Project page: https://open-air-sun.github.io/mars/.

Comments:
- CICAI 2023, project page with code:
  https://open-air-sun.github.io/mars/

---

## Seal-3D: Interactive Pixel-Level Editing for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-27 | Xiangyu Wang, Jingsen Zhu, Qi Ye, Yuchi Huo, Yunlong Ran, Zhihua Zhong, Jiming Chen | cs.CV | [PDF](http://arxiv.org/pdf/2307.15131v2){: .btn .btn-green } |

**Abstract**: With the popularity of implicit neural representations, or neural radiance
fields (NeRF), there is a pressing need for editing methods to interact with
the implicit 3D models for tasks like post-processing reconstructed scenes and
3D content creation. While previous works have explored NeRF editing from
various perspectives, they are restricted in editing flexibility, quality, and
speed, failing to offer direct editing response and instant preview. The key
challenge is to conceive a locally editable neural representation that can
directly reflect the editing instructions and update instantly. To bridge the
gap, we propose a new interactive editing method and system for implicit
representations, called Seal-3D, which allows users to edit NeRF models in a
pixel-level and free manner with a wide range of NeRF-like backbone and preview
the editing effects instantly. To achieve the effects, the challenges are
addressed by our proposed proxy function mapping the editing instructions to
the original space of NeRF models in the teacher model and a two-stage training
strategy for the student model with local pretraining and global finetuning. A
NeRF editing system is built to showcase various editing types. Our system can
achieve compelling editing effects with an interactive speed of about 1 second.

Comments:
- Accepted by ICCV2023. Project Page:
  https://windingwind.github.io/seal-3d/ Code:
  https://github.com/windingwind/seal-3d/

---

## Points-to-3D: Bridging the Gap between Sparse Points and  Shape-Controllable Text-to-3D Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-26 | Chaohui Yu, Qiang Zhou, Jingliang Li, Zhe Zhang, Zhibin Wang, Fan Wang | cs.CV | [PDF](http://arxiv.org/pdf/2307.13908v1){: .btn .btn-green } |

**Abstract**: Text-to-3D generation has recently garnered significant attention, fueled by
2D diffusion models trained on billions of image-text pairs. Existing methods
primarily rely on score distillation to leverage the 2D diffusion priors to
supervise the generation of 3D models, e.g., NeRF. However, score distillation
is prone to suffer the view inconsistency problem, and implicit NeRF modeling
can also lead to an arbitrary shape, thus leading to less realistic and
uncontrollable 3D generation. In this work, we propose a flexible framework of
Points-to-3D to bridge the gap between sparse yet freely available 3D points
and realistic shape-controllable 3D generation by distilling the knowledge from
both 2D and 3D diffusion models. The core idea of Points-to-3D is to introduce
controllable sparse 3D points to guide the text-to-3D generation. Specifically,
we use the sparse point cloud generated from the 3D diffusion model, Point-E,
as the geometric prior, conditioned on a single reference image. To better
utilize the sparse 3D points, we propose an efficient point cloud guidance loss
to adaptively drive the NeRF's geometry to align with the shape of the sparse
3D points. In addition to controlling the geometry, we propose to optimize the
NeRF for a more view-consistent appearance. To be specific, we perform score
distillation to the publicly available 2D image diffusion model ControlNet,
conditioned on text as well as depth map of the learned compact geometry.
Qualitative and quantitative comparisons demonstrate that Points-to-3D improves
view consistency and achieves good shape controllability for text-to-3D
generation. Points-to-3D provides users with a new way to improve and control
text-to-3D generation.

Comments:
- Accepted by ACMMM 2023

---

## Dyn-E: Local Appearance Editing of Dynamic Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-24 | Shangzhan Zhang, Sida Peng, Yinji ShenTu, Qing Shuai, Tianrun Chen, Kaicheng Yu, Hujun Bao, Xiaowei Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2307.12909v1){: .btn .btn-green } |

**Abstract**: Recently, the editing of neural radiance fields (NeRFs) has gained
considerable attention, but most prior works focus on static scenes while
research on the appearance editing of dynamic scenes is relatively lacking. In
this paper, we propose a novel framework to edit the local appearance of
dynamic NeRFs by manipulating pixels in a single frame of training video.
Specifically, to locally edit the appearance of dynamic NeRFs while preserving
unedited regions, we introduce a local surface representation of the edited
region, which can be inserted into and rendered along with the original NeRF
and warped to arbitrary other frames through a learned invertible motion
representation network. By employing our method, users without professional
expertise can easily add desired content to the appearance of a dynamic scene.
We extensively evaluate our approach on various scenes and show that our
approach achieves spatially and temporally consistent editing results. Notably,
our approach is versatile and applicable to different variants of dynamic NeRF
representations.

Comments:
- project page: https://dyn-e.github.io/

---

## CarPatch: A Synthetic Benchmark for Radiance Field Evaluation on Vehicle  Components

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-24 | Davide Di Nucci, Alessandro Simoni, Matteo Tomei, Luca Ciuffreda, Roberto Vezzani, Rita Cucchiara | cs.CV | [PDF](http://arxiv.org/pdf/2307.12718v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have gained widespread recognition as a highly
effective technique for representing 3D reconstructions of objects and scenes
derived from sets of images. Despite their efficiency, NeRF models can pose
challenges in certain scenarios such as vehicle inspection, where the lack of
sufficient data or the presence of challenging elements (e.g. reflections)
strongly impact the accuracy of the reconstruction. To this aim, we introduce
CarPatch, a novel synthetic benchmark of vehicles. In addition to a set of
images annotated with their intrinsic and extrinsic camera parameters, the
corresponding depth maps and semantic segmentation masks have been generated
for each view. Global and part-based metrics have been defined and used to
evaluate, compare, and better characterize some state-of-the-art techniques.
The dataset is publicly released at
https://aimagelab.ing.unimore.it/go/carpatch and can be used as an evaluation
guide and as a baseline for future work on this challenging topic.

Comments:
- Accepted at ICIAP2023

---

## TransHuman: A Transformer-based Human Representation for Generalizable  Neural Human Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-23 | Xiao Pan, Zongxin Yang, Jianxin Ma, Chang Zhou, Yi Yang | cs.CV | [PDF](http://arxiv.org/pdf/2307.12291v1){: .btn .btn-green } |

**Abstract**: In this paper, we focus on the task of generalizable neural human rendering
which trains conditional Neural Radiance Fields (NeRF) from multi-view videos
of different characters. To handle the dynamic human motion, previous methods
have primarily used a SparseConvNet (SPC)-based human representation to process
the painted SMPL. However, such SPC-based representation i) optimizes under the
volatile observation space which leads to the pose-misalignment between
training and inference stages, and ii) lacks the global relationships among
human parts that is critical for handling the incomplete painted SMPL. Tackling
these issues, we present a brand-new framework named TransHuman, which learns
the painted SMPL under the canonical space and captures the global
relationships between human parts with transformers. Specifically, TransHuman
is mainly composed of Transformer-based Human Encoding (TransHE), Deformable
Partial Radiance Fields (DPaRF), and Fine-grained Detail Integration (FDI).
TransHE first processes the painted SMPL under the canonical space via
transformers for capturing the global relationships between human parts. Then,
DPaRF binds each output token with a deformable radiance field for encoding the
query point under the observation space. Finally, the FDI is employed to
further integrate fine-grained information from reference images. Extensive
experiments on ZJU-MoCap and H36M show that our TransHuman achieves a
significantly new state-of-the-art performance with high efficiency. Project
page: https://pansanity666.github.io/TransHuman/

Comments:
- Accepted by ICCV 2023

---

## CopyRNeRF: Protecting the CopyRight of Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-21 | Ziyuan Luo, Qing Guo, Ka Chun Cheung, Simon See, Renjie Wan | cs.CV | [PDF](http://arxiv.org/pdf/2307.11526v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have the potential to be a major representation
of media. Since training a NeRF has never been an easy task, the protection of
its model copyright should be a priority. In this paper, by analyzing the pros
and cons of possible copyright protection solutions, we propose to protect the
copyright of NeRF models by replacing the original color representation in NeRF
with a watermarked color representation. Then, a distortion-resistant rendering
scheme is designed to guarantee robust message extraction in 2D renderings of
NeRF. Our proposed method can directly protect the copyright of NeRF models
while maintaining high rendering quality and bit accuracy when compared among
optional solutions.

Comments:
- 11 pages, 6 figures, accepted by ICCV 2023 non-camera-ready version

---

## FaceCLIPNeRF: Text-driven 3D Face Manipulation using Deformable Neural  Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-21 | Sungwon Hwang, Junha Hyung, Daejin Kim, Min-Jung Kim, Jaegul Choo | cs.CV | [PDF](http://arxiv.org/pdf/2307.11418v3){: .btn .btn-green } |

**Abstract**: As recent advances in Neural Radiance Fields (NeRF) have enabled
high-fidelity 3D face reconstruction and novel view synthesis, its manipulation
also became an essential task in 3D vision. However, existing manipulation
methods require extensive human labor, such as a user-provided semantic mask
and manual attribute search unsuitable for non-expert users. Instead, our
approach is designed to require a single text to manipulate a face
reconstructed with NeRF. To do so, we first train a scene manipulator, a latent
code-conditional deformable NeRF, over a dynamic scene to control a face
deformation using the latent code. However, representing a scene deformation
with a single latent code is unfavorable for compositing local deformations
observed in different instances. As so, our proposed Position-conditional
Anchor Compositor (PAC) learns to represent a manipulated scene with spatially
varying latent codes. Their renderings with the scene manipulator are then
optimized to yield high cosine similarity to a target text in CLIP embedding
space for text-driven manipulation. To the best of our knowledge, our approach
is the first to address the text-driven manipulation of a face reconstructed
with NeRF. Extensive results, comparisons, and ablation studies demonstrate the
effectiveness of our approach.

Comments:
- ICCV 2023 project page at https://faceclipnerf.github.io

---

## Tri-MipRF: Tri-Mip Representation for Efficient Anti-Aliasing Neural  Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-21 | Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao, Xiao Liu, Yuewen Ma | cs.CV | [PDF](http://arxiv.org/pdf/2307.11335v1){: .btn .btn-green } |

**Abstract**: Despite the tremendous progress in neural radiance fields (NeRF), we still
face a dilemma of the trade-off between quality and efficiency, e.g., MipNeRF
presents fine-detailed and anti-aliased renderings but takes days for training,
while Instant-ngp can accomplish the reconstruction in a few minutes but
suffers from blurring or aliasing when rendering at various distances or
resolutions due to ignoring the sampling area. To this end, we propose a novel
Tri-Mip encoding that enables both instant reconstruction and anti-aliased
high-fidelity rendering for neural radiance fields. The key is to factorize the
pre-filtered 3D feature spaces in three orthogonal mipmaps. In this way, we can
efficiently perform 3D area sampling by taking advantage of 2D pre-filtered
feature maps, which significantly elevates the rendering quality without
sacrificing efficiency. To cope with the novel Tri-Mip representation, we
propose a cone-casting rendering technique to efficiently sample anti-aliased
3D features with the Tri-Mip encoding considering both pixel imaging and
observing distance. Extensive experiments on both synthetic and real-world
datasets demonstrate our method achieves state-of-the-art rendering quality and
reconstruction speed while maintaining a compact representation that reduces
25% model size compared against Instant-ngp.

Comments:
- Accepted to ICCV 2023 Project page:
  https://wbhu.github.io/projects/Tri-MipRF

---

## Urban Radiance Field Representation with Deformable Neural Mesh  Primitives

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-20 | Fan Lu, Yan Xu, Guang Chen, Hongsheng Li, Kwan-Yee Lin, Changjun Jiang | cs.CV | [PDF](http://arxiv.org/pdf/2307.10776v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have achieved great success in the past few
years. However, most current methods still require intensive resources due to
ray marching-based rendering. To construct urban-level radiance fields
efficiently, we design Deformable Neural Mesh Primitive~(DNMP), and propose to
parameterize the entire scene with such primitives. The DNMP is a flexible and
compact neural variant of classic mesh representation, which enjoys both the
efficiency of rasterization-based rendering and the powerful neural
representation capability for photo-realistic image synthesis. Specifically, a
DNMP consists of a set of connected deformable mesh vertices with paired vertex
features to parameterize the geometry and radiance information of a local area.
To constrain the degree of freedom for optimization and lower the storage
budgets, we enforce the shape of each primitive to be decoded from a relatively
low-dimensional latent space. The rendering colors are decoded from the vertex
features (interpolated with rasterization) by a view-dependent MLP. The DNMP
provides a new paradigm for urban-level scene representation with appealing
properties: $(1)$ High-quality rendering. Our method achieves leading
performance for novel view synthesis in urban scenarios. $(2)$ Low
computational costs. Our representation enables fast rendering (2.07ms/1k
pixels) and low peak memory usage (110MB/1k pixels). We also present a
lightweight version that can run 33$\times$ faster than vanilla NeRFs, and
comparable to the highly-optimized Instant-NGP (0.61 vs 0.71ms/1k pixels).
Project page: \href{https://dnmp.github.io/}{https://dnmp.github.io/}.

Comments:
- Accepted to ICCV2023

---

## Lighting up NeRF via Unsupervised Decomposition and Enhancement

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-20 | Haoyuan Wang, Xiaogang Xu, Ke Xu, Rynson WH. Lau | cs.CV | [PDF](http://arxiv.org/pdf/2307.10664v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) is a promising approach for synthesizing novel
views, given a set of images and the corresponding camera poses of a scene.
However, images photographed from a low-light scene can hardly be used to train
a NeRF model to produce high-quality results, due to their low pixel
intensities, heavy noise, and color distortion. Combining existing low-light
image enhancement methods with NeRF methods also does not work well due to the
view inconsistency caused by the individual 2D enhancement process. In this
paper, we propose a novel approach, called Low-Light NeRF (or LLNeRF), to
enhance the scene representation and synthesize normal-light novel views
directly from sRGB low-light images in an unsupervised manner. The core of our
approach is a decomposition of radiance field learning, which allows us to
enhance the illumination, reduce noise and correct the distorted colors jointly
with the NeRF optimization process. Our method is able to produce novel view
images with proper lighting and vivid colors and details, given a collection of
camera-finished low dynamic range (8-bits/channel) images from a low-light
scene. Experiments demonstrate that our method outperforms existing low-light
enhancement methods and NeRF methods.

Comments:
- ICCV 2023. Project website: https://whyy.site/paper/llnerf

---

## Magic NeRF Lens: Interactive Fusion of Neural Radiance Fields for  Virtual Facility Inspection

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-19 | Ke Li, Susanne Schmidt, Tim Rolff, Reinhard Bacher, Wim Leemans, Frank Steinicke | cs.GR | [PDF](http://arxiv.org/pdf/2307.09860v1){: .btn .btn-green } |

**Abstract**: Large industrial facilities such as particle accelerators and nuclear power
plants are critical infrastructures for scientific research and industrial
processes. These facilities are complex systems that not only require regular
maintenance and upgrades but are often inaccessible to humans due to various
safety hazards. Therefore, a virtual reality (VR) system that can quickly
replicate real-world remote environments to provide users with a high level of
spatial and situational awareness is crucial for facility maintenance planning.
However, the exact 3D shapes of these facilities are often too complex to be
accurately modeled with geometric primitives through the traditional
rasterization pipeline.
  In this work, we develop Magic NeRF Lens, an interactive framework to support
facility inspection in immersive VR using neural radiance fields (NeRF) and
volumetric rendering. We introduce a novel data fusion approach that combines
the complementary strengths of volumetric rendering and geometric
rasterization, allowing a NeRF model to be merged with other conventional 3D
data, such as a computer-aided design model. We develop two novel 3D magic lens
effects to optimize NeRF rendering by exploiting the properties of human vision
and context-aware visualization. We demonstrate the high usability of our
framework and methods through a technical benchmark, a visual search user
study, and expert reviews. In addition, the source code of our VR NeRF
framework is made publicly available for future research and development.

Comments:
- This work has been submitted to the IEEE TVCG for possible
  publication. Copyright may be transferred without notice, after which this
  version may no longer be accessible

---

## Efficient Region-Aware Neural Radiance Fields for High-Fidelity Talking  Portrait Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-18 | Jiahe Li, Jiawei Zhang, Xiao Bai, Jun Zhou, Lin Gu | cs.CV | [PDF](http://arxiv.org/pdf/2307.09323v2){: .btn .btn-green } |

**Abstract**: This paper presents ER-NeRF, a novel conditional Neural Radiance Fields
(NeRF) based architecture for talking portrait synthesis that can concurrently
achieve fast convergence, real-time rendering, and state-of-the-art performance
with small model size. Our idea is to explicitly exploit the unequal
contribution of spatial regions to guide talking portrait modeling.
Specifically, to improve the accuracy of dynamic head reconstruction, a compact
and expressive NeRF-based Tri-Plane Hash Representation is introduced by
pruning empty spatial regions with three planar hash encoders. For speech
audio, we propose a Region Attention Module to generate region-aware condition
feature via an attention mechanism. Different from existing methods that
utilize an MLP-based encoder to learn the cross-modal relation implicitly, the
attention mechanism builds an explicit connection between audio features and
spatial regions to capture the priors of local motions. Moreover, a direct and
fast Adaptive Pose Encoding is introduced to optimize the head-torso separation
problem by mapping the complex transformation of the head pose into spatial
coordinates. Extensive experiments demonstrate that our method renders better
high-fidelity and audio-lips synchronized talking portrait videos, with
realistic details and high efficiency compared to previous methods.

Comments:
- Accepted by ICCV 2023. Project page:
  https://fictionarry.github.io/ER-NeRF/

---

## OPHAvatars: One-shot Photo-realistic Head Avatars



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-18 | Shaoxu Li | cs.CV | [PDF](http://arxiv.org/pdf/2307.09153v2){: .btn .btn-green } |

**Abstract**: We propose a method for synthesizing photo-realistic digital avatars from
only one portrait as the reference. Given a portrait, our method synthesizes a
coarse talking head video using driving keypoints features. And with the coarse
video, our method synthesizes a coarse talking head avatar with a deforming
neural radiance field. With rendered images of the coarse avatar, our method
updates the low-quality images with a blind face restoration model. With
updated images, we retrain the avatar for higher quality. After several
iterations, our method can synthesize a photo-realistic animatable 3D neural
head avatar. The motivation of our method is deformable neural radiance field
can eliminate the unnatural distortion caused by the image2video method. Our
method outperforms state-of-the-art methods in quantitative and qualitative
studies on various subjects.

Comments:
- code: https://github.com/lsx0101/OPHAvatars

---

## PixelHuman: Animatable Neural Radiance Fields from Few Images



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-18 | Gyumin Shim, Jaeseong Lee, Junha Hyung, Jaegul Choo | cs.CV | [PDF](http://arxiv.org/pdf/2307.09070v1){: .btn .btn-green } |

**Abstract**: In this paper, we propose PixelHuman, a novel human rendering model that
generates animatable human scenes from a few images of a person with unseen
identity, views, and poses. Previous work have demonstrated reasonable
performance in novel view and pose synthesis, but they rely on a large number
of images to train and are trained per scene from videos, which requires
significant amount of time to produce animatable scenes from unseen human
images. Our method differs from existing methods in that it can generalize to
any input image for animatable human synthesis. Given a random pose sequence,
our method synthesizes each target scene using a neural radiance field that is
conditioned on a canonical representation and pose-aware pixel-aligned
features, both of which can be obtained through deformation fields learned in a
data-driven manner. Our experiments show that our method achieves
state-of-the-art performance in multiview and novel pose synthesis from
few-shot images.

Comments:
- 8 pages

---

## Cross-Ray Neural Radiance Fields for Novel-view Synthesis from  Unconstrained Image Collections

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-16 | Yifan Yang, Shuhai Zhang, Zixiong Huang, Yubing Zhang, Mingkui Tan | cs.CV | [PDF](http://arxiv.org/pdf/2307.08093v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) is a revolutionary approach for rendering
scenes by sampling a single ray per pixel and it has demonstrated impressive
capabilities in novel-view synthesis from static scene images. However, in
practice, we usually need to recover NeRF from unconstrained image collections,
which poses two challenges: 1) the images often have dynamic changes in
appearance because of different capturing time and camera settings; 2) the
images may contain transient objects such as humans and cars, leading to
occlusion and ghosting artifacts. Conventional approaches seek to address these
challenges by locally utilizing a single ray to synthesize a color of a pixel.
In contrast, humans typically perceive appearance and objects by globally
utilizing information across multiple pixels. To mimic the perception process
of humans, in this paper, we propose Cross-Ray NeRF (CR-NeRF) that leverages
interactive information across multiple rays to synthesize occlusion-free novel
views with the same appearances as the images. Specifically, to model varying
appearances, we first propose to represent multiple rays with a novel cross-ray
feature and then recover the appearance by fusing global statistics, i.e.,
feature covariance of the rays and the image appearance. Moreover, to avoid
occlusion introduced by transient objects, we propose a transient objects
handler and introduce a grid sampling strategy for masking out the transient
objects. We theoretically find that leveraging correlation across multiple rays
promotes capturing more global information. Moreover, extensive experimental
results on large real-world datasets verify the effectiveness of CR-NeRF.

Comments:
- ICCV 2023 Oral

---

## Improving NeRF with Height Data for Utilization of GIS Data

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-15 | Hinata Aoki, Takao Yamanaka | cs.CV | [PDF](http://arxiv.org/pdf/2307.07729v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) has been applied to various tasks related to
representations of 3D scenes. Most studies based on NeRF have focused on a
small object, while a few studies have tried to reconstruct large-scale scenes
although these methods tend to require large computational cost. For the
application of NeRF to large-scale scenes, a method based on NeRF is proposed
in this paper to effectively use height data which can be obtained from GIS
(Geographic Information System). For this purpose, the scene space was divided
into multiple objects and a background using the height data to represent them
with separate neural networks. In addition, an adaptive sampling method is also
proposed by using the height data. As a result, the accuracy of image rendering
was improved with faster training speed.

Comments:
- ICIP2023

---

## Transient Neural Radiance Fields for Lidar View Synthesis and 3D  Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-14 | Anagh Malik, Parsa Mirdehghan, Sotiris Nousias, Kiriakos N. Kutulakos, David B. Lindell | cs.CV | [PDF](http://arxiv.org/pdf/2307.09555v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) have become a ubiquitous tool for modeling
scene appearance and geometry from multiview imagery. Recent work has also
begun to explore how to use additional supervision from lidar or depth sensor
measurements in the NeRF framework. However, previous lidar-supervised NeRFs
focus on rendering conventional camera imagery and use lidar-derived point
cloud data as auxiliary supervision; thus, they fail to incorporate the
underlying image formation model of the lidar. Here, we propose a novel method
for rendering transient NeRFs that take as input the raw, time-resolved photon
count histograms measured by a single-photon lidar system, and we seek to
render such histograms from novel views. Different from conventional NeRFs, the
approach relies on a time-resolved version of the volume rendering equation to
render the lidar measurements and capture transient light transport phenomena
at picosecond timescales. We evaluate our method on a first-of-its-kind dataset
of simulated and captured transient multiview scans from a prototype
single-photon lidar. Overall, our work brings NeRFs to a new dimension of
imaging at transient timescales, newly enabling rendering of transient imagery
from novel views. Additionally, we show that our approach recovers improved
geometry and conventional appearance compared to point cloud-based supervision
when training on few input viewpoints. Transient NeRFs may be especially useful
for applications which seek to simulate raw lidar measurements for downstream
tasks in autonomous driving, robotics, and remote sensing.

---

## CeRF: Convolutional Neural Radiance Fields for New View Synthesis with  Derivatives of Ray Modeling



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-14 | Xiaoyan Yang, Dingbo Lu, Yang Li, Chenhui Li, Changbo Wang | cs.CV | [PDF](http://arxiv.org/pdf/2307.07125v2){: .btn .btn-green } |

**Abstract**: In recent years, novel view synthesis has gained popularity in generating
high-fidelity images. While demonstrating superior performance in the task of
synthesizing novel views, the majority of these methods are still based on the
conventional multi-layer perceptron for scene embedding. Furthermore, light
field models suffer from geometric blurring during pixel rendering, while
radiance field-based volume rendering methods have multiple solutions for a
certain target of density distribution integration. To address these issues, we
introduce the Convolutional Neural Radiance Fields to model the derivatives of
radiance along rays. Based on 1D convolutional operations, our proposed method
effectively extracts potential ray representations through a structured neural
network architecture. Besides, with the proposed ray modeling, a proposed
recurrent module is employed to solve geometric ambiguity in the fully neural
rendering process. Extensive experiments demonstrate the promising results of
our proposed model compared with existing state-of-the-art methods.

Comments:
- 16 pages, 11 figures

---

## SAR-NeRF: Neural Radiance Fields for Synthetic Aperture Radar Multi-View  Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-11 | Zhengxin Lei, Feng Xu, Jiangtao Wei, Feng Cai, Feng Wang, Ya-Qiu Jin | cs.CV | [PDF](http://arxiv.org/pdf/2307.05087v1){: .btn .btn-green } |

**Abstract**: SAR images are highly sensitive to observation configurations, and they
exhibit significant variations across different viewing angles, making it
challenging to represent and learn their anisotropic features. As a result,
deep learning methods often generalize poorly across different view angles.
Inspired by the concept of neural radiance fields (NeRF), this study combines
SAR imaging mechanisms with neural networks to propose a novel NeRF model for
SAR image generation. Following the mapping and projection pinciples, a set of
SAR images is modeled implicitly as a function of attenuation coefficients and
scattering intensities in the 3D imaging space through a differentiable
rendering equation. SAR-NeRF is then constructed to learn the distribution of
attenuation coefficients and scattering intensities of voxels, where the
vectorized form of 3D voxel SAR rendering equation and the sampling
relationship between the 3D space voxels and the 2D view ray grids are
analytically derived. Through quantitative experiments on various datasets, we
thoroughly assess the multi-view representation and generalization capabilities
of SAR-NeRF. Additionally, it is found that SAR-NeRF augumented dataset can
significantly improve SAR target classification performance under few-shot
learning setup, where a 10-type classification accuracy of 91.6\% can be
achieved by using only 12 images per class.

---

## RGB-D Mapping and Tracking in a Plenoxel Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-07 | Andreas L. Teigen, Yeonsoo Park, Annette Stahl, Rudolf Mester | cs.CV | [PDF](http://arxiv.org/pdf/2307.03404v2){: .btn .btn-green } |

**Abstract**: The widespread adoption of Neural Radiance Fields (NeRFs) have ensured
significant advances in the domain of novel view synthesis in recent years.
These models capture a volumetric radiance field of a scene, creating highly
convincing, dense, photorealistic models through the use of simple,
differentiable rendering equations. Despite their popularity, these algorithms
suffer from severe ambiguities in visual data inherent to the RGB sensor, which
means that although images generated with view synthesis can visually appear
very believable, the underlying 3D model will often be wrong. This considerably
limits the usefulness of these models in practical applications like Robotics
and Extended Reality (XR), where an accurate dense 3D reconstruction otherwise
would be of significant value. In this paper, we present the vital differences
between view synthesis models and 3D reconstruction models. We also comment on
why a depth sensor is essential for modeling accurate geometry in general
outward-facing scenes using the current paradigm of novel view synthesis
methods. Focusing on the structure-from-motion task, we practically demonstrate
this need by extending the Plenoxel radiance field model: Presenting an
analytical differential approach for dense mapping and tracking with radiance
fields based on RGB-D data without a neural network. Our method achieves
state-of-the-art results in both mapping and tracking tasks, while also being
faster than competing neural network-based approaches. The code is available
at: https://github.com/ysus33/RGB-D_Plenoxel_Mapping_Tracking.git

Comments:
- 2024 IEEE/CVF Winter Conference on Applications of Computer Vision
  (WACV) *The first two authors contributed equally to this paper

---

## NOFA: NeRF-based One-shot Facial Avatar Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-07-07 | Wangbo Yu, Yanbo Fan, Yong Zhang, Xuan Wang, Fei Yin, Yunpeng Bai, Yan-Pei Cao, Ying Shan, Yang Wu, Zhongqian Sun, Baoyuan Wu | cs.CV | [PDF](http://arxiv.org/pdf/2307.03441v1){: .btn .btn-green } |

**Abstract**: 3D facial avatar reconstruction has been a significant research topic in
computer graphics and computer vision, where photo-realistic rendering and
flexible controls over poses and expressions are necessary for many related
applications. Recently, its performance has been greatly improved with the
development of neural radiance fields (NeRF). However, most existing NeRF-based
facial avatars focus on subject-specific reconstruction and reenactment,
requiring multi-shot images containing different views of the specific subject
for training, and the learned model cannot generalize to new identities,
limiting its further applications. In this work, we propose a one-shot 3D
facial avatar reconstruction framework that only requires a single source image
to reconstruct a high-fidelity 3D facial avatar. For the challenges of lacking
generalization ability and missing multi-view information, we leverage the
generative prior of 3D GAN and develop an efficient encoder-decoder network to
reconstruct the canonical neural volume of the source image, and further
propose a compensation network to complement facial details. To enable
fine-grained control over facial dynamics, we propose a deformation field to
warp the canonical volume into driven expressions. Through extensive
experimental comparisons, we achieve superior synthesis results compared to
several state-of-the-art methods.