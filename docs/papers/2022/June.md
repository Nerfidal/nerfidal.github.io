---
layout: default
title: June
parent: 2022
nav_order: 6
---
<!---metadata--->

## Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in  Robotic Surgery

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-30 | Yuehao Wang, Yonghao Long, Siu Hin Fan, Qi Dou | cs.CV | [PDF](http://arxiv.org/pdf/2206.15255v1){: .btn .btn-green } |

**Abstract**: Reconstruction of the soft tissues in robotic surgery from endoscopic stereo
videos is important for many applications such as intra-operative navigation
and image-guided robotic surgery automation. Previous works on this task mainly
rely on SLAM-based approaches, which struggle to handle complex surgical
scenes. Inspired by recent progress in neural rendering, we present a novel
framework for deformable tissue reconstruction from binocular captures in
robotic surgery under the single-viewpoint setting. Our framework adopts
dynamic neural radiance fields to represent deformable surgical scenes in MLPs
and optimize shapes and deformations in a learning-based manner. In addition to
non-rigid deformations, tool occlusion and poor 3D clues from a single
viewpoint are also particular challenges in soft tissue reconstruction. To
overcome these difficulties, we present a series of strategies of tool
mask-guided ray casting, stereo depth-cueing ray marching and stereo
depth-supervised optimization. With experiments on DaVinci robotic surgery
videos, our method significantly outperforms the current state-of-the-art
reconstruction method for handling various complex non-rigid deformations. To
our best knowledge, this is the first work leveraging neural rendering for
surgical scene 3D reconstruction with remarkable potential demonstrated. Code
is available at: https://github.com/med-air/EndoNeRF.

Comments:
- 11 pages, 4 figures, conference

---

## Regularization of NeRFs using differential geometry

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-29 | Thibaud Ehret, Roger Marí, Gabriele Facciolo | cs.CV | [PDF](http://arxiv.org/pdf/2206.14938v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields, or NeRF, represent a breakthrough in the field of
novel view synthesis and 3D modeling of complex scenes from multi-view image
collections. Numerous recent works have shown the importance of making NeRF
models more robust, by means of regularization, in order to train with possibly
inconsistent and/or very sparse data. In this work, we explore how differential
geometry can provide elegant regularization tools for robustly training
NeRF-like models, which are modified so as to represent continuous and
infinitely differentiable functions. In particular, we present a generic
framework for regularizing different types of NeRFs observations to improve the
performance in challenging conditions. We also show how the same formalism can
also be used to natively encourage the regularity of surfaces by means of
Gaussian or mean curvatures.

---

## Ev-NeRF: Event Based Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-24 | Inwoo Hwang, Junho Kim, Young Min Kim | cs.CV | [PDF](http://arxiv.org/pdf/2206.12455v2){: .btn .btn-green } |

**Abstract**: We present Ev-NeRF, a Neural Radiance Field derived from event data. While
event cameras can measure subtle brightness changes in high frame rates, the
measurements in low lighting or extreme motion suffer from significant domain
discrepancy with complex noise. As a result, the performance of event-based
vision tasks does not transfer to challenging environments, where the event
cameras are expected to thrive over normal cameras. We find that the multi-view
consistency of NeRF provides a powerful self-supervision signal for eliminating
the spurious measurements and extracting the consistent underlying structure
despite highly noisy input. Instead of posed images of the original NeRF, the
input to Ev-NeRF is the event measurements accompanied by the movements of the
sensors. Using the loss function that reflects the measurement model of the
sensor, Ev-NeRF creates an integrated neural volume that summarizes the
unstructured and sparse data points captured for about 2-4 seconds. The
generated neural volume can also produce intensity images from novel views with
reasonable depth estimates, which can serve as a high-quality input to various
vision-based tasks. Our results show that Ev-NeRF achieves competitive
performance for intensity image reconstruction under extreme noise conditions
and high-dynamic-range imaging.

Comments:
- Accepted to WACV 2023

---

## UNeRF: Time and Memory Conscious U-Shaped Network for Training Neural  Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-23 | Abiramy Kuganesan, Shih-yang Su, James J. Little, Helge Rhodin | cs.CV | [PDF](http://arxiv.org/pdf/2206.11952v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) increase reconstruction detail for novel view
synthesis and scene reconstruction, with applications ranging from large static
scenes to dynamic human motion. However, the increased resolution and
model-free nature of such neural fields come at the cost of high training times
and excessive memory requirements. Recent advances improve the inference time
by using complementary data structures yet these methods are ill-suited for
dynamic scenes and often increase memory consumption. Little has been done to
reduce the resources required at training time. We propose a method to exploit
the redundancy of NeRF's sample-based computations by partially sharing
evaluations across neighboring sample points. Our UNeRF architecture is
inspired by the UNet, where spatial resolution is reduced in the middle of the
network and information is shared between adjacent samples. Although this
change violates the strict and conscious separation of view-dependent
appearance and view-independent density estimation in the NeRF method, we show
that it improves novel view synthesis. We also introduce an alternative
subsampling strategy which shares computation while minimizing any violation of
view invariance. UNeRF is a plug-in module for the original NeRF network. Our
major contributions include reduction of the memory footprint, improved
accuracy, and reduced amortized processing time both during training and
inference. With only weak assumptions on locality, we achieve improved resource
utilization on a variety of neural radiance fields tasks. We demonstrate
applications to the novel view synthesis of static scenes as well as dynamic
human shape and motion.

---

## EventNeRF: Neural Radiance Fields from a Single Colour Event Camera

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-23 | Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, Vladislav Golyanik | cs.CV | [PDF](http://arxiv.org/pdf/2206.11896v3){: .btn .btn-green } |

**Abstract**: Asynchronously operating event cameras find many applications due to their
high dynamic range, vanishingly low motion blur, low latency and low data
bandwidth. The field saw remarkable progress during the last few years, and
existing event-based 3D reconstruction approaches recover sparse point clouds
of the scene. However, such sparsity is a limiting factor in many cases,
especially in computer vision and graphics, that has not been addressed
satisfactorily so far. Accordingly, this paper proposes the first approach for
3D-consistent, dense and photorealistic novel view synthesis using just a
single colour event stream as input. At its core is a neural radiance field
trained entirely in a self-supervised manner from events while preserving the
original resolution of the colour event channels. Next, our ray sampling
strategy is tailored to events and allows for data-efficient training. At test,
our method produces results in the RGB space at unprecedented quality. We
evaluate our method qualitatively and numerically on several challenging
synthetic and real scenes and show that it produces significantly denser and
more visually appealing renderings than the existing methods. We also
demonstrate robustness in challenging scenarios with fast motion and under low
lighting conditions. We release the newly recorded dataset and our source code
to facilitate the research field, see https://4dqv.mpi-inf.mpg.de/EventNeRF.

Comments:
- 19 pages, 21 figures, 3 tables; CVPR 2023

---

## KiloNeuS: A Versatile Neural Implicit Surface Representation for  Real-Time Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-22 | Stefano Esposito, Daniele Baieri, Stefan Zellmann, André Hinkenjann, Emanuele Rodolà | cs.CV | [PDF](http://arxiv.org/pdf/2206.10885v2){: .btn .btn-green } |

**Abstract**: NeRF-based techniques fit wide and deep multi-layer perceptrons (MLPs) to a
continuous radiance field that can be rendered from any unseen viewpoint.
However, the lack of surface and normals definition and high rendering times
limit their usage in typical computer graphics applications. Such limitations
have recently been overcome separately, but solving them together remains an
open problem. We present KiloNeuS, a neural representation reconstructing an
implicit surface represented as a signed distance function (SDF) from
multi-view images and enabling real-time rendering by partitioning the space
into thousands of tiny MLPs fast to inference. As we learn the implicit surface
locally using independent models, resulting in a globally coherent geometry is
non-trivial and needs to be addressed during training. We evaluate rendering
performance on a GPU-accelerated ray-caster with in-shader neural network
inference, resulting in an average of 46 FPS at high resolution, proving a
satisfying tradeoff between storage costs and rendering quality. In fact, our
evaluation for rendering quality and surface recovery shows that KiloNeuS
outperforms its single-MLP counterpart. Finally, to exhibit the versatility of
KiloNeuS, we integrate it into an interactive path-tracer taking full advantage
of its surface normals. We consider our work a crucial first step toward
real-time rendering of implicit neural representations under global
illumination.

Comments:
- 9 pages, 8 figures

---

## FWD: Real-time Novel View Synthesis with Forward Warping and Depth

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-16 | Ang Cao, Chris Rockwell, Justin Johnson | cs.CV | [PDF](http://arxiv.org/pdf/2206.08355v3){: .btn .btn-green } |

**Abstract**: Novel view synthesis (NVS) is a challenging task requiring systems to
generate photorealistic images of scenes from new viewpoints, where both
quality and speed are important for applications. Previous image-based
rendering (IBR) methods are fast, but have poor quality when input views are
sparse. Recent Neural Radiance Fields (NeRF) and generalizable variants give
impressive results but are not real-time. In our paper, we propose a
generalizable NVS method with sparse inputs, called FWD, which gives
high-quality synthesis in real-time. With explicit depth and differentiable
rendering, it achieves competitive results to the SOTA methods with 130-1000x
speedup and better perceptual quality. If available, we can seamlessly
integrate sensor depth during either training or inference to improve image
quality while retaining real-time speed. With the growing prevalence of depths
sensors, we hope that methods making use of depth will become increasingly
useful.

Comments:
- CVPR 2022. Project website https://caoang327.github.io/FWD/

---

## Controllable 3D Face Synthesis with Conditional Generative Occupancy  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-16 | Keqiang Sun, Shangzhe Wu, Zhaoyang Huang, Ning Zhang, Quan Wang, HongSheng Li | cs.CV | [PDF](http://arxiv.org/pdf/2206.08361v2){: .btn .btn-green } |

**Abstract**: Capitalizing on the recent advances in image generation models, existing
controllable face image synthesis methods are able to generate high-fidelity
images with some levels of controllability, e.g., controlling the shapes,
expressions, textures, and poses of the generated face images. However, these
methods focus on 2D image generative models, which are prone to producing
inconsistent face images under large expression and pose changes. In this
paper, we propose a new NeRF-based conditional 3D face synthesis framework,
which enables 3D controllability over the generated face images by imposing
explicit 3D conditions from 3D face priors. At its core is a conditional
Generative Occupancy Field (cGOF) that effectively enforces the shape of the
generated face to commit to a given 3D Morphable Model (3DMM) mesh. To achieve
accurate control over fine-grained 3D face shapes of the synthesized image, we
additionally incorporate a 3D landmark loss as well as a volume warping loss
into our synthesis algorithm. Experiments validate the effectiveness of the
proposed method, which is able to generate high-fidelity face images and shows
more precise 3D controllability than state-of-the-art 2D-based controllable
face synthesis methods. Find code and demo at
https://keqiangsun.github.io/projects/cgof.

---

## Neural Deformable Voxel Grid for Fast Optimization of Dynamic View  Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-15 | Xiang Guo, Guanying Chen, Yuchao Dai, Xiaoqing Ye, Jiadai Sun, Xiao Tan, Errui Ding | cs.CV | [PDF](http://arxiv.org/pdf/2206.07698v2){: .btn .btn-green } |

**Abstract**: Recently, Neural Radiance Fields (NeRF) is revolutionizing the task of novel
view synthesis (NVS) for its superior performance. In this paper, we propose to
synthesize dynamic scenes. Extending the methods for static scenes to dynamic
scenes is not straightforward as both the scene geometry and appearance change
over time, especially under monocular setup. Also, the existing dynamic NeRF
methods generally require a lengthy per-scene training procedure, where
multi-layer perceptrons (MLP) are fitted to model both motions and radiance. In
this paper, built on top of the recent advances in voxel-grid optimization, we
propose a fast deformable radiance field method to handle dynamic scenes. Our
method consists of two modules. The first module adopts a deformation grid to
store 3D dynamic features, and a light-weight MLP for decoding the deformation
that maps a 3D point in the observation space to the canonical space using the
interpolated features. The second module contains a density and a color grid to
model the geometry and density of the scene. The occlusion is explicitly
modeled to further improve the rendering quality. Experimental results show
that our method achieves comparable performance to D-NeRF using only 20 minutes
for training, which is more than 70x faster than D-NeRF, clearly demonstrating
the efficiency of our proposed method.

Comments:
- Technical Report: 29 pages; project page:
  https://npucvr.github.io/NDVG

---

## Physics Informed Neural Fields for Smoke Reconstruction with Sparse Data



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-14 | Mengyu Chu, Lingjie Liu, Quan Zheng, Erik Franz, Hans-Peter Seidel, Christian Theobalt, Rhaleb Zayer | cs.GR | [PDF](http://arxiv.org/pdf/2206.06577v1){: .btn .btn-green } |

**Abstract**: High-fidelity reconstruction of fluids from sparse multiview RGB videos
remains a formidable challenge due to the complexity of the underlying physics
as well as complex occlusion and lighting in captures. Existing solutions
either assume knowledge of obstacles and lighting, or only focus on simple
fluid scenes without obstacles or complex lighting, and thus are unsuitable for
real-world scenes with unknown lighting or arbitrary obstacles. We present the
first method to reconstruct dynamic fluid by leveraging the governing physics
(ie, Navier -Stokes equations) in an end-to-end optimization from sparse videos
without taking lighting conditions, geometry information, or boundary
conditions as input. We provide a continuous spatio-temporal scene
representation using neural networks as the ansatz of density and velocity
solution functions for fluids as well as the radiance field for static objects.
With a hybrid architecture that separates static and dynamic contents, fluid
interactions with static obstacles are reconstructed for the first time without
additional geometry input or human labeling. By augmenting time-varying neural
radiance fields with physics-informed deep learning, our method benefits from
the supervision of images and physical priors. To achieve robust optimization
from sparse views, we introduced a layer-by-layer growing strategy to
progressively increase the network capacity. Using progressively growing models
with a new regularization term, we manage to disentangle density-color
ambiguity in radiance fields without overfitting. A pretrained
density-to-velocity fluid model is leveraged in addition as the data prior to
avoid suboptimal velocity which underestimates vorticity but trivially fulfills
physical equations. Our method exhibits high-quality results with relaxed
constraints and strong flexibility on a representative set of synthetic and
real flow captures.

Comments:
- accepted to ACM Transactions On Graphics (SIGGRAPH 2022), further
  info:\url{https://people.mpi-inf.mpg.de/~mchu/projects/PI-NeRF/}

---

## RigNeRF: Fully Controllable Neural 3D Portraits

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-13 | ShahRukh Athar, Zexiang Xu, Kalyan Sunkavalli, Eli Shechtman, Zhixin Shu | cs.CV | [PDF](http://arxiv.org/pdf/2206.06481v1){: .btn .btn-green } |

**Abstract**: Volumetric neural rendering methods, such as neural radiance fields (NeRFs),
have enabled photo-realistic novel view synthesis. However, in their standard
form, NeRFs do not support the editing of objects, such as a human head, within
a scene. In this work, we propose RigNeRF, a system that goes beyond just novel
view synthesis and enables full control of head pose and facial expressions
learned from a single portrait video. We model changes in head pose and facial
expressions using a deformation field that is guided by a 3D morphable face
model (3DMM). The 3DMM effectively acts as a prior for RigNeRF that learns to
predict only residuals to the 3DMM deformations and allows us to render novel
(rigid) poses and (non-rigid) expressions that were not present in the input
sequence. Using only a smartphone-captured short video of a subject for
training, we demonstrate the effectiveness of our method on free view synthesis
of a portrait scene with explicit head pose and expression controls. The
project page can be found here:
http://shahrukhathar.github.io/2022/06/06/RigNeRF.html

Comments:
- The project page can be found here:
  http://shahrukhathar.github.io/2022/06/06/RigNeRF.html

---

## SNeS: Learning Probably Symmetric Neural Surfaces from Incomplete Data

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-13 | Eldar Insafutdinov, Dylan Campbell, João F. Henriques, Andrea Vedaldi | cs.CV | [PDF](http://arxiv.org/pdf/2206.06340v1){: .btn .btn-green } |

**Abstract**: We present a method for the accurate 3D reconstruction of partly-symmetric
objects. We build on the strengths of recent advances in neural reconstruction
and rendering such as Neural Radiance Fields (NeRF). A major shortcoming of
such approaches is that they fail to reconstruct any part of the object which
is not clearly visible in the training image, which is often the case for
in-the-wild images and videos. When evidence is lacking, structural priors such
as symmetry can be used to complete the missing information. However,
exploiting such priors in neural rendering is highly non-trivial: while
geometry and non-reflective materials may be symmetric, shadows and reflections
from the ambient scene are not symmetric in general. To address this, we apply
a soft symmetry constraint to the 3D geometry and material properties, having
factored appearance into lighting, albedo colour and reflectivity. We evaluate
our method on the recently introduced CO3D dataset, focusing on the car
category due to the challenge of reconstructing highly-reflective materials. We
show that it can reconstruct unobserved regions with high fidelity and render
high-quality novel view images.

Comments:
- First two authors contributed equally

---

## AR-NeRF: Unsupervised Learning of Depth and Defocus Effects from Natural  Images with Aperture Rendering Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-13 | Takuhiro Kaneko | cs.CV | [PDF](http://arxiv.org/pdf/2206.06100v1){: .btn .btn-green } |

**Abstract**: Fully unsupervised 3D representation learning has gained attention owing to
its advantages in data collection. A successful approach involves a
viewpoint-aware approach that learns an image distribution based on generative
models (e.g., generative adversarial networks (GANs)) while generating various
view images based on 3D-aware models (e.g., neural radiance fields (NeRFs)).
However, they require images with various views for training, and consequently,
their application to datasets with few or limited viewpoints remains a
challenge. As a complementary approach, an aperture rendering GAN (AR-GAN) that
employs a defocus cue was proposed. However, an AR-GAN is a CNN-based model and
represents a defocus independently from a viewpoint change despite its high
correlation, which is one of the reasons for its performance. As an alternative
to an AR-GAN, we propose an aperture rendering NeRF (AR-NeRF), which can
utilize viewpoint and defocus cues in a unified manner by representing both
factors in a common ray-tracing framework. Moreover, to learn defocus-aware and
defocus-independent representations in a disentangled manner, we propose
aperture randomized training, for which we learn to generate images while
randomizing the aperture size and latent codes independently. During our
experiments, we applied AR-NeRF to various natural image datasets, including
flower, bird, and face images, the results of which demonstrate the utility of
AR-NeRF for unsupervised learning of the depth and defocus effects.

Comments:
- Accepted to CVPR 2022. Project page:
  https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/ar-nerf/

---

## Generalizable Neural Radiance Fields for Novel View Synthesis with  Transformer

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-10 | Dan Wang, Xinrui Cui, Septimiu Salcudean, Z. Jane Wang | cs.CV | [PDF](http://arxiv.org/pdf/2206.05375v1){: .btn .btn-green } |

**Abstract**: We propose a Transformer-based NeRF (TransNeRF) to learn a generic neural
radiance field conditioned on observed-view images for the novel view synthesis
task. By contrast, existing MLP-based NeRFs are not able to directly receive
observed views with an arbitrary number and require an auxiliary pooling-based
operation to fuse source-view information, resulting in the missing of
complicated relationships between source views and the target rendering view.
Furthermore, current approaches process each 3D point individually and ignore
the local consistency of a radiance field scene representation. These
limitations potentially can reduce their performance in challenging real-world
applications where large differences between source views and a novel rendering
view may exist. To address these challenges, our TransNeRF utilizes the
attention mechanism to naturally decode deep associations of an arbitrary
number of source views into a coordinate-based scene representation. Local
consistency of shape and appearance are considered in the ray-cast space and
the surrounding-view space within a unified Transformer network. Experiments
demonstrate that our TransNeRF, trained on a wide variety of scenes, can
achieve better performance in comparison to state-of-the-art image-based neural
rendering methods in both scene-agnostic and per-scene finetuning scenarios
especially when there is a considerable gap between source views and a
rendering view.

---

## NeRF-In: Free-Form NeRF Inpainting with RGB-D Priors

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-10 | Hao-Kang Liu, I-Chao Shen, Bing-Yu Chen | cs.CV | [PDF](http://arxiv.org/pdf/2206.04901v1){: .btn .btn-green } |

**Abstract**: Though Neural Radiance Field (NeRF) demonstrates compelling novel view
synthesis results, it is still unintuitive to edit a pre-trained NeRF because
the neural network's parameters and the scene geometry/appearance are often not
explicitly associated. In this paper, we introduce the first framework that
enables users to remove unwanted objects or retouch undesired regions in a 3D
scene represented by a pre-trained NeRF without any category-specific data and
training. The user first draws a free-form mask to specify a region containing
unwanted objects over a rendered view from the pre-trained NeRF. Our framework
first transfers the user-provided mask to other rendered views and estimates
guiding color and depth images within these transferred masked regions. Next,
we formulate an optimization problem that jointly inpaints the image content in
all masked regions across multiple views by updating the NeRF model's
parameters. We demonstrate our framework on diverse scenes and show it obtained
visual plausible and structurally consistent results across multiple views
using shorter time and less user manual efforts.

Comments:
- Hao-Kang Liu and I-Chao Shen contributed equally to the paper.
  Project page: https://jdily.github.io/proj_site/nerfin_proj.html

---

## Improved Direct Voxel Grid Optimization for Radiance Fields  Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-10 | Cheng Sun, Min Sun, Hwann-Tzong Chen | cs.GR | [PDF](http://arxiv.org/pdf/2206.05085v4){: .btn .btn-green } |

**Abstract**: In this technical report, we improve the DVGO framework (called DVGOv2),
which is based on Pytorch and uses the simplest dense grid representation.
First, we re-implement part of the Pytorch operations with cuda, achieving 2-3x
speedup. The cuda extension is automatically compiled just in time. Second, we
extend DVGO to support Forward-facing and Unbounded Inward-facing capturing.
Third, we improve the space time complexity of the distortion loss proposed by
mip-NeRF 360 from O(N^2) to O(N). The distortion loss improves our quality and
training speed. Our efficient implementation could allow more future works to
benefit from the loss.

Comments:
- Project page https://sunset1995.github.io/dvgo/ ; Code
  https://github.com/sunset1995/DirectVoxGO ; Results updated

---

## Beyond RGB: Scene-Property Synthesis with Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-09 | Mingtong Zhang, Shuhong Zheng, Zhipeng Bao, Martial Hebert, Yu-Xiong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2206.04669v1){: .btn .btn-green } |

**Abstract**: Comprehensive 3D scene understanding, both geometrically and semantically, is
important for real-world applications such as robot perception. Most of the
existing work has focused on developing data-driven discriminative models for
scene understanding. This paper provides a new approach to scene understanding,
from a synthesis model perspective, by leveraging the recent progress on
implicit 3D representation and neural rendering. Building upon the great
success of Neural Radiance Fields (NeRFs), we introduce Scene-Property
Synthesis with NeRF (SS-NeRF) that is able to not only render photo-realistic
RGB images from novel viewpoints, but also render various accurate scene
properties (e.g., appearance, geometry, and semantics). By doing so, we
facilitate addressing a variety of scene understanding tasks under a unified
framework, including semantic segmentation, surface normal estimation,
reshading, keypoint detection, and edge detection. Our SS-NeRF framework can be
a powerful tool for bridging generative learning and discriminative learning,
and thus be beneficial to the investigation of a wide range of interesting
problems, such as studying task relationships within a synthesis paradigm,
transferring knowledge to novel tasks, facilitating downstream discriminative
tasks as ways of data augmentation, and serving as auto-labeller for data
creation.

---

## ObPose: Leveraging Pose for Object-Centric Scene Inference and  Generation in 3D

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-07 | Yizhe Wu, Oiwi Parker Jones, Ingmar Posner | cs.CV | [PDF](http://arxiv.org/pdf/2206.03591v3){: .btn .btn-green } |

**Abstract**: We present ObPose, an unsupervised object-centric inference and generation
model which learns 3D-structured latent representations from RGB-D scenes.
Inspired by prior art in 2D representation learning, ObPose considers a
factorised latent space, separately encoding object location (where) and
appearance (what). ObPose further leverages an object's pose (i.e. location and
orientation), defined via a minimum volume principle, as a novel inductive bias
for learning the where component. To achieve this, we propose an efficient,
voxelised approximation approach to recover the object shape directly from a
neural radiance field (NeRF). As a consequence, ObPose models each scene as a
composition of NeRFs, richly representing individual objects. To evaluate the
quality of the learned representations, ObPose is evaluated quantitatively on
the YCB, MultiShapeNet, and CLEVR datatasets for unsupervised scene
segmentation, outperforming the current state-of-the-art in 3D scene inference
(ObSuRF) by a significant margin. Generative results provide qualitative
demonstration that the same ObPose model can both generate novel scenes and
flexibly edit the objects in them. These capacities again reflect the quality
of the learned latents and the benefits of disentangling the where and what
components of a scene. Key design choices made in the ObPose encoder are
validated with ablations.

Comments:
- 14 pages, 4 figures

---

## Reinforcement Learning with Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-03 | Danny Driess, Ingmar Schubert, Pete Florence, Yunzhu Li, Marc Toussaint | cs.LG | [PDF](http://arxiv.org/pdf/2206.01634v1){: .btn .btn-green } |

**Abstract**: It is a long-standing problem to find effective representations for training
reinforcement learning (RL) agents. This paper demonstrates that learning state
representations with supervision from Neural Radiance Fields (NeRFs) can
improve the performance of RL compared to other learned representations or even
low-dimensional, hand-engineered state information. Specifically, we propose to
train an encoder that maps multiple image observations to a latent space
describing the objects in the scene. The decoder built from a
latent-conditioned NeRF serves as the supervision signal to learn the latent
space. An RL algorithm then operates on the learned latent space as its state
representation. We call this NeRF-RL. Our experiments indicate that NeRF as
supervision leads to a latent space better suited for the downstream RL tasks
involving robotic object manipulations like hanging mugs on hooks, pushing
objects, or opening doors. Video: https://dannydriess.github.io/nerf-rl

---

## Points2NeRF: Generating Neural Radiance Fields from 3D point cloud

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-02 | D. Zimny, T. Trzciński, P. Spurek | cs.CV | [PDF](http://arxiv.org/pdf/2206.01290v2){: .btn .btn-green } |

**Abstract**: Contemporary registration devices for 3D visual information, such as LIDARs
and various depth cameras, capture data as 3D point clouds. In turn, such
clouds are challenging to be processed due to their size and complexity.
Existing methods address this problem by fitting a mesh to the point cloud and
rendering it instead. This approach, however, leads to the reduced fidelity of
the resulting visualization and misses color information of the objects crucial
in computer graphics applications. In this work, we propose to mitigate this
challenge by representing 3D objects as Neural Radiance Fields (NeRFs). We
leverage a hypernetwork paradigm and train the model to take a 3D point cloud
with the associated color values and return a NeRF network's weights that
reconstruct 3D objects from input 2D images. Our method provides efficient 3D
object representation and offers several advantages over the existing
approaches, including the ability to condition NeRFs and improved
generalization beyond objects seen in training. The latter we also confirmed in
the results of our empirical evaluation.

Comments:
- arXiv admin note: text overlap with arXiv:2003.08934 by other authors

---

## EfficientNeRF: Efficient Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-06-02 | Tao Hu, Shu Liu, Yilun Chen, Tiancheng Shen, Jiaya Jia | cs.CV | [PDF](http://arxiv.org/pdf/2206.00878v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) has been wildly applied to various tasks for
its high-quality representation of 3D scenes. It takes long per-scene training
time and per-image testing time. In this paper, we present EfficientNeRF as an
efficient NeRF-based method to represent 3D scene and synthesize novel-view
images. Although several ways exist to accelerate the training or testing
process, it is still difficult to much reduce time for both phases
simultaneously. We analyze the density and weight distribution of the sampled
points then propose valid and pivotal sampling at the coarse and fine stage,
respectively, to significantly improve sampling efficiency. In addition, we
design a novel data structure to cache the whole scene during testing to
accelerate the rendering speed. Overall, our method can reduce over 88\% of
training time, reach rendering speed of over 200 FPS, while still achieving
competitive accuracy. Experiments prove that our method promotes the
practicality of NeRF in the real world and enables many applications.