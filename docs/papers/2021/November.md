---
layout: default
title: November
parent: 2021
nav_order: 11
---
<!---metadata--->

## Hallucinated Neural Radiance Fields in the Wild

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-30 | Xingyu Chen, Qi Zhang, Xiaoyu Li, Yue Chen, Ying Feng, Xuan Wang, Jue Wang | cs.CV | [PDF](http://arxiv.org/pdf/2111.15246v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) has recently gained popularity for its
impressive novel view synthesis ability. This paper studies the problem of
hallucinated NeRF: i.e., recovering a realistic NeRF at a different time of day
from a group of tourism images. Existing solutions adopt NeRF with a
controllable appearance embedding to render novel views under various
conditions, but they cannot render view-consistent images with an unseen
appearance. To solve this problem, we present an end-to-end framework for
constructing a hallucinated NeRF, dubbed as Ha-NeRF. Specifically, we propose
an appearance hallucination module to handle time-varying appearances and
transfer them to novel views. Considering the complex occlusions of tourism
images, we introduce an anti-occlusion module to decompose the static subjects
for visibility accurately. Experimental results on synthetic data and real
tourism photo collections demonstrate that our method can hallucinate the
desired appearances and render occlusion-free images from different views. The
project and supplementary materials are available at
https://rover-xingyu.github.io/Ha-NeRF/.

Comments:
- Accepted by CVPR 2022. Project website:
  https://rover-xingyu.github.io/Ha-NeRF/

---

## NeRFReN: Neural Radiance Fields with Reflections

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-30 | Yuan-Chen Guo, Di Kang, Linchao Bao, Yu He, Song-Hai Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2111.15234v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) has achieved unprecedented view synthesis
quality using coordinate-based neural scene representations. However, NeRF's
view dependency can only handle simple reflections like highlights but cannot
deal with complex reflections such as those from glass and mirrors. In these
scenarios, NeRF models the virtual image as real geometries which leads to
inaccurate depth estimation, and produces blurry renderings when the multi-view
consistency is violated as the reflected objects may only be seen under some of
the viewpoints. To overcome these issues, we introduce NeRFReN, which is built
upon NeRF to model scenes with reflections. Specifically, we propose to split a
scene into transmitted and reflected components, and model the two components
with separate neural radiance fields. Considering that this decomposition is
highly under-constrained, we exploit geometric priors and apply
carefully-designed training strategies to achieve reasonable decomposition
results. Experiments on various self-captured scenes show that our method
achieves high-quality novel view synthesis and physically sound depth
estimation results while enabling scene editing applications.

Comments:
- Accepted to CVPR 2022. Project page:
  https://bennyguo.github.io/nerfren/

---

## FENeRF: Face Editing in Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-30 | Jingxiang Sun, Xuan Wang, Yong Zhang, Xiaoyu Li, Qi Zhang, Yebin Liu, Jue Wang | cs.CV | [PDF](http://arxiv.org/pdf/2111.15490v2){: .btn .btn-green } |

**Abstract**: Previous portrait image generation methods roughly fall into two categories:
2D GANs and 3D-aware GANs. 2D GANs can generate high fidelity portraits but
with low view consistency. 3D-aware GAN methods can maintain view consistency
but their generated images are not locally editable. To overcome these
limitations, we propose FENeRF, a 3D-aware generator that can produce
view-consistent and locally-editable portrait images. Our method uses two
decoupled latent codes to generate corresponding facial semantics and texture
in a spatial aligned 3D volume with shared geometry. Benefiting from such
underlying 3D representation, FENeRF can jointly render the boundary-aligned
image and semantic mask and use the semantic mask to edit the 3D volume via GAN
inversion. We further show such 3D representation can be learned from widely
available monocular image and semantic mask pairs. Moreover, we reveal that
joint learning semantics and texture helps to generate finer geometry. Our
experiments demonstrate that FENeRF outperforms state-of-the-art methods in
various face editing tasks.

Comments:
- Accepted to CVPR 2022. Project: https://mrtornado24.github.io/FENeRF/

---

## NeuSample: Neural Sample Field for Efficient View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-30 | Jiemin Fang, Lingxi Xie, Xinggang Wang, Xiaopeng Zhang, Wenyu Liu, Qi Tian | cs.CV | [PDF](http://arxiv.org/pdf/2111.15552v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) have shown great potentials in representing 3D
scenes and synthesizing novel views, but the computational overhead of NeRF at
the inference stage is still heavy. To alleviate the burden, we delve into the
coarse-to-fine, hierarchical sampling procedure of NeRF and point out that the
coarse stage can be replaced by a lightweight module which we name a neural
sample field. The proposed sample field maps rays into sample distributions,
which can be transformed into point coordinates and fed into radiance fields
for volume rendering. The overall framework is named as NeuSample. We perform
experiments on Realistic Synthetic 360$^{\circ}$ and Real Forward-Facing, two
popular 3D scene sets, and show that NeuSample achieves better rendering
quality than NeRF while enjoying a faster inference speed. NeuSample is further
compressed with a proposed sample field extraction method towards a better
trade-off between quality and speed.

Comments:
- Project page: https://jaminfong.cn/neusample/

---

## Urban Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-29 | Konstantinos Rematas, Andrew Liu, Pratul P. Srinivasan, Jonathan T. Barron, Andrea Tagliasacchi, Thomas Funkhouser, Vittorio Ferrari | cs.CV | [PDF](http://arxiv.org/pdf/2111.14643v1){: .btn .btn-green } |

**Abstract**: The goal of this work is to perform 3D reconstruction and novel view
synthesis from data captured by scanning platforms commonly deployed for world
mapping in urban outdoor environments (e.g., Street View). Given a sequence of
posed RGB images and lidar sweeps acquired by cameras and scanners moving
through an outdoor scene, we produce a model from which 3D surfaces can be
extracted and novel RGB images can be synthesized. Our approach extends Neural
Radiance Fields, which has been demonstrated to synthesize realistic novel
images for small scenes in controlled settings, with new methods for leveraging
asynchronously captured lidar data, for addressing exposure variation between
captured images, and for leveraging predicted image segmentations to supervise
densities on rays pointing at the sky. Each of these three extensions provides
significant performance improvements in experiments on Street View data. Our
system produces state-of-the-art 3D surface reconstructions and synthesizes
higher quality novel views in comparison to both traditional methods
(e.g.~COLMAP) and recent neural representations (e.g.~Mip-NeRF).

Comments:
- Project: https://urban-radiance-fields.github.io/

---

## HDR-NeRF: High Dynamic Range Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-29 | Xin Huang, Qi Zhang, Ying Feng, Hongdong Li, Xuan Wang, Qing Wang | cs.CV | [PDF](http://arxiv.org/pdf/2111.14451v4){: .btn .btn-green } |

**Abstract**: We present High Dynamic Range Neural Radiance Fields (HDR-NeRF) to recover an
HDR radiance field from a set of low dynamic range (LDR) views with different
exposures. Using the HDR-NeRF, we are able to generate both novel HDR views and
novel LDR views under different exposures. The key to our method is to model
the physical imaging process, which dictates that the radiance of a scene point
transforms to a pixel value in the LDR image with two implicit functions: a
radiance field and a tone mapper. The radiance field encodes the scene radiance
(values vary from 0 to +infty), which outputs the density and radiance of a ray
by giving corresponding ray origin and ray direction. The tone mapper models
the mapping process that a ray hitting on the camera sensor becomes a pixel
value. The color of the ray is predicted by feeding the radiance and the
corresponding exposure time into the tone mapper. We use the classic volume
rendering technique to project the output radiance, colors, and densities into
HDR and LDR images, while only the input LDR images are used as the
supervision. We collect a new forward-facing HDR dataset to evaluate the
proposed method. Experimental results on synthetic and real-world scenes
validate that our method can not only accurately control the exposures of
synthesized views but also render views with a high dynamic range.

Comments:
- Accepted to CVPR 2022. Project page:
  https://xhuangcv.github.io/hdr-nerf/

---

## Deblur-NeRF: Neural Radiance Fields from Blurry Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-29 | Li Ma, Xiaoyu Li, Jing Liao, Qi Zhang, Xuan Wang, Jue Wang, Pedro V. Sander | cs.CV | [PDF](http://arxiv.org/pdf/2111.14292v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has gained considerable attention recently for
3D scene reconstruction and novel view synthesis due to its remarkable
synthesis quality. However, image blurriness caused by defocus or motion, which
often occurs when capturing scenes in the wild, significantly degrades its
reconstruction quality. To address this problem, We propose Deblur-NeRF, the
first method that can recover a sharp NeRF from blurry input. We adopt an
analysis-by-synthesis approach that reconstructs blurry views by simulating the
blurring process, thus making NeRF robust to blurry inputs. The core of this
simulation is a novel Deformable Sparse Kernel (DSK) module that models
spatially-varying blur kernels by deforming a canonical sparse kernel at each
spatial location. The ray origin of each kernel point is jointly optimized,
inspired by the physical blurring process. This module is parameterized as an
MLP that has the ability to be generalized to various blur types. Jointly
optimizing the NeRF and the DSK module allows us to restore a sharp NeRF. We
demonstrate that our method can be used on both camera motion blur and defocus
blur: the two most common types of blur in real scenes. Evaluation results on
both synthetic and real-world data show that our method outperforms several
baselines. The synthetic and real datasets along with the source code is
publicly available at https://limacv.github.io/deblurnerf/

Comments:
- accepted in CVPR2022

---

## NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw  Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-26 | Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T. Barron | cs.CV | [PDF](http://arxiv.org/pdf/2111.13679v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) is a technique for high quality novel view
synthesis from a collection of posed input images. Like most view synthesis
methods, NeRF uses tonemapped low dynamic range (LDR) as input; these images
have been processed by a lossy camera pipeline that smooths detail, clips
highlights, and distorts the simple noise distribution of raw sensor data. We
modify NeRF to instead train directly on linear raw images, preserving the
scene's full dynamic range. By rendering raw output images from the resulting
NeRF, we can perform novel high dynamic range (HDR) view synthesis tasks. In
addition to changing the camera viewpoint, we can manipulate focus, exposure,
and tonemapping after the fact. Although a single raw image appears
significantly more noisy than a postprocessed one, we show that NeRF is highly
robust to the zero-mean distribution of raw noise. When optimized over many
noisy raw inputs (25-200), NeRF produces a scene representation so accurate
that its rendered novel views outperform dedicated single and multi-image deep
raw denoisers run on the same wide baseline input images. As a result, our
method, which we call RawNeRF, can reconstruct scenes from extremely noisy
images captured in near-darkness.

Comments:
- Project page: https://bmild.github.io/rawnerf/

---

## GeoNeRF: Generalizing NeRF with Geometry Priors

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-26 | Mohammad Mahdi Johari, Yann Lepoittevin, François Fleuret | cs.CV | [PDF](http://arxiv.org/pdf/2111.13539v2){: .btn .btn-green } |

**Abstract**: We present GeoNeRF, a generalizable photorealistic novel view synthesis
method based on neural radiance fields. Our approach consists of two main
stages: a geometry reasoner and a renderer. To render a novel view, the
geometry reasoner first constructs cascaded cost volumes for each nearby source
view. Then, using a Transformer-based attention mechanism and the cascaded cost
volumes, the renderer infers geometry and appearance, and renders detailed
images via classical volume rendering techniques. This architecture, in
particular, allows sophisticated occlusion reasoning, gathering information
from consistent source views. Moreover, our method can easily be fine-tuned on
a single scene, and renders competitive results with per-scene optimized neural
rendering methods with a fraction of computational cost. Experiments show that
GeoNeRF outperforms state-of-the-art generalizable neural rendering models on
various synthetic and real datasets. Lastly, with a slight modification to the
geometry reasoner, we also propose an alternative model that adapts to RGBD
images. This model directly exploits the depth information often available
thanks to depth sensors. The implementation code is available at
https://www.idiap.ch/paper/geonerf.

Comments:
- CVPR2022

---

## VaxNeRF: Revisiting the Classic for Voxel-Accelerated Neural Radiance  Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-25 | Naruya Kondo, Yuya Ikeda, Andrea Tagliasacchi, Yutaka Matsuo, Yoichi Ochiai, Shixiang Shane Gu | cs.CV | [PDF](http://arxiv.org/pdf/2111.13112v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) is a popular method in data-driven 3D
reconstruction. Given its simplicity and high quality rendering, many NeRF
applications are being developed. However, NeRF's big limitation is its slow
speed. Many attempts are made to speeding up NeRF training and inference,
including intricate code-level optimization and caching, use of sophisticated
data structures, and amortization through multi-task and meta learning. In this
work, we revisit the basic building blocks of NeRF through the lens of classic
techniques before NeRF. We propose Voxel-Accelearated NeRF (VaxNeRF),
integrating NeRF with visual hull, a classic 3D reconstruction technique only
requiring binary foreground-background pixel labels per image. Visual hull,
which can be optimized in about 10 seconds, can provide coarse in-out field
separation to omit substantial amounts of network evaluations in NeRF. We
provide a clean fully-pythonic, JAX-based implementation on the popular JaxNeRF
codebase, consisting of only about 30 lines of code changes and a modular
visual hull subroutine, and achieve about 2-8x faster learning on top of the
highly-performative JaxNeRF baseline with zero degradation in rendering
quality. With sufficient compute, this effectively brings down full NeRF
training from hours to 30 minutes. We hope VaxNeRF -- a careful combination of
a classic technique with a deep method (that arguably replaced it) -- can
empower and accelerate new NeRF extensions and applications, with its
simplicity, portability, and reliable performance gains. Codes are available at
https://github.com/naruya/VaxNeRF .

---

## Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-23 | Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, Peter Hedman | cs.CV | [PDF](http://arxiv.org/pdf/2111.12077v3){: .btn .btn-green } |

**Abstract**: Though neural radiance fields (NeRF) have demonstrated impressive view
synthesis results on objects and small bounded regions of space, they struggle
on "unbounded" scenes, where the camera may point in any direction and content
may exist at any distance. In this setting, existing NeRF-like models often
produce blurry or low-resolution renderings (due to the unbalanced detail and
scale of nearby and distant objects), are slow to train, and may exhibit
artifacts due to the inherent ambiguity of the task of reconstructing a large
scene from a small set of images. We present an extension of mip-NeRF (a NeRF
variant that addresses sampling and aliasing) that uses a non-linear scene
parameterization, online distillation, and a novel distortion-based regularizer
to overcome the challenges presented by unbounded scenes. Our model, which we
dub "mip-NeRF 360" as we target scenes in which the camera rotates 360 degrees
around a point, reduces mean-squared error by 57% compared to mip-NeRF, and is
able to produce realistic synthesized views and detailed depth maps for highly
intricate, unbounded real-world scenes.

Comments:
- https://jonbarron.info/mipnerf360/

---

## Direct Voxel Grid Optimization: Super-fast Convergence for Radiance  Fields Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-22 | Cheng Sun, Min Sun, Hwann-Tzong Chen | cs.CV | [PDF](http://arxiv.org/pdf/2111.11215v2){: .btn .btn-green } |

**Abstract**: We present a super-fast convergence approach to reconstructing the per-scene
radiance field from a set of images that capture the scene with known poses.
This task, which is often applied to novel view synthesis, is recently
revolutionized by Neural Radiance Field (NeRF) for its state-of-the-art quality
and flexibility. However, NeRF and its variants require a lengthy training time
ranging from hours to days for a single scene. In contrast, our approach
achieves NeRF-comparable quality and converges rapidly from scratch in less
than 15 minutes with a single GPU. We adopt a representation consisting of a
density voxel grid for scene geometry and a feature voxel grid with a shallow
network for complex view-dependent appearance. Modeling with explicit and
discretized volume representations is not new, but we propose two simple yet
non-trivial techniques that contribute to fast convergence speed and
high-quality output. First, we introduce the post-activation interpolation on
voxel density, which is capable of producing sharp surfaces in lower grid
resolution. Second, direct voxel density optimization is prone to suboptimal
geometry solutions, so we robustify the optimization process by imposing
several priors. Finally, evaluation on five inward-facing benchmarks shows that
our method matches, if not surpasses, NeRF's quality, yet it only takes about
15 minutes to train from scratch for a new scene.

Comments:
- Project page at https://sunset1995.github.io/dvgo/ ; Code at
  https://github.com/sunset1995/DirectVoxGO

---

## LOLNeRF: Learn from One Look

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-19 | Daniel Rebain, Mark Matthews, Kwang Moo Yi, Dmitry Lagun, Andrea Tagliasacchi | cs.CV | [PDF](http://arxiv.org/pdf/2111.09996v2){: .btn .btn-green } |

**Abstract**: We present a method for learning a generative 3D model based on neural
radiance fields, trained solely from data with only single views of each
object. While generating realistic images is no longer a difficult task,
producing the corresponding 3D structure such that they can be rendered from
different views is non-trivial. We show that, unlike existing methods, one does
not need multi-view data to achieve this goal. Specifically, we show that by
reconstructing many images aligned to an approximate canonical pose with a
single network conditioned on a shared latent space, you can learn a space of
radiance fields that models shape and appearance for a class of objects. We
demonstrate this by training models to reconstruct object categories using
datasets that contain only one view of each subject without depth or geometry
information. Our experiments show that we achieve state-of-the-art results in
novel view synthesis and high-quality results for monocular depth prediction.

Comments:
- See https://lolnerf.github.io for additional results

---

## DIVeR: Real-time and Accurate Neural Radiance Fields with Deterministic  Integration for Volume Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-19 | Liwen Wu, Jae Yong Lee, Anand Bhattad, Yuxiong Wang, David Forsyth | cs.CV | [PDF](http://arxiv.org/pdf/2111.10427v2){: .btn .btn-green } |

**Abstract**: DIVeR builds on the key ideas of NeRF and its variants -- density models and
volume rendering -- to learn 3D object models that can be rendered
realistically from small numbers of images. In contrast to all previous NeRF
methods, DIVeR uses deterministic rather than stochastic estimates of the
volume rendering integral. DIVeR's representation is a voxel based field of
features. To compute the volume rendering integral, a ray is broken into
intervals, one per voxel; components of the volume rendering integral are
estimated from the features for each interval using an MLP, and the components
are aggregated. As a result, DIVeR can render thin translucent structures that
are missed by other integrators. Furthermore, DIVeR's representation has
semantics that is relatively exposed compared to other such methods -- moving
feature vectors around in the voxel space results in natural edits. Extensive
qualitative and quantitative comparisons to current state-of-the-art methods
show that DIVeR produces models that (1) render at or above state-of-the-art
quality, (2) are very small without being baked, (3) render very fast without
being baked, and (4) can be edited in natural ways.

---

## LVAC: Learned Volumetric Attribute Compression for Point Clouds using  Coordinate Based Networks



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-17 | Berivan Isik, Philip A. Chou, Sung Jin Hwang, Nick Johnston, George Toderici | cs.GR | [PDF](http://arxiv.org/pdf/2111.08988v1){: .btn .btn-green } |

**Abstract**: We consider the attributes of a point cloud as samples of a vector-valued
volumetric function at discrete positions. To compress the attributes given the
positions, we compress the parameters of the volumetric function. We model the
volumetric function by tiling space into blocks, and representing the function
over each block by shifts of a coordinate-based, or implicit, neural network.
Inputs to the network include both spatial coordinates and a latent vector per
block. We represent the latent vectors using coefficients of the
region-adaptive hierarchical transform (RAHT) used in the MPEG geometry-based
point cloud codec G-PCC. The coefficients, which are highly compressible, are
rate-distortion optimized by back-propagation through a rate-distortion
Lagrangian loss in an auto-decoder configuration. The result outperforms RAHT
by 2--4 dB. This is the first work to compress volumetric functions represented
by local coordinate-based neural networks. As such, we expect it to be
applicable beyond point clouds, for example to compression of high-resolution
neural radiance fields.

Comments:
- 30 pages, 29 figures

---

## Template NeRF: Towards Modeling Dense Shape Correspondences from  Category-Specific Object Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-11-08 | Jianfei Guo, Zhiyuan Yang, Xi Lin, Qingfu Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2111.04237v1){: .btn .btn-green } |

**Abstract**: We present neural radiance fields (NeRF) with templates, dubbed
Template-NeRF, for modeling appearance and geometry and generating dense shape
correspondences simultaneously among objects of the same category from only
multi-view posed images, without the need of either 3D supervision or
ground-truth correspondence knowledge. The learned dense correspondences can be
readily used for various image-based tasks such as keypoint detection, part
segmentation, and texture transfer that previously require specific model
designs. Our method can also accommodate annotation transfer in a one or
few-shot manner, given only one or a few instances of the category. Using
periodic activation and feature-wise linear modulation (FiLM) conditioning, we
introduce deep implicit templates on 3D data into the 3D-aware image synthesis
pipeline NeRF. By representing object instances within the same category as
shape and appearance variation of a shared NeRF template, our proposed method
can achieve dense shape correspondences reasoning on images for a wide range of
object classes. We demonstrate the results and applications on both synthetic
and real-world data with competitive results compared with other methods based
on 3D information.

Comments:
- 10 pages, 8 figures