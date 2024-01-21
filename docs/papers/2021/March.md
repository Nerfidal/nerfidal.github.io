---
layout: default
title: March
parent: 2021
nav_order: 3
---
<!---metadata--->

## CAMPARI: Camera-Aware Decomposed Generative Neural Radiance Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-31 | Michael Niemeyer, Andreas Geiger | cs.CV | [PDF](http://arxiv.org/pdf/2103.17269v1){: .btn .btn-green } |

**Abstract**: Tremendous progress in deep generative models has led to photorealistic image
synthesis. While achieving compelling results, most approaches operate in the
two-dimensional image domain, ignoring the three-dimensional nature of our
world. Several recent works therefore propose generative models which are
3D-aware, i.e., scenes are modeled in 3D and then rendered differentiably to
the image plane. This leads to impressive 3D consistency, but incorporating
such a bias comes at a price: the camera needs to be modeled as well. Current
approaches assume fixed intrinsics and a predefined prior over camera pose
ranges. As a result, parameter tuning is typically required for real-world
data, and results degrade if the data distribution is not matched. Our key
hypothesis is that learning a camera generator jointly with the image generator
leads to a more principled approach to 3D-aware image synthesis. Further, we
propose to decompose the scene into a background and foreground model, leading
to more efficient and disentangled scene representations. While training from
raw, unposed image collections, we learn a 3D- and camera-aware generative
model which faithfully recovers not only the image but also the camera data
distribution. At test time, our model generates images with explicit control
over the camera as well as the shape and appearance of the scene.

---

## FoV-NeRF: Foveated Neural Radiance Fields for Virtual Reality

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-30 | Nianchen Deng, Zhenyi He, Jiannan Ye, Budmonde Duinkharjav, Praneeth Chakravarthula, Xubo Yang, Qi Sun | cs.GR | [PDF](http://arxiv.org/pdf/2103.16365v2){: .btn .btn-green } |

**Abstract**: Virtual Reality (VR) is becoming ubiquitous with the rise of consumer
displays and commercial VR platforms. Such displays require low latency and
high quality rendering of synthetic imagery with reduced compute overheads.
Recent advances in neural rendering showed promise of unlocking new
possibilities in 3D computer graphics via image-based representations of
virtual or physical environments. Specifically, the neural radiance fields
(NeRF) demonstrated that photo-realistic quality and continuous view changes of
3D scenes can be achieved without loss of view-dependent effects. While NeRF
can significantly benefit rendering for VR applications, it faces unique
challenges posed by high field-of-view, high resolution, and
stereoscopic/egocentric viewing, typically causing low quality and high latency
of the rendered images. In VR, this not only harms the interaction experience
but may also cause sickness. To tackle these problems toward
six-degrees-of-freedom, egocentric, and stereo NeRF in VR, we present the first
gaze-contingent 3D neural representation and view synthesis method. We
incorporate the human psychophysics of visual- and stereo-acuity into an
egocentric neural representation of 3D scenery. We then jointly optimize the
latency/performance and visual quality while mutually bridging human perception
and neural scene synthesis to achieve perceptually high-quality immersive
interaction. We conducted both objective analysis and subjective studies to
evaluate the effectiveness of our approach. We find that our method
significantly reduces latency (up to 99% time reduction compared with NeRF)
without loss of high-fidelity rendering (perceptually identical to
full-resolution ground truth). The presented approach may serve as the first
step toward future VR/AR systems that capture, teleport, and visualize remote
environments in real-time.

Comments:
- 9 pages

---

## In-Place Scene Labelling and Understanding with Implicit Scene  Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-29 | Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, Andrew J. Davison | cs.CV | [PDF](http://arxiv.org/pdf/2103.15875v2){: .btn .btn-green } |

**Abstract**: Semantic labelling is highly correlated with geometry and radiance
reconstruction, as scene entities with similar shape and appearance are more
likely to come from similar classes. Recent implicit neural reconstruction
techniques are appealing as they do not require prior training data, but the
same fully self-supervised approach is not possible for semantics because
labels are human-defined properties.
  We extend neural radiance fields (NeRF) to jointly encode semantics with
appearance and geometry, so that complete and accurate 2D semantic labels can
be achieved using a small amount of in-place annotations specific to the scene.
The intrinsic multi-view consistency and smoothness of NeRF benefit semantics
by enabling sparse labels to efficiently propagate. We show the benefit of this
approach when labels are either sparse or very noisy in room-scale scenes. We
demonstrate its advantageous properties in various interesting applications
such as an efficient scene labelling tool, novel semantic view synthesis, label
denoising, super-resolution, label interpolation and multi-view semantic label
fusion in visual semantic mapping systems.

Comments:
- Camera ready version. To be published in Proceedings of IEEE
  International Conference on Computer Vision (ICCV 2021) as Oral Presentation.
  Project page with more videos: https://shuaifengzhi.com/Semantic-NeRF/

---

## GNeRF: GAN-based Neural Radiance Field without Posed Camera

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-29 | Quan Meng, Anpei Chen, Haimin Luo, Minye Wu, Hao Su, Lan Xu, Xuming He, Jingyi Yu | cs.CV | [PDF](http://arxiv.org/pdf/2103.15606v3){: .btn .btn-green } |

**Abstract**: We introduce GNeRF, a framework to marry Generative Adversarial Networks
(GAN) with Neural Radiance Field (NeRF) reconstruction for the complex
scenarios with unknown and even randomly initialized camera poses. Recent
NeRF-based advances have gained popularity for remarkable realistic novel view
synthesis. However, most of them heavily rely on accurate camera poses
estimation, while few recent methods can only optimize the unknown camera poses
in roughly forward-facing scenes with relatively short camera trajectories and
require rough camera poses initialization. Differently, our GNeRF only utilizes
randomly initialized poses for complex outside-in scenarios. We propose a novel
two-phases end-to-end framework. The first phase takes the use of GANs into the
new realm for optimizing coarse camera poses and radiance fields jointly, while
the second phase refines them with additional photometric loss. We overcome
local minima using a hybrid and iterative optimization scheme. Extensive
experiments on a variety of synthetic and natural scenes demonstrate the
effectiveness of GNeRF. More impressively, our approach outperforms the
baselines favorably in those scenes with repeated patterns or even low textures
that are regarded as extremely challenging before.

Comments:
- ICCV 2021 (Oral)

---

## MVSNeRF: Fast Generalizable Radiance Field Reconstruction from  Multi-View Stereo

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-29 | Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, Hao Su | cs.CV | [PDF](http://arxiv.org/pdf/2103.15595v2){: .btn .btn-green } |

**Abstract**: We present MVSNeRF, a novel neural rendering approach that can efficiently
reconstruct neural radiance fields for view synthesis. Unlike prior works on
neural radiance fields that consider per-scene optimization on densely captured
images, we propose a generic deep neural network that can reconstruct radiance
fields from only three nearby input views via fast network inference. Our
approach leverages plane-swept cost volumes (widely used in multi-view stereo)
for geometry-aware scene reasoning, and combines this with physically based
volume rendering for neural radiance field reconstruction. We train our network
on real objects in the DTU dataset, and test it on three different datasets to
evaluate its effectiveness and generalizability. Our approach can generalize
across scenes (even indoor scenes, completely different from our training
scenes of objects) and generate realistic view synthesis results using only
three input images, significantly outperforming concurrent works on
generalizable radiance field reconstruction. Moreover, if dense images are
captured, our estimated radiance field representation can be easily fine-tuned;
this leads to fast per-scene reconstruction with higher rendering quality and
substantially less optimization time than NeRF.

Comments:
- Project Page: https://apchenstu.github.io/mvsnerf/
  Code:https://github.com/apchenstu/mvsnerf

---

## MINE: Towards Continuous Depth MPI with NeRF for Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-27 | Jiaxin Li, Zijian Feng, Qi She, Henghui Ding, Changhu Wang, Gim Hee Lee | cs.CV | [PDF](http://arxiv.org/pdf/2103.14910v3){: .btn .btn-green } |

**Abstract**: In this paper, we propose MINE to perform novel view synthesis and depth
estimation via dense 3D reconstruction from a single image. Our approach is a
continuous depth generalization of the Multiplane Images (MPI) by introducing
the NEural radiance fields (NeRF). Given a single image as input, MINE predicts
a 4-channel image (RGB and volume density) at arbitrary depth values to jointly
reconstruct the camera frustum and fill in occluded contents. The reconstructed
and inpainted frustum can then be easily rendered into novel RGB or depth views
using differentiable rendering. Extensive experiments on RealEstate10K, KITTI
and Flowers Light Fields show that our MINE outperforms state-of-the-art by a
large margin in novel view synthesis. We also achieve competitive results in
depth estimation on iBims-1 and NYU-v2 without annotated depth supervision. Our
source code is available at https://github.com/vincentfung13/MINE

Comments:
- ICCV 2021. Main paper and supplementary materials

---

## Baking Neural Radiance Fields for Real-Time View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-26 | Peter Hedman, Pratul P. Srinivasan, Ben Mildenhall, Jonathan T. Barron, Paul Debevec | cs.CV | [PDF](http://arxiv.org/pdf/2103.14645v1){: .btn .btn-green } |

**Abstract**: Neural volumetric representations such as Neural Radiance Fields (NeRF) have
emerged as a compelling technique for learning to represent 3D scenes from
images with the goal of rendering photorealistic images of the scene from
unobserved viewpoints. However, NeRF's computational requirements are
prohibitive for real-time applications: rendering views from a trained NeRF
requires querying a multilayer perceptron (MLP) hundreds of times per ray. We
present a method to train a NeRF, then precompute and store (i.e. "bake") it as
a novel representation called a Sparse Neural Radiance Grid (SNeRG) that
enables real-time rendering on commodity hardware. To achieve this, we
introduce 1) a reformulation of NeRF's architecture, and 2) a sparse voxel grid
representation with learned feature vectors. The resulting scene representation
retains NeRF's ability to render fine geometric details and view-dependent
appearance, is compact (averaging less than 90 MB per scene), and can be
rendered in real-time (higher than 30 frames per second on a laptop GPU).
Actual screen captures are shown in our video.

Comments:
- Project page: https://nerf.live

---

## PlenOctrees for Real-time Rendering of Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-25 | Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, Angjoo Kanazawa | cs.CV | [PDF](http://arxiv.org/pdf/2103.14024v2){: .btn .btn-green } |

**Abstract**: We introduce a method to render Neural Radiance Fields (NeRFs) in real time
using PlenOctrees, an octree-based 3D representation which supports
view-dependent effects. Our method can render 800x800 images at more than 150
FPS, which is over 3000 times faster than conventional NeRFs. We do so without
sacrificing quality while preserving the ability of NeRFs to perform
free-viewpoint rendering of scenes with arbitrary geometry and view-dependent
effects. Real-time performance is achieved by pre-tabulating the NeRF into a
PlenOctree. In order to preserve view-dependent effects such as specularities,
we factorize the appearance via closed-form spherical basis functions.
Specifically, we show that it is possible to train NeRFs to predict a spherical
harmonic representation of radiance, removing the viewing direction as an input
to the neural network. Furthermore, we show that PlenOctrees can be directly
optimized to further minimize the reconstruction loss, which leads to equal or
better quality compared to competing methods. Moreover, this octree
optimization step can be used to reduce the training time, as we no longer need
to wait for the NeRF training to converge fully. Our real-time neural rendering
approach may potentially enable new applications such as 6-DOF industrial and
product visualizations, as well as next generation AR/VR systems. PlenOctrees
are amenable to in-browser rendering as well; please visit the project page for
the interactive online demo, as well as video and code:
https://alexyu.net/plenoctrees

Comments:
- ICCV 2021 (Oral)

---

## KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-25 | Christian Reiser, Songyou Peng, Yiyi Liao, Andreas Geiger | cs.CV | [PDF](http://arxiv.org/pdf/2103.13744v2){: .btn .btn-green } |

**Abstract**: NeRF synthesizes novel views of a scene with unprecedented quality by fitting
a neural radiance field to RGB images. However, NeRF requires querying a deep
Multi-Layer Perceptron (MLP) millions of times, leading to slow rendering
times, even on modern GPUs. In this paper, we demonstrate that real-time
rendering is possible by utilizing thousands of tiny MLPs instead of one single
large MLP. In our setting, each individual MLP only needs to represent parts of
the scene, thus smaller and faster-to-evaluate MLPs can be used. By combining
this divide-and-conquer strategy with further optimizations, rendering is
accelerated by three orders of magnitude compared to the original NeRF model
without incurring high storage costs. Further, using teacher-student
distillation for training, we show that this speed-up can be achieved without
sacrificing visual quality.

Comments:
- ICCV 2021. Code, pretrained models and an interactive viewer are
  available at https://github.com/creiser/kilonerf/

---

## Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-24 | Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, Pratul P. Srinivasan | cs.CV | [PDF](http://arxiv.org/pdf/2103.13415v3){: .btn .btn-green } |

**Abstract**: The rendering procedure used by neural radiance fields (NeRF) samples a scene
with a single ray per pixel and may therefore produce renderings that are
excessively blurred or aliased when training or testing images observe scene
content at different resolutions. The straightforward solution of supersampling
by rendering with multiple rays per pixel is impractical for NeRF, because
rendering each ray requires querying a multilayer perceptron hundreds of times.
Our solution, which we call "mip-NeRF" (a la "mipmap"), extends NeRF to
represent the scene at a continuously-valued scale. By efficiently rendering
anti-aliased conical frustums instead of rays, mip-NeRF reduces objectionable
aliasing artifacts and significantly improves NeRF's ability to represent fine
details, while also being 7% faster than NeRF and half the size. Compared to
NeRF, mip-NeRF reduces average error rates by 17% on the dataset presented with
NeRF and by 60% on a challenging multiscale variant of that dataset that we
present. Mip-NeRF is also able to match the accuracy of a brute-force
supersampled NeRF on our multiscale dataset while being 22x faster.

---

## UltraSR: Spatial Encoding is a Missing Key for Implicit Image  Function-based Arbitrary-Scale Super-Resolution

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-23 | Xingqian Xu, Zhangyang Wang, Humphrey Shi | cs.CV | [PDF](http://arxiv.org/pdf/2103.12716v2){: .btn .btn-green } |

**Abstract**: The recent success of NeRF and other related implicit neural representation
methods has opened a new path for continuous image representation, where pixel
values no longer need to be looked up from stored discrete 2D arrays but can be
inferred from neural network models on a continuous spatial domain. Although
the recent work LIIF has demonstrated that such novel approaches can achieve
good performance on the arbitrary-scale super-resolution task, their upscaled
images frequently show structural distortion due to the inaccurate prediction
of high-frequency textures. In this work, we propose UltraSR, a simple yet
effective new network design based on implicit image functions in which we
deeply integrated spatial coordinates and periodic encoding with the implicit
neural representation. Through extensive experiments and ablation studies, we
show that spatial encoding is a missing key toward the next-stage
high-performing implicit image function. Our UltraSR sets new state-of-the-art
performance on the DIV2K benchmark under all super-resolution scales compared
to previous state-of-the-art methods. UltraSR also achieves superior
performance on other standard benchmark datasets in which it outperforms prior
works in almost all experiments.

---

## AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-20 | Yudong Guo, Keyu Chen, Sen Liang, Yong-Jin Liu, Hujun Bao, Juyong Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2103.11078v3){: .btn .btn-green } |

**Abstract**: Generating high-fidelity talking head video by fitting with the input audio
sequence is a challenging problem that receives considerable attentions
recently. In this paper, we address this problem with the aid of neural scene
representation networks. Our method is completely different from existing
methods that rely on intermediate representations like 2D landmarks or 3D face
models to bridge the gap between audio input and video output. Specifically,
the feature of input audio signal is directly fed into a conditional implicit
function to generate a dynamic neural radiance field, from which a
high-fidelity talking-head video corresponding to the audio signal is
synthesized using volume rendering. Another advantage of our framework is that
not only the head (with hair) region is synthesized as previous methods did,
but also the upper body is generated via two individual neural radiance fields.
Experimental results demonstrate that our novel framework can (1) produce
high-fidelity and natural results, and (2) support free adjustment of audio
signals, viewing directions, and background images. Code is available at
https://github.com/YudongGuo/AD-NeRF.

Comments:
- Project: https://yudongguo.github.io/ADNeRF/ Code:
  https://github.com/YudongGuo/AD-NeRF

---

## FastNeRF: High-Fidelity Neural Rendering at 200FPS

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-18 | Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, Julien Valentin | cs.CV | [PDF](http://arxiv.org/pdf/2103.10380v2){: .btn .btn-green } |

**Abstract**: Recent work on Neural Radiance Fields (NeRF) showed how neural networks can
be used to encode complex 3D environments that can be rendered
photorealistically from novel viewpoints. Rendering these images is very
computationally demanding and recent improvements are still a long way from
enabling interactive rates, even on high-end hardware. Motivated by scenarios
on mobile and mixed reality devices, we propose FastNeRF, the first NeRF-based
system capable of rendering high fidelity photorealistic images at 200Hz on a
high-end consumer GPU. The core of our method is a graphics-inspired
factorization that allows for (i) compactly caching a deep radiance map at each
position in space, (ii) efficiently querying that map using ray directions to
estimate the pixel values in the rendered image. Extensive experiments show
that the proposed method is 3000 times faster than the original NeRF algorithm
and at least an order of magnitude faster than existing work on accelerating
NeRF, while maintaining visual quality and extensibility.

Comments:
- main paper: 10 pages, 6 figures; supplementary: 10 pages, 17 figures

---

## DONeRF: Towards Real-Time Rendering of Compact Neural Radiance Fields  using Depth Oracle Networks

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-04 | Thomas Neff, Pascal Stadlbauer, Mathias Parger, Andreas Kurz, Joerg H. Mueller, Chakravarty R. Alla Chaitanya, Anton Kaplanyan, Markus Steinberger | cs.CV | [PDF](http://arxiv.org/pdf/2103.03231v4){: .btn .btn-green } |

**Abstract**: The recent research explosion around implicit neural representations, such as
NeRF, shows that there is immense potential for implicitly storing high-quality
scene and lighting information in compact neural networks. However, one major
limitation preventing the use of NeRF in real-time rendering applications is
the prohibitive computational cost of excessive network evaluations along each
view ray, requiring dozens of petaFLOPS. In this work, we bring compact neural
representations closer to practical rendering of synthetic content in real-time
applications, such as games and virtual reality. We show that the number of
samples required for each view ray can be significantly reduced when samples
are placed around surfaces in the scene without compromising image quality. To
this end, we propose a depth oracle network that predicts ray sample locations
for each view ray with a single network evaluation. We show that using a
classification network around logarithmically discretized and spherically
warped depth values is essential to encode surface locations rather than
directly estimating depth. The combination of these techniques leads to DONeRF,
our compact dual network design with a depth oracle network as its first step
and a locally sampled shading network for ray accumulation. With DONeRF, we
reduce the inference costs by up to 48x compared to NeRF when conditioning on
available ground truth depth information. Compared to concurrent acceleration
methods for raymarching-based neural representations, DONeRF does not require
additional memory for explicit caching or acceleration structures, and can
render interactively (20 frames per second) on a single GPU.

Comments:
- Accepted to EGSR 2021 in the CGF track; Project website:
  https://depthoraclenerf.github.io/

---

## Neural 3D Video Synthesis from Multi-view Video



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-03 | Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, Zhaoyang Lv | cs.CV | [PDF](http://arxiv.org/pdf/2103.02597v2){: .btn .btn-green } |

**Abstract**: We propose a novel approach for 3D video synthesis that is able to represent
multi-view video recordings of a dynamic real-world scene in a compact, yet
expressive representation that enables high-quality view synthesis and motion
interpolation. Our approach takes the high quality and compactness of static
neural radiance fields in a new direction: to a model-free, dynamic setting. At
the core of our approach is a novel time-conditioned neural radiance field that
represents scene dynamics using a set of compact latent codes. We are able to
significantly boost the training speed and perceptual quality of the generated
imagery by a novel hierarchical training scheme in combination with ray
importance sampling. Our learned representation is highly compact and able to
represent a 10 second 30 FPS multiview video recording by 18 cameras with a
model size of only 28MB. We demonstrate that our method can render
high-fidelity wide-angle novel views at over 1K resolution, even for complex
and dynamic scenes. We perform an extensive qualitative and quantitative
evaluation that shows that our approach outperforms the state of the art.
Project website: https://neural-3d-video.github.io/.

Comments:
- Accepted as an oral presentation for CVPR 2022. Project website:
  https://neural-3d-video.github.io/

---

## Mixture of Volumetric Primitives for Efficient Neural Rendering



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-03-02 | Stephen Lombardi, Tomas Simon, Gabriel Schwartz, Michael Zollhoefer, Yaser Sheikh, Jason Saragih | cs.GR | [PDF](http://arxiv.org/pdf/2103.01954v2){: .btn .btn-green } |

**Abstract**: Real-time rendering and animation of humans is a core function in games,
movies, and telepresence applications. Existing methods have a number of
drawbacks we aim to address with our work. Triangle meshes have difficulty
modeling thin structures like hair, volumetric representations like Neural
Volumes are too low-resolution given a reasonable memory budget, and
high-resolution implicit representations like Neural Radiance Fields are too
slow for use in real-time applications. We present Mixture of Volumetric
Primitives (MVP), a representation for rendering dynamic 3D content that
combines the completeness of volumetric representations with the efficiency of
primitive-based rendering, e.g., point-based or mesh-based methods. Our
approach achieves this by leveraging spatially shared computation with a
deconvolutional architecture and by minimizing computation in empty regions of
space with volumetric primitives that can move to cover only occupied regions.
Our parameterization supports the integration of correspondence and tracking
constraints, while being robust to areas where classical tracking fails, such
as around thin or translucent structures and areas with large topological
variability. MVP is a hybrid that generalizes both volumetric and
primitive-based representations. Through a series of extensive experiments we
demonstrate that it inherits the strengths of each, while avoiding many of
their limitations. We also compare our approach to several state-of-the-art
methods and demonstrate that MVP produces superior results in terms of quality
and runtime performance.

Comments:
- 13 pages; SIGGRAPH 2021