---
layout: default
title: April
parent: 2022
nav_order: 4
---
<!---metadata--->

## AE-NeRF: Auto-Encoding Neural Radiance Fields for 3D-Aware Object  Manipulation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-28 | Mira Kim, Jaehoon Ko, Kyusun Cho, Junmyeong Choi, Daewon Choi, Seungryong Kim | cs.CV | [PDF](http://arxiv.org/pdf/2204.13426v1){: .btn .btn-green } |

**Abstract**: We propose a novel framework for 3D-aware object manipulation, called
Auto-Encoding Neural Radiance Fields (AE-NeRF). Our model, which is formulated
in an auto-encoder architecture, extracts disentangled 3D attributes such as 3D
shape, appearance, and camera pose from an image, and a high-quality image is
rendered from the attributes through disentangled generative Neural Radiance
Fields (NeRF). To improve the disentanglement ability, we present two losses,
global-local attribute consistency loss defined between input and output, and
swapped-attribute classification loss. Since training such auto-encoding
networks from scratch without ground-truth shape and appearance information is
non-trivial, we present a stage-wise training scheme, which dramatically helps
to boost the performance. We conduct experiments to demonstrate the
effectiveness of the proposed model over the latest methods and provide
extensive ablation studies.

---

## NeurMiPs: Neural Mixture of Planar Experts for View Synthesis



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-28 | Zhi-Hao Lin, Wei-Chiu Ma, Hao-Yu Hsu, Yu-Chiang Frank Wang, Shenlong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2204.13696v1){: .btn .btn-green } |

**Abstract**: We present Neural Mixtures of Planar Experts (NeurMiPs), a novel planar-based
scene representation for modeling geometry and appearance. NeurMiPs leverages a
collection of local planar experts in 3D space as the scene representation.
Each planar expert consists of the parameters of the local rectangular shape
representing geometry and a neural radiance field modeling the color and
opacity. We render novel views by calculating ray-plane intersections and
composite output colors and densities at intersected points to the image.
NeurMiPs blends the efficiency of explicit mesh rendering and flexibility of
the neural radiance field. Experiments demonstrate superior performance and
speed of our proposed method, compared to other 3D representations in novel
view synthesis.

Comments:
- CVPR 2022. Project page: https://zhihao-lin.github.io/neurmips/

---

## Generalizable Neural Performer: Learning Robust Radiance Fields for  Human Novel View Synthesis



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-25 | Wei Cheng, Su Xu, Jingtan Piao, Chen Qian, Wayne Wu, Kwan-Yee Lin, Hongsheng Li | cs.CV | [PDF](http://arxiv.org/pdf/2204.11798v1){: .btn .btn-green } |

**Abstract**: This work targets at using a general deep learning framework to synthesize
free-viewpoint images of arbitrary human performers, only requiring a sparse
number of camera views as inputs and skirting per-case fine-tuning. The large
variation of geometry and appearance, caused by articulated body poses, shapes
and clothing types, are the key bottlenecks of this task. To overcome these
challenges, we present a simple yet powerful framework, named Generalizable
Neural Performer (GNR), that learns a generalizable and robust neural body
representation over various geometry and appearance. Specifically, we compress
the light fields for novel view human rendering as conditional implicit neural
radiance fields from both geometry and appearance aspects. We first introduce
an Implicit Geometric Body Embedding strategy to enhance the robustness based
on both parametric 3D human body model and multi-view images hints. We further
propose a Screen-Space Occlusion-Aware Appearance Blending technique to
preserve the high-quality appearance, through interpolating source view
appearance to the radiance fields with a relax but approximate geometric
guidance.
  To evaluate our method, we present our ongoing effort of constructing a
dataset with remarkable complexity and diversity. The dataset GeneBody-1.0,
includes over 360M frames of 370 subjects under multi-view cameras capturing,
performing a large variety of pose actions, along with diverse body shapes,
clothing, accessories and hairdos. Experiments on GeneBody-1.0 and ZJU-Mocap
show better robustness of our methods than recent state-of-the-art
generalizable methods among all cross-dataset, unseen subjects and unseen poses
settings. We also demonstrate the competitiveness of our model compared with
cutting-edge case-specific ones. Dataset, code and model will be made publicly
available.

Comments:
- Project Page: https://generalizable-neural-performer.github.io/
  Dataset: https://generalizable-neural-performer.github.io/genebody.html/

---

## Control-NeRF: Editable Feature Volumes for Scene Rendering and  Manipulation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-22 | Verica Lazova, Vladimir Guzov, Kyle Olszewski, Sergey Tulyakov, Gerard Pons-Moll | cs.CV | [PDF](http://arxiv.org/pdf/2204.10850v1){: .btn .btn-green } |

**Abstract**: We present a novel method for performing flexible, 3D-aware image content
manipulation while enabling high-quality novel view synthesis. While NeRF-based
approaches are effective for novel view synthesis, such models memorize the
radiance for every point in a scene within a neural network. Since these models
are scene-specific and lack a 3D scene representation, classical editing such
as shape manipulation, or combining scenes is not possible. Hence, editing and
combining NeRF-based scenes has not been demonstrated. With the aim of
obtaining interpretable and controllable scene representations, our model
couples learnt scene-specific feature volumes with a scene agnostic neural
rendering network. With this hybrid representation, we decouple neural
rendering from scene-specific geometry and appearance. We can generalize to
novel scenes by optimizing only the scene-specific 3D feature representation,
while keeping the parameters of the rendering network fixed. The rendering
function learnt during the initial training stage can thus be easily applied to
new scenes, making our approach more flexible. More importantly, since the
feature volumes are independent of the rendering model, we can manipulate and
combine scenes by editing their corresponding feature volumes. The edited
volume can then be plugged into the rendering model to synthesize high-quality
novel views. We demonstrate various scene manipulations, including mixing
scenes, deforming objects and inserting objects into scenes, while still
producing photo-realistic results.

---

## Implicit Object Mapping With Noisy Data

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-22 | Jad Abou-Chakra, Feras Dayoub, Niko Sünderhauf | cs.RO | [PDF](http://arxiv.org/pdf/2204.10516v2){: .btn .btn-green } |

**Abstract**: Modelling individual objects in a scene as Neural Radiance Fields (NeRFs)
provides an alternative geometric scene representation that may benefit
downstream robotics tasks such as scene understanding and object manipulation.
However, we identify three challenges to using real-world training data
collected by a robot to train a NeRF: (i) The camera trajectories are
constrained, and full visual coverage is not guaranteed - especially when
obstructions to the objects of interest are present; (ii) the poses associated
with the images are noisy due to odometry or localization noise; (iii) the
objects are not easily isolated from the background. This paper evaluates the
extent to which above factors degrade the quality of the learnt implicit object
representation. We introduce a pipeline that decomposes a scene into multiple
individual object-NeRFs, using noisy object instance masks and bounding boxes,
and evaluate the sensitivity of this pipeline with respect to noisy poses,
instance masks, and the number of training images. We uncover that the
sensitivity to noisy instance masks can be partially alleviated with depth
supervision and quantify the importance of including the camera extrinsics in
the NeRF optimisation process.

---

## SILVR: A Synthetic Immersive Large-Volume Plenoptic Dataset

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-20 | Martijn Courteaux, Julie Artois, Stijn De Pauw, Peter Lambert, Glenn Van Wallendael | cs.GR | [PDF](http://arxiv.org/pdf/2204.09523v1){: .btn .btn-green } |

**Abstract**: In six-degrees-of-freedom light-field (LF) experiences, the viewer's freedom
is limited by the extent to which the plenoptic function was sampled. Existing
LF datasets represent only small portions of the plenoptic function, such that
they either cover a small volume, or they have limited field of view.
Therefore, we propose a new LF image dataset "SILVR" that allows for
six-degrees-of-freedom navigation in much larger volumes while maintaining full
panoramic field of view. We rendered three different virtual scenes in various
configurations, where the number of views ranges from 642 to 2226. One of these
scenes (called Zen Garden) is a novel scene, and is made publicly available. We
chose to position the virtual cameras closely together in large cuboid and
spherical organisations ($2.2m^3$ to $48m^3$), equipped with 180{\deg} fish-eye
lenses. Every view is rendered to a color image and depth map of 2048px
$\times$ 2048px. Additionally, we present the software used to automate the
multi-view rendering process, as well as a lens-reprojection tool that converts
between images with panoramic or fish-eye projection to a standard rectilinear
(i.e., perspective) projection. Finally, we demonstrate how the proposed
dataset and software can be used to evaluate LF coding/rendering techniques(in
this case for training NeRFs with instant-ngp). As such, we provide the first
publicly-available LF dataset for large volumes of light with full panoramic
field of view

Comments:
- In 13th ACM Multimedia Systems Conference (MMSys '22), June 14-17,
  2022, Athlone, Ireland. ACM, New York, NY, USA, 6 pages

---

## Modeling Indirect Illumination for Inverse Rendering



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-14 | Yuanqing Zhang, Jiaming Sun, Xingyi He, Huan Fu, Rongfei Jia, Xiaowei Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2204.06837v1){: .btn .btn-green } |

**Abstract**: Recent advances in implicit neural representations and differentiable
rendering make it possible to simultaneously recover the geometry and materials
of an object from multi-view RGB images captured under unknown static
illumination. Despite the promising results achieved, indirect illumination is
rarely modeled in previous methods, as it requires expensive recursive path
tracing which makes the inverse rendering computationally intractable. In this
paper, we propose a novel approach to efficiently recovering spatially-varying
indirect illumination. The key insight is that indirect illumination can be
conveniently derived from the neural radiance field learned from input images
instead of being estimated jointly with direct illumination and materials. By
properly modeling the indirect illumination and visibility of direct
illumination, interreflection- and shadow-free albedo can be recovered. The
experiments on both synthetic and real data demonstrate the superior
performance of our approach compared to previous work and its capability to
synthesize realistic renderings under novel viewpoints and illumination. Our
code and data are available at https://zju3dv.github.io/invrender/.

---

## GARF: Gaussian Activated Radiance Fields for High Fidelity  Reconstruction and Pose Estimation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-12 | Shin-Fang Chng, Sameera Ramasinghe, Jamie Sherrah, Simon Lucey | cs.CV | [PDF](http://arxiv.org/pdf/2204.05735v1){: .btn .btn-green } |

**Abstract**: Despite Neural Radiance Fields (NeRF) showing compelling results in
photorealistic novel views synthesis of real-world scenes, most existing
approaches require accurate prior camera poses. Although approaches for jointly
recovering the radiance field and camera pose exist (BARF), they rely on a
cumbersome coarse-to-fine auxiliary positional embedding to ensure good
performance. We present Gaussian Activated neural Radiance Fields (GARF), a new
positional embedding-free neural radiance field architecture - employing
Gaussian activations - that outperforms the current state-of-the-art in terms
of high fidelity reconstruction and pose estimation.

Comments:
- Project page: https://sfchng.github.io/garf/

---

## NAN: Noise-Aware NeRFs for Burst-Denoising

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-10 | Naama Pearl, Tali Treibitz, Simon Korman | cs.CV | [PDF](http://arxiv.org/pdf/2204.04668v2){: .btn .btn-green } |

**Abstract**: Burst denoising is now more relevant than ever, as computational photography
helps overcome sensitivity issues inherent in mobile phones and small cameras.
A major challenge in burst-denoising is in coping with pixel misalignment,
which was so far handled with rather simplistic assumptions of simple motion,
or the ability to align in pre-processing. Such assumptions are not realistic
in the presence of large motion and high levels of noise. We show that Neural
Radiance Fields (NeRFs), originally suggested for physics-based novel-view
rendering, can serve as a powerful framework for burst denoising. NeRFs have an
inherent capability of handling noise as they integrate information from
multiple images, but they are limited in doing so, mainly since they build on
pixel-wise operations which are suitable to ideal imaging conditions. Our
approach, termed NAN, leverages inter-view and spatial information in NeRFs to
better deal with noise. It achieves state-of-the-art results in burst denoising
and is especially successful in coping with large movement and occlusions,
under very high levels of noise. With the rapid advances in accelerating NeRFs,
it could provide a powerful platform for denoising in challenging environments.

Comments:
- to appear at CVPR 2022

---

## Gravitationally Lensed Black Hole Emission Tomography

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-07 | Aviad Levis, Pratul P. Srinivasan, Andrew A. Chael, Ren Ng, Katherine L. Bouman | cs.CV | [PDF](http://arxiv.org/pdf/2204.03715v1){: .btn .btn-green } |

**Abstract**: Measurements from the Event Horizon Telescope enabled the visualization of
light emission around a black hole for the first time. So far, these
measurements have been used to recover a 2D image under the assumption that the
emission field is static over the period of acquisition. In this work, we
propose BH-NeRF, a novel tomography approach that leverages gravitational
lensing to recover the continuous 3D emission field near a black hole. Compared
to other 3D reconstruction or tomography settings, this task poses two
significant challenges: first, rays near black holes follow curved paths
dictated by general relativity, and second, we only observe measurements from a
single viewpoint. Our method captures the unknown emission field using a
continuous volumetric function parameterized by a coordinate-based neural
network, and uses knowledge of Keplerian orbital dynamics to establish
correspondence between 3D points over time. Together, these enable BH-NeRF to
recover accurate 3D emission fields, even in challenging situations with sparse
measurements and uncertain orbital dynamics. This work takes the first steps in
showing how future measurements from the Event Horizon Telescope could be used
to recover evolving 3D emission around the supermassive black hole in our
Galactic center.

Comments:
- To appear in the IEEE Proceedings of the Conference on Computer
  Vision and Pattern Recognition (CVPR), 2022. Supplemental material including
  accompanying pdf, code, and video highlight can be found in the project page:
  http://imaging.cms.caltech.edu/bhnerf/

---

## SqueezeNeRF: Further factorized FastNeRF for memory-efficient inference

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-06 | Krishna Wadhwani, Tamaki Kojima | cs.CV | [PDF](http://arxiv.org/pdf/2204.02585v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) has emerged as the state-of-the-art method for
novel view generation of complex scenes, but is very slow during inference.
Recently, there have been multiple works on speeding up NeRF inference, but the
state of the art methods for real-time NeRF inference rely on caching the
neural network output, which occupies several giga-bytes of disk space that
limits their real-world applicability. As caching the neural network of
original NeRF network is not feasible, Garbin et al. proposed "FastNeRF" which
factorizes the problem into 2 sub-networks - one which depends only on the 3D
coordinate of a sample point and one which depends only on the 2D camera
viewing direction. Although this factorization enables them to reduce the cache
size and perform inference at over 200 frames per second, the memory overhead
is still substantial. In this work, we propose SqueezeNeRF, which is more than
60 times memory-efficient than the sparse cache of FastNeRF and is still able
to render at more than 190 frames per second on a high spec GPU during
inference.

Comments:
- 9 pages, 3 figures, 5 tables. Presented in the "5th Efficient Deep
  Learning for Computer Vision" CVPR 2022 Workshop"

---

## Unified Implicit Neural Stylization



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-05 | Zhiwen Fan, Yifan Jiang, Peihao Wang, Xinyu Gong, Dejia Xu, Zhangyang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2204.01943v3){: .btn .btn-green } |

**Abstract**: Representing visual signals by implicit representation (e.g., a coordinate
based deep network) has prevailed among many vision tasks. This work explores a
new intriguing direction: training a stylized implicit representation, using a
generalized approach that can apply to various 2D and 3D scenarios. We conduct
a pilot study on a variety of implicit functions, including 2D coordinate-based
representation, neural radiance field, and signed distance function. Our
solution is a Unified Implicit Neural Stylization framework, dubbed INS. In
contrary to vanilla implicit representation, INS decouples the ordinary
implicit function into a style implicit module and a content implicit module,
in order to separately encode the representations from the style image and
input scenes. An amalgamation module is then applied to aggregate these
information and synthesize the stylized output. To regularize the geometry in
3D scenes, we propose a novel self-distillation geometry consistency loss which
preserves the geometry fidelity of the stylized scenes. Comprehensive
experiments are conducted on multiple task settings, including novel view
synthesis of complex scenes, stylization for implicit surfaces, and fitting
images using MLPs. We further demonstrate that the learned representation is
continuous not only spatially but also style-wise, leading to effortlessly
interpolating between different styles and generating images with new mixed
styles. Please refer to the video on our project page for more view synthesis
results: https://zhiwenfan.github.io/INS.

---

## Neural Rendering of Humans in Novel View and Pose from Monocular Video



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-04 | Tiantian Wang, Nikolaos Sarafianos, Ming-Hsuan Yang, Tony Tung | cs.CV | [PDF](http://arxiv.org/pdf/2204.01218v2){: .btn .btn-green } |

**Abstract**: We introduce a new method that generates photo-realistic humans under novel
views and poses given a monocular video as input. Despite the significant
progress recently on this topic, with several methods exploring shared
canonical neural radiance fields in dynamic scene scenarios, learning a
user-controlled model for unseen poses remains a challenging task. To tackle
this problem, we introduce an effective method to a) integrate observations
across several frames and b) encode the appearance at each individual frame. We
accomplish this by utilizing both the human pose that models the body shape as
well as point clouds that partially cover the human as input. Our approach
simultaneously learns a shared set of latent codes anchored to the human pose
among several frames, and an appearance-dependent code anchored to incomplete
point clouds generated by each frame and its predicted depth. The former human
pose-based code models the shape of the performer whereas the latter point
cloud-based code predicts fine-level details and reasons about missing
structures at the unseen poses. To further recover non-visible regions in query
frames, we employ a temporal transformer to integrate features of points in
query frames and tracked body points from automatically-selected key frames.
Experiments on various sequences of dynamic humans from different datasets
including ZJU-MoCap show that our method significantly outperforms existing
approaches under unseen poses and novel views given monocular videos as input.

Comments:
- 10 pages

---

## SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single  Image

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-04-02 | Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Humphrey Shi, Zhangyang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2204.00928v2){: .btn .btn-green } |

**Abstract**: Despite the rapid development of Neural Radiance Field (NeRF), the necessity
of dense covers largely prohibits its wider applications. While several recent
works have attempted to address this issue, they either operate with sparse
views (yet still, a few of them) or on simple objects/scenes. In this work, we
consider a more ambitious task: training neural radiance field, over
realistically complex visual scenes, by "looking only once", i.e., using only a
single view. To attain this goal, we present a Single View NeRF (SinNeRF)
framework consisting of thoughtfully designed semantic and geometry
regularizations. Specifically, SinNeRF constructs a semi-supervised learning
process, where we introduce and propagate geometry pseudo labels and semantic
pseudo labels to guide the progressive training process. Extensive experiments
are conducted on complex scene benchmarks, including NeRF synthetic dataset,
Local Light Field Fusion dataset, and DTU dataset. We show that even without
pre-training on multi-view datasets, SinNeRF can yield photo-realistic
novel-view synthesis results. Under the single image setting, SinNeRF
significantly outperforms the current state-of-the-art NeRF baselines in all
cases. Project page: https://vita-group.github.io/SinNeRF/

Comments:
- Project page: https://vita-group.github.io/SinNeRF/