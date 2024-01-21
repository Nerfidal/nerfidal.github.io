---
layout: default
title: November
parent: 2022
nav_order: 11
---
<!---metadata--->

## NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-shot Real  Image Animation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-30 | Yu Yin, Kamran Ghasedi, HsiangTao Wu, Jiaolong Yang, Xin Tong, Yun Fu | cs.CV | [PDF](http://arxiv.org/pdf/2211.17235v1){: .btn .btn-green } |

**Abstract**: Nerf-based Generative models have shown impressive capacity in generating
high-quality images with consistent 3D geometry. Despite successful synthesis
of fake identity images randomly sampled from latent space, adopting these
models for generating face images of real subjects is still a challenging task
due to its so-called inversion issue. In this paper, we propose a universal
method to surgically fine-tune these NeRF-GAN models in order to achieve
high-fidelity animation of real subjects only by a single image. Given the
optimized latent code for an out-of-domain real image, we employ 2D loss
functions on the rendered image to reduce the identity gap. Furthermore, our
method leverages explicit and implicit 3D regularizations using the in-domain
neighborhood samples around the optimized latent code to remove geometrical and
visual artifacts. Our experiments confirm the effectiveness of our method in
realistic, high-fidelity, and 3D consistent animation of real faces on multiple
NeRF-GAN models across different datasets.

---

## DINER: Depth-aware Image-based NEural Radiance fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-29 | Malte Prinzler, Otmar Hilliges, Justus Thies | cs.CV | [PDF](http://arxiv.org/pdf/2211.16630v2){: .btn .btn-green } |

**Abstract**: We present Depth-aware Image-based NEural Radiance fields (DINER). Given a
sparse set of RGB input views, we predict depth and feature maps to guide the
reconstruction of a volumetric scene representation that allows us to render 3D
objects under novel views. Specifically, we propose novel techniques to
incorporate depth information into feature fusion and efficient scene sampling.
In comparison to the previous state of the art, DINER achieves higher synthesis
quality and can process input views with greater disparity. This allows us to
capture scenes more completely without changing capturing hardware requirements
and ultimately enables larger viewpoint changes during novel view synthesis. We
evaluate our method by synthesizing novel views, both for human heads and for
general objects, and observe significantly improved qualitative results and
increased perceptual metrics compared to the previous state of the art. The
code is publicly available for research purposes.

Comments:
- Website: https://malteprinzler.github.io/projects/diner/diner.html ;
  Video: https://www.youtube.com/watch?v=iI_fpjY5k8Y&t=1s

---

## NeuralLift-360: Lifting An In-the-wild 2D Photo to A 3D Object with  360° Views

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-29 | Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Yi Wang, Zhangyang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2211.16431v2){: .btn .btn-green } |

**Abstract**: Virtual reality and augmented reality (XR) bring increasing demand for 3D
content. However, creating high-quality 3D content requires tedious work that a
human expert must do. In this work, we study the challenging task of lifting a
single image to a 3D object and, for the first time, demonstrate the ability to
generate a plausible 3D object with 360{\deg} views that correspond well with
the given reference image. By conditioning on the reference image, our model
can fulfill the everlasting curiosity for synthesizing novel views of objects
from images. Our technique sheds light on a promising direction of easing the
workflows for 3D artists and XR designers. We propose a novel framework, dubbed
NeuralLift-360, that utilizes a depth-aware neural radiance representation
(NeRF) and learns to craft the scene guided by denoising diffusion models. By
introducing a ranking loss, our NeuralLift-360 can be guided with rough depth
estimation in the wild. We also adopt a CLIP-guided sampling strategy for the
diffusion prior to provide coherent guidance. Extensive experiments demonstrate
that our NeuralLift-360 significantly outperforms existing state-of-the-art
baselines. Project page: https://vita-group.github.io/NeuralLift-360/

Comments:
- Project page: https://vita-group.github.io/NeuralLift-360/

---

## Compressing Volumetric Radiance Fields to 1 MB

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-29 | Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, Liefeng Bo | cs.CV | [PDF](http://arxiv.org/pdf/2211.16386v1){: .btn .btn-green } |

**Abstract**: Approximating radiance fields with volumetric grids is one of promising
directions for improving NeRF, represented by methods like Plenoxels and DVGO,
which achieve super-fast training convergence and real-time rendering. However,
these methods typically require a tremendous storage overhead, costing up to
hundreds of megabytes of disk space and runtime memory for a single scene. We
address this issue in this paper by introducing a simple yet effective
framework, called vector quantized radiance fields (VQRF), for compressing
these volume-grid-based radiance fields. We first present a robust and adaptive
metric for estimating redundancy in grid models and performing voxel pruning by
better exploring intermediate outputs of volumetric rendering. A trainable
vector quantization is further proposed to improve the compactness of grid
models. In combination with an efficient joint tuning strategy and
post-processing, our method can achieve a compression ratio of 100$\times$ by
reducing the overall model size to 1 MB with negligible loss on visual quality.
Extensive experiments demonstrate that the proposed framework is capable of
achieving unrivaled performance and well generalization across multiple methods
with distinct volumetric structures, facilitating the wide use of volumetric
radiance fields methods in real-world applications. Code Available at
\url{https://github.com/AlgoHunt/VQRF}

---

## One is All: Bridging the Gap Between Neural Radiance Fields  Architectures with Progressive Volume Distillation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-29 | Shuangkang Fang, Weixin Xu, Heng Wang, Yi Yang, Yufeng Wang, Shuchang Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2211.15977v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) methods have proved effective as compact,
high-quality and versatile representations for 3D scenes, and enable downstream
tasks such as editing, retrieval, navigation, etc. Various neural architectures
are vying for the core structure of NeRF, including the plain Multi-Layer
Perceptron (MLP), sparse tensors, low-rank tensors, hashtables and their
compositions. Each of these representations has its particular set of
trade-offs. For example, the hashtable-based representations admit faster
training and rendering but their lack of clear geometric meaning hampers
downstream tasks like spatial-relation-aware editing. In this paper, we propose
Progressive Volume Distillation (PVD), a systematic distillation method that
allows any-to-any conversions between different architectures, including MLP,
sparse or low-rank tensors, hashtables and their compositions. PVD consequently
empowers downstream applications to optimally adapt the neural representations
for the task at hand in a post hoc fashion. The conversions are fast, as
distillation is progressively performed on different levels of volume
representations, from shallower to deeper. We also employ special treatment of
density to deal with its specific numerical instability problem. Empirical
evidence is presented to validate our method on the NeRF-Synthetic, LLFF and
TanksAndTemples datasets. For example, with PVD, an MLP-based NeRF model can be
distilled from a hashtable-based Instant-NGP model at a 10X~20X faster speed
than being trained the original NeRF from scratch, while achieving a superior
level of synthesis quality. Code is available at
https://github.com/megvii-research/AAAI2023-PVD.

Comments:
- Accepted by AAAI2023. Project Page: https://sk-fun.fun/PVD

---

## In-Hand 3D Object Scanning from an RGB Sequence

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-28 | Shreyas Hampali, Tomas Hodan, Luan Tran, Lingni Ma, Cem Keskin, Vincent Lepetit | cs.CV | [PDF](http://arxiv.org/pdf/2211.16193v2){: .btn .btn-green } |

**Abstract**: We propose a method for in-hand 3D scanning of an unknown object with a
monocular camera. Our method relies on a neural implicit surface representation
that captures both the geometry and the appearance of the object, however, by
contrast with most NeRF-based methods, we do not assume that the camera-object
relative poses are known. Instead, we simultaneously optimize both the object
shape and the pose trajectory. As direct optimization over all shape and pose
parameters is prone to fail without coarse-level initialization, we propose an
incremental approach that starts by splitting the sequence into carefully
selected overlapping segments within which the optimization is likely to
succeed. We reconstruct the object shape and track its poses independently
within each segment, then merge all the segments before performing a global
optimization. We show that our method is able to reconstruct the shape and
color of both textured and challenging texture-less objects, outperforms
classical methods that rely only on appearance features, and that its
performance is close to recent methods that assume known camera poses.

Comments:
- CVPR 2023

---

## High-fidelity Facial Avatar Reconstruction from Monocular Video with  Generative Priors

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-28 | Yunpeng Bai, Yanbo Fan, Xuan Wang, Yong Zhang, Jingxiang Sun, Chun Yuan, Ying Shan | cs.CV | [PDF](http://arxiv.org/pdf/2211.15064v2){: .btn .btn-green } |

**Abstract**: High-fidelity facial avatar reconstruction from a monocular video is a
significant research problem in computer graphics and computer vision.
Recently, Neural Radiance Field (NeRF) has shown impressive novel view
rendering results and has been considered for facial avatar reconstruction.
However, the complex facial dynamics and missing 3D information in monocular
videos raise significant challenges for faithful facial reconstruction. In this
work, we propose a new method for NeRF-based facial avatar reconstruction that
utilizes 3D-aware generative prior. Different from existing works that depend
on a conditional deformation field for dynamic modeling, we propose to learn a
personalized generative prior, which is formulated as a local and low
dimensional subspace in the latent space of 3D-GAN. We propose an efficient
method to construct the personalized generative prior based on a small set of
facial images of a given individual. After learning, it allows for
photo-realistic rendering with novel views and the face reenactment can be
realized by performing navigation in the latent space. Our proposed method is
applicable for different driven signals, including RGB images, 3DMM
coefficients, and audios. Compared with existing works, we obtain superior
novel view synthesis results and faithfully face reenactment performance.

Comments:
- 8 pages, 7 figures

---

## 3D Scene Creation and Rendering via Rough Meshes: A Lighting Transfer  Avenue

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-27 | Yujie Li, Bowen Cai, Yuqin Liang, Rongfei Jia, Binqiang Zhao, Mingming Gong, Huan Fu | cs.CV | [PDF](http://arxiv.org/pdf/2211.14823v2){: .btn .btn-green } |

**Abstract**: This paper studies how to flexibly integrate reconstructed 3D models into
practical 3D modeling pipelines such as 3D scene creation and rendering. Due to
the technical difficulty, one can only obtain rough 3D models (R3DMs) for most
real objects using existing 3D reconstruction techniques. As a result,
physically-based rendering (PBR) would render low-quality images or videos for
scenes that are constructed by R3DMs. One promising solution would be
representing real-world objects as Neural Fields such as NeRFs, which are able
to generate photo-realistic renderings of an object under desired viewpoints.
However, a drawback is that the synthesized views through Neural Fields
Rendering (NFR) cannot reflect the simulated lighting details on R3DMs in PBR
pipelines, especially when object interactions in the 3D scene creation cause
local shadows. To solve this dilemma, we propose a lighting transfer network
(LighTNet) to bridge NFR and PBR, such that they can benefit from each other.
LighTNet reasons about a simplified image composition model, remedies the
uneven surface issue caused by R3DMs, and is empowered by several
perceptual-motivated constraints and a new Lab angle loss which enhances the
contrast between lighting strength and colors. Comparisons demonstrate that
LighTNet is superior in synthesizing impressive lighting, and is promising in
pushing NFR further in practical 3D modeling workflows. Project page:
https://3d-front-future.github.io/LighTNet .

---

## Sampling Neural Radiance Fields for Refractive Objects

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-27 | Jen-I Pan, Jheng-Wei Su, Kai-Wen Hsiao, Ting-Yu Yen, Hung-Kuo Chu | cs.CV | [PDF](http://arxiv.org/pdf/2211.14799v1){: .btn .btn-green } |

**Abstract**: Recently, differentiable volume rendering in neural radiance fields (NeRF)
has gained a lot of popularity, and its variants have attained many impressive
results. However, existing methods usually assume the scene is a homogeneous
volume so that a ray is cast along the straight path. In this work, the scene
is instead a heterogeneous volume with a piecewise-constant refractive index,
where the path will be curved if it intersects the different refractive
indices. For novel view synthesis of refractive objects, our NeRF-based
framework aims to optimize the radiance fields of bounded volume and boundary
from multi-view posed images with refractive object silhouettes. To tackle this
challenging problem, the refractive index of a scene is reconstructed from
silhouettes. Given the refractive index, we extend the stratified and
hierarchical sampling techniques in NeRF to allow drawing samples along a
curved path tracked by the Eikonal equation. The results indicate that our
framework outperforms the state-of-the-art method both quantitatively and
qualitatively, demonstrating better performance on the perceptual similarity
metric and an apparent improvement in the rendering quality on several
synthetic and real scenes.

Comments:
- SIGGRAPH Asia 2022 Technical Communications. 4 pages, 4 figures, 1
  table. Project: https://alexkeroro86.github.io/SampleNeRFRO/ Code:
  https://github.com/alexkeroro86/SampleNeRFRO

---

## SuNeRF: Validation of a 3D Global Reconstruction of the Solar Corona  Using Simulated EUV Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-27 | Kyriaki-Margarita Bintsi, Robert Jarolim, Benoit Tremblay, Miraflor Santos, Anna Jungbluth, James Paul Mason, Sairam Sundaresan, Angelos Vourlidas, Cooper Downs, Ronald M. Caplan, Andrés Muñoz Jaramillo | astro-ph.SR | [PDF](http://arxiv.org/pdf/2211.14879v1){: .btn .btn-green } |

**Abstract**: Extreme Ultraviolet (EUV) light emitted by the Sun impacts satellite
operations and communications and affects the habitability of planets.
Currently, EUV-observing instruments are constrained to viewing the Sun from
its equator (i.e., ecliptic), limiting our ability to forecast EUV emission for
other viewpoints (e.g. solar poles), and to generalize our knowledge of the
Sun-Earth system to other host stars. In this work, we adapt Neural Radiance
Fields (NeRFs) to the physical properties of the Sun and demonstrate that
non-ecliptic viewpoints could be reconstructed from observations limited to the
solar ecliptic. To validate our approach, we train on simulations of solar EUV
emission that provide a ground truth for all viewpoints. Our model accurately
reconstructs the simulated 3D structure of the Sun, achieving a peak
signal-to-noise ratio of 43.3 dB and a mean absolute relative error of 0.3\%
for non-ecliptic viewpoints. Our method provides a consistent 3D reconstruction
of the Sun from a limited number of viewpoints, thus highlighting the potential
to create a virtual instrument for satellite observations of the Sun. Its
extension to real observations will provide the missing link to compare the Sun
to other stars and to improve space-weather forecasting.

Comments:
- Accepted at Machine Learning and the Physical Sciences workshop,
  NeurIPS 2022

---

## ResNeRF: Geometry-Guided Residual Neural Radiance Field for Indoor Scene  Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-26 | Yuting Xiao, Yiqun Zhao, Yanyu Xu, Shenghua Gao | cs.CV | [PDF](http://arxiv.org/pdf/2211.16211v3){: .btn .btn-green } |

**Abstract**: We represent the ResNeRF, a novel geometry-guided two-stage framework for
indoor scene novel view synthesis. Be aware of that a good geometry would
greatly boost the performance of novel view synthesis, and to avoid the
geometry ambiguity issue, we propose to characterize the density distribution
of the scene based on a base density estimated from scene geometry and a
residual density parameterized by the geometry. In the first stage, we focus on
geometry reconstruction based on SDF representation, which would lead to a good
geometry surface of the scene and also a sharp density. In the second stage,
the residual density is learned based on the SDF learned in the first stage for
encoding more details about the appearance. In this way, our method can better
learn the density distribution with the geometry prior for high-fidelity novel
view synthesis while preserving the 3D structures. Experiments on large-scale
indoor scenes with many less-observed and textureless areas show that with the
good 3D surface, our method achieves state-of-the-art performance for novel
view synthesis.

Comments:
- This is an incomplete paper

---

## 3DDesigner: Towards Photorealistic 3D Object Generation and Editing with  Text-guided Diffusion Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-25 | Gang Li, Heliang Zheng, Chaoyue Wang, Chang Li, Changwen Zheng, Dacheng Tao | cs.CV | [PDF](http://arxiv.org/pdf/2211.14108v3){: .btn .btn-green } |

**Abstract**: Text-guided diffusion models have shown superior performance in image/video
generation and editing. While few explorations have been performed in 3D
scenarios. In this paper, we discuss three fundamental and interesting problems
on this topic. First, we equip text-guided diffusion models to achieve
3D-consistent generation. Specifically, we integrate a NeRF-like neural field
to generate low-resolution coarse results for a given camera view. Such results
can provide 3D priors as condition information for the following diffusion
process. During denoising diffusion, we further enhance the 3D consistency by
modeling cross-view correspondences with a novel two-stream (corresponding to
two different views) asynchronous diffusion process. Second, we study 3D local
editing and propose a two-step solution that can generate 360-degree
manipulated results by editing an object from a single view. Step 1, we propose
to perform 2D local editing by blending the predicted noises. Step 2, we
conduct a noise-to-text inversion process that maps 2D blended noises into the
view-independent text embedding space. Once the corresponding text embedding is
obtained, 360-degree images can be generated. Last but not least, we extend our
model to perform one-shot novel view synthesis by fine-tuning on a single
image, firstly showing the potential of leveraging text guidance for novel view
synthesis. Extensive experiments and various applications show the prowess of
our 3DDesigner. The project page is available at
https://3ddesigner-diffusion.github.io/.

Comments:
- Submitted to IJCV

---

## ShadowNeuS: Neural SDF Reconstruction by Shadow Ray Supervision

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-25 | Jingwang Ling, Zhibo Wang, Feng Xu | cs.CV | [PDF](http://arxiv.org/pdf/2211.14086v2){: .btn .btn-green } |

**Abstract**: By supervising camera rays between a scene and multi-view image planes, NeRF
reconstructs a neural scene representation for the task of novel view
synthesis. On the other hand, shadow rays between the light source and the
scene have yet to be considered. Therefore, we propose a novel shadow ray
supervision scheme that optimizes both the samples along the ray and the ray
location. By supervising shadow rays, we successfully reconstruct a neural SDF
of the scene from single-view images under multiple lighting conditions. Given
single-view binary shadows, we train a neural network to reconstruct a complete
scene not limited by the camera's line of sight. By further modeling the
correlation between the image colors and the shadow rays, our technique can
also be effectively extended to RGB inputs. We compare our method with previous
works on challenging tasks of shape reconstruction from single-view binary
shadow or RGB images and observe significant improvements. The code and data
are available at https://github.com/gerwang/ShadowNeuS.

Comments:
- CVPR 2023. Project page: https://gerwang.github.io/shadowneus/

---

## Dynamic Neural Portraits

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-25 | Michail Christos Doukas, Stylianos Ploumpis, Stefanos Zafeiriou | cs.CV | [PDF](http://arxiv.org/pdf/2211.13994v1){: .btn .btn-green } |

**Abstract**: We present Dynamic Neural Portraits, a novel approach to the problem of
full-head reenactment. Our method generates photo-realistic video portraits by
explicitly controlling head pose, facial expressions and eye gaze. Our proposed
architecture is different from existing methods that rely on GAN-based
image-to-image translation networks for transforming renderings of 3D faces
into photo-realistic images. Instead, we build our system upon a 2D
coordinate-based MLP with controllable dynamics. Our intuition to adopt a
2D-based representation, as opposed to recent 3D NeRF-like systems, stems from
the fact that video portraits are captured by monocular stationary cameras,
therefore, only a single viewpoint of the scene is available. Primarily, we
condition our generative model on expression blendshapes, nonetheless, we show
that our system can be successfully driven by audio features as well. Our
experiments demonstrate that the proposed method is 270 times faster than
recent NeRF-based reenactment methods, with our networks achieving speeds of 24
fps for resolutions up to 1024 x 1024, while outperforming prior works in terms
of visual quality.

Comments:
- In IEEE/CVF Winter Conference on Applications of Computer Vision
  (WACV) 2023

---

## Unsupervised Continual Semantic Adaptation through Neural Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-25 | Zhizheng Liu, Francesco Milano, Jonas Frey, Roland Siegwart, Hermann Blum, Cesar Cadena | cs.CV | [PDF](http://arxiv.org/pdf/2211.13969v2){: .btn .btn-green } |

**Abstract**: An increasing amount of applications rely on data-driven models that are
deployed for perception tasks across a sequence of scenes. Due to the mismatch
between training and deployment data, adapting the model on the new scenes is
often crucial to obtain good performance. In this work, we study continual
multi-scene adaptation for the task of semantic segmentation, assuming that no
ground-truth labels are available during deployment and that performance on the
previous scenes should be maintained. We propose training a Semantic-NeRF
network for each scene by fusing the predictions of a segmentation model and
then using the view-consistent rendered semantic labels as pseudo-labels to
adapt the model. Through joint training with the segmentation model, the
Semantic-NeRF model effectively enables 2D-3D knowledge transfer. Furthermore,
due to its compact size, it can be stored in a long-term memory and
subsequently used to render data from arbitrary viewpoints to reduce
forgetting. We evaluate our approach on ScanNet, where we outperform both a
voxel-based baseline and a state-of-the-art unsupervised domain adaptation
method.

Comments:
- Accepted by the IEEE/CVF Conference on Computer Vision and Pattern
  Recognition (CVPR) 2023. Zhizheng Liu and Francesco Milano share first
  authorship. Hermann Blum and Cesar Cadena share senior authorship. 18 pages,
  8 figures, 9 tables

---

## TPA-Net: Generate A Dataset for Text to Physics-based Animation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-25 | Yuxing Qiu, Feng Gao, Minchen Li, Govind Thattai, Yin Yang, Chenfanfu Jiang | cs.AI | [PDF](http://arxiv.org/pdf/2211.13887v1){: .btn .btn-green } |

**Abstract**: Recent breakthroughs in Vision-Language (V&L) joint research have achieved
remarkable results in various text-driven tasks. High-quality Text-to-video
(T2V), a task that has been long considered mission-impossible, was proven
feasible with reasonably good results in latest works. However, the resulting
videos often have undesired artifacts largely because the system is purely
data-driven and agnostic to the physical laws. To tackle this issue and further
push T2V towards high-level physical realism, we present an autonomous data
generation technique and a dataset, which intend to narrow the gap with a large
number of multi-modal, 3D Text-to-Video/Simulation (T2V/S) data. In the
dataset, we provide high-resolution 3D physical simulations for both solids and
fluids, along with textual descriptions of the physical phenomena. We take
advantage of state-of-the-art physical simulation methods (i) Incremental
Potential Contact (IPC) and (ii) Material Point Method (MPM) to simulate
diverse scenarios, including elastic deformations, material fractures,
collisions, turbulence, etc. Additionally, high-quality, multi-view rendering
videos are supplied for the benefit of T2V, Neural Radiance Fields (NeRF), and
other communities. This work is the first step towards fully automated
Text-to-Video/Simulation (T2V/S). Live examples and subsequent work are at
https://sites.google.com/view/tpa-net.

---

## ScanNeRF: a Scalable Benchmark for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-24 | Luca De Luigi, Damiano Bolognini, Federico Domeniconi, Daniele De Gregorio, Matteo Poggi, Luigi Di Stefano | cs.CV | [PDF](http://arxiv.org/pdf/2211.13762v2){: .btn .btn-green } |

**Abstract**: In this paper, we propose the first-ever real benchmark thought for
evaluating Neural Radiance Fields (NeRFs) and, in general, Neural Rendering
(NR) frameworks. We design and implement an effective pipeline for scanning
real objects in quantity and effortlessly. Our scan station is built with less
than 500$ hardware budget and can collect roughly 4000 images of a scanned
object in just 5 minutes. Such a platform is used to build ScanNeRF, a dataset
characterized by several train/val/test splits aimed at benchmarking the
performance of modern NeRF methods under different conditions. Accordingly, we
evaluate three cutting-edge NeRF variants on it to highlight their strengths
and weaknesses. The dataset is available on our project page, together with an
online benchmark to foster the development of better and better NeRFs.

Comments:
- WACV 2023. The first three authors contributed equally. Project page:
  https://eyecan-ai.github.io/scannerf/

---

## Immersive Neural Graphics Primitives

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-24 | Ke Li, Tim Rolff, Susanne Schmidt, Reinhard Bacher, Simone Frintrop, Wim Leemans, Frank Steinicke | cs.CV | [PDF](http://arxiv.org/pdf/2211.13494v1){: .btn .btn-green } |

**Abstract**: Neural radiance field (NeRF), in particular its extension by instant neural
graphics primitives, is a novel rendering method for view synthesis that uses
real-world images to build photo-realistic immersive virtual scenes. Despite
its potential, research on the combination of NeRF and virtual reality (VR)
remains sparse. Currently, there is no integration into typical VR systems
available, and the performance and suitability of NeRF implementations for VR
have not been evaluated, for instance, for different scene complexities or
screen resolutions. In this paper, we present and evaluate a NeRF-based
framework that is capable of rendering scenes in immersive VR allowing users to
freely move their heads to explore complex real-world scenes. We evaluate our
framework by benchmarking three different NeRF scenes concerning their
rendering performance at different scene complexities and resolutions.
Utilizing super-resolution, our approach can yield a frame rate of 30 frames
per second with a resolution of 1280x720 pixels per eye. We discuss potential
applications of our framework and provide an open source implementation online.

Comments:
- Submitted to IEEE VR, currently under review

---

## BAD-NeRF: Bundle Adjusted Deblur Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-23 | Peng Wang, Lingzhe Zhao, Ruijie Ma, Peidong Liu | cs.CV | [PDF](http://arxiv.org/pdf/2211.12853v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have received considerable attention recently,
due to its impressive capability in photo-realistic 3D reconstruction and novel
view synthesis, given a set of posed camera images. Earlier work usually
assumes the input images are of good quality. However, image degradation (e.g.
image motion blur in low-light conditions) can easily happen in real-world
scenarios, which would further affect the rendering quality of NeRF. In this
paper, we present a novel bundle adjusted deblur Neural Radiance Fields
(BAD-NeRF), which can be robust to severe motion blurred images and inaccurate
camera poses. Our approach models the physical image formation process of a
motion blurred image, and jointly learns the parameters of NeRF and recovers
the camera motion trajectories during exposure time. In experiments, we show
that by directly modeling the real physical image formation process, BAD-NeRF
achieves superior performance over prior works on both synthetic and real
datasets. Code and data are available at https://github.com/WU-CVGL/BAD-NeRF.

Comments:
- Accepted to CVPR 2023, Project page:
  https://wangpeng000.github.io/BAD-NeRF/

---

## ActiveRMAP: Radiance Field for Active Mapping And Planning

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-23 | Huangying Zhan, Jiyang Zheng, Yi Xu, Ian Reid, Hamid Rezatofighi | cs.CV | [PDF](http://arxiv.org/pdf/2211.12656v1){: .btn .btn-green } |

**Abstract**: A high-quality 3D reconstruction of a scene from a collection of 2D images
can be achieved through offline/online mapping methods. In this paper, we
explore active mapping from the perspective of implicit representations, which
have recently produced compelling results in a variety of applications. One of
the most popular implicit representations - Neural Radiance Field (NeRF), first
demonstrated photorealistic rendering results using multi-layer perceptrons,
with promising offline 3D reconstruction as a by-product of the radiance field.
More recently, researchers also applied this implicit representation for online
reconstruction and localization (i.e. implicit SLAM systems). However, the
study on using implicit representation for active vision tasks is still very
limited. In this paper, we are particularly interested in applying the neural
radiance field for active mapping and planning problems, which are closely
coupled tasks in an active system. We, for the first time, present an RGB-only
active vision framework using radiance field representation for active 3D
reconstruction and planning in an online manner. Specifically, we formulate
this joint task as an iterative dual-stage optimization problem, where we
alternatively optimize for the radiance field representation and path planning.
Experimental results suggest that the proposed method achieves competitive
results compared to other offline methods and outperforms active reconstruction
methods using NeRFs.

Comments:
- Under review

---

## PANeRF: Pseudo-view Augmentation for Improved Neural Radiance Fields  Based on Few-shot Inputs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-23 | Young Chun Ahn, Seokhwan Jang, Sungheon Park, Ji-Yeon Kim, Nahyup Kang | cs.CV | [PDF](http://arxiv.org/pdf/2211.12758v1){: .btn .btn-green } |

**Abstract**: The method of neural radiance fields (NeRF) has been developed in recent
years, and this technology has promising applications for synthesizing novel
views of complex scenes. However, NeRF requires dense input views, typically
numbering in the hundreds, for generating high-quality images. With a decrease
in the number of input views, the rendering quality of NeRF for unseen
viewpoints tends to degenerate drastically. To overcome this challenge, we
propose pseudo-view augmentation of NeRF, a scheme that expands a sufficient
amount of data by considering the geometry of few-shot inputs. We first
initialized the NeRF network by leveraging the expanded pseudo-views, which
efficiently minimizes uncertainty when rendering unseen views. Subsequently, we
fine-tuned the network by utilizing sparse-view inputs containing precise
geometry and color information. Through experiments under various settings, we
verified that our model faithfully synthesizes novel-view images of superior
quality and outperforms existing methods for multi-view datasets.

---

## CGOF++: Controllable 3D Face Synthesis with Conditional Generative  Occupancy Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-23 | Keqiang Sun, Shangzhe Wu, Ning Zhang, Zhaoyang Huang, Quan Wang, Hongsheng Li | cs.CV | [PDF](http://arxiv.org/pdf/2211.13251v2){: .btn .btn-green } |

**Abstract**: Capitalizing on the recent advances in image generation models, existing
controllable face image synthesis methods are able to generate high-fidelity
images with some levels of controllability, e.g., controlling the shapes,
expressions, textures, and poses of the generated face images. However,
previous methods focus on controllable 2D image generative models, which are
prone to producing inconsistent face images under large expression and pose
changes. In this paper, we propose a new NeRF-based conditional 3D face
synthesis framework, which enables 3D controllability over the generated face
images by imposing explicit 3D conditions from 3D face priors. At its core is a
conditional Generative Occupancy Field (cGOF++) that effectively enforces the
shape of the generated face to conform to a given 3D Morphable Model (3DMM)
mesh, built on top of EG3D [1], a recent tri-plane-based generative model. To
achieve accurate control over fine-grained 3D face shapes of the synthesized
images, we additionally incorporate a 3D landmark loss as well as a volume
warping loss into our synthesis framework. Experiments validate the
effectiveness of the proposed method, which is able to generate high-fidelity
face images and shows more precise 3D controllability than state-of-the-art
2D-based controllable face synthesis methods.

Comments:
- Accepted to IEEE Transactions on Pattern Analysis and Machine
  Intelligence (TPAMI). This article is an extension of the NeurIPS'22 paper
  arXiv:2206.08361

---

## AvatarMAV: Fast 3D Head Avatar Reconstruction Using Motion-Aware Neural  Voxels

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-23 | Yuelang Xu, Lizhen Wang, Xiaochen Zhao, Hongwen Zhang, Yebin Liu | cs.CV | [PDF](http://arxiv.org/pdf/2211.13206v3){: .btn .btn-green } |

**Abstract**: With NeRF widely used for facial reenactment, recent methods can recover
photo-realistic 3D head avatar from just a monocular video. Unfortunately, the
training process of the NeRF-based methods is quite time-consuming, as MLP used
in the NeRF-based methods is inefficient and requires too many iterations to
converge. To overcome this problem, we propose AvatarMAV, a fast 3D head avatar
reconstruction method using Motion-Aware Neural Voxels. AvatarMAV is the first
to model both the canonical appearance and the decoupled expression motion by
neural voxels for head avatar. In particular, the motion-aware neural voxels is
generated from the weighted concatenation of multiple 4D tensors. The 4D
tensors semantically correspond one-to-one with 3DMM expression basis and share
the same weights as 3DMM expression coefficients. Benefiting from our novel
representation, the proposed AvatarMAV can recover photo-realistic head avatars
in just 5 minutes (implemented with pure PyTorch), which is significantly
faster than the state-of-the-art facial reenactment methods. Project page:
https://www.liuyebin.com/avatarmav.

Comments:
- Accepted by SIGGRAPH 2023

---

## ClimateNeRF: Extreme Weather Synthesis in Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-23 | Yuan Li, Zhi-Hao Lin, David Forsyth, Jia-Bin Huang, Shenlong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2211.13226v3){: .btn .btn-green } |

**Abstract**: Physical simulations produce excellent predictions of weather effects. Neural
radiance fields produce SOTA scene models. We describe a novel NeRF-editing
procedure that can fuse physical simulations with NeRF models of scenes,
producing realistic movies of physical phenomena in those scenes. Our
application -- Climate NeRF -- allows people to visualize what climate change
outcomes will do to them. ClimateNeRF allows us to render realistic weather
effects, including smog, snow, and flood. Results can be controlled with
physically meaningful variables like water level. Qualitative and quantitative
studies show that our simulated results are significantly more realistic than
those from SOTA 2D image editing and SOTA 3D NeRF stylization.

Comments:
- project page: https://climatenerf.github.io/

---

## ONeRF: Unsupervised 3D Object Segmentation from Multiple Views

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-22 | Shengnan Liang, Yichen Liu, Shangzhe Wu, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2211.12038v1){: .btn .btn-green } |

**Abstract**: We present ONeRF, a method that automatically segments and reconstructs
object instances in 3D from multi-view RGB images without any additional manual
annotations. The segmented 3D objects are represented using separate Neural
Radiance Fields (NeRFs) which allow for various 3D scene editing and novel view
rendering. At the core of our method is an unsupervised approach using the
iterative Expectation-Maximization algorithm, which effectively aggregates 2D
visual features and the corresponding 3D cues from multi-views for joint 3D
object segmentation and reconstruction. Unlike existing approaches that can
only handle simple objects, our method produces segmented full 3D NeRFs of
individual objects with complex shapes, topologies and appearance. The
segmented ONeRfs enable a range of 3D scene editing, such as object
transformation, insertion and deletion.

---

## DP-NeRF: Deblurred Neural Radiance Field with Physical Scene Priors

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-22 | Dogyoon Lee, Minhyeok Lee, Chajin Shin, Sangyoun Lee | cs.CV | [PDF](http://arxiv.org/pdf/2211.12046v4){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has exhibited outstanding three-dimensional (3D)
reconstruction quality via the novel view synthesis from multi-view images and
paired calibrated camera parameters. However, previous NeRF-based systems have
been demonstrated under strictly controlled settings, with little attention
paid to less ideal scenarios, including with the presence of noise such as
exposure, illumination changes, and blur. In particular, though blur frequently
occurs in real situations, NeRF that can handle blurred images has received
little attention. The few studies that have investigated NeRF for blurred
images have not considered geometric and appearance consistency in 3D space,
which is one of the most important factors in 3D reconstruction. This leads to
inconsistency and the degradation of the perceptual quality of the constructed
scene. Hence, this paper proposes a DP-NeRF, a novel clean NeRF framework for
blurred images, which is constrained with two physical priors. These priors are
derived from the actual blurring process during image acquisition by the
camera. DP-NeRF proposes rigid blurring kernel to impose 3D consistency
utilizing the physical priors and adaptive weight proposal to refine the color
composition error in consideration of the relationship between depth and blur.
We present extensive experimental results for synthetic and real scenes with
two types of blur: camera motion blur and defocus blur. The results demonstrate
that DP-NeRF successfully improves the perceptual quality of the constructed
NeRF ensuring 3D geometric and appearance consistency. We further demonstrate
the effectiveness of our model with comprehensive ablation analysis.

Comments:
- Accepted at CVPR 2023, Code: https://github.com/dogyoonlee/DP-NeRF,
  Project page: https://dogyoonlee.github.io/dpnerf/

---

## SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with Neural  Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-22 | Ashkan Mirzaei, Tristan Aumentado-Armstrong, Konstantinos G. Derpanis, Jonathan Kelly, Marcus A. Brubaker, Igor Gilitschenski, Alex Levinshtein | cs.CV | [PDF](http://arxiv.org/pdf/2211.12254v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have emerged as a popular approach for novel
view synthesis. While NeRFs are quickly being adapted for a wider set of
applications, intuitively editing NeRF scenes is still an open challenge. One
important editing task is the removal of unwanted objects from a 3D scene, such
that the replaced region is visually plausible and consistent with its context.
We refer to this task as 3D inpainting. In 3D, solutions must be both
consistent across multiple views and geometrically valid. In this paper, we
propose a novel 3D inpainting method that addresses these challenges. Given a
small set of posed images and sparse annotations in a single input image, our
framework first rapidly obtains a 3D segmentation mask for a target object.
Using the mask, a perceptual optimizationbased approach is then introduced that
leverages learned 2D image inpainters, distilling their information into 3D
space, while ensuring view consistency. We also address the lack of a diverse
benchmark for evaluating 3D scene inpainting methods by introducing a dataset
comprised of challenging real-world scenes. In particular, our dataset contains
views of the same scene with and without a target object, enabling more
principled benchmarking of the 3D inpainting task. We first demonstrate the
superiority of our approach on multiview segmentation, comparing to NeRFbased
methods and 2D segmentation approaches. We then evaluate on the task of 3D
inpainting, establishing state-ofthe-art performance against other NeRF
manipulation algorithms, as well as a strong 2D image inpainter baseline.
Project Page: https://spinnerf3d.github.io

Comments:
- Project Page: https://spinnerf3d.github.io

---

## Exact-NeRF: An Exploration of a Precise Volumetric Parameterization for  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-22 | Brian K. S. Isaac-Medina, Chris G. Willcocks, Toby P. Breckon | cs.CV | [PDF](http://arxiv.org/pdf/2211.12285v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have attracted significant attention due to
their ability to synthesize novel scene views with great accuracy. However,
inherent to their underlying formulation, the sampling of points along a ray
with zero width may result in ambiguous representations that lead to further
rendering artifacts such as aliasing in the final scene. To address this issue,
the recent variant mip-NeRF proposes an Integrated Positional Encoding (IPE)
based on a conical view frustum. Although this is expressed with an integral
formulation, mip-NeRF instead approximates this integral as the expected value
of a multivariate Gaussian distribution. This approximation is reliable for
short frustums but degrades with highly elongated regions, which arises when
dealing with distant scene objects under a larger depth of field. In this
paper, we explore the use of an exact approach for calculating the IPE by using
a pyramid-based integral formulation instead of an approximated conical-based
one. We denote this formulation as Exact-NeRF and contribute the first approach
to offer a precise analytical solution to the IPE within the NeRF domain. Our
exploratory work illustrates that such an exact formulation Exact-NeRF matches
the accuracy of mip-NeRF and furthermore provides a natural extension to more
challenging scenarios without further modification, such as in the case of
unbounded scenes. Our contribution aims to both address the hitherto unexplored
issues of frustum approximation in earlier NeRF work and additionally provide
insight into the potential future consideration of analytical solutions in
future NeRF extensions.

Comments:
- 15 pages,10 figures

---

## Real-time Neural Radiance Talking Portrait Synthesis via Audio-spatial  Decomposition

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-22 | Jiaxiang Tang, Kaisiyuan Wang, Hang Zhou, Xiaokang Chen, Dongliang He, Tianshu Hu, Jingtuo Liu, Gang Zeng, Jingdong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2211.12368v1){: .btn .btn-green } |

**Abstract**: While dynamic Neural Radiance Fields (NeRF) have shown success in
high-fidelity 3D modeling of talking portraits, the slow training and inference
speed severely obstruct their potential usage. In this paper, we propose an
efficient NeRF-based framework that enables real-time synthesizing of talking
portraits and faster convergence by leveraging the recent success of grid-based
NeRF. Our key insight is to decompose the inherently high-dimensional talking
portrait representation into three low-dimensional feature grids. Specifically,
a Decomposed Audio-spatial Encoding Module models the dynamic head with a 3D
spatial grid and a 2D audio grid. The torso is handled with another 2D grid in
a lightweight Pseudo-3D Deformable Module. Both modules focus on efficiency
under the premise of good rendering quality. Extensive experiments demonstrate
that our method can generate realistic and audio-lips synchronized talking
portrait videos, while also being highly efficient compared to previous
methods.

Comments:
- Project page: https://me.kiui.moe/radnerf/

---

## Instant Volumetric Head Avatars



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-22 | Wojciech Zielonka, Timo Bolkart, Justus Thies | cs.CV | [PDF](http://arxiv.org/pdf/2211.12499v2){: .btn .btn-green } |

**Abstract**: We present Instant Volumetric Head Avatars (INSTA), a novel approach for
reconstructing photo-realistic digital avatars instantaneously. INSTA models a
dynamic neural radiance field based on neural graphics primitives embedded
around a parametric face model. Our pipeline is trained on a single monocular
RGB portrait video that observes the subject under different expressions and
views. While state-of-the-art methods take up to several days to train an
avatar, our method can reconstruct a digital avatar in less than 10 minutes on
modern GPU hardware, which is orders of magnitude faster than previous
solutions. In addition, it allows for the interactive rendering of novel poses
and expressions. By leveraging the geometry prior of the underlying parametric
face model, we demonstrate that INSTA extrapolates to unseen poses. In
quantitative and qualitative studies on various subjects, INSTA outperforms
state-of-the-art methods regarding rendering quality and training time.

Comments:
- Website: https://zielon.github.io/insta/ Video:
  https://youtu.be/HOgaeWTih7Q Accepted to CVPR2023

---

## Zero NeRF: Registration with Zero Overlap

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-22 | Casey Peat, Oliver Batchelor, Richard Green, James Atlas | cs.CV | [PDF](http://arxiv.org/pdf/2211.12544v1){: .btn .btn-green } |

**Abstract**: We present Zero-NeRF, a projective surface registration method that, to the
best of our knowledge, offers the first general solution capable of alignment
between scene representations with minimal or zero visual correspondence. To do
this, we enforce consistency between visible surfaces of partial and complete
reconstructions, which allows us to constrain occluded geometry. We use a NeRF
as our surface representation and the NeRF rendering pipeline to perform this
alignment. To demonstrate the efficacy of our method, we register real-world
scenes from opposite sides with infinitesimal overlaps that cannot be
accurately registered using prior methods, and we compare these results against
widely used registration methods.

---

## Dynamic Depth-Supervised NeRF for Multi-View RGB-D Operating Room Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-22 | Beerend G. A. Gerats, Jelmer M. Wolterink, Ivo A. M. J. Broeders | cs.CV | [PDF](http://arxiv.org/pdf/2211.12436v2){: .btn .btn-green } |

**Abstract**: The operating room (OR) is an environment of interest for the development of
sensing systems, enabling the detection of people, objects, and their semantic
relations. Due to frequent occlusions in the OR, these systems often rely on
input from multiple cameras. While increasing the number of cameras generally
increases algorithm performance, there are hard limitations to the number and
locations of cameras in the OR. Neural Radiance Fields (NeRF) can be used to
render synthetic views from arbitrary camera positions, virtually enlarging the
number of cameras in the dataset. In this work, we explore the use of NeRF for
view synthesis of dynamic scenes in the OR, and we show that regularisation
with depth supervision from RGB-D sensor data results in higher image quality.
We optimise a dynamic depth-supervised NeRF with up to six synchronised cameras
that capture the surgical field in five distinct phases before and during a
knee replacement surgery. We qualitatively inspect views rendered by a virtual
camera that moves 180 degrees around the surgical field at differing time
values. Quantitatively, we evaluate view synthesis from an unseen camera
position in terms of PSNR, SSIM and LPIPS for the colour channels and in MAE
and error percentage for the estimated depth. We find that NeRFs can be used to
generate geometrically consistent views, also from interpolated camera
positions and at interpolated time intervals. Views are generated from an
unseen camera pose with an average PSNR of 18.2 and a depth estimation error of
2.0%. Our results show the potential of a dynamic NeRF for view synthesis in
the OR and stress the relevance of depth supervision in a clinical setting.

Comments:
- Accepted to the Workshop on Ambient Intelligence for HealthCare 2023

---

## NeRF-RPN: A general framework for object detection in NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-21 | Benran Hu, Junkai Huang, Yichen Liu, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2211.11646v3){: .btn .btn-green } |

**Abstract**: This paper presents the first significant object detection framework,
NeRF-RPN, which directly operates on NeRF. Given a pre-trained NeRF model,
NeRF-RPN aims to detect all bounding boxes of objects in a scene. By exploiting
a novel voxel representation that incorporates multi-scale 3D neural volumetric
features, we demonstrate it is possible to regress the 3D bounding boxes of
objects in NeRF directly without rendering the NeRF at any viewpoint. NeRF-RPN
is a general framework and can be applied to detect objects without class
labels. We experimented NeRF-RPN with various backbone architectures, RPN head
designs and loss functions. All of them can be trained in an end-to-end manner
to estimate high quality 3D bounding boxes. To facilitate future research in
object detection for NeRF, we built a new benchmark dataset which consists of
both synthetic and real-world data with careful labeling and clean up. Code and
dataset are available at https://github.com/lyclyc52/NeRF_RPN.

Comments:
- Accepted by CVPR 2023

---

## FLNeRF: 3D Facial Landmarks Estimation in Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-21 | Hao Zhang, Tianyuan Dai, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2211.11202v3){: .btn .btn-green } |

**Abstract**: This paper presents the first significant work on directly predicting 3D face
landmarks on neural radiance fields (NeRFs). Our 3D coarse-to-fine Face
Landmarks NeRF (FLNeRF) model efficiently samples from a given face NeRF with
individual facial features for accurate landmarks detection. Expression
augmentation is applied to facial features in a fine scale to simulate large
emotions range including exaggerated facial expressions (e.g., cheek blowing,
wide opening mouth, eye blinking) for training FLNeRF. Qualitative and
quantitative comparison with related state-of-the-art 3D facial landmark
estimation methods demonstrate the efficacy of FLNeRF, which contributes to
downstream tasks such as high-quality face editing and swapping with direct
control using our NeRF landmarks. Code and data will be available. Github link:
https://github.com/ZHANG1023/FLNeRF.

Comments:
- Hao Zhang and Tianyuan Dai contributed equally. Project website:
  https://github.com/ZHANG1023/FLNeRF

---

## SegNeRF: 3D Part Segmentation with Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-21 | Jesus Zarzar, Sara Rojas, Silvio Giancola, Bernard Ghanem | cs.CV | [PDF](http://arxiv.org/pdf/2211.11215v2){: .btn .btn-green } |

**Abstract**: Recent advances in Neural Radiance Fields (NeRF) boast impressive
performances for generative tasks such as novel view synthesis and 3D
reconstruction. Methods based on neural radiance fields are able to represent
the 3D world implicitly by relying exclusively on posed images. Yet, they have
seldom been explored in the realm of discriminative tasks such as 3D part
segmentation. In this work, we attempt to bridge that gap by proposing SegNeRF:
a neural field representation that integrates a semantic field along with the
usual radiance field. SegNeRF inherits from previous works the ability to
perform novel view synthesis and 3D reconstruction, and enables 3D part
segmentation from a few images. Our extensive experiments on PartNet show that
SegNeRF is capable of simultaneously predicting geometry, appearance, and
semantic information from posed images, even for unseen objects. The predicted
semantic fields allow SegNeRF to achieve an average mIoU of $\textbf{30.30%}$
for 2D novel view segmentation, and $\textbf{37.46%}$ for 3D part segmentation,
boasting competitive performance against point-based methods by using only a
few posed images. Additionally, SegNeRF is able to generate an explicit 3D
model from a single image of an object taken in the wild, with its
corresponding part segmentation.

Comments:
- Fixed abstract typo

---

## Local-to-Global Registration for Bundle-Adjusting Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-21 | Yue Chen, Xingyu Chen, Xuan Wang, Qi Zhang, Yu Guo, Ying Shan, Fei Wang | cs.CV | [PDF](http://arxiv.org/pdf/2211.11505v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have achieved photorealistic novel views
synthesis; however, the requirement of accurate camera poses limits its
application. Despite analysis-by-synthesis extensions for jointly learning
neural 3D representations and registering camera frames exist, they are
susceptible to suboptimal solutions if poorly initialized. We propose L2G-NeRF,
a Local-to-Global registration method for bundle-adjusting Neural Radiance
Fields: first, a pixel-wise flexible alignment, followed by a frame-wise
constrained parametric alignment. Pixel-wise local alignment is learned in an
unsupervised way via a deep network which optimizes photometric reconstruction
errors. Frame-wise global alignment is performed using differentiable parameter
estimation solvers on the pixel-wise correspondences to find a global
transformation. Experiments on synthetic and real-world data show that our
method outperforms the current state-of-the-art in terms of high-fidelity
reconstruction and resolving large camera pose misalignment. Our module is an
easy-to-use plugin that can be applied to NeRF variants and other neural field
applications. The Code and supplementary materials are available at
https://rover-xingyu.github.io/L2G-NeRF/.

Comments:
- Accepted to CVPR 2023

---

## Shape, Pose, and Appearance from a Single Image via Bootstrapped  Radiance Field Inversion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-21 | Dario Pavllo, David Joseph Tan, Marie-Julie Rakotosaona, Federico Tombari | cs.CV | [PDF](http://arxiv.org/pdf/2211.11674v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) coupled with GANs represent a promising
direction in the area of 3D reconstruction from a single view, owing to their
ability to efficiently model arbitrary topologies. Recent work in this area,
however, has mostly focused on synthetic datasets where exact ground-truth
poses are known, and has overlooked pose estimation, which is important for
certain downstream applications such as augmented reality (AR) and robotics. We
introduce a principled end-to-end reconstruction framework for natural images,
where accurate ground-truth poses are not available. Our approach recovers an
SDF-parameterized 3D shape, pose, and appearance from a single image of an
object, without exploiting multiple views during training. More specifically,
we leverage an unconditional 3D-aware generator, to which we apply a hybrid
inversion scheme where a model produces a first guess of the solution which is
then refined via optimization. Our framework can de-render an image in as few
as 10 steps, enabling its use in practical scenarios. We demonstrate
state-of-the-art results on a variety of real and synthetic benchmarks.

Comments:
- CVPR 2023. Code and models are available at
  https://github.com/google-research/nerf-from-image

---

## ESLAM: Efficient Dense SLAM System Based on Hybrid Representation of  Signed Distance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-21 | Mohammad Mahdi Johari, Camilla Carta, François Fleuret | cs.CV | [PDF](http://arxiv.org/pdf/2211.11704v2){: .btn .btn-green } |

**Abstract**: We present ESLAM, an efficient implicit neural representation method for
Simultaneous Localization and Mapping (SLAM). ESLAM reads RGB-D frames with
unknown camera poses in a sequential manner and incrementally reconstructs the
scene representation while estimating the current camera position in the scene.
We incorporate the latest advances in Neural Radiance Fields (NeRF) into a SLAM
system, resulting in an efficient and accurate dense visual SLAM method. Our
scene representation consists of multi-scale axis-aligned perpendicular feature
planes and shallow decoders that, for each point in the continuous space,
decode the interpolated features into Truncated Signed Distance Field (TSDF)
and RGB values. Our extensive experiments on three standard datasets, Replica,
ScanNet, and TUM RGB-D show that ESLAM improves the accuracy of 3D
reconstruction and camera localization of state-of-the-art dense visual SLAM
methods by more than 50%, while it runs up to 10 times faster and does not
require any pre-training.

Comments:
- CVPR 2023 Highlight. Project page: https://www.idiap.ch/paper/eslam/

---

## SPARF: Neural Radiance Fields from Sparse and Noisy Poses

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-21 | Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt, Federico Tombari | cs.CV | [PDF](http://arxiv.org/pdf/2211.11738v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has recently emerged as a powerful
representation to synthesize photorealistic novel views. While showing
impressive performance, it relies on the availability of dense input views with
highly accurate camera poses, thus limiting its application in real-world
scenarios. In this work, we introduce Sparse Pose Adjusting Radiance Field
(SPARF), to address the challenge of novel-view synthesis given only few
wide-baseline input images (as low as 3) with noisy camera poses. Our approach
exploits multi-view geometry constraints in order to jointly learn the NeRF and
refine the camera poses. By relying on pixel matches extracted between the
input views, our multi-view correspondence objective enforces the optimized
scene and camera poses to converge to a global and geometrically accurate
solution. Our depth consistency loss further encourages the reconstructed scene
to be consistent from any viewpoint. Our approach sets a new state of the art
in the sparse-view regime on multiple challenging datasets.

Comments:
- Code is released at https://github.com/google-research/sparf.
  Published at CVPR 2023 as a Highlight

---

## Towards Live 3D Reconstruction from Wearable Video: An Evaluation of  V-SLAM, NeRF, and Videogrammetry Techniques

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-21 | David Ramirez, Suren Jayasuriya, Andreas Spanias | eess.IV | [PDF](http://arxiv.org/pdf/2211.11836v1){: .btn .btn-green } |

**Abstract**: Mixed reality (MR) is a key technology which promises to change the future of
warfare. An MR hybrid of physical outdoor environments and virtual military
training will enable engagements with long distance enemies, both real and
simulated. To enable this technology, a large-scale 3D model of a physical
environment must be maintained based on live sensor observations. 3D
reconstruction algorithms should utilize the low cost and pervasiveness of
video camera sensors, from both overhead and soldier-level perspectives.
Mapping speed and 3D quality can be balanced to enable live MR training in
dynamic environments. Given these requirements, we survey several 3D
reconstruction algorithms for large-scale mapping for military applications
given only live video. We measure 3D reconstruction performance from common
structure from motion, visual-SLAM, and photogrammetry techniques. This
includes the open source algorithms COLMAP, ORB-SLAM3, and NeRF using
Instant-NGP. We utilize the autonomous driving academic benchmark KITTI, which
includes both dashboard camera video and lidar produced 3D ground truth. With
the KITTI data, our primary contribution is a quantitative evaluation of 3D
reconstruction computational speed when considering live video.

Comments:
- Accepted to 2022 Interservice/Industry Training, Simulation, and
  Education Conference (I/ITSEC), 13 pages

---

## DynIBaR: Neural Dynamic Image-Based Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-20 | Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, Noah Snavely | cs.CV | [PDF](http://arxiv.org/pdf/2211.11082v3){: .btn .btn-green } |

**Abstract**: We address the problem of synthesizing novel views from a monocular video
depicting a complex dynamic scene. State-of-the-art methods based on temporally
varying Neural Radiance Fields (aka dynamic NeRFs) have shown impressive
results on this task. However, for long videos with complex object motions and
uncontrolled camera trajectories, these methods can produce blurry or
inaccurate renderings, hampering their use in real-world applications. Instead
of encoding the entire dynamic scene within the weights of MLPs, we present a
new approach that addresses these limitations by adopting a volumetric
image-based rendering framework that synthesizes new viewpoints by aggregating
features from nearby views in a scene-motion-aware manner. Our system retains
the advantages of prior methods in its ability to model complex scenes and
view-dependent effects, but also enables synthesizing photo-realistic novel
views from long videos featuring complex scene dynamics with unconstrained
camera trajectories. We demonstrate significant improvements over
state-of-the-art methods on dynamic scene datasets, and also apply our approach
to in-the-wild videos with challenging camera and object motion, where prior
methods fail to produce high-quality renderings. Our project webpage is at
dynibar.github.io.

Comments:
- Award Candidate, CVPR 2023 Project page: dynibar.github.io

---

## Magic3D: High-Resolution Text-to-3D Content Creation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-18 | Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, Tsung-Yi Lin | cs.CV | [PDF](http://arxiv.org/pdf/2211.10440v2){: .btn .btn-green } |

**Abstract**: DreamFusion has recently demonstrated the utility of a pre-trained
text-to-image diffusion model to optimize Neural Radiance Fields (NeRF),
achieving remarkable text-to-3D synthesis results. However, the method has two
inherent limitations: (a) extremely slow optimization of NeRF and (b)
low-resolution image space supervision on NeRF, leading to low-quality 3D
models with a long processing time. In this paper, we address these limitations
by utilizing a two-stage optimization framework. First, we obtain a coarse
model using a low-resolution diffusion prior and accelerate with a sparse 3D
hash grid structure. Using the coarse representation as the initialization, we
further optimize a textured 3D mesh model with an efficient differentiable
renderer interacting with a high-resolution latent diffusion model. Our method,
dubbed Magic3D, can create high quality 3D mesh models in 40 minutes, which is
2x faster than DreamFusion (reportedly taking 1.5 hours on average), while also
achieving higher resolution. User studies show 61.7% raters to prefer our
approach over DreamFusion. Together with the image-conditioned generation
capabilities, we provide users with new ways to control 3D synthesis, opening
up new avenues to various creative applications.

Comments:
- Accepted to CVPR 2023 as highlight. Project website:
  https://research.nvidia.com/labs/dir/magic3d

---

## Neural Fields for Fast and Scalable Interpolation of Geophysical Ocean  Variables

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-18 | J. Emmanuel Johnson, Redouane Lguensat, Ronan Fablet, Emmanuel Cosme, Julien Le Sommer | physics.ao-ph | [PDF](http://arxiv.org/pdf/2211.10444v1){: .btn .btn-green } |

**Abstract**: Optimal Interpolation (OI) is a widely used, highly trusted algorithm for
interpolation and reconstruction problems in geosciences. With the influx of
more satellite missions, we have access to more and more observations and it is
becoming more pertinent to take advantage of these observations in applications
such as forecasting and reanalysis. With the increase in the volume of
available data, scalability remains an issue for standard OI and it prevents
many practitioners from effectively and efficiently taking advantage of these
large sums of data to learn the model hyperparameters. In this work, we
leverage recent advances in Neural Fields (NerFs) as an alternative to the OI
framework where we show how they can be easily applied to standard
reconstruction problems in physical oceanography. We illustrate the relevance
of NerFs for gap-filling of sparse measurements of sea surface height (SSH) via
satellite altimetry and demonstrate how NerFs are scalable with comparable
results to the standard OI. We find that NerFs are a practical set of methods
that can be readily applied to geoscience interpolation problems and we
anticipate a wider adoption in the future.

Comments:
- Machine Learning and the Physical Sciences workshop, NeurIPS 2022

---

## AligNeRF: High-Fidelity Neural Radiance Fields via Alignment-Aware  Training

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-17 | Yifan Jiang, Peter Hedman, Ben Mildenhall, Dejia Xu, Jonathan T. Barron, Zhangyang Wang, Tianfan Xue | cs.CV | [PDF](http://arxiv.org/pdf/2211.09682v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) are a powerful representation for modeling a
3D scene as a continuous function. Though NeRF is able to render complex 3D
scenes with view-dependent effects, few efforts have been devoted to exploring
its limits in a high-resolution setting. Specifically, existing NeRF-based
methods face several limitations when reconstructing high-resolution real
scenes, including a very large number of parameters, misaligned input data, and
overly smooth details. In this work, we conduct the first pilot study on
training NeRF with high-resolution data and propose the corresponding
solutions: 1) marrying the multilayer perceptron (MLP) with convolutional
layers which can encode more neighborhood information while reducing the total
number of parameters; 2) a novel training strategy to address misalignment
caused by moving objects or small camera calibration errors; and 3) a
high-frequency aware loss. Our approach is nearly free without introducing
obvious training/testing costs, while experiments on different datasets
demonstrate that it can recover more high-frequency details compared with the
current state-of-the-art NeRF models. Project page:
\url{https://yifanjiang.net/alignerf.}

---

## CoNFies: Controllable Neural Face Avatars

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-16 | Heng Yu, Koichiro Niinuma, Laszlo A. Jeni | cs.CV | [PDF](http://arxiv.org/pdf/2211.08610v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) are compelling techniques for modeling dynamic
3D scenes from 2D image collections. These volumetric representations would be
well suited for synthesizing novel facial expressions but for two problems.
First, deformable NeRFs are object agnostic and model holistic movement of the
scene: they can replay how the motion changes over time, but they cannot alter
it in an interpretable way. Second, controllable volumetric representations
typically require either time-consuming manual annotations or 3D supervision to
provide semantic meaning to the scene. We propose a controllable neural
representation for face self-portraits (CoNFies), that solves both of these
problems within a common framework, and it can rely on automated processing. We
use automated facial action recognition (AFAR) to characterize facial
expressions as a combination of action units (AU) and their intensities. AUs
provide both the semantic locations and control labels for the system. CoNFies
outperformed competing methods for novel view and expression synthesis in terms
of visual and anatomic fidelity of expressions.

Comments:
- accepted by FG2023

---

## NeRFFaceEditing: Disentangled Face Editing in Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-15 | Kaiwen Jiang, Shu-Yu Chen, Feng-Lin Liu, Hongbo Fu, Lin Gao | cs.GR | [PDF](http://arxiv.org/pdf/2211.07968v1){: .btn .btn-green } |

**Abstract**: Recent methods for synthesizing 3D-aware face images have achieved rapid
development thanks to neural radiance fields, allowing for high quality and
fast inference speed. However, existing solutions for editing facial geometry
and appearance independently usually require retraining and are not optimized
for the recent work of generation, thus tending to lag behind the generation
process. To address these issues, we introduce NeRFFaceEditing, which enables
editing and decoupling geometry and appearance in the pretrained
tri-plane-based neural radiance field while retaining its high quality and fast
inference speed. Our key idea for disentanglement is to use the statistics of
the tri-plane to represent the high-level appearance of its corresponding
facial volume. Moreover, we leverage a generated 3D-continuous semantic mask as
an intermediary for geometry editing. We devise a geometry decoder (whose
output is unchanged when the appearance changes) and an appearance decoder. The
geometry decoder aligns the original facial volume with the semantic mask
volume. We also enhance the disentanglement by explicitly regularizing rendered
images with the same appearance but different geometry to be similar in terms
of color distribution for each facial component separately. Our method allows
users to edit via semantic masks with decoupled control of geometry and
appearance. Both qualitative and quantitative evaluations show the superior
geometry and appearance control abilities of our method compared to existing
and alternative solutions.

---

## Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-14 | Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, Daniel Cohen-Or | cs.CV | [PDF](http://arxiv.org/pdf/2211.07600v1){: .btn .btn-green } |

**Abstract**: Text-guided image generation has progressed rapidly in recent years,
inspiring major breakthroughs in text-guided shape generation. Recently, it has
been shown that using score distillation, one can successfully text-guide a
NeRF model to generate a 3D object. We adapt the score distillation to the
publicly available, and computationally efficient, Latent Diffusion Models,
which apply the entire diffusion process in a compact latent space of a
pretrained autoencoder. As NeRFs operate in image space, a naive solution for
guiding them with latent score distillation would require encoding to the
latent space at each guidance step. Instead, we propose to bring the NeRF to
the latent space, resulting in a Latent-NeRF. Analyzing our Latent-NeRF, we
show that while Text-to-3D models can generate impressive results, they are
inherently unconstrained and may lack the ability to guide or enforce a
specific 3D structure. To assist and direct the 3D generation, we propose to
guide our Latent-NeRF using a Sketch-Shape: an abstract geometry that defines
the coarse structure of the desired object. Then, we present means to integrate
such a constraint directly into a Latent-NeRF. This unique combination of text
and shape guidance allows for increased control over the generation process. We
also show that latent score distillation can be successfully applied directly
on 3D meshes. This allows for generating high-quality textures on a given
geometry. Our experiments validate the power of our different forms of guidance
and the efficiency of using latent rendering. Implementation is available at
https://github.com/eladrich/latent-nerf

---

## 3D-Aware Encoding for Style-based Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-12 | Yu-Jhe Li, Tao Xu, Bichen Wu, Ningyuan Zheng, Xiaoliang Dai, Albert Pumarola, Peizhao Zhang, Peter Vajda, Kris Kitani | cs.CV | [PDF](http://arxiv.org/pdf/2211.06583v1){: .btn .btn-green } |

**Abstract**: We tackle the task of NeRF inversion for style-based neural radiance fields,
(e.g., StyleNeRF). In the task, we aim to learn an inversion function to
project an input image to the latent space of a NeRF generator and then
synthesize novel views of the original image based on the latent code. Compared
with GAN inversion for 2D generative models, NeRF inversion not only needs to
1) preserve the identity of the input image, but also 2) ensure 3D consistency
in generated novel views. This requires the latent code obtained from the
single-view image to be invariant across multiple views. To address this new
challenge, we propose a two-stage encoder for style-based NeRF inversion. In
the first stage, we introduce a base encoder that converts the input image to a
latent code. To ensure the latent code is view-invariant and is able to
synthesize 3D consistent novel view images, we utilize identity contrastive
learning to train the base encoder. Second, to better preserve the identity of
the input image, we introduce a refining encoder to refine the latent code and
add finer details to the output image. Importantly note that the novelty of
this model lies in the design of its first-stage encoder which produces the
closest latent code lying on the latent manifold and thus the refinement in the
second stage would be close to the NeRF manifold. Through extensive
experiments, we demonstrate that our proposed two-stage encoder qualitatively
and quantitatively exhibits superiority over the existing encoders for
inversion in both image reconstruction and novel-view rendering.

Comments:
- 21 pages (under review)

---

## ParticleNeRF: A Particle-Based Encoding for Online Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-08 | Jad Abou-Chakra, Feras Dayoub, Niko Sünderhauf | cs.CV | [PDF](http://arxiv.org/pdf/2211.04041v4){: .btn .btn-green } |

**Abstract**: While existing Neural Radiance Fields (NeRFs) for dynamic scenes are offline
methods with an emphasis on visual fidelity, our paper addresses the online use
case that prioritises real-time adaptability. We present ParticleNeRF, a new
approach that dynamically adapts to changes in the scene geometry by learning
an up-to-date representation online, every 200ms. ParticleNeRF achieves this
using a novel particle-based parametric encoding. We couple features to
particles in space and backpropagate the photometric reconstruction loss into
the particles' position gradients, which are then interpreted as velocity
vectors. Governed by a lightweight physics system to handle collisions, this
lets the features move freely with the changing scene geometry. We demonstrate
ParticleNeRF on various dynamic scenes containing translating, rotating,
articulated, and deformable objects. ParticleNeRF is the first online dynamic
NeRF and achieves fast adaptability with better visual fidelity than
brute-force online InstantNGP and other baseline approaches on dynamic scenes
with online constraints. Videos of our system can be found at our project
website https://sites.google.com/view/particlenerf.

---

## Common Pets in 3D: Dynamic New-View Synthesis of Real-Life Deformable  Categories

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-07 | Samarth Sinha, Roman Shapovalov, Jeremy Reizenstein, Ignacio Rocco, Natalia Neverova, Andrea Vedaldi, David Novotny | cs.CV | [PDF](http://arxiv.org/pdf/2211.03889v1){: .btn .btn-green } |

**Abstract**: Obtaining photorealistic reconstructions of objects from sparse views is
inherently ambiguous and can only be achieved by learning suitable
reconstruction priors. Earlier works on sparse rigid object reconstruction
successfully learned such priors from large datasets such as CO3D. In this
paper, we extend this approach to dynamic objects. We use cats and dogs as a
representative example and introduce Common Pets in 3D (CoP3D), a collection of
crowd-sourced videos showing around 4,200 distinct pets. CoP3D is one of the
first large-scale datasets for benchmarking non-rigid 3D reconstruction "in the
wild". We also propose Tracker-NeRF, a method for learning 4D reconstruction
from our dataset. At test time, given a small number of video frames of an
unseen object, Tracker-NeRF predicts the trajectories of its 3D points and
generates new views, interpolating viewpoint and time. Results on CoP3D reveal
significantly better non-rigid new-view synthesis performance than existing
baselines.

---

## Learning-based Inverse Rendering of Complex Indoor Scenes with  Differentiable Monte Carlo Raytracing



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-06 | Jingsen Zhu, Fujun Luan, Yuchi Huo, Zihao Lin, Zhihua Zhong, Dianbing Xi, Jiaxiang Zheng, Rui Tang, Hujun Bao, Rui Wang | cs.CV | [PDF](http://arxiv.org/pdf/2211.03017v2){: .btn .btn-green } |

**Abstract**: Indoor scenes typically exhibit complex, spatially-varying appearance from
global illumination, making inverse rendering a challenging ill-posed problem.
This work presents an end-to-end, learning-based inverse rendering framework
incorporating differentiable Monte Carlo raytracing with importance sampling.
The framework takes a single image as input to jointly recover the underlying
geometry, spatially-varying lighting, and photorealistic materials.
Specifically, we introduce a physically-based differentiable rendering layer
with screen-space ray tracing, resulting in more realistic specular reflections
that match the input photo. In addition, we create a large-scale,
photorealistic indoor scene dataset with significantly richer details like
complex furniture and dedicated decorations. Further, we design a novel
out-of-view lighting network with uncertainty-aware refinement leveraging
hypernetwork-based neural radiance fields to predict lighting outside the view
of the input photo. Through extensive evaluations on common benchmark datasets,
we demonstrate superior inverse rendering quality of our method compared to
state-of-the-art baselines, enabling various applications such as complex
object insertion and material editing with high fidelity. Code and data will be
made available at \url{https://jingsenzhu.github.io/invrend}.

---

## nerf2nerf: Pairwise Registration of Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-11-03 | Lily Goli, Daniel Rebain, Sara Sabour, Animesh Garg, Andrea Tagliasacchi | cs.CV | [PDF](http://arxiv.org/pdf/2211.01600v1){: .btn .btn-green } |

**Abstract**: We introduce a technique for pairwise registration of neural fields that
extends classical optimization-based local registration (i.e. ICP) to operate
on Neural Radiance Fields (NeRF) -- neural 3D scene representations trained
from collections of calibrated images. NeRF does not decompose illumination and
color, so to make registration invariant to illumination, we introduce the
concept of a ''surface field'' -- a field distilled from a pre-trained NeRF
model that measures the likelihood of a point being on the surface of an
object. We then cast nerf2nerf registration as a robust optimization that
iteratively seeks a rigid transformation that aligns the surface fields of the
two scenes. We evaluate the effectiveness of our technique by introducing a
dataset of pre-trained NeRF scenes -- our synthetic scenes enable quantitative
evaluations and comparisons to classical registration techniques, while our
real scenes demonstrate the validity of our technique in real-world scenarios.
Additional results available at: https://nerf2nerf.github.io