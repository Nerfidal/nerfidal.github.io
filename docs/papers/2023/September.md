---
layout: default
title: September
parent: 2023
nav_order: 9
---
<!---metadata--->

## MMPI: a Flexible Radiance Field Representation by Multiple Multi-plane  Images Blending

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-30 | Yuze He, Peng Wang, Yubin Hu, Wang Zhao, Ran Yi, Yong-Jin Liu, Wenping Wang | cs.CV | [PDF](http://arxiv.org/pdf/2310.00249v1){: .btn .btn-green } |

**Abstract**: This paper presents a flexible representation of neural radiance fields based
on multi-plane images (MPI), for high-quality view synthesis of complex scenes.
MPI with Normalized Device Coordinate (NDC) parameterization is widely used in
NeRF learning for its simple definition, easy calculation, and powerful ability
to represent unbounded scenes. However, existing NeRF works that adopt MPI
representation for novel view synthesis can only handle simple forward-facing
unbounded scenes, where the input cameras are all observing in similar
directions with small relative translations. Hence, extending these MPI-based
methods to more complex scenes like large-range or even 360-degree scenes is
very challenging. In this paper, we explore the potential of MPI and show that
MPI can synthesize high-quality novel views of complex scenes with diverse
camera distributions and view directions, which are not only limited to simple
forward-facing scenes. Our key idea is to encode the neural radiance field with
multiple MPIs facing different directions and blend them with an adaptive
blending operation. For each region of the scene, the blending operation gives
larger blending weights to those advantaged MPIs with stronger local
representation abilities while giving lower weights to those with weaker
representation abilities. Such blending operation automatically modulates the
multiple MPIs to appropriately represent the diverse local density and color
information. Experiments on the KITTI dataset and ScanNet dataset demonstrate
that our proposed MMPI synthesizes high-quality images from diverse camera pose
distributions and is fast to train, outperforming the previous fast-training
NeRF methods for novel view synthesis. Moreover, we show that MMPI can encode
extremely long trajectories and produce novel view renderings, demonstrating
its potential in applications like autonomous driving.

---

## Multi-task View Synthesis with Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-29 | Shuhong Zheng, Zhipeng Bao, Martial Hebert, Yu-Xiong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2309.17450v1){: .btn .btn-green } |

**Abstract**: Multi-task visual learning is a critical aspect of computer vision. Current
research, however, predominantly concentrates on the multi-task dense
prediction setting, which overlooks the intrinsic 3D world and its multi-view
consistent structures, and lacks the capability for versatile imagination. In
response to these limitations, we present a novel problem setting -- multi-task
view synthesis (MTVS), which reinterprets multi-task prediction as a set of
novel-view synthesis tasks for multiple scene properties, including RGB. To
tackle the MTVS problem, we propose MuvieNeRF, a framework that incorporates
both multi-task and cross-view knowledge to simultaneously synthesize multiple
scene properties. MuvieNeRF integrates two key modules, the Cross-Task
Attention (CTA) and Cross-View Attention (CVA) modules, enabling the efficient
use of information across multiple views and tasks. Extensive evaluation on
both synthetic and realistic benchmarks demonstrates that MuvieNeRF is capable
of simultaneously synthesizing different scene properties with promising visual
quality, even outperforming conventional discriminative models in various
settings. Notably, we show that MuvieNeRF exhibits universal applicability
across a range of NeRF backbones. Our code is available at
https://github.com/zsh2000/MuvieNeRF.

Comments:
- ICCV 2023, Website: https://zsh2000.github.io/mtvs.github.io/

---

## Forward Flow for Novel View Synthesis of Dynamic Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-29 | Xiang Guo, Jiadai Sun, Yuchao Dai, Guanying Chen, Xiaoqing Ye, Xiao Tan, Errui Ding, Yumeng Zhang, Jingdong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2309.17390v1){: .btn .btn-green } |

**Abstract**: This paper proposes a neural radiance field (NeRF) approach for novel view
synthesis of dynamic scenes using forward warping. Existing methods often adopt
a static NeRF to represent the canonical space, and render dynamic images at
other time steps by mapping the sampled 3D points back to the canonical space
with the learned backward flow field. However, this backward flow field is
non-smooth and discontinuous, which is difficult to be fitted by commonly used
smooth motion models. To address this problem, we propose to estimate the
forward flow field and directly warp the canonical radiance field to other time
steps. Such forward flow field is smooth and continuous within the object
region, which benefits the motion model learning. To achieve this goal, we
represent the canonical radiance field with voxel grids to enable efficient
forward warping, and propose a differentiable warping process, including an
average splatting operation and an inpaint network, to resolve the many-to-one
and one-to-many mapping issues. Thorough experiments show that our method
outperforms existing methods in both novel view rendering and motion modeling,
demonstrating the effectiveness of our forward flow motion modeling. Project
page: https://npucvr.github.io/ForwardFlowDNeRF

Comments:
- Accepted by ICCV2023 as oral. Project page:
  https://npucvr.github.io/ForwardFlowDNeRF

---

## HAvatar: High-fidelity Head Avatar via Facial Model Conditioned Neural  Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-29 | Xiaochen Zhao, Lizhen Wang, Jingxiang Sun, Hongwen Zhang, Jinli Suo, Yebin Liu | cs.CV | [PDF](http://arxiv.org/pdf/2309.17128v1){: .btn .btn-green } |

**Abstract**: The problem of modeling an animatable 3D human head avatar under light-weight
setups is of significant importance but has not been well solved. Existing 3D
representations either perform well in the realism of portrait images synthesis
or the accuracy of expression control, but not both. To address the problem, we
introduce a novel hybrid explicit-implicit 3D representation, Facial Model
Conditioned Neural Radiance Field, which integrates the expressiveness of NeRF
and the prior information from the parametric template. At the core of our
representation, a synthetic-renderings-based condition method is proposed to
fuse the prior information from the parametric model into the implicit field
without constraining its topological flexibility. Besides, based on the hybrid
representation, we properly overcome the inconsistent shape issue presented in
existing methods and improve the animation stability. Moreover, by adopting an
overall GAN-based architecture using an image-to-image translation network, we
achieve high-resolution, realistic and view-consistent synthesis of dynamic
head appearance. Experiments demonstrate that our method can achieve
state-of-the-art performance for 3D head avatar animation compared with
previous methods.

---

## MatrixCity: A Large-scale City Dataset for City-scale Neural Rendering  and Beyond

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-28 | Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, Bo Dai | cs.CV | [PDF](http://arxiv.org/pdf/2309.16553v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) and its subsequent variants have led to
remarkable progress in neural rendering. While most of recent neural rendering
works focus on objects and small-scale scenes, developing neural rendering
methods for city-scale scenes is of great potential in many real-world
applications. However, this line of research is impeded by the absence of a
comprehensive and high-quality dataset, yet collecting such a dataset over real
city-scale scenes is costly, sensitive, and technically difficult. To this end,
we build a large-scale, comprehensive, and high-quality synthetic dataset for
city-scale neural rendering researches. Leveraging the Unreal Engine 5 City
Sample project, we develop a pipeline to easily collect aerial and street city
views, accompanied by ground-truth camera poses and a range of additional data
modalities. Flexible controls over environmental factors like light, weather,
human and car crowd are also available in our pipeline, supporting the need of
various tasks covering city-scale neural rendering and beyond. The resulting
pilot dataset, MatrixCity, contains 67k aerial images and 452k street images
from two city maps of total size $28km^2$. On top of MatrixCity, a thorough
benchmark is also conducted, which not only reveals unique challenges of the
task of city-scale neural rendering, but also highlights potential improvements
for future works. The dataset and code will be publicly available at our
project page: https://city-super.github.io/matrixcity/.

Comments:
- Accepted to ICCV 2023. Project page:
  $\href{https://city-super.github.io/matrixcity/}{this\, https\, URL}$

---

## Learning Effective NeRFs and SDFs Representations with 3D Generative  Adversarial Networks for 3D Object Generation: Technical Report for ICCV 2023  OmniObject3D Challenge

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-28 | Zheyuan Yang, Yibo Liu, Guile Wu, Tongtong Cao, Yuan Ren, Yang Liu, Bingbing Liu | cs.CV | [PDF](http://arxiv.org/pdf/2309.16110v1){: .btn .btn-green } |

**Abstract**: In this technical report, we present a solution for 3D object generation of
ICCV 2023 OmniObject3D Challenge. In recent years, 3D object generation has
made great process and achieved promising results, but it remains a challenging
task due to the difficulty of generating complex, textured and high-fidelity
results. To resolve this problem, we study learning effective NeRFs and SDFs
representations with 3D Generative Adversarial Networks (GANs) for 3D object
generation. Specifically, inspired by recent works, we use the efficient
geometry-aware 3D GANs as the backbone incorporating with label embedding and
color mapping, which enables to train the model on different taxonomies
simultaneously. Then, through a decoder, we aggregate the resulting features to
generate Neural Radiance Fields (NeRFs) based representations for rendering
high-fidelity synthetic images. Meanwhile, we optimize Signed Distance
Functions (SDFs) to effectively represent objects with 3D meshes. Besides, we
observe that this model can be effectively trained with only a few images of
each object from a variety of classes, instead of using a great number of
images per object or training one model per class. With this pipeline, we can
optimize an effective model for 3D object generation. This solution is one of
the final top-3-place solutions in the ICCV 2023 OmniObject3D Challenge.

---

## FG-NeRF: Flow-GAN based Probabilistic Neural Radiance Field for  Independence-Assumption-Free Uncertainty Estimation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-28 | Songlin Wei, Jiazhao Zhang, Yang Wang, Fanbo Xiang, Hao Su, He Wang | cs.CV | [PDF](http://arxiv.org/pdf/2309.16364v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields with stochasticity have garnered significant interest
by enabling the sampling of plausible radiance fields and quantifying
uncertainty for downstream tasks. Existing works rely on the independence
assumption of points in the radiance field or the pixels in input views to
obtain tractable forms of the probability density function. However, this
assumption inadvertently impacts performance when dealing with intricate
geometry and texture. In this work, we propose an independence-assumption-free
probabilistic neural radiance field based on Flow-GAN. By combining the
generative capability of adversarial learning and the powerful expressivity of
normalizing flow, our method explicitly models the density-radiance
distribution of the whole scene. We represent our probabilistic NeRF as a
mean-shifted probabilistic residual neural model. Our model is trained without
an explicit likelihood function, thereby avoiding the independence assumption.
Specifically, We downsample the training images with different strides and
centers to form fixed-size patches which are used to train the generator with
patch-based adversarial learning. Through extensive experiments, our method
demonstrates state-of-the-art performance by predicting lower rendering errors
and more reliable uncertainty on both synthetic and real-world datasets.

---

## DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content  Creation

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-28 | Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, Gang Zeng | cs.CV | [PDF](http://arxiv.org/pdf/2309.16653v1){: .btn .btn-green } |

**Abstract**: Recent advances in 3D content creation mostly leverage optimization-based 3D
generation via score distillation sampling (SDS). Though promising results have
been exhibited, these methods often suffer from slow per-sample optimization,
limiting their practical usage. In this paper, we propose DreamGaussian, a
novel 3D content generation framework that achieves both efficiency and quality
simultaneously. Our key insight is to design a generative 3D Gaussian Splatting
model with companioned mesh extraction and texture refinement in UV space. In
contrast to the occupancy pruning used in Neural Radiance Fields, we
demonstrate that the progressive densification of 3D Gaussians converges
significantly faster for 3D generative tasks. To further enhance the texture
quality and facilitate downstream applications, we introduce an efficient
algorithm to convert 3D Gaussians into textured meshes and apply a fine-tuning
stage to refine the details. Extensive experiments demonstrate the superior
efficiency and competitive generation quality of our proposed approach.
Notably, DreamGaussian produces high-quality textured meshes in just 2 minutes
from a single-view image, achieving approximately 10 times acceleration
compared to existing methods.

Comments:
- project page: https://dreamgaussian.github.io/

---

## Augmenting Heritage: An Open-Source Multiplatform AR Application

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-28 | Corrie Green | cs.HC | [PDF](http://arxiv.org/pdf/2310.13700v1){: .btn .btn-green } |

**Abstract**: AI NeRF algorithms, capable of cloud processing, have significantly reduced
hardware requirements and processing efficiency in photogrammetry pipelines.
This accessibility has unlocked the potential for museums, charities, and
cultural heritage sites worldwide to leverage mobile devices for artifact
scanning and processing. However, the adoption of augmented reality platforms
often necessitates the installation of proprietary applications on users'
mobile devices, which adds complexity to development and limits global
availability. This paper presents a case study that demonstrates a
cost-effective pipeline for visualizing scanned museum artifacts using mobile
augmented reality, leveraging an open-source embedded solution on a website.

---

## Text-to-3D using Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-28 | Zilong Chen, Feng Wang, Huaping Liu | cs.CV | [PDF](http://arxiv.org/pdf/2309.16585v3){: .btn .btn-green } |

**Abstract**: In this paper, we present Gaussian Splatting based text-to-3D generation
(GSGEN), a novel approach for generating high-quality 3D objects. Previous
methods suffer from inaccurate geometry and limited fidelity due to the absence
of 3D prior and proper representation. We leverage 3D Gaussian Splatting, a
recent state-of-the-art representation, to address existing shortcomings by
exploiting the explicit nature that enables the incorporation of 3D prior.
Specifically, our method adopts a progressive optimization strategy, which
includes a geometry optimization stage and an appearance refinement stage. In
geometry optimization, a coarse representation is established under a 3D
geometry prior along with the ordinary 2D SDS loss, ensuring a sensible and
3D-consistent rough shape. Subsequently, the obtained Gaussians undergo an
iterative refinement to enrich details. In this stage, we increase the number
of Gaussians by compactness-based densification to enhance continuity and
improve fidelity. With these designs, our approach can generate 3D content with
delicate details and more accurate geometry. Extensive evaluations demonstrate
the effectiveness of our method, especially for capturing high-frequency
components. Video results are provided at https://gsgen3d.github.io. Our code
is available at https://github.com/gsgen3d/gsgen

Comments:
- Project page: https://gsgen3d.github.io. Code:
  https://github.com/gsgen3d/gsgen

---

## Preface: A Data-driven Volumetric Prior for Few-shot Ultra  High-resolution Face Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-28 | Marcel C. Bühler, Kripasindhu Sarkar, Tanmay Shah, Gengyan Li, Daoye Wang, Leonhard Helminger, Sergio Orts-Escolano, Dmitry Lagun, Otmar Hilliges, Thabo Beeler, Abhimitra Meka | cs.CV | [PDF](http://arxiv.org/pdf/2309.16859v1){: .btn .btn-green } |

**Abstract**: NeRFs have enabled highly realistic synthesis of human faces including
complex appearance and reflectance effects of hair and skin. These methods
typically require a large number of multi-view input images, making the process
hardware intensive and cumbersome, limiting applicability to unconstrained
settings. We propose a novel volumetric human face prior that enables the
synthesis of ultra high-resolution novel views of subjects that are not part of
the prior's training distribution. This prior model consists of an
identity-conditioned NeRF, trained on a dataset of low-resolution multi-view
images of diverse humans with known camera calibration. A simple sparse
landmark-based 3D alignment of the training dataset allows our model to learn a
smooth latent space of geometry and appearance despite a limited number of
training identities. A high-quality volumetric representation of a novel
subject can be obtained by model fitting to 2 or 3 camera views of arbitrary
resolution. Importantly, our method requires as few as two views of casually
captured images as input at inference time.

Comments:
- In Proceedings of the IEEE/CVF International Conference on Computer
  Vision, 2023

---

## P2I-NET: Mapping Camera Pose to Image via Adversarial Learning for New  View Synthesis in Real Indoor Environments

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-27 | Xujie Kang, Kanglin Liu, Jiang Duan, Yuanhao Gong, Guoping Qiu | cs.CV | [PDF](http://arxiv.org/pdf/2309.15526v1){: .btn .btn-green } |

**Abstract**: Given a new $6DoF$ camera pose in an indoor environment, we study the
challenging problem of predicting the view from that pose based on a set of
reference RGBD views. Existing explicit or implicit 3D geometry construction
methods are computationally expensive while those based on learning have
predominantly focused on isolated views of object categories with regular
geometric structure. Differing from the traditional \textit{render-inpaint}
approach to new view synthesis in the real indoor environment, we propose a
conditional generative adversarial neural network (P2I-NET) to directly predict
the new view from the given pose. P2I-NET learns the conditional distribution
of the images of the environment for establishing the correspondence between
the camera pose and its view of the environment, and achieves this through a
number of innovative designs in its architecture and training lost function.
Two auxiliary discriminator constraints are introduced for enforcing the
consistency between the pose of the generated image and that of the
corresponding real world image in both the latent feature space and the real
world pose space. Additionally a deep convolutional neural network (CNN) is
introduced to further reinforce this consistency in the pixel space. We have
performed extensive new view synthesis experiments on real indoor datasets.
Results show that P2I-NET has superior performance against a number of NeRF
based strong baseline models. In particular, we show that P2I-NET is 40 to 100
times faster than these competitor techniques while synthesising similar
quality images. Furthermore, we contribute a new publicly available indoor
environment dataset containing 22 high resolution RGBD videos where each frame
also has accurate camera pose parameters.

---

## NeuRBF: A Neural Fields Representation with Adaptive Radial Basis  Functions



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-27 | Zhang Chen, Zhong Li, Liangchen Song, Lele Chen, Jingyi Yu, Junsong Yuan, Yi Xu | cs.CV | [PDF](http://arxiv.org/pdf/2309.15426v1){: .btn .btn-green } |

**Abstract**: We present a novel type of neural fields that uses general radial bases for
signal representation. State-of-the-art neural fields typically rely on
grid-based representations for storing local neural features and N-dimensional
linear kernels for interpolating features at continuous query points. The
spatial positions of their neural features are fixed on grid nodes and cannot
well adapt to target signals. Our method instead builds upon general radial
bases with flexible kernel position and shape, which have higher spatial
adaptivity and can more closely fit target signals. To further improve the
channel-wise capacity of radial basis functions, we propose to compose them
with multi-frequency sinusoid functions. This technique extends a radial basis
to multiple Fourier radial bases of different frequency bands without requiring
extra parameters, facilitating the representation of details. Moreover, by
marrying adaptive radial bases with grid-based ones, our hybrid combination
inherits both adaptivity and interpolation smoothness. We carefully designed
weighting schemes to let radial bases adapt to different types of signals
effectively. Our experiments on 2D image and 3D signed distance field
representation demonstrate the higher accuracy and compactness of our method
than prior arts. When applied to neural radiance field reconstruction, our
method achieves state-of-the-art rendering quality, with small model size and
comparable training speed.

Comments:
- Accepted to ICCV 2023 Oral. Project page:
  https://oppo-us-research.github.io/NeuRBF-website/

---

## BASED: Bundle-Adjusting Surgical Endoscopic Dynamic Video Reconstruction  using Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-27 | Shreya Saha, Sainan Liu, Shan Lin, Jingpei Lu, Michael Yip | cs.CV | [PDF](http://arxiv.org/pdf/2309.15329v1){: .btn .btn-green } |

**Abstract**: Reconstruction of deformable scenes from endoscopic videos is important for
many applications such as intraoperative navigation, surgical visual
perception, and robotic surgery. It is a foundational requirement for realizing
autonomous robotic interventions for minimally invasive surgery. However,
previous approaches in this domain have been limited by their modular nature
and are confined to specific camera and scene settings. Our work adopts the
Neural Radiance Fields (NeRF) approach to learning 3D implicit representations
of scenes that are both dynamic and deformable over time, and furthermore with
unknown camera poses. We demonstrate this approach on endoscopic surgical
scenes from robotic surgery. This work removes the constraints of known camera
poses and overcomes the drawbacks of the state-of-the-art unstructured dynamic
scene reconstruction technique, which relies on the static part of the scene
for accurate reconstruction. Through several experimental datasets, we
demonstrate the versatility of our proposed model to adapt to diverse camera
and scene settings, and show its promise for both current and future robotic
surgical systems.

---

## 3D Density-Gradient based Edge Detection on Neural Radiance Fields  (NeRFs) for Geometric Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-26 | Miriam Jäger, Boris Jutzi | cs.CV | [PDF](http://arxiv.org/pdf/2309.14800v1){: .btn .btn-green } |

**Abstract**: Generating geometric 3D reconstructions from Neural Radiance Fields (NeRFs)
is of great interest. However, accurate and complete reconstructions based on
the density values are challenging. The network output depends on input data,
NeRF network configuration and hyperparameter. As a result, the direct usage of
density values, e.g. via filtering with global density thresholds, usually
requires empirical investigations. Under the assumption that the density
increases from non-object to object area, the utilization of density gradients
from relative values is evident. As the density represents a position-dependent
parameter it can be handled anisotropically, therefore processing of the
voxelized 3D density field is justified. In this regard, we address geometric
3D reconstructions based on density gradients, whereas the gradients result
from 3D edge detection filters of the first and second derivatives, namely
Sobel, Canny and Laplacian of Gaussian. The gradients rely on relative
neighboring density values in all directions, thus are independent from
absolute magnitudes. Consequently, gradient filters are able to extract edges
along a wide density range, almost independent from assumptions and empirical
investigations. Our approach demonstrates the capability to achieve geometric
3D reconstructions with high geometric accuracy on object surfaces and
remarkable object completeness. Notably, Canny filter effectively eliminates
gaps, delivers a uniform point density, and strikes a favorable balance between
correctness and completeness across the scenes.

Comments:
- 8 pages, 4 figures, 2 tables. Will be published in the ISPRS The
  International Archives of Photogrammetry, Remote Sensing and Spatial
  Information Sciences

---

## NAS-NeRF: Generative Neural Architecture Search for Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-25 | Saeejith Nair, Yuhao Chen, Mohammad Javad Shafiee, Alexander Wong | cs.CV | [PDF](http://arxiv.org/pdf/2309.14293v3){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) enable high-quality novel view synthesis, but
their high computational complexity limits deployability. While existing
neural-based solutions strive for efficiency, they use one-size-fits-all
architectures regardless of scene complexity. The same architecture may be
unnecessarily large for simple scenes but insufficient for complex ones. Thus,
there is a need to dynamically optimize the neural network component of NeRFs
to achieve a balance between computational complexity and specific targets for
synthesis quality. We introduce NAS-NeRF, a generative neural architecture
search strategy that generates compact, scene-specialized NeRF architectures by
balancing architecture complexity and target synthesis quality metrics. Our
method incorporates constraints on target metrics and budgets to guide the
search towards architectures tailored for each scene. Experiments on the
Blender synthetic dataset show the proposed NAS-NeRF can generate architectures
up to 5.74$\times$ smaller, with 4.19$\times$ fewer FLOPs, and 1.93$\times$
faster on a GPU than baseline NeRFs, without suffering a drop in SSIM.
Furthermore, we illustrate that NAS-NeRF can also achieve architectures up to
23$\times$ smaller, with 22$\times$ fewer FLOPs, and 4.7$\times$ faster than
baseline NeRFs with only a 5.3% average SSIM drop. Our source code is also made
publicly available at https://saeejithnair.github.io/NAS-NeRF.

Comments:
- 8 pages

---

## Variational Inference for Scalable 3D Object-centric Learning

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-25 | Tianyu Wang, Kee Siong Ng, Miaomiao Liu | cs.CV | [PDF](http://arxiv.org/pdf/2309.14010v1){: .btn .btn-green } |

**Abstract**: We tackle the task of scalable unsupervised object-centric representation
learning on 3D scenes. Existing approaches to object-centric representation
learning show limitations in generalizing to larger scenes as their learning
processes rely on a fixed global coordinate system. In contrast, we propose to
learn view-invariant 3D object representations in localized object coordinate
systems. To this end, we estimate the object pose and appearance representation
separately and explicitly map object representations across views while
maintaining object identities. We adopt an amortized variational inference
pipeline that can process sequential input and scalably update object latent
distributions online. To handle large-scale scenes with a varying number of
objects, we further introduce a Cognitive Map that allows the registration and
query of objects on a per-scene global map to achieve scalable representation
learning. We explore the object-centric neural radiance field (NeRF) as our 3D
scene representation, which is jointly modeled within our unsupervised
object-centric learning framework. Experimental results on synthetic and real
datasets show that our proposed method can infer and maintain object-centric
representations of 3D scenes and outperforms previous models.

---

## Tiled Multiplane Images for Practical 3D Photography



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-25 | Numair Khan, Douglas Lanman, Lei Xiao | cs.CV | [PDF](http://arxiv.org/pdf/2309.14291v1){: .btn .btn-green } |

**Abstract**: The task of synthesizing novel views from a single image has useful
applications in virtual reality and mobile computing, and a number of
approaches to the problem have been proposed in recent years. A Multiplane
Image (MPI) estimates the scene as a stack of RGBA layers, and can model
complex appearance effects, anti-alias depth errors and synthesize soft edges
better than methods that use textured meshes or layered depth images. And
unlike neural radiance fields, an MPI can be efficiently rendered on graphics
hardware. However, MPIs are highly redundant and require a large number of
depth layers to achieve plausible results. Based on the observation that the
depth complexity in local image regions is lower than that over the entire
image, we split an MPI into many small, tiled regions, each with only a few
depth planes. We call this representation a Tiled Multiplane Image (TMPI). We
propose a method for generating a TMPI with adaptive depth planes for
single-view 3D photography in the wild. Our synthesized results are comparable
to state-of-the-art single-view MPI methods while having lower computational
overhead.

Comments:
- ICCV 2023

---

## MM-NeRF: Multimodal-Guided 3D Multi-Style Transfer of Neural Radiance  Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-24 | Zijiang Yang, Zhongwei Qiu, Chang Xu, Dongmei Fu | cs.CV | [PDF](http://arxiv.org/pdf/2309.13607v2){: .btn .btn-green } |

**Abstract**: 3D style transfer aims to generate stylized views of 3D scenes with specified
styles, which requires high-quality generating and keeping multi-view
consistency. Existing methods still suffer the challenges of high-quality
stylization with texture details and stylization with multimodal guidance. In
this paper, we reveal that the common training method of stylization with NeRF,
which generates stylized multi-view supervision by 2D style transfer models,
causes the same object in supervision to show various states (color tone,
details, etc.) in different views, leading NeRF to tend to smooth the texture
details, further resulting in low-quality rendering for 3D multi-style
transfer. To tackle these problems, we propose a novel Multimodal-guided 3D
Multi-style transfer of NeRF, termed MM-NeRF. First, MM-NeRF projects
multimodal guidance into a unified space to keep the multimodal styles
consistency and extracts multimodal features to guide the 3D stylization.
Second, a novel multi-head learning scheme is proposed to relieve the
difficulty of learning multi-style transfer, and a multi-view style consistent
loss is proposed to track the inconsistency of multi-view supervision data.
Finally, a novel incremental learning mechanism to generalize MM-NeRF to any
new style with small costs. Extensive experiments on several real-world
datasets show that MM-NeRF achieves high-quality 3D multi-style stylization
with multimodal guidance, and keeps multi-view consistency and style
consistency between multimodal guidance. Codes will be released.

---

## NeRF-Enhanced Outpainting for Faithful Field-of-View Extrapolation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-23 | Rui Yu, Jiachen Liu, Zihan Zhou, Sharon X. Huang | cs.CV | [PDF](http://arxiv.org/pdf/2309.13240v1){: .btn .btn-green } |

**Abstract**: In various applications, such as robotic navigation and remote visual
assistance, expanding the field of view (FOV) of the camera proves beneficial
for enhancing environmental perception. Unlike image outpainting techniques
aimed solely at generating aesthetically pleasing visuals, these applications
demand an extended view that faithfully represents the scene. To achieve this,
we formulate a new problem of faithful FOV extrapolation that utilizes a set of
pre-captured images as prior knowledge of the scene. To address this problem,
we present a simple yet effective solution called NeRF-Enhanced Outpainting
(NEO) that uses extended-FOV images generated through NeRF to train a
scene-specific image outpainting model. To assess the performance of NEO, we
conduct comprehensive evaluations on three photorealistic datasets and one
real-world dataset. Extensive experiments on the benchmark datasets showcase
the robustness and potential of our method in addressing this challenge. We
believe our work lays a strong foundation for future exploration within the
research community.

---

## NeRRF: 3D Reconstruction and View Synthesis for Transparent and Specular  Objects with Neural Refractive-Reflective Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-22 | Xiaoxue Chen, Junchen Liu, Hao Zhao, Guyue Zhou, Ya-Qin Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2309.13039v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) have revolutionized the field of image-based
view synthesis. However, NeRF uses straight rays and fails to deal with
complicated light path changes caused by refraction and reflection. This
prevents NeRF from successfully synthesizing transparent or specular objects,
which are ubiquitous in real-world robotics and A/VR applications. In this
paper, we introduce the refractive-reflective field. Taking the object
silhouette as input, we first utilize marching tetrahedra with a progressive
encoding to reconstruct the geometry of non-Lambertian objects and then model
refraction and reflection effects of the object in a unified framework using
Fresnel terms. Meanwhile, to achieve efficient and effective anti-aliasing, we
propose a virtual cone supersampling technique. We benchmark our method on
different shapes, backgrounds and Fresnel terms on both real-world and
synthetic datasets. We also qualitatively and quantitatively benchmark the
rendering results of various editing applications, including material editing,
object replacement/insertion, and environment illumination estimation. Codes
and data are publicly available at https://github.com/dawning77/NeRRF.

---

## Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene  Reconstruction



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-22 | Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, Xiaogang Jin | cs.CV | [PDF](http://arxiv.org/pdf/2309.13101v2){: .btn .btn-green } |

**Abstract**: Implicit neural representation has paved the way for new approaches to
dynamic scene reconstruction and rendering. Nonetheless, cutting-edge dynamic
neural rendering methods rely heavily on these implicit representations, which
frequently struggle to capture the intricate details of objects in the scene.
Furthermore, implicit methods have difficulty achieving real-time rendering in
general dynamic scenes, limiting their use in a variety of tasks. To address
the issues, we propose a deformable 3D Gaussians Splatting method that
reconstructs scenes using 3D Gaussians and learns them in canonical space with
a deformation field to model monocular dynamic scenes. We also introduce an
annealing smoothing training mechanism with no extra overhead, which can
mitigate the impact of inaccurate poses on the smoothness of time interpolation
tasks in real-world datasets. Through a differential Gaussian rasterizer, the
deformable 3D Gaussians not only achieve higher rendering quality but also
real-time rendering speed. Experiments show that our method outperforms
existing methods significantly in terms of both rendering quality and speed,
making it well-suited for tasks such as novel-view synthesis, time
interpolation, and real-time rendering.

---

## RHINO: Regularizing the Hash-based Implicit Neural Representation



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-22 | Hao Zhu, Fengyi Liu, Qi Zhang, Xun Cao, Zhan Ma | cs.CV | [PDF](http://arxiv.org/pdf/2309.12642v1){: .btn .btn-green } |

**Abstract**: The use of Implicit Neural Representation (INR) through a hash-table has
demonstrated impressive effectiveness and efficiency in characterizing
intricate signals. However, current state-of-the-art methods exhibit
insufficient regularization, often yielding unreliable and noisy results during
interpolations. We find that this issue stems from broken gradient flow between
input coordinates and indexed hash-keys, where the chain rule attempts to model
discrete hash-keys, rather than the continuous coordinates. To tackle this
concern, we introduce RHINO, in which a continuous analytical function is
incorporated to facilitate regularization by connecting the input coordinate
and the network additionally without modifying the architecture of current
hash-based INRs. This connection ensures a seamless backpropagation of
gradients from the network's output back to the input coordinates, thereby
enhancing regularization. Our experimental results not only showcase the
broadened regularization capability across different hash-based INRs like DINER
and Instant NGP, but also across a variety of tasks such as image fitting,
representation of signed distance functions, and optimization of 5D static / 6D
dynamic neural radiance fields. Notably, RHINO outperforms current
state-of-the-art techniques in both quality and speed, affirming its
superiority.

Comments:
- 17 pages, 11 figures

---

## MarkNerf:Watermarking for Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-21 | Lifeng Chen, Jia Liu, Yan Ke, Wenquan Sun, Weina Dong, Xiaozhong Pan | cs.CR | [PDF](http://arxiv.org/pdf/2309.11747v1){: .btn .btn-green } |

**Abstract**: A watermarking algorithm is proposed in this paper to address the copyright
protection issue of implicit 3D models. The algorithm involves embedding
watermarks into the images in the training set through an embedding network,
and subsequently utilizing the NeRF model for 3D modeling. A copyright verifier
is employed to generate a backdoor image by providing a secret perspective as
input to the neural radiation field. Subsequently, a watermark extractor is
devised using the hyperparameterization method of the neural network to extract
the embedded watermark image from that perspective. In a black box scenario, if
there is a suspicion that the 3D model has been used without authorization, the
verifier can extract watermarks from a secret perspective to verify network
copyright. Experimental results demonstrate that the proposed algorithm
effectively safeguards the copyright of 3D models. Furthermore, the extracted
watermarks exhibit favorable visual effects and demonstrate robust resistance
against various types of noise attacks.

---

## Rendering stable features improves sampling-based localisation with  Neural radiance fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-21 | Boxuan Zhang, Lindsay Kleeman, Michael Burke | cs.RO | [PDF](http://arxiv.org/pdf/2309.11698v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) are a powerful tool for implicit scene
representations, allowing for differentiable rendering and the ability to make
predictions about previously unseen viewpoints. From a robotics perspective,
there has been growing interest in object and scene-based localisation using
NeRFs, with a number of recent works relying on sampling-based or Monte-Carlo
localisation schemes. Unfortunately, these can be extremely computationally
expensive, requiring multiple network forward passes to infer camera or object
pose. To alleviate this, a variety of sampling strategies have been applied,
many relying on keypoint recognition techniques from classical computer vision.
This work conducts a systematic empirical comparison of these approaches and
shows that in contrast to conventional feature matching approaches for
geometry-based localisation, sampling-based localisation using NeRFs benefits
significantly from stable features. Results show that rendering stable features
can result in a tenfold reduction in the number of forward passes required, a
significant speed improvement.

---

## NeuralLabeling: A versatile toolset for labeling vision datasets using  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-21 | Floris Erich, Naoya Chiba, Yusuke Yoshiyasu, Noriaki Ando, Ryo Hanai, Yukiyasu Domae | cs.CV | [PDF](http://arxiv.org/pdf/2309.11966v1){: .btn .btn-green } |

**Abstract**: We present NeuralLabeling, a labeling approach and toolset for annotating a
scene using either bounding boxes or meshes and generating segmentation masks,
affordance maps, 2D bounding boxes, 3D bounding boxes, 6DOF object poses, depth
maps and object meshes. NeuralLabeling uses Neural Radiance Fields (NeRF) as
renderer, allowing labeling to be performed using 3D spatial tools while
incorporating geometric clues such as occlusions, relying only on images
captured from multiple viewpoints as input. To demonstrate the applicability of
NeuralLabeling to a practical problem in robotics, we added ground truth depth
maps to 30000 frames of transparent object RGB and noisy depth maps of glasses
placed in a dishwasher captured using an RGBD sensor, yielding the
Dishwasher30k dataset. We show that training a simple deep neural network with
supervision using the annotated depth maps yields a higher reconstruction
performance than training with the previously applied weakly supervised
approach.

Comments:
- 8 pages, project website:
  https://florise.github.io/neural_labeling_web/

---

## Fast Satellite Tensorial Radiance Field for Multi-date Satellite Imagery  of Large Size

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-21 | Tongtong Zhang, Yuanxiang Li | cs.CV | [PDF](http://arxiv.org/pdf/2309.11767v1){: .btn .btn-green } |

**Abstract**: Existing NeRF models for satellite images suffer from slow speeds, mandatory
solar information as input, and limitations in handling large satellite images.
In response, we present SatensoRF, which significantly accelerates the entire
process while employing fewer parameters for satellite imagery of large size.
Besides, we observed that the prevalent assumption of Lambertian surfaces in
neural radiance fields falls short for vegetative and aquatic elements. In
contrast to the traditional hierarchical MLP-based scene representation, we
have chosen a multiscale tensor decomposition approach for color, volume
density, and auxiliary variables to model the lightfield with specular color.
Additionally, to rectify inconsistencies in multi-date imagery, we incorporate
total variation loss to restore the density tensor field and treat the problem
as a denosing task.To validate our approach, we conducted assessments of
SatensoRF using subsets from the spacenet multi-view dataset, which includes
both multi-date and single-date multi-view RGB images. Our results clearly
demonstrate that SatensoRF surpasses the state-of-the-art Sat-NeRF series in
terms of novel view synthesis performance. Significantly, SatensoRF requires
fewer parameters for training, resulting in faster training and inference
speeds and reduced computational demands.

---

## ORTexME: Occlusion-Robust Human Shape and Pose via Temporal Average  Texture and Mesh Encoding

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-21 | Yu Cheng, Bo Wang, Robby T. Tan | cs.CV | [PDF](http://arxiv.org/pdf/2309.12183v1){: .btn .btn-green } |

**Abstract**: In 3D human shape and pose estimation from a monocular video, models trained
with limited labeled data cannot generalize well to videos with occlusion,
which is common in the wild videos. The recent human neural rendering
approaches focusing on novel view synthesis initialized by the off-the-shelf
human shape and pose methods have the potential to correct the initial human
shape. However, the existing methods have some drawbacks such as, erroneous in
handling occlusion, sensitive to inaccurate human segmentation, and ineffective
loss computation due to the non-regularized opacity field. To address these
problems, we introduce ORTexME, an occlusion-robust temporal method that
utilizes temporal information from the input video to better regularize the
occluded body parts. While our ORTexME is based on NeRF, to determine the
reliable regions for the NeRF ray sampling, we utilize our novel average
texture learning approach to learn the average appearance of a person, and to
infer a mask based on the average texture. In addition, to guide the
opacity-field updates in NeRF to suppress blur and noise, we propose the use of
human body mesh. The quantitative evaluation demonstrates that our method
achieves significant improvement on the challenging multi-person 3DPW dataset,
where our method achieves 1.8 P-MPJPE error reduction. The SOTA rendering-based
methods fail and enlarge the error up to 5.6 on the same dataset.

Comments:
- 8 pages, 8 figures

---

## GenLayNeRF: Generalizable Layered Representations with 3D Model  Alignment for Multi-Human View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-20 | Youssef Abdelkareem, Shady Shehata, Fakhri Karray | cs.CV | [PDF](http://arxiv.org/pdf/2309.11627v1){: .btn .btn-green } |

**Abstract**: Novel view synthesis (NVS) of multi-human scenes imposes challenges due to
the complex inter-human occlusions. Layered representations handle the
complexities by dividing the scene into multi-layered radiance fields, however,
they are mainly constrained to per-scene optimization making them inefficient.
Generalizable human view synthesis methods combine the pre-fitted 3D human
meshes with image features to reach generalization, yet they are mainly
designed to operate on single-human scenes. Another drawback is the reliance on
multi-step optimization techniques for parametric pre-fitting of the 3D body
models that suffer from misalignment with the images in sparse view settings
causing hallucinations in synthesized views. In this work, we propose,
GenLayNeRF, a generalizable layered scene representation for free-viewpoint
rendering of multiple human subjects which requires no per-scene optimization
and very sparse views as input. We divide the scene into multi-human layers
anchored by the 3D body meshes. We then ensure pixel-level alignment of the
body models with the input views through a novel end-to-end trainable module
that carries out iterative parametric correction coupled with multi-view
feature fusion to produce aligned 3D models. For NVS, we extract point-wise
image-aligned and human-anchored features which are correlated and fused using
self-attention and cross-attention modules. We augment low-level RGB values
into the features with an attention-based RGB fusion module. To evaluate our
approach, we construct two multi-human view synthesis datasets; DeepMultiSyn
and ZJU-MultiHuman. The results indicate that our proposed approach outperforms
generalizable and non-human per-scene NeRF methods while performing at par with
layered per-scene methods without test time optimization.

Comments:
- Accepted to GCPR 2023

---

## Language-driven Object Fusion into Neural Radiance Fields with  Pose-Conditioned Dataset Updates



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-20 | Ka Chun Shum, Jaeyeon Kim, Binh-Son Hua, Duc Thanh Nguyen, Sai-Kit Yeung | cs.CV | [PDF](http://arxiv.org/pdf/2309.11281v2){: .btn .btn-green } |

**Abstract**: Neural radiance field is an emerging rendering method that generates
high-quality multi-view consistent images from a neural scene representation
and volume rendering. Although neural radiance field-based techniques are
robust for scene reconstruction, their ability to add or remove objects remains
limited. This paper proposes a new language-driven approach for object
manipulation with neural radiance fields through dataset updates. Specifically,
to insert a new foreground object represented by a set of multi-view images
into a background radiance field, we use a text-to-image diffusion model to
learn and generate combined images that fuse the object of interest into the
given background across views. These combined images are then used for refining
the background radiance field so that we can render view-consistent images
containing both the object and the background. To ensure view consistency, we
propose a dataset updates strategy that prioritizes radiance field training
with camera views close to the already-trained views prior to propagating the
training to remaining views. We show that under the same dataset updates
strategy, we can easily adapt our method for object insertion using data from
text-to-3D models as well as object removal. Experimental results show that our
method generates photorealistic images of the edited scenes, and outperforms
state-of-the-art methods in 3D reconstruction and neural radiance field
blending.

---

## Light Field Diffusion for Single-View Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-20 | Yifeng Xiong, Haoyu Ma, Shanlin Sun, Kun Han, Xiaohui Xie | cs.CV | [PDF](http://arxiv.org/pdf/2309.11525v2){: .btn .btn-green } |

**Abstract**: Single-view novel view synthesis, the task of generating images from new
viewpoints based on a single reference image, is an important but challenging
task in computer vision. Recently, Denoising Diffusion Probabilistic Model
(DDPM) has become popular in this area due to its strong ability to generate
high-fidelity images. However, current diffusion-based methods directly rely on
camera pose matrices as viewing conditions, globally and implicitly introducing
3D constraints. These methods may suffer from inconsistency among generated
images from different perspectives, especially in regions with intricate
textures and structures. In this work, we present Light Field Diffusion (LFD),
a conditional diffusion-based model for single-view novel view synthesis.
Unlike previous methods that employ camera pose matrices, LFD transforms the
camera view information into light field encoding and combines it with the
reference image. This design introduces local pixel-wise constraints within the
diffusion models, thereby encouraging better multi-view consistency.
Experiments on several datasets show that our LFD can efficiently generate
high-fidelity images and maintain better 3D consistency even in intricate
regions. Our method can generate images with higher quality than NeRF-based
models, and we obtain sample quality similar to other diffusion-based models
but with only one-third of the model size.

---

## Controllable Dynamic Appearance for Neural 3D Portraits

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-20 | ShahRukh Athar, Zhixin Shu, Zexiang Xu, Fujun Luan, Sai Bi, Kalyan Sunkavalli, Dimitris Samaras | cs.CV | [PDF](http://arxiv.org/pdf/2309.11009v2){: .btn .btn-green } |

**Abstract**: Recent advances in Neural Radiance Fields (NeRFs) have made it possible to
reconstruct and reanimate dynamic portrait scenes with control over head-pose,
facial expressions and viewing direction. However, training such models assumes
photometric consistency over the deformed region e.g. the face must be evenly
lit as it deforms with changing head-pose and facial expression. Such
photometric consistency across frames of a video is hard to maintain, even in
studio environments, thus making the created reanimatable neural portraits
prone to artifacts during reanimation. In this work, we propose CoDyNeRF, a
system that enables the creation of fully controllable 3D portraits in
real-world capture conditions. CoDyNeRF learns to approximate illumination
dependent effects via a dynamic appearance model in the canonical space that is
conditioned on predicted surface normals and the facial expressions and
head-pose deformations. The surface normals prediction is guided using 3DMM
normals that act as a coarse prior for the normals of the human head, where
direct prediction of normals is hard due to rigid and non-rigid deformations
induced by head-pose and facial expression changes. Using only a
smartphone-captured short video of a subject for training, we demonstrate the
effectiveness of our method on free view synthesis of a portrait scene with
explicit head pose and expression controls, and realistic lighting effects. The
project page can be found here:
http://shahrukhathar.github.io/2023/08/22/CoDyNeRF.html

---

## SpikingNeRF: Making Bio-inspired Neural Networks See through the Real  World

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-20 | Xingting Yao, Qinghao Hu, Tielong Liu, Zitao Mo, Zeyu Zhu, Zhengyang Zhuge, Jian Cheng | cs.NE | [PDF](http://arxiv.org/pdf/2309.10987v3){: .btn .btn-green } |

**Abstract**: Spiking neural networks (SNNs) have been thriving on numerous tasks to
leverage their promising energy efficiency and exploit their potentialities as
biologically plausible intelligence. Meanwhile, the Neural Radiance Fields
(NeRF) render high-quality 3D scenes with massive energy consumption, but few
works delve into the energy-saving solution with a bio-inspired approach. In
this paper, we propose SpikingNeRF, which aligns the radiance ray with the
temporal dimension of SNN, to naturally accommodate the SNN to the
reconstruction of Radiance Fields. Thus, the computation turns into a
spike-based, multiplication-free manner, reducing the energy consumption. In
SpikingNeRF, each sampled point on the ray is matched onto a particular time
step, and represented in a hybrid manner where the voxel grids are maintained
as well. Based on the voxel grids, sampled points are determined whether to be
masked for better training and inference. However, this operation also incurs
irregular temporal length. We propose the temporal padding strategy to tackle
the masked samples to maintain regular temporal length, i.e., regular tensors,
and the temporal condensing strategy to form a denser data structure for
hardware-friendly computation. Extensive experiments on various datasets
demonstrate that our method reduces the 70.79% energy consumption on average
and obtains comparable synthesis quality with the ANN baseline.

---

## Steganography for Neural Radiance Fields by Backdooring

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-19 | Weina Dong, Jia Liu, Yan Ke, Lifeng Chen, Wenquan Sun, Xiaozhong Pan | cs.CR | [PDF](http://arxiv.org/pdf/2309.10503v1){: .btn .btn-green } |

**Abstract**: The utilization of implicit representation for visual data (such as images,
videos, and 3D models) has recently gained significant attention in computer
vision research. In this letter, we propose a novel model steganography scheme
with implicit neural representation. The message sender leverages Neural
Radiance Fields (NeRF) and its viewpoint synthesis capabilities by introducing
a viewpoint as a key. The NeRF model generates a secret viewpoint image, which
serves as a backdoor. Subsequently, we train a message extractor using
overfitting to establish a one-to-one mapping between the secret message and
the secret viewpoint image. The sender delivers the trained NeRF model and the
message extractor to the receiver over the open channel, and the receiver
utilizes the key shared by both parties to obtain the rendered image in the
secret view from the NeRF model, and then obtains the secret message through
the message extractor. The inherent complexity of the viewpoint information
prevents attackers from stealing the secret message accurately. Experimental
results demonstrate that the message extractor trained in this letter achieves
high-capacity steganography with fast performance, achieving a 100\% accuracy
in message extraction. Furthermore, the extensive viewpoint key space of NeRF
ensures the security of the steganography scheme.

Comments:
- 6 pages, 7 figures

---

## Locally Stylized Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-19 | Hong-Wing Pang, Binh-Son Hua, Sai-Kit Yeung | cs.CV | [PDF](http://arxiv.org/pdf/2309.10684v1){: .btn .btn-green } |

**Abstract**: In recent years, there has been increasing interest in applying stylization
on 3D scenes from a reference style image, in particular onto neural radiance
fields (NeRF). While performing stylization directly on NeRF guarantees
appearance consistency over arbitrary novel views, it is a challenging problem
to guide the transfer of patterns from the style image onto different parts of
the NeRF scene. In this work, we propose a stylization framework for NeRF based
on local style transfer. In particular, we use a hash-grid encoding to learn
the embedding of the appearance and geometry components, and show that the
mapping defined by the hash table allows us to control the stylization to a
certain extent. Stylization is then achieved by optimizing the appearance
branch while keeping the geometry branch fixed. To support local style
transfer, we propose a new loss function that utilizes a segmentation network
and bipartite matching to establish region correspondences between the style
image and the content images obtained from volume rendering. Our experiments
show that our method yields plausible stylization results with novel view
synthesis while having flexible controllability via manipulating and
customizing the region correspondences.

Comments:
- ICCV 2023

---

## RenderOcc: Vision-Centric 3D Occupancy Prediction with 2D Rendering  Supervision

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-18 | Mingjie Pan, Jiaming Liu, Renrui Zhang, Peixiang Huang, Xiaoqi Li, Li Liu, Shanghang Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2309.09502v1){: .btn .btn-green } |

**Abstract**: 3D occupancy prediction holds significant promise in the fields of robot
perception and autonomous driving, which quantifies 3D scenes into grid cells
with semantic labels. Recent works mainly utilize complete occupancy labels in
3D voxel space for supervision. However, the expensive annotation process and
sometimes ambiguous labels have severely constrained the usability and
scalability of 3D occupancy models. To address this, we present RenderOcc, a
novel paradigm for training 3D occupancy models only using 2D labels.
Specifically, we extract a NeRF-style 3D volume representation from multi-view
images, and employ volume rendering techniques to establish 2D renderings, thus
enabling direct 3D supervision from 2D semantics and depth labels.
Additionally, we introduce an Auxiliary Ray method to tackle the issue of
sparse viewpoints in autonomous driving scenarios, which leverages sequential
frames to construct comprehensive 2D rendering for each object. To our best
knowledge, RenderOcc is the first attempt to train multi-view 3D occupancy
models only using 2D labels, reducing the dependence on costly 3D occupancy
annotations. Extensive experiments demonstrate that RenderOcc achieves
comparable performance to models fully supervised with 3D labels, underscoring
the significance of this approach in real-world applications.

---

## Instant Photorealistic Style Transfer: A Lightweight and Adaptive  Approach

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-18 | Rong Liu, Enyu Zhao, Zhiyuan Liu, Andrew Feng, Scott John Easley | cs.CV | [PDF](http://arxiv.org/pdf/2309.10011v2){: .btn .btn-green } |

**Abstract**: In this paper, we propose an Instant Photorealistic Style Transfer (IPST)
approach, designed to achieve instant photorealistic style transfer on
super-resolution inputs without the need for pre-training on pair-wise datasets
or imposing extra constraints. Our method utilizes a lightweight StyleNet to
enable style transfer from a style image to a content image while preserving
non-color information. To further enhance the style transfer process, we
introduce an instance-adaptive optimization to prioritize the photorealism of
outputs and accelerate the convergence of the style network, leading to a rapid
training completion within seconds. Moreover, IPST is well-suited for
multi-frame style transfer tasks, as it retains temporal and multi-view
consistency of the multi-frame inputs such as video and Neural Radiance Field
(NeRF). Experimental results demonstrate that IPST requires less GPU memory
usage, offers faster multi-frame transfer speed, and generates photorealistic
outputs, making it a promising solution for various photorealistic transfer
applications.

Comments:
- 8 pages (reference excluded), 6 figures, 4 tables

---

## NeRF-VINS: A Real-time Neural Radiance Field Map-based Visual-Inertial  Navigation System

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-17 | Saimouli Katragadda, Woosik Lee, Yuxiang Peng, Patrick Geneva, Chuchu Chen, Chao Guo, Mingyang Li, Guoquan Huang | cs.RO | [PDF](http://arxiv.org/pdf/2309.09295v1){: .btn .btn-green } |

**Abstract**: Achieving accurate, efficient, and consistent localization within an a priori
environment map remains a fundamental challenge in robotics and computer
vision. Conventional map-based keyframe localization often suffers from
sub-optimal viewpoints due to limited field of view (FOV), thus degrading its
performance. To address this issue, in this paper, we design a real-time
tightly-coupled Neural Radiance Fields (NeRF)-aided visual-inertial navigation
system (VINS), termed NeRF-VINS. By effectively leveraging NeRF's potential to
synthesize novel views, essential for addressing limited viewpoints, the
proposed NeRF-VINS optimally fuses IMU and monocular image measurements along
with synthetically rendered images within an efficient filter-based framework.
This tightly coupled integration enables 3D motion tracking with bounded error.
We extensively compare the proposed NeRF-VINS against the state-of-the-art
methods that use prior map information, which is shown to achieve superior
performance. We also demonstrate the proposed method is able to perform
real-time estimation at 15 Hz, on a resource-constrained Jetson AGX Orin
embedded platform with impressive accuracy.

Comments:
- 6 pages, 7 figures

---

## DynaMoN: Motion-Aware Fast And Robust Camera Localization for Dynamic  NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-16 | Mert Asim Karaoglu, Hannah Schieber, Nicolas Schischka, Melih Görgülü, Florian Grötzner, Alexander Ladikos, Daniel Roth, Nassir Navab, Benjamin Busam | cs.CV | [PDF](http://arxiv.org/pdf/2309.08927v1){: .btn .btn-green } |

**Abstract**: Dynamic reconstruction with neural radiance fields (NeRF) requires accurate
camera poses. These are often hard to retrieve with existing
structure-from-motion (SfM) pipelines as both camera and scene content can
change. We propose DynaMoN that leverages simultaneous localization and mapping
(SLAM) jointly with motion masking to handle dynamic scene content. Our robust
SLAM-based tracking module significantly accelerates the training process of
the dynamic NeRF while improving the quality of synthesized views at the same
time. Extensive experimental validation on TUM RGB-D, BONN RGB-D Dynamic and
the DyCheck's iPhone dataset, three real-world datasets, shows the advantages
of DynaMoN both for camera pose estimation and novel view synthesis.

Comments:
- 6 pages, 4 figures

---

## Robust e-NeRF: NeRF from Sparse & Noisy Events under Non-Uniform Motion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-15 | Weng Fei Low, Gim Hee Lee | cs.CV | [PDF](http://arxiv.org/pdf/2309.08596v1){: .btn .btn-green } |

**Abstract**: Event cameras offer many advantages over standard cameras due to their
distinctive principle of operation: low power, low latency, high temporal
resolution and high dynamic range. Nonetheless, the success of many downstream
visual applications also hinges on an efficient and effective scene
representation, where Neural Radiance Field (NeRF) is seen as the leading
candidate. Such promise and potential of event cameras and NeRF inspired recent
works to investigate on the reconstruction of NeRF from moving event cameras.
However, these works are mainly limited in terms of the dependence on dense and
low-noise event streams, as well as generalization to arbitrary contrast
threshold values and camera speed profiles. In this work, we propose Robust
e-NeRF, a novel method to directly and robustly reconstruct NeRFs from moving
event cameras under various real-world conditions, especially from sparse and
noisy events generated under non-uniform motion. It consists of two key
components: a realistic event generation model that accounts for various
intrinsic parameters (e.g. time-independent, asymmetric threshold and
refractory period) and non-idealities (e.g. pixel-to-pixel threshold
variation), as well as a complementary pair of normalized reconstruction losses
that can effectively generalize to arbitrary speed profiles and intrinsic
parameter values without such prior knowledge. Experiments on real and novel
realistically simulated sequences verify our effectiveness. Our code, synthetic
dataset and improved event simulator are public.

Comments:
- Accepted to ICCV 2023. Project website is accessible at
  https://wengflow.github.io/robust-e-nerf

---

## Breathing New Life into 3D Assets with Generative Repainting



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-15 | Tianfu Wang, Menelaos Kanakis, Konrad Schindler, Luc Van Gool, Anton Obukhov | cs.CV | [PDF](http://arxiv.org/pdf/2309.08523v2){: .btn .btn-green } |

**Abstract**: Diffusion-based text-to-image models ignited immense attention from the
vision community, artists, and content creators. Broad adoption of these models
is due to significant improvement in the quality of generations and efficient
conditioning on various modalities, not just text. However, lifting the rich
generative priors of these 2D models into 3D is challenging. Recent works have
proposed various pipelines powered by the entanglement of diffusion models and
neural fields. We explore the power of pretrained 2D diffusion models and
standard 3D neural radiance fields as independent, standalone tools and
demonstrate their ability to work together in a non-learned fashion. Such
modularity has the intrinsic advantage of eased partial upgrades, which became
an important property in such a fast-paced domain. Our pipeline accepts any
legacy renderable geometry, such as textured or untextured meshes, orchestrates
the interaction between 2D generative refinement and 3D consistency enforcement
tools, and outputs a painted input geometry in several formats. We conduct a
large-scale study on a wide range of objects and categories from the
ShapeNetSem dataset and demonstrate the advantages of our approach, both
qualitatively and quantitatively. Project page:
https://www.obukhov.ai/repainting_3d_assets

---

## Deformable Neural Radiance Fields using RGB and Event Cameras



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-15 | Qi Ma, Danda Pani Paudel, Ajad Chhatkuli, Luc Van Gool | cs.CV | [PDF](http://arxiv.org/pdf/2309.08416v2){: .btn .btn-green } |

**Abstract**: Modeling Neural Radiance Fields for fast-moving deformable objects from
visual data alone is a challenging problem. A major issue arises due to the
high deformation and low acquisition rates. To address this problem, we propose
to use event cameras that offer very fast acquisition of visual change in an
asynchronous manner. In this work, we develop a novel method to model the
deformable neural radiance fields using RGB and event cameras. The proposed
method uses the asynchronous stream of events and calibrated sparse RGB frames.
In our setup, the camera pose at the individual events required to integrate
them into the radiance fields remains unknown. Our method jointly optimizes
these poses and the radiance field. This happens efficiently by leveraging the
collection of events at once and actively sampling the events during learning.
Experiments conducted on both realistically rendered graphics and real-world
datasets demonstrate a significant benefit of the proposed method over the
state-of-the-art and the compared baseline.
  This shows a promising direction for modeling deformable neural radiance
fields in real-world dynamic scenes.

---

## Indoor Scene Reconstruction with Fine-Grained Details Using Hybrid  Representation and Normal Prior Enhancement



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-14 | Sheng Ye, Yubin Hu, Matthieu Lin, Yu-Hui Wen, Wang Zhao, Yong-Jin Liu, Wenping Wang | cs.CV | [PDF](http://arxiv.org/pdf/2309.07640v2){: .btn .btn-green } |

**Abstract**: The reconstruction of indoor scenes from multi-view RGB images is challenging
due to the coexistence of flat and texture-less regions alongside delicate and
fine-grained regions. Recent methods leverage neural radiance fields aided by
predicted surface normal priors to recover the scene geometry. These methods
excel in producing complete and smooth results for floor and wall areas.
However, they struggle to capture complex surfaces with high-frequency
structures due to the inadequate neural representation and the inaccurately
predicted normal priors. This work aims to reconstruct high-fidelity surfaces
with fine-grained details by addressing the above limitations. To improve the
capacity of the implicit representation, we propose a hybrid architecture to
represent low-frequency and high-frequency regions separately. To enhance the
normal priors, we introduce a simple yet effective image sharpening and
denoising technique, coupled with a network that estimates the pixel-wise
uncertainty of the predicted surface normal vectors. Identifying such
uncertainty can prevent our model from being misled by unreliable surface
normal supervisions that hinder the accurate reconstruction of intricate
geometries. Experiments on the benchmark datasets show that our method
outperforms existing methods in terms of reconstruction quality. Furthermore,
the proposed method also generalizes well to real-world indoor scenarios
captured by our hand-held mobile phones. Our code is publicly available at:
https://github.com/yec22/Fine-Grained-Indoor-Recon.

---

## CoRF : Colorizing Radiance Fields using Knowledge Distillation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-14 | Ankit Dhiman, R Srinath, Srinjay Sarkar, Lokesh R Boregowda, R Venkatesh Babu | cs.CV | [PDF](http://arxiv.org/pdf/2309.07668v1){: .btn .btn-green } |

**Abstract**: Neural radiance field (NeRF) based methods enable high-quality novel-view
synthesis for multi-view images. This work presents a method for synthesizing
colorized novel views from input grey-scale multi-view images. When we apply
image or video-based colorization methods on the generated grey-scale novel
views, we observe artifacts due to inconsistency across views. Training a
radiance field network on the colorized grey-scale image sequence also does not
solve the 3D consistency issue. We propose a distillation based method to
transfer color knowledge from the colorization networks trained on natural
images to the radiance field network. Specifically, our method uses the
radiance field network as a 3D representation and transfers knowledge from
existing 2D colorization methods. The experimental results demonstrate that the
proposed method produces superior colorized novel views for indoor and outdoor
scenes while maintaining cross-view consistency than baselines. Further, we
show the efficacy of our method on applications like colorization of radiance
field network trained from 1.) Infra-Red (IR) multi-view images and 2.) Old
grey-scale multi-view image sequences.

Comments:
- AI3DCC @ ICCV 2023

---

## DT-NeRF: Decomposed Triplane-Hash Neural Radiance Fields for  High-Fidelity Talking Portrait Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-14 | Yaoyu Su, Shaohui Wang, Haoqian Wang | cs.CV | [PDF](http://arxiv.org/pdf/2309.07752v1){: .btn .btn-green } |

**Abstract**: In this paper, we present the decomposed triplane-hash neural radiance fields
(DT-NeRF), a framework that significantly improves the photorealistic rendering
of talking faces and achieves state-of-the-art results on key evaluation
datasets. Our architecture decomposes the facial region into two specialized
triplanes: one specialized for representing the mouth, and the other for the
broader facial features. We introduce audio features as residual terms and
integrate them as query vectors into our model through an audio-mouth-face
transformer. Additionally, our method leverages the capabilities of Neural
Radiance Fields (NeRF) to enrich the volumetric representation of the entire
face through additive volumetric rendering techniques. Comprehensive
experimental evaluations corroborate the effectiveness and superiority of our
proposed approach.

Comments:
- 5 pages, 5 figures. Submitted to ICASSP 2024

---

## MC-NeRF: Multi-Camera Neural Radiance Fields for Multi-Camera Image  Acquisition Systems

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-14 | Yu Gao, Lutong Su, Hao Liang, Yufeng Yue, Yi Yang, Mengyin Fu | cs.CV | [PDF](http://arxiv.org/pdf/2309.07846v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) employ multi-view images for 3D scene
representation and have shown remarkable performance. As one of the primary
sources of multi-view images, multi-camera systems encounter challenges such as
varying intrinsic parameters and frequent pose changes. Most previous
NeRF-based methods often assume a global unique camera and seldom consider
scenarios with multiple cameras. Besides, some pose-robust methods still remain
susceptible to suboptimal solutions when poses are poor initialized. In this
paper, we propose MC-NeRF, a method can jointly optimize both intrinsic and
extrinsic parameters for bundle-adjusting Neural Radiance Fields. Firstly, we
conduct a theoretical analysis to tackle the degenerate case and coupling issue
that arise from the joint optimization between intrinsic and extrinsic
parameters. Secondly, based on the proposed solutions, we introduce an
efficient calibration image acquisition scheme for multi-camera systems,
including the design of calibration object. Lastly, we present a global
end-to-end network with training sequence that enables the regression of
intrinsic and extrinsic parameters, along with the rendering network. Moreover,
most existing datasets are designed for unique camera, we create a new dataset
that includes four different styles of multi-camera acquisition systems,
allowing readers to generate custom datasets. Experiments confirm the
effectiveness of our method when each image corresponds to different camera
parameters. Specifically, we adopt up to 110 images with 110 different
intrinsic and extrinsic parameters, to achieve 3D scene representation without
providing initial poses. The Code and supplementary materials are available at
https://in2-viaun.github.io/MC-NeRF.

Comments:
- This manuscript is currently under review

---

## Gradient based Grasp Pose Optimization on a NeRF that Approximates Grasp  Success

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-14 | Gergely Sóti, Björn Hein, Christian Wurll | cs.RO | [PDF](http://arxiv.org/pdf/2309.08040v1){: .btn .btn-green } |

**Abstract**: Current robotic grasping methods often rely on estimating the pose of the
target object, explicitly predicting grasp poses, or implicitly estimating
grasp success probabilities. In this work, we propose a novel approach that
directly maps gripper poses to their corresponding grasp success values,
without considering objectness. Specifically, we leverage a Neural Radiance
Field (NeRF) architecture to learn a scene representation and use it to train a
grasp success estimator that maps each pose in the robot's task space to a
grasp success value. We employ this learned estimator to tune its inputs, i.e.,
grasp poses, by gradient-based optimization to obtain successful grasp poses.
Contrary to other NeRF-based methods which enhance existing grasp pose
estimation approaches by relying on NeRF's rendering capabilities or directly
estimate grasp poses in a discretized space using NeRF's scene representation
capabilities, our approach uniquely sidesteps both the need for rendering and
the limitation of discretization. We demonstrate the effectiveness of our
approach on four simulated 3DoF (Degree of Freedom) robotic grasping tasks and
show that it can generalize to novel objects. Our best model achieves an
average translation error of 3mm from valid grasp poses. This work opens the
door for future research to apply our approach to higher DoF grasps and
real-world scenarios.

---

## Spec-NeRF: Multi-spectral Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-14 | Jiabao Li, Yuqi Li, Ciliang Sun, Chong Wang, Jinhui Xiang | eess.IV | [PDF](http://arxiv.org/pdf/2310.12987v1){: .btn .btn-green } |

**Abstract**: We propose Multi-spectral Neural Radiance Fields(Spec-NeRF) for jointly
reconstructing a multispectral radiance field and spectral sensitivity
functions(SSFs) of the camera from a set of color images filtered by different
filters. The proposed method focuses on modeling the physical imaging process,
and applies the estimated SSFs and radiance field to synthesize novel views of
multispectral scenes. In this method, the data acquisition requires only a
low-cost trichromatic camera and several off-the-shelf color filters, making it
more practical than using specialized 3D scanning and spectral imaging
equipment. Our experiments on both synthetic and real scenario datasets
demonstrate that utilizing filtered RGB images with learnable NeRF and SSFs can
achieve high fidelity and promising spectral reconstruction while retaining the
inherent capability of NeRF to comprehend geometric structures. Code is
available at https://github.com/CPREgroup/SpecNeRF-v2.

---

## Text-Guided Generation and Editing of Compositional 3D Avatars

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-13 | Hao Zhang, Yao Feng, Peter Kulits, Yandong Wen, Justus Thies, Michael J. Black | cs.CV | [PDF](http://arxiv.org/pdf/2309.07125v1){: .btn .btn-green } |

**Abstract**: Our goal is to create a realistic 3D facial avatar with hair and accessories
using only a text description. While this challenge has attracted significant
recent interest, existing methods either lack realism, produce unrealistic
shapes, or do not support editing, such as modifications to the hairstyle. We
argue that existing methods are limited because they employ a monolithic
modeling approach, using a single representation for the head, face, hair, and
accessories. Our observation is that the hair and face, for example, have very
different structural qualities that benefit from different representations.
Building on this insight, we generate avatars with a compositional model, in
which the head, face, and upper body are represented with traditional 3D
meshes, and the hair, clothing, and accessories with neural radiance fields
(NeRF). The model-based mesh representation provides a strong geometric prior
for the face region, improving realism while enabling editing of the person's
appearance. By using NeRFs to represent the remaining components, our method is
able to model and synthesize parts with complex geometry and appearance, such
as curly hair and fluffy scarves. Our novel system synthesizes these
high-quality compositional avatars from text descriptions. The experimental
results demonstrate that our method, Text-guided generation and Editing of
Compositional Avatars (TECA), produces avatars that are more realistic than
those of recent methods while being editable because of their compositional
nature. For example, our TECA enables the seamless transfer of compositional
features like hairstyles, scarves, and other accessories between avatars. This
capability supports applications such as virtual try-on.

Comments:
- Home page: https://yfeng95.github.io/teca

---

## Dynamic NeRFs for Soccer Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-13 | Sacha Lewin, Maxime Vandegar, Thomas Hoyoux, Olivier Barnich, Gilles Louppe | cs.CV | [PDF](http://arxiv.org/pdf/2309.06802v1){: .btn .btn-green } |

**Abstract**: The long-standing problem of novel view synthesis has many applications,
notably in sports broadcasting. Photorealistic novel view synthesis of soccer
actions, in particular, is of enormous interest to the broadcast industry. Yet
only a few industrial solutions have been proposed, and even fewer that achieve
near-broadcast quality of the synthetic replays. Except for their setup of
multiple static cameras around the playfield, the best proprietary systems
disclose close to no information about their inner workings. Leveraging
multiple static cameras for such a task indeed presents a challenge rarely
tackled in the literature, for a lack of public datasets: the reconstruction of
a large-scale, mostly static environment, with small, fast-moving elements.
Recently, the emergence of neural radiance fields has induced stunning progress
in many novel view synthesis applications, leveraging deep learning principles
to produce photorealistic results in the most challenging settings. In this
work, we investigate the feasibility of basing a solution to the task on
dynamic NeRFs, i.e., neural models purposed to reconstruct general dynamic
content. We compose synthetic soccer environments and conduct multiple
experiments using them, identifying key components that help reconstruct soccer
scenes with dynamic NeRFs. We show that, although this approach cannot fully
meet the quality requirements for the target application, it suggests promising
avenues toward a cost-efficient, automatic solution. We also make our work
dataset and code publicly available, with the goal to encourage further efforts
from the research community on the task of novel view synthesis for dynamic
soccer scenes. For code, data, and video results, please see
https://soccernerfs.isach.be.

Comments:
- Accepted at the 6th International ACM Workshop on Multimedia Content
  Analysis in Sports. 8 pages, 9 figures. Project page:
  https://soccernerfs.isach.be

---

## Learning Disentangled Avatars with Hybrid 3D Representations



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-12 | Yao Feng, Weiyang Liu, Timo Bolkart, Jinlong Yang, Marc Pollefeys, Michael J. Black | cs.CV | [PDF](http://arxiv.org/pdf/2309.06441v1){: .btn .btn-green } |

**Abstract**: Tremendous efforts have been made to learn animatable and photorealistic
human avatars. Towards this end, both explicit and implicit 3D representations
are heavily studied for a holistic modeling and capture of the whole human
(e.g., body, clothing, face and hair), but neither representation is an optimal
choice in terms of representation efficacy since different parts of the human
avatar have different modeling desiderata. For example, meshes are generally
not suitable for modeling clothing and hair. Motivated by this, we present
Disentangled Avatars~(DELTA), which models humans with hybrid explicit-implicit
3D representations. DELTA takes a monocular RGB video as input, and produces a
human avatar with separate body and clothing/hair layers. Specifically, we
demonstrate two important applications for DELTA. For the first one, we
consider the disentanglement of the human body and clothing and in the second,
we disentangle the face and hair. To do so, DELTA represents the body or face
with an explicit mesh-based parametric 3D model and the clothing or hair with
an implicit neural radiance field. To make this possible, we design an
end-to-end differentiable renderer that integrates meshes into volumetric
rendering, enabling DELTA to learn directly from monocular videos without any
3D supervision. Finally, we show that how these two applications can be easily
combined to model full-body avatars, such that the hair, face, body and
clothing can be fully disentangled yet jointly rendered. Such a disentanglement
enables hair and clothing transfer to arbitrary body shapes. We empirically
validate the effectiveness of DELTA's disentanglement by demonstrating its
promising performance on disentangled reconstruction, virtual clothing try-on
and hairstyle transfer. To facilitate future research, we also release an
open-sourced pipeline for the study of hybrid human avatar modeling.

Comments:
- home page: https://yfeng95.github.io/delta. arXiv admin note: text
  overlap with arXiv:2210.01868

---

## Federated Learning for Large-Scale Scene Modeling with Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-12 | Teppei Suzuki | cs.CV | [PDF](http://arxiv.org/pdf/2309.06030v1){: .btn .btn-green } |

**Abstract**: We envision a system to continuously build and maintain a map based on
earth-scale neural radiance fields (NeRF) using data collected from vehicles
and drones in a lifelong learning manner. However, existing large-scale
modeling by NeRF has problems in terms of scalability and maintainability when
modeling earth-scale environments. Therefore, to address these problems, we
propose a federated learning pipeline for large-scale modeling with NeRF. We
tailor the model aggregation pipeline in federated learning for NeRF, thereby
allowing local updates of NeRF. In the aggregation step, the accuracy of the
clients' global pose is critical. Thus, we also propose global pose alignment
to align the noisy global pose of clients before the aggregation step. In
experiments, we show the effectiveness of the proposed pose alignment and the
federated learning pipeline on the large-scale scene dataset, Mill19.

---

## PAg-NeRF: Towards fast and efficient end-to-end panoptic 3D  representations for agricultural robotics

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-11 | Claus Smitt, Michael Halstead, Patrick Zimmer, Thomas Läbe, Esra Guclu, Cyrill Stachniss, Chris McCool | cs.RO | [PDF](http://arxiv.org/pdf/2309.05339v1){: .btn .btn-green } |

**Abstract**: Precise scene understanding is key for most robot monitoring and intervention
tasks in agriculture. In this work we present PAg-NeRF which is a novel
NeRF-based system that enables 3D panoptic scene understanding. Our
representation is trained using an image sequence with noisy robot odometry
poses and automatic panoptic predictions with inconsistent IDs between frames.
Despite this noisy input, our system is able to output scene geometry,
photo-realistic renders and 3D consistent panoptic representations with
consistent instance IDs. We evaluate this novel system in a very challenging
horticultural scenario and in doing so demonstrate an end-to-end trainable
system that can make use of noisy robot poses rather than precise poses that
have to be pre-calculated. Compared to a baseline approach the peak signal to
noise ratio is improved from 21.34dB to 23.37dB while the panoptic quality
improves from 56.65% to 70.08%. Furthermore, our approach is faster and can be
tuned to improve inference time by more than a factor of 2 while being memory
efficient with approximately 12 times fewer parameters.

---

## Editing 3D Scenes via Text Prompts without Retraining

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-10 | Shuangkang Fang, Yufeng Wang, Yi Yang, Yi-Hsuan Tsai, Wenrui Ding, Shuchang Zhou, Ming-Hsuan Yang | cs.CV | [PDF](http://arxiv.org/pdf/2309.04917v3){: .btn .btn-green } |

**Abstract**: Numerous diffusion models have recently been applied to image synthesis and
editing. However, editing 3D scenes is still in its early stages. It poses
various challenges, such as the requirement to design specific methods for
different editing types, retraining new models for various 3D scenes, and the
absence of convenient human interaction during editing. To tackle these issues,
we introduce a text-driven editing method, termed DN2N, which allows for the
direct acquisition of a NeRF model with universal editing capabilities,
eliminating the requirement for retraining. Our method employs off-the-shelf
text-based editing models of 2D images to modify the 3D scene images, followed
by a filtering process to discard poorly edited images that disrupt 3D
consistency. We then consider the remaining inconsistency as a problem of
removing noise perturbation, which can be solved by generating training data
with similar perturbation characteristics for training. We further propose
cross-view regularization terms to help the generalized NeRF model mitigate
these perturbations. Our text-driven method allows users to edit a 3D scene
with their desired description, which is more friendly, intuitive, and
practical than prior works. Empirical results show that our method achieves
multiple editing types, including but not limited to appearance editing,
weather transition, material changing, and style transfer. Most importantly,
our method generalizes well with editing abilities shared among a set of model
parameters without requiring a customized editing model for some specific
scenes, thus inferring novel views with editing effects directly from user
input. The project website is available at https://sk-fun.fun/DN2N

Comments:
- Project Website: https://sk-fun.fun/DN2N

---

## SC-NeRF: Self-Correcting Neural Radiance Field with Sparse Views

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-10 | Liang Song, Guangming Wang, Jiuming Liu, Zhenyang Fu, Yanzi Miao,  Hesheng | cs.CV | [PDF](http://arxiv.org/pdf/2309.05028v1){: .btn .btn-green } |

**Abstract**: In recent studies, the generalization of neural radiance fields for novel
view synthesis task has been widely explored. However, existing methods are
limited to objects and indoor scenes. In this work, we extend the
generalization task to outdoor scenes, trained only on object-level datasets.
This approach presents two challenges. Firstly, the significant distributional
shift between training and testing scenes leads to black artifacts in rendering
results. Secondly, viewpoint changes in outdoor scenes cause ghosting or
missing regions in rendered images. To address these challenges, we propose a
geometric correction module and an appearance correction module based on
multi-head attention mechanisms. We normalize rendered depth and combine it
with light direction as query in the attention mechanism. Our network
effectively corrects varying scene structures and geometric features in outdoor
scenes, generalizing well from object-level to unseen outdoor scenes.
Additionally, we use appearance correction module to correct appearance
features, preventing rendering artifacts like blank borders and ghosting due to
viewpoint changes. By combining these modules, our approach successfully
tackles the challenges of outdoor scene generalization, producing high-quality
rendering results. When evaluated on four datasets (Blender, DTU, LLFF,
Spaces), our network outperforms previous methods. Notably, compared to
MVSNeRF, our network improves average PSNR from 19.369 to 25.989, SSIM from
0.838 to 0.889, and reduces LPIPS from 0.265 to 0.224 on Spaces outdoor scenes.

---

## Mirror-Aware Neural Humans

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-09 | Daniel Ajisafe, James Tang, Shih-Yang Su, Bastian Wandt, Helge Rhodin | cs.CV | [PDF](http://arxiv.org/pdf/2309.04750v1){: .btn .btn-green } |

**Abstract**: Human motion capture either requires multi-camera systems or is unreliable
using single-view input due to depth ambiguities. Meanwhile, mirrors are
readily available in urban environments and form an affordable alternative by
recording two views with only a single camera. However, the mirror setting
poses the additional challenge of handling occlusions of real and mirror image.
Going beyond existing mirror approaches for 3D human pose estimation, we
utilize mirrors for learning a complete body model, including shape and dense
appearance. Our main contributions are extending articulated neural radiance
fields to include a notion of a mirror, making it sample-efficient over
potential occlusion regions. Together, our contributions realize a
consumer-level 3D motion capture system that starts from off-the-shelf 2D poses
by automatically calibrating the camera, estimating mirror orientation, and
subsequently lifting 2D keypoint detections to 3D skeleton pose that is used to
condition the mirror-aware NeRF. We empirically demonstrate the benefit of
learning a body model and accounting for occlusion in challenging mirror
scenes.

Comments:
- Project website:
  https://danielajisafe.github.io/mirror-aware-neural-humans/

---

## DeformToon3D: Deformable 3D Toonification from Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-08 | Junzhe Zhang, Yushi Lan, Shuai Yang, Fangzhou Hong, Quan Wang, Chai Kiat Yeo, Ziwei Liu, Chen Change Loy | cs.CV | [PDF](http://arxiv.org/pdf/2309.04410v1){: .btn .btn-green } |

**Abstract**: In this paper, we address the challenging problem of 3D toonification, which
involves transferring the style of an artistic domain onto a target 3D face
with stylized geometry and texture. Although fine-tuning a pre-trained 3D GAN
on the artistic domain can produce reasonable performance, this strategy has
limitations in the 3D domain. In particular, fine-tuning can deteriorate the
original GAN latent space, which affects subsequent semantic editing, and
requires independent optimization and storage for each new style, limiting
flexibility and efficient deployment. To overcome these challenges, we propose
DeformToon3D, an effective toonification framework tailored for hierarchical 3D
GAN. Our approach decomposes 3D toonification into subproblems of geometry and
texture stylization to better preserve the original latent space. Specifically,
we devise a novel StyleField that predicts conditional 3D deformation to align
a real-space NeRF to the style space for geometry stylization. Thanks to the
StyleField formulation, which already handles geometry stylization well,
texture stylization can be achieved conveniently via adaptive style mixing that
injects information of the artistic domain into the decoder of the pre-trained
3D GAN. Due to the unique design, our method enables flexible style degree
control and shape-texture-specific style swap. Furthermore, we achieve
efficient training without any real-world 2D-3D training pairs but proxy
samples synthesized from off-the-shelf 2D toonification models.

Comments:
- ICCV 2023. Code: https://github.com/junzhezhang/DeformToon3D Project
  page: https://www.mmlab-ntu.com/project/deformtoon3d/

---

## Dynamic Mesh-Aware Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-08 | Yi-Ling Qiao, Alexander Gao, Yiran Xu, Yue Feng, Jia-Bin Huang, Ming C. Lin | cs.GR | [PDF](http://arxiv.org/pdf/2309.04581v1){: .btn .btn-green } |

**Abstract**: Embedding polygonal mesh assets within photorealistic Neural Radience Fields
(NeRF) volumes, such that they can be rendered and their dynamics simulated in
a physically consistent manner with the NeRF, is under-explored from the system
perspective of integrating NeRF into the traditional graphics pipeline. This
paper designs a two-way coupling between mesh and NeRF during rendering and
simulation. We first review the light transport equations for both mesh and
NeRF, then distill them into an efficient algorithm for updating radiance and
throughput along a cast ray with an arbitrary number of bounces. To resolve the
discrepancy between the linear color space that the path tracer assumes and the
sRGB color space that standard NeRF uses, we train NeRF with High Dynamic Range
(HDR) images. We also present a strategy to estimate light sources and cast
shadows on the NeRF. Finally, we consider how the hybrid surface-volumetric
formulation can be efficiently integrated with a high-performance physics
simulator that supports cloth, rigid and soft bodies. The full rendering and
simulation system can be run on a GPU at interactive rates. We show that a
hybrid system approach outperforms alternatives in visual realism for mesh
insertion, because it allows realistic light transport from volumetric NeRF
media onto surfaces, which affects the appearance of reflective/refractive
surfaces and illumination of diffuse surfaces informed by the dynamic scene.

Comments:
- ICCV 2023

---

## SimpleNeRF: Regularizing Sparse Input Neural Radiance Fields with  Simpler Solutions

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-07 | Nagabhushan Somraj, Adithyan Karanayil, Rajiv Soundararajan | cs.CV | [PDF](http://arxiv.org/pdf/2309.03955v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) show impressive performance for the
photorealistic free-view rendering of scenes. However, NeRFs require dense
sampling of images in the given scene, and their performance degrades
significantly when only a sparse set of views are available. Researchers have
found that supervising the depth estimated by the NeRF helps train it
effectively with fewer views. The depth supervision is obtained either using
classical approaches or neural networks pre-trained on a large dataset. While
the former may provide only sparse supervision, the latter may suffer from
generalization issues. As opposed to the earlier approaches, we seek to learn
the depth supervision by designing augmented models and training them along
with the NeRF. We design augmented models that encourage simpler solutions by
exploring the role of positional encoding and view-dependent radiance in
training the few-shot NeRF. The depth estimated by these simpler models is used
to supervise the NeRF depth estimates. Since the augmented models can be
inaccurate in certain regions, we design a mechanism to choose only reliable
depth estimates for supervision. Finally, we add a consistency loss between the
coarse and fine multi-layer perceptrons of the NeRF to ensure better
utilization of hierarchical sampling. We achieve state-of-the-art
view-synthesis performance on two popular datasets by employing the above
regularizations. The source code for our model can be found on our project
page: https://nagabhushansn95.github.io/publications/2023/SimpleNeRF.html

Comments:
- SIGGRAPH Asia 2023

---

## BluNF: Blueprint Neural Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-07 | Robin Courant, Xi Wang, Marc Christie, Vicky Kalogeiton | cs.CV | [PDF](http://arxiv.org/pdf/2309.03933v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have revolutionized scene novel view
synthesis, offering visually realistic, precise, and robust implicit
reconstructions. While recent approaches enable NeRF editing, such as object
removal, 3D shape modification, or material property manipulation, the manual
annotation prior to such edits makes the process tedious. Additionally,
traditional 2D interaction tools lack an accurate sense of 3D space, preventing
precise manipulation and editing of scenes. In this paper, we introduce a novel
approach, called Blueprint Neural Field (BluNF), to address these editing
issues. BluNF provides a robust and user-friendly 2D blueprint, enabling
intuitive scene editing. By leveraging implicit neural representation, BluNF
constructs a blueprint of a scene using prior semantic and depth information.
The generated blueprint allows effortless editing and manipulation of NeRF
representations. We demonstrate BluNF's editability through an intuitive
click-and-change mechanism, enabling 3D manipulations, such as masking,
appearance modification, and object removal. Our approach significantly
contributes to visual content creation, paving the way for further research in
this area.

Comments:
- ICCV-W (AI3DCC) 2023. Project page with videos and code:
  https://www.lix.polytechnique.fr/vista/projects/2023_iccvw_courant/

---

## Text2Control3D: Controllable 3D Avatar Generation in Neural Radiance  Fields using Geometry-Guided Text-to-Image Diffusion Model

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-07 | Sungwon Hwang, Junha Hyung, Jaegul Choo | cs.CV | [PDF](http://arxiv.org/pdf/2309.03550v1){: .btn .btn-green } |

**Abstract**: Recent advances in diffusion models such as ControlNet have enabled
geometrically controllable, high-fidelity text-to-image generation. However,
none of them addresses the question of adding such controllability to
text-to-3D generation. In response, we propose Text2Control3D, a controllable
text-to-3D avatar generation method whose facial expression is controllable
given a monocular video casually captured with hand-held camera. Our main
strategy is to construct the 3D avatar in Neural Radiance Fields (NeRF)
optimized with a set of controlled viewpoint-aware images that we generate from
ControlNet, whose condition input is the depth map extracted from the input
video. When generating the viewpoint-aware images, we utilize cross-reference
attention to inject well-controlled, referential facial expression and
appearance via cross attention. We also conduct low-pass filtering of Gaussian
latent of the diffusion model in order to ameliorate the viewpoint-agnostic
texture problem we observed from our empirical analysis, where the
viewpoint-aware images contain identical textures on identical pixel positions
that are incomprehensible in 3D. Finally, to train NeRF with the images that
are viewpoint-aware yet are not strictly consistent in geometry, our approach
considers per-image geometric variation as a view of deformation from a shared
3D canonical space. Consequently, we construct the 3D avatar in a canonical
space of deformable NeRF by learning a set of per-image deformation via
deformation field table. We demonstrate the empirical results and discuss the
effectiveness of our method.

Comments:
- Project page: https://text2control3d.github.io/

---

## Bayes' Rays: Uncertainty Quantification for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-06 | Lily Goli, Cody Reading, Silvia Sellán, Alec Jacobson, Andrea Tagliasacchi | cs.CV | [PDF](http://arxiv.org/pdf/2309.03185v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have shown promise in applications like view
synthesis and depth estimation, but learning from multiview images faces
inherent uncertainties. Current methods to quantify them are either heuristic
or computationally demanding. We introduce BayesRays, a post-hoc framework to
evaluate uncertainty in any pre-trained NeRF without modifying the training
process. Our method establishes a volumetric uncertainty field using spatial
perturbations and a Bayesian Laplace approximation. We derive our algorithm
statistically and show its superior performance in key metrics and
applications. Additional results available at: https://bayesrays.github.io.

---

## ResFields: Residual Neural Fields for Spatiotemporal Signals

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-06 | Marko Mihajlovic, Sergey Prokudin, Marc Pollefeys, Siyu Tang | cs.CV | [PDF](http://arxiv.org/pdf/2309.03160v2){: .btn .btn-green } |

**Abstract**: Neural fields, a category of neural networks trained to represent
high-frequency signals, have gained significant attention in recent years due
to their impressive performance in modeling complex 3D data, especially large
neural signed distance (SDFs) or radiance fields (NeRFs) via a single
multi-layer perceptron (MLP). However, despite the power and simplicity of
representing signals with an MLP, these methods still face challenges when
modeling large and complex temporal signals due to the limited capacity of
MLPs. In this paper, we propose an effective approach to address this
limitation by incorporating temporal residual layers into neural fields, dubbed
ResFields, a novel class of networks specifically designed to effectively
represent complex temporal signals. We conduct a comprehensive analysis of the
properties of ResFields and propose a matrix factorization technique to reduce
the number of trainable parameters and enhance generalization capabilities.
Importantly, our formulation seamlessly integrates with existing techniques and
consistently improves results across various challenging tasks: 2D video
approximation, dynamic shape modeling via temporal SDFs, and dynamic NeRF
reconstruction. Lastly, we demonstrate the practical utility of ResFields by
showcasing its effectiveness in capturing dynamic 3D scenes from sparse sensory
inputs of a lightweight capture system.

Comments:
- Project page and code at https://markomih.github.io/ResFields/

---

## Adv3D: Generating 3D Adversarial Examples in Driving Scenarios with NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-04 | Leheng Li, Qing Lian, Ying-Cong Chen | cs.CV | [PDF](http://arxiv.org/pdf/2309.01351v1){: .btn .btn-green } |

**Abstract**: Deep neural networks (DNNs) have been proven extremely susceptible to
adversarial examples, which raises special safety-critical concerns for
DNN-based autonomous driving stacks (i.e., 3D object detection). Although there
are extensive works on image-level attacks, most are restricted to 2D pixel
spaces, and such attacks are not always physically realistic in our 3D world.
Here we present Adv3D, the first exploration of modeling adversarial examples
as Neural Radiance Fields (NeRFs). Advances in NeRF provide photorealistic
appearances and 3D accurate generation, yielding a more realistic and
realizable adversarial example. We train our adversarial NeRF by minimizing the
surrounding objects' confidence predicted by 3D detectors on the training set.
Then we evaluate Adv3D on the unseen validation set and show that it can cause
a large performance reduction when rendering NeRF in any sampled pose. To
generate physically realizable adversarial examples, we propose primitive-aware
sampling and semantic-guided regularization that enable 3D patch attacks with
camouflage adversarial texture. Experimental results demonstrate that the
trained adversarial NeRF generalizes well to different poses, scenes, and 3D
detectors. Finally, we provide a defense method to our attacks that involves
adversarial training through data augmentation. Project page:
https://len-li.github.io/adv3d-web

---

## Instant Continual Learning of Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-04 | Ryan Po, Zhengyang Dong, Alexander W. Bergman, Gordon Wetzstein | cs.CV | [PDF](http://arxiv.org/pdf/2309.01811v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) have emerged as an effective method for
novel-view synthesis and 3D scene reconstruction. However, conventional
training methods require access to all training views during scene
optimization. This assumption may be prohibitive in continual learning
scenarios, where new data is acquired in a sequential manner and a continuous
update of the NeRF is desired, as in automotive or remote sensing applications.
When naively trained in such a continual setting, traditional scene
representation frameworks suffer from catastrophic forgetting, where previously
learned knowledge is corrupted after training on new data. Prior works in
alleviating forgetting with NeRFs suffer from low reconstruction quality and
high latency, making them impractical for real-world application. We propose a
continual learning framework for training NeRFs that leverages replay-based
methods combined with a hybrid explicit--implicit scene representation. Our
method outperforms previous methods in reconstruction quality when trained in a
continual setting, while having the additional benefit of being an order of
magnitude faster.

Comments:
- For project page please visit https://ryanpo.com/icngp/

---

## SparseSat-NeRF: Dense Depth Supervised Neural Radiance Fields for Sparse  Satellite Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-09-01 | Lulin Zhang, Ewelina Rupnik | cs.CV | [PDF](http://arxiv.org/pdf/2309.00277v1){: .btn .btn-green } |

**Abstract**: Digital surface model generation using traditional multi-view stereo matching
(MVS) performs poorly over non-Lambertian surfaces, with asynchronous
acquisitions, or at discontinuities. Neural radiance fields (NeRF) offer a new
paradigm for reconstructing surface geometries using continuous volumetric
representation. NeRF is self-supervised, does not require ground truth geometry
for training, and provides an elegant way to include in its representation
physical parameters about the scene, thus potentially remedying the challenging
scenarios where MVS fails. However, NeRF and its variants require many views to
produce convincing scene's geometries which in earth observation satellite
imaging is rare. In this paper we present SparseSat-NeRF (SpS-NeRF) - an
extension of Sat-NeRF adapted to sparse satellite views. SpS-NeRF employs dense
depth supervision guided by crosscorrelation similarity metric provided by
traditional semi-global MVS matching. We demonstrate the effectiveness of our
approach on stereo and tri-stereo Pleiades 1B/WorldView-3 images, and compare
against NeRF and Sat-NeRF. The code is available at
https://github.com/LulinZhang/SpS-NeRF

Comments:
- ISPRS Annals 2023