---
layout: default
title: March
parent: 2023
nav_order: 3
---
<!---metadata--->

## VDN-NeRF: Resolving Shape-Radiance Ambiguity via View-Dependence  Normalization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-31 | Bingfan Zhu, Yanchao Yang, Xulong Wang, Youyi Zheng, Leonidas Guibas | cs.CV | [PDF](http://arxiv.org/pdf/2303.17968v1){: .btn .btn-green } |

**Abstract**: We propose VDN-NeRF, a method to train neural radiance fields (NeRFs) for
better geometry under non-Lambertian surface and dynamic lighting conditions
that cause significant variation in the radiance of a point when viewed from
different angles. Instead of explicitly modeling the underlying factors that
result in the view-dependent phenomenon, which could be complex yet not
inclusive, we develop a simple and effective technique that normalizes the
view-dependence by distilling invariant information already encoded in the
learned NeRFs. We then jointly train NeRFs for view synthesis with
view-dependence normalization to attain quality geometry. Our experiments show
that even though shape-radiance ambiguity is inevitable, the proposed
normalization can minimize its effect on geometry, which essentially aligns the
optimal capacity needed for explaining view-dependent variations. Our method
applies to various baselines and significantly improves geometry without
changing the volume rendering pipeline, even if the data is captured under a
moving light source. Code is available at: https://github.com/BoifZ/VDN-NeRF.

---

## NeILF++: Inter-Reflectable Light Fields for Geometry and Material  Estimation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-30 | Jingyang Zhang, Yao Yao, Shiwei Li, Jingbo Liu, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan | cs.CV | [PDF](http://arxiv.org/pdf/2303.17147v1){: .btn .btn-green } |

**Abstract**: We present a novel differentiable rendering framework for joint geometry,
material, and lighting estimation from multi-view images. In contrast to
previous methods which assume a simplified environment map or co-located
flashlights, in this work, we formulate the lighting of a static scene as one
neural incident light field (NeILF) and one outgoing neural radiance field
(NeRF). The key insight of the proposed method is the union of the incident and
outgoing light fields through physically-based rendering and inter-reflections
between surfaces, making it possible to disentangle the scene geometry,
material, and lighting from image observations in a physically-based manner.
The proposed incident light and inter-reflection framework can be easily
applied to other NeRF systems. We show that our method can not only decompose
the outgoing radiance into incident lights and surface materials, but also
serve as a surface refinement module that further improves the reconstruction
detail of the neural surface. We demonstrate on several datasets that the
proposed method is able to achieve state-of-the-art results in terms of
geometry reconstruction quality, material estimation accuracy, and the fidelity
of novel view rendering.

Comments:
- Project page: \url{https://yoyo000.github.io/NeILF_pp}

---

## Enhanced Stable View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-30 | Nishant Jain, Suryansh Kumar, Luc Van Gool | cs.CV | [PDF](http://arxiv.org/pdf/2303.17094v1){: .btn .btn-green } |

**Abstract**: We introduce an approach to enhance the novel view synthesis from images
taken from a freely moving camera. The introduced approach focuses on outdoor
scenes where recovering accurate geometric scaffold and camera pose is
challenging, leading to inferior results using the state-of-the-art stable view
synthesis (SVS) method. SVS and related methods fail for outdoor scenes
primarily due to (i) over-relying on the multiview stereo (MVS) for geometric
scaffold recovery and (ii) assuming COLMAP computed camera poses as the best
possible estimates, despite it being well-studied that MVS 3D reconstruction
accuracy is limited to scene disparity and camera-pose accuracy is sensitive to
key-point correspondence selection. This work proposes a principled way to
enhance novel view synthesis solutions drawing inspiration from the basics of
multiple view geometry. By leveraging the complementary behavior of MVS and
monocular depth, we arrive at a better scene depth per view for nearby and far
points, respectively. Moreover, our approach jointly refines camera poses with
image-based rendering via multiple rotation averaging graph optimization. The
recovered scene depth and the camera-pose help better view-dependent on-surface
feature aggregation of the entire scene. Extensive evaluation of our approach
on the popular benchmark dataset, such as Tanks and Temples, shows substantial
improvement in view synthesis results compared to the prior art. For instance,
our method shows 1.5 dB of PSNR improvement on the Tank and Temples. Similar
statistics are observed when tested on other benchmark datasets such as FVS,
Mip-NeRF 360, and DTU.

Comments:
- Accepted to IEEE/CVF CVPR 2023. Draft info: 13 pages, 6 Figures, 7
  Tables

---

## SynBody: Synthetic Dataset with Layered Human Models for 3D Human  Perception and Modeling

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-30 | Zhitao Yang, Zhongang Cai, Haiyi Mei, Shuai Liu, Zhaoxi Chen, Weiye Xiao, Yukun Wei, Zhongfei Qing, Chen Wei, Bo Dai, Wayne Wu, Chen Qian, Dahua Lin, Ziwei Liu, Lei Yang | cs.CV | [PDF](http://arxiv.org/pdf/2303.17368v2){: .btn .btn-green } |

**Abstract**: Synthetic data has emerged as a promising source for 3D human research as it
offers low-cost access to large-scale human datasets. To advance the diversity
and annotation quality of human models, we introduce a new synthetic dataset,
SynBody, with three appealing features: 1) a clothed parametric human model
that can generate a diverse range of subjects; 2) the layered human
representation that naturally offers high-quality 3D annotations to support
multiple tasks; 3) a scalable system for producing realistic data to facilitate
real-world tasks. The dataset comprises 1.2M images with corresponding accurate
3D annotations, covering 10,000 human body models, 1,187 actions, and various
viewpoints. The dataset includes two subsets for human pose and shape
estimation as well as human neural rendering. Extensive experiments on SynBody
indicate that it substantially enhances both SMPL and SMPL-X estimation.
Furthermore, the incorporation of layered annotations offers a valuable
training resource for investigating the Human Neural Radiance Fields (NeRF).

Comments:
- Accepted by ICCV 2023. Project webpage: https://synbody.github.io/

---

## NeRF-Supervised Deep Stereo

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-30 | Fabio Tosi, Alessio Tonioni, Daniele De Gregorio, Matteo Poggi | cs.CV | [PDF](http://arxiv.org/pdf/2303.17603v1){: .btn .btn-green } |

**Abstract**: We introduce a novel framework for training deep stereo networks effortlessly
and without any ground-truth. By leveraging state-of-the-art neural rendering
solutions, we generate stereo training data from image sequences collected with
a single handheld camera. On top of them, a NeRF-supervised training procedure
is carried out, from which we exploit rendered stereo triplets to compensate
for occlusions and depth maps as proxy labels. This results in stereo networks
capable of predicting sharp and detailed disparity maps. Experimental results
show that models trained under this regime yield a 30-40% improvement over
existing self-supervised methods on the challenging Middlebury dataset, filling
the gap to supervised models and, most times, outperforming them at zero-shot
generalization.

Comments:
- CVPR 2023. Project page: https://nerfstereo.github.io/ Code:
  https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo

---

## Instant Neural Radiance Fields Stylization



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-29 | Shaoxu Li, Ye Pan | cs.CV | [PDF](http://arxiv.org/pdf/2303.16884v1){: .btn .btn-green } |

**Abstract**: We present Instant Neural Radiance Fields Stylization, a novel approach for
multi-view image stylization for the 3D scene. Our approach models a neural
radiance field based on neural graphics primitives, which use a hash
table-based position encoder for position embedding. We split the position
encoder into two parts, the content and style sub-branches, and train the
network for normal novel view image synthesis with the content and style
targets. In the inference stage, we execute AdaIN to the output features of the
position encoder, with content and style voxel grid features as reference. With
the adjusted features, the stylization of novel view images could be obtained.
Our method extends the style target from style images to image sets of scenes
and does not require additional network training for stylization. Given a set
of images of 3D scenes and a style target(a style image or another set of 3D
scenes), our method can generate stylized novel views with a consistent
appearance at various view angles in less than 10 minutes on modern GPU
hardware. Extensive experimental results demonstrate the validity and
superiority of our method.

---

## TriVol: Point Cloud Rendering via Triple Volumes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-29 | Tao Hu, Xiaogang Xu, Ruihang Chu, Jiaya Jia | cs.CV | [PDF](http://arxiv.org/pdf/2303.16485v1){: .btn .btn-green } |

**Abstract**: Existing learning-based methods for point cloud rendering adopt various 3D
representations and feature querying mechanisms to alleviate the sparsity
problem of point clouds. However, artifacts still appear in rendered images,
due to the challenges in extracting continuous and discriminative 3D features
from point clouds. In this paper, we present a dense while lightweight 3D
representation, named TriVol, that can be combined with NeRF to render
photo-realistic images from point clouds. Our TriVol consists of triple slim
volumes, each of which is encoded from the point cloud. TriVol has two
advantages. First, it fuses respective fields at different scales and thus
extracts local and non-local features for discriminative representation.
Second, since the volume size is greatly reduced, our 3D decoder can be
efficiently inferred, allowing us to increase the resolution of the 3D space to
render more point details. Extensive experiments on different benchmarks with
varying kinds of scenes/objects demonstrate our framework's effectiveness
compared with current approaches. Moreover, our framework has excellent
generalization ability to render a category of scenes/objects without
fine-tuning.

---

## Point2Pix: Photo-Realistic Point Cloud Rendering via Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-29 | Tao Hu, Xiaogang Xu, Shu Liu, Jiaya Jia | cs.CV | [PDF](http://arxiv.org/pdf/2303.16482v1){: .btn .btn-green } |

**Abstract**: Synthesizing photo-realistic images from a point cloud is challenging because
of the sparsity of point cloud representation. Recent Neural Radiance Fields
and extensions are proposed to synthesize realistic images from 2D input. In
this paper, we present Point2Pix as a novel point renderer to link the 3D
sparse point clouds with 2D dense image pixels. Taking advantage of the point
cloud 3D prior and NeRF rendering pipeline, our method can synthesize
high-quality images from colored point clouds, generally for novel indoor
scenes. To improve the efficiency of ray sampling, we propose point-guided
sampling, which focuses on valid samples. Also, we present Point Encoding to
build Multi-scale Radiance Fields that provide discriminative 3D point
features. Finally, we propose Fusion Encoding to efficiently synthesize
high-quality images. Extensive experiments on the ScanNet and ArkitScenes
datasets demonstrate the effectiveness and generalization.

---

## Flow supervision for Deformable NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-28 | Chaoyang Wang, Lachlan Ewen MacDonald, Laszlo A. Jeni, Simon Lucey | cs.CV | [PDF](http://arxiv.org/pdf/2303.16333v1){: .btn .btn-green } |

**Abstract**: In this paper we present a new method for deformable NeRF that can directly
use optical flow as supervision. We overcome the major challenge with respect
to the computationally inefficiency of enforcing the flow constraints to the
backward deformation field, used by deformable NeRFs. Specifically, we show
that inverting the backward deformation function is actually not needed for
computing scene flows between frames. This insight dramatically simplifies the
problem, as one is no longer constrained to deformation functions that can be
analytically inverted. Instead, thanks to the weak assumptions required by our
derivation based on the inverse function theorem, our approach can be extended
to a broad class of commonly used backward deformation field. We present
results on monocular novel view synthesis with rapid object motion, and
demonstrate significant improvements over baselines without flow supervision.

---

## CuNeRF: Cube-Based Neural Radiance Field for Zero-Shot Medical Image  Arbitrary-Scale Super Resolution

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-28 | Zixuan Chen, Jian-Huang Lai, Lingxiao Yang, Xiaohua Xie | eess.IV | [PDF](http://arxiv.org/pdf/2303.16242v3){: .btn .btn-green } |

**Abstract**: Medical image arbitrary-scale super-resolution (MIASSR) has recently gained
widespread attention, aiming to super sample medical volumes at arbitrary
scales via a single model. However, existing MIASSR methods face two major
limitations: (i) reliance on high-resolution (HR) volumes and (ii) limited
generalization ability, which restricts their application in various scenarios.
To overcome these limitations, we propose Cube-based Neural Radiance Field
(CuNeRF), a zero-shot MIASSR framework that can yield medical images at
arbitrary scales and viewpoints in a continuous domain. Unlike existing MIASSR
methods that fit the mapping between low-resolution (LR) and HR volumes, CuNeRF
focuses on building a coordinate-intensity continuous representation from LR
volumes without the need for HR references. This is achieved by the proposed
differentiable modules: including cube-based sampling, isotropic volume
rendering, and cube-based hierarchical rendering. Through extensive experiments
on magnetic resource imaging (MRI) and computed tomography (CT) modalities, we
demonstrate that CuNeRF outperforms state-of-the-art MIASSR methods. CuNeRF
yields better visual verisimilitude and reduces aliasing artifacts at various
upsampling factors. Moreover, our CuNeRF does not need any LR-HR training
pairs, which is more flexible and easier to be used than others. Our code will
be publicly available soon.

Comments:
- This paper is accepted by the International Conference on Computer
  Vision (ICCV) 2023

---

## SparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-28 | Guangcong Wang, Zhaoxi Chen, Chen Change Loy, Ziwei Liu | cs.CV | [PDF](http://arxiv.org/pdf/2303.16196v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) significantly degrades when only a limited
number of views are available. To complement the lack of 3D information,
depth-based models, such as DSNeRF and MonoSDF, explicitly assume the
availability of accurate depth maps of multiple views. They linearly scale the
accurate depth maps as supervision to guide the predicted depth of few-shot
NeRFs. However, accurate depth maps are difficult and expensive to capture due
to wide-range depth distances in the wild.
  In this work, we present a new Sparse-view NeRF (SparseNeRF) framework that
exploits depth priors from real-world inaccurate observations. The inaccurate
depth observations are either from pre-trained depth models or coarse depth
maps of consumer-level depth sensors. Since coarse depth maps are not strictly
scaled to the ground-truth depth maps, we propose a simple yet effective
constraint, a local depth ranking method, on NeRFs such that the expected depth
ranking of the NeRF is consistent with that of the coarse depth maps in local
patches. To preserve the spatial continuity of the estimated depth of NeRF, we
further propose a spatial continuity constraint to encourage the consistency of
the expected depth continuity of NeRF with coarse depth maps. Surprisingly,
with simple depth ranking constraints, SparseNeRF outperforms all
state-of-the-art few-shot NeRF methods (including depth-based models) on
standard LLFF and DTU datasets. Moreover, we collect a new dataset NVS-RGBD
that contains real-world depth maps from Azure Kinect, ZED 2, and iPhone 13
Pro. Extensive experiments on NVS-RGBD dataset also validate the superiority
and generalizability of SparseNeRF. Code and dataset are available at
https://sparsenerf.github.io/.

Comments:
- Accepted by ICCV 2023, Project page: https://sparsenerf.github.io/

---

## Adaptive Voronoi NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-28 | Tim Elsner, Victor Czech, Julia Berger, Zain Selman, Isaak Lim, Leif Kobbelt | cs.CV | [PDF](http://arxiv.org/pdf/2303.16001v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) learn to represent a 3D scene from just a set
of registered images. Increasing sizes of a scene demands more complex
functions, typically represented by neural networks, to capture all details.
Training and inference then involves querying the neural network millions of
times per image, which becomes impractically slow. Since such complex functions
can be replaced by multiple simpler functions to improve speed, we show that a
hierarchy of Voronoi diagrams is a suitable choice to partition the scene. By
equipping each Voronoi cell with its own NeRF, our approach is able to quickly
learn a scene representation. We propose an intuitive partitioning of the space
that increases quality gains during training by distributing information evenly
among the networks and avoids artifacts through a top-down adaptive refinement.
Our framework is agnostic to the underlying NeRF method and easy to implement,
which allows it to be applied to various NeRF variants for improved learning
and rendering speeds.

---

## F$^{2}$-NeRF: Fast Neural Radiance Field Training with Free Camera  Trajectories

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-28 | Peng Wang, Yuan Liu, Zhaoxi Chen, Lingjie Liu, Ziwei Liu, Taku Komura, Christian Theobalt, Wenping Wang | cs.CV | [PDF](http://arxiv.org/pdf/2303.15951v1){: .btn .btn-green } |

**Abstract**: This paper presents a novel grid-based NeRF called F2-NeRF (Fast-Free-NeRF)
for novel view synthesis, which enables arbitrary input camera trajectories and
only costs a few minutes for training. Existing fast grid-based NeRF training
frameworks, like Instant-NGP, Plenoxels, DVGO, or TensoRF, are mainly designed
for bounded scenes and rely on space warping to handle unbounded scenes.
Existing two widely-used space-warping methods are only designed for the
forward-facing trajectory or the 360-degree object-centric trajectory but
cannot process arbitrary trajectories. In this paper, we delve deep into the
mechanism of space warping to handle unbounded scenes. Based on our analysis,
we further propose a novel space-warping method called perspective warping,
which allows us to handle arbitrary trajectories in the grid-based NeRF
framework. Extensive experiments demonstrate that F2-NeRF is able to use the
same perspective warping to render high-quality images on two standard datasets
and a new free trajectory dataset collected by us. Project page:
https://totoro97.github.io/projects/f2-nerf.

Comments:
- CVPR 2023. Project page: https://totoro97.github.io/projects/f2-nerf

---

## VMesh: Hybrid Volume-Mesh Representation for Efficient View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-28 | Yuan-Chen Guo, Yan-Pei Cao, Chen Wang, Yu He, Ying Shan, Xiaohu Qie, Song-Hai Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2303.16184v1){: .btn .btn-green } |

**Abstract**: With the emergence of neural radiance fields (NeRFs), view synthesis quality
has reached an unprecedented level. Compared to traditional mesh-based assets,
this volumetric representation is more powerful in expressing scene geometry
but inevitably suffers from high rendering costs and can hardly be involved in
further processes like editing, posing significant difficulties in combination
with the existing graphics pipeline. In this paper, we present a hybrid
volume-mesh representation, VMesh, which depicts an object with a textured mesh
along with an auxiliary sparse volume. VMesh retains the advantages of
mesh-based assets, such as efficient rendering, compact storage, and easy
editing, while also incorporating the ability to represent subtle geometric
structures provided by the volumetric counterpart. VMesh can be obtained from
multi-view images of an object and renders at 2K 60FPS on common consumer
devices with high fidelity, unleashing new opportunities for real-time
immersive applications.

Comments:
- Project page: https://bennyguo.github.io/vmesh/

---

## JAWS: Just A Wild Shot for Cinematic Transfer in Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-27 | Xi Wang, Robin Courant, Jinglei Shi, Eric Marchand, Marc Christie | cs.CV | [PDF](http://arxiv.org/pdf/2303.15427v1){: .btn .btn-green } |

**Abstract**: This paper presents JAWS, an optimization-driven approach that achieves the
robust transfer of visual cinematic features from a reference in-the-wild video
clip to a newly generated clip. To this end, we rely on an
implicit-neural-representation (INR) in a way to compute a clip that shares the
same cinematic features as the reference clip. We propose a general formulation
of a camera optimization problem in an INR that computes extrinsic and
intrinsic camera parameters as well as timing. By leveraging the
differentiability of neural representations, we can back-propagate our designed
cinematic losses measured on proxy estimators through a NeRF network to the
proposed cinematic parameters directly. We also introduce specific enhancements
such as guidance maps to improve the overall quality and efficiency. Results
display the capacity of our system to replicate well known camera sequences
from movies, adapting the framing, camera parameters and timing of the
generated video clip to maximize the similarity with the reference clip.

Comments:
- CVPR 2023. Project page with videos and code:
  http://www.lix.polytechnique.fr/vista/projects/2023_cvpr_wang

---

## Generalizable Neural Voxels for Fast Human Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-27 | Taoran Yi, Jiemin Fang, Xinggang Wang, Wenyu Liu | cs.CV | [PDF](http://arxiv.org/pdf/2303.15387v1){: .btn .btn-green } |

**Abstract**: Rendering moving human bodies at free viewpoints only from a monocular video
is quite a challenging problem. The information is too sparse to model
complicated human body structures and motions from both view and pose
dimensions. Neural radiance fields (NeRF) have shown great power in novel view
synthesis and have been applied to human body rendering. However, most current
NeRF-based methods bear huge costs for both training and rendering, which
impedes the wide applications in real-life scenarios. In this paper, we propose
a rendering framework that can learn moving human body structures extremely
quickly from a monocular video. The framework is built by integrating both
neural fields and neural voxels. Especially, a set of generalizable neural
voxels are constructed. With pretrained on various human bodies, these general
voxels represent a basic skeleton and can provide strong geometric priors. For
the fine-tuning process, individual voxels are constructed for learning
differential textures, complementary to general voxels. Thus learning a novel
body can be further accelerated, taking only a few minutes. Our method shows
significantly higher training efficiency compared with previous methods, while
maintaining similar rendering quality. The project page is at
https://taoranyi.com/gneuvox .

Comments:
- Project page: http://taoranyi.com/gneuvox

---

## NeUDF: Learning Unsigned Distance Fields from Multi-view Images for  Reconstructing Non-watertight Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-27 | Fei Hou, Jukai Deng, Xuhui Chen, Wencheng Wang, Ying He | cs.CV | [PDF](http://arxiv.org/pdf/2303.15368v1){: .btn .btn-green } |

**Abstract**: Volume rendering-based 3D reconstruction from multi-view images has gained
popularity in recent years, largely due to the success of neural radiance
fields (NeRF). A number of methods have been developed that build upon NeRF and
use neural volume rendering to learn signed distance fields (SDFs) for
reconstructing 3D models. However, SDF-based methods cannot represent
non-watertight models and, therefore, cannot capture open boundaries. This
paper proposes a new algorithm for learning an accurate unsigned distance field
(UDF) from multi-view images, which is specifically designed for reconstructing
non-watertight, textureless models. The proposed method, called NeUDF,
addresses the limitations of existing UDF-based methods by introducing a simple
and approximately unbiased and occlusion-aware density function. In addition, a
smooth and differentiable UDF representation is presented to make the learning
process easier and more efficient. Experiments on both texture-rich and
textureless models demonstrate the robustness and effectiveness of the proposed
approach, making it a promising solution for reconstructing challenging 3D
models from multi-view images.

---

## 3D-Aware Multi-Class Image-to-Image Translation with NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-27 | Senmao Li, Joost van de Weijer, Yaxing Wang, Fahad Shahbaz Khan, Meiqin Liu, Jian Yang | cs.CV | [PDF](http://arxiv.org/pdf/2303.15012v1){: .btn .btn-green } |

**Abstract**: Recent advances in 3D-aware generative models (3D-aware GANs) combined with
Neural Radiance Fields (NeRF) have achieved impressive results. However no
prior works investigate 3D-aware GANs for 3D consistent multi-class
image-to-image (3D-aware I2I) translation. Naively using 2D-I2I translation
methods suffers from unrealistic shape/identity change. To perform 3D-aware
multi-class I2I translation, we decouple this learning process into a
multi-class 3D-aware GAN step and a 3D-aware I2I translation step. In the first
step, we propose two novel techniques: a new conditional architecture and an
effective training strategy. In the second step, based on the well-trained
multi-class 3D-aware GAN architecture, that preserves view-consistency, we
construct a 3D-aware I2I translation system. To further reduce the
view-consistency problems, we propose several new techniques, including a
U-net-like adaptor network design, a hierarchical representation constrain and
a relative regularization loss. In extensive experiments on two datasets,
quantitative and qualitative results demonstrate that we successfully perform
3D-aware I2I translation with multi-view consistency.

Comments:
- Accepted by CVPR2023

---

## Clean-NeRF: Reformulating NeRF to account for View-Dependent  Observations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-26 | Xinhang Liu, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2303.14707v1){: .btn .btn-green } |

**Abstract**: While Neural Radiance Fields (NeRFs) had achieved unprecedented novel view
synthesis results, they have been struggling in dealing with large-scale
cluttered scenes with sparse input views and highly view-dependent appearances.
Specifically, existing NeRF-based models tend to produce blurry rendering with
the volumetric reconstruction often inaccurate, where a lot of reconstruction
errors are observed in the form of foggy "floaters" hovering within the entire
volume of an opaque 3D scene. Such inaccuracies impede NeRF's potential for
accurate 3D NeRF registration, object detection, segmentation, etc., which
possibly accounts for only limited significant research effort so far to
directly address these important 3D fundamental computer vision problems to
date. This paper analyzes the NeRF's struggles in such settings and proposes
Clean-NeRF for accurate 3D reconstruction and novel view rendering in complex
scenes. Our key insights consist of enforcing effective appearance and geometry
constraints, which are absent in the conventional NeRF reconstruction, by 1)
automatically detecting and modeling view-dependent appearances in the training
views to prevent them from interfering with density estimation, which is
complete with 2) a geometric correction procedure performed on each traced ray
during inference. Clean-NeRF can be implemented as a plug-in that can
immediately benefit existing NeRF-based methods without additional input. Codes
will be released.

---

## DBARF: Deep Bundle-Adjusting Generalizable Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-25 | Yu Chen, Gim Hee Lee | cs.CV | [PDF](http://arxiv.org/pdf/2303.14478v1){: .btn .btn-green } |

**Abstract**: Recent works such as BARF and GARF can bundle adjust camera poses with neural
radiance fields (NeRF) which is based on coordinate-MLPs. Despite the
impressive results, these methods cannot be applied to Generalizable NeRFs
(GeNeRFs) which require image feature extractions that are often based on more
complicated 3D CNN or transformer architectures. In this work, we first analyze
the difficulties of jointly optimizing camera poses with GeNeRFs, and then
further propose our DBARF to tackle these issues. Our DBARF which bundle
adjusts camera poses by taking a cost feature map as an implicit cost function
can be jointly trained with GeNeRFs in a self-supervised manner. Unlike BARF
and its follow-up works, which can only be applied to per-scene optimized NeRFs
and need accurate initial camera poses with the exception of forward-facing
scenes, our method can generalize across scenes and does not require any good
initialization. Experiments show the effectiveness and generalization ability
of our DBARF when evaluated on real-world datasets. Our code is available at
\url{https://aibluefisher.github.io/dbarf}.

---

## NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-25 | Zhiwen Yan, Chen Li, Gim Hee Lee | cs.CV | [PDF](http://arxiv.org/pdf/2303.14435v1){: .btn .btn-green } |

**Abstract**: Dynamic Neural Radiance Field (NeRF) is a powerful algorithm capable of
rendering photo-realistic novel view images from a monocular RGB video of a
dynamic scene. Although it warps moving points across frames from the
observation spaces to a common canonical space for rendering, dynamic NeRF does
not model the change of the reflected color during the warping. As a result,
this approach often fails drastically on challenging specular objects in
motion. We address this limitation by reformulating the neural radiance field
function to be conditioned on surface position and orientation in the
observation space. This allows the specular surface at different poses to keep
the different reflected colors when mapped to the common canonical space.
Additionally, we add the mask of moving objects to guide the deformation field.
As the specular surface changes color during motion, the mask mitigates the
problem of failure to find temporal correspondences with only RGB supervision.
We evaluate our model based on the novel view synthesis quality with a
self-collected dataset of different moving specular objects in realistic
environments. The experimental results demonstrate that our method
significantly improves the reconstruction quality of moving specular objects
from monocular RGB videos compared to the existing NeRF models. Our code and
data are available at the project website https://github.com/JokerYan/NeRF-DS.

Comments:
- CVPR 2023

---

## SUDS: Scalable Urban Dynamic Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-25 | Haithem Turki, Jason Y. Zhang, Francesco Ferroni, Deva Ramanan | cs.CV | [PDF](http://arxiv.org/pdf/2303.14536v1){: .btn .btn-green } |

**Abstract**: We extend neural radiance fields (NeRFs) to dynamic large-scale urban scenes.
Prior work tends to reconstruct single video clips of short durations (up to 10
seconds). Two reasons are that such methods (a) tend to scale linearly with the
number of moving objects and input videos because a separate model is built for
each and (b) tend to require supervision via 3D bounding boxes and panoptic
labels, obtained manually or via category-specific models. As a step towards
truly open-world reconstructions of dynamic cities, we introduce two key
innovations: (a) we factorize the scene into three separate hash table data
structures to efficiently encode static, dynamic, and far-field radiance
fields, and (b) we make use of unlabeled target signals consisting of RGB
images, sparse LiDAR, off-the-shelf self-supervised 2D descriptors, and most
importantly, 2D optical flow.
  Operationalizing such inputs via photometric, geometric, and feature-metric
reconstruction losses enables SUDS to decompose dynamic scenes into the static
background, individual objects, and their motions. When combined with our
multi-branch table representation, such reconstructions can be scaled to tens
of thousands of objects across 1.2 million frames from 1700 videos spanning
geospatial footprints of hundreds of kilometers, (to our knowledge) the largest
dynamic NeRF built to date.
  We present qualitative initial results on a variety of tasks enabled by our
representations, including novel-view synthesis of dynamic urban scenes,
unsupervised 3D instance segmentation, and unsupervised 3D cuboid detection. To
compare to prior work, we also evaluate on KITTI and Virtual KITTI 2,
surpassing state-of-the-art methods that rely on ground truth 3D bounding box
annotations while being 10x quicker to train.

Comments:
- CVPR 2023 Project page: https://haithemturki.com/suds/

---

## TEGLO: High Fidelity Canonical Texture Mapping from Single-View Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-24 | Vishal Vinod, Tanmay Shah, Dmitry Lagun | cs.CV | [PDF](http://arxiv.org/pdf/2303.13743v1){: .btn .btn-green } |

**Abstract**: Recent work in Neural Fields (NFs) learn 3D representations from
class-specific single view image collections. However, they are unable to
reconstruct the input data preserving high-frequency details. Further, these
methods do not disentangle appearance from geometry and hence are not suitable
for tasks such as texture transfer and editing. In this work, we propose TEGLO
(Textured EG3D-GLO) for learning 3D representations from single view
in-the-wild image collections for a given class of objects. We accomplish this
by training a conditional Neural Radiance Field (NeRF) without any explicit 3D
supervision. We equip our method with editing capabilities by creating a dense
correspondence mapping to a 2D canonical space. We demonstrate that such
mapping enables texture transfer and texture editing without requiring meshes
with shared topology. Our key insight is that by mapping the input image pixels
onto the texture space we can achieve near perfect reconstruction (>= 74 dB
PSNR at 1024^2 resolution). Our formulation allows for high quality 3D
consistent novel view synthesis with high-frequency details at megapixel image
resolution.

---

## Grid-guided Neural Radiance Fields for Large Urban Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-24 | Linning Xu, Yuanbo Xiangli, Sida Peng, Xingang Pan, Nanxuan Zhao, Christian Theobalt, Bo Dai, Dahua Lin | cs.CV | [PDF](http://arxiv.org/pdf/2303.14001v1){: .btn .btn-green } |

**Abstract**: Purely MLP-based neural radiance fields (NeRF-based methods) often suffer
from underfitting with blurred renderings on large-scale scenes due to limited
model capacity. Recent approaches propose to geographically divide the scene
and adopt multiple sub-NeRFs to model each region individually, leading to
linear scale-up in training costs and the number of sub-NeRFs as the scene
expands. An alternative solution is to use a feature grid representation, which
is computationally efficient and can naturally scale to a large scene with
increased grid resolutions. However, the feature grid tends to be less
constrained and often reaches suboptimal solutions, producing noisy artifacts
in renderings, especially in regions with complex geometry and texture. In this
work, we present a new framework that realizes high-fidelity rendering on large
urban scenes while being computationally efficient. We propose to use a compact
multiresolution ground feature plane representation to coarsely capture the
scene, and complement it with positional encoding inputs through another NeRF
branch for rendering in a joint learning fashion. We show that such an
integration can utilize the advantages of two alternative solutions: a
light-weighted NeRF is sufficient, under the guidance of the feature grid
representation, to render photorealistic novel views with fine details; and the
jointly optimized ground feature planes, can meanwhile gain further
refinements, forming a more accurate and compact feature space and output much
more natural rendering results.

Comments:
- CVPR2023, Project page at https://city-super.github.io/gridnerf/

---

## Perceptual Quality Assessment of NeRF and Neural View Synthesis Methods  for Front-Facing Views

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-24 | Hanxue Liang, Tianhao Wu, Param Hanji, Francesco Banterle, Hongyun Gao, Rafal Mantiuk, Cengiz Oztireli | cs.CV | [PDF](http://arxiv.org/pdf/2303.15206v3){: .btn .btn-green } |

**Abstract**: Neural view synthesis (NVS) is one of the most successful techniques for
synthesizing free viewpoint videos, capable of achieving high fidelity from
only a sparse set of captured images. This success has led to many variants of
the techniques, each evaluated on a set of test views typically using image
quality metrics such as PSNR, SSIM, or LPIPS. There has been a lack of research
on how NVS methods perform with respect to perceived video quality. We present
the first study on perceptual evaluation of NVS and NeRF variants. For this
study, we collected two datasets of scenes captured in a controlled lab
environment as well as in-the-wild. In contrast to existing datasets, these
scenes come with reference video sequences, allowing us to test for temporal
artifacts and subtle distortions that are easily overlooked when viewing only
static images. We measured the quality of videos synthesized by several NVS
methods in a well-controlled perceptual quality assessment experiment as well
as with many existing state-of-the-art image/video quality metrics. We present
a detailed analysis of the results and recommendations for dataset and metric
selection for NVS evaluation.

---

## CompoNeRF: Text-guided Multi-object Compositional NeRF with Editable 3D  Scene Layout

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-24 | Haotian Bai, Yuanhuiyi Lyu, Lutao Jiang, Sijia Li, Haonan Lu, Xiaodong Lin, Lin Wang | cs.CV | [PDF](http://arxiv.org/pdf/2303.13843v3){: .btn .btn-green } |

**Abstract**: Recent advances have shown promise in merging neural radiance fields (NeRFs)
with pre-trained diffusion models for text-to-3D object generation. However,
one enduring challenge is their inadequate capability to accurately parse and
regenerate consistent multi-object environments. Specifically, these models
encounter difficulties in accurately representing quantity and style prompted
by multi-object texts, often resulting in a collapse of the rendering fidelity
that fails to match the semantic intricacies. Moreover, amalgamating these
elements into a coherent 3D scene is a substantial challenge, stemming from
generic distribution inherent in diffusion models. To tackle the issue of
'guidance collapse' and enhance consistency, we propose a novel framework,
dubbed CompoNeRF, by integrating an editable 3D scene layout with object
specific and scene-wide guidance mechanisms. It initiates by interpreting a
complex text into an editable 3D layout populated with multiple NeRFs, each
paired with a corresponding subtext prompt for precise object depiction. Next,
a tailored composition module seamlessly blends these NeRFs, promoting
consistency, while the dual-level text guidance reduces ambiguity and boosts
accuracy. Noticeably, the unique modularity of CompoNeRF permits NeRF
decomposition. This enables flexible scene editing and recomposition into new
scenes based on the edited layout or text prompts. Utilizing the open source
Stable Diffusion model, CompoNeRF not only generates scenes with high fidelity
but also paves the way for innovative multi-object composition using editable
3D layouts. Remarkably, our framework achieves up to a 54\% improvement in
performance, as measured by the multi-view CLIP score metric. Code is available
at https://github.com/hbai98/Componerf.

---

## HandNeRF: Neural Radiance Fields for Animatable Interacting Hands

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-24 | Zhiyang Guo, Wengang Zhou, Min Wang, Li Li, Houqiang Li | cs.CV | [PDF](http://arxiv.org/pdf/2303.13825v1){: .btn .btn-green } |

**Abstract**: We propose a novel framework to reconstruct accurate appearance and geometry
with neural radiance fields (NeRF) for interacting hands, enabling the
rendering of photo-realistic images and videos for gesture animation from
arbitrary views. Given multi-view images of a single hand or interacting hands,
an off-the-shelf skeleton estimator is first employed to parameterize the hand
poses. Then we design a pose-driven deformation field to establish
correspondence from those different poses to a shared canonical space, where a
pose-disentangled NeRF for one hand is optimized. Such unified modeling
efficiently complements the geometry and texture cues in rarely-observed areas
for both hands. Meanwhile, we further leverage the pose priors to generate
pseudo depth maps as guidance for occlusion-aware density learning. Moreover, a
neural feature distillation method is proposed to achieve cross-domain
alignment for color optimization. We conduct extensive experiments to verify
the merits of our proposed HandNeRF and report a series of state-of-the-art
results both qualitatively and quantitatively on the large-scale InterHand2.6M
dataset.

Comments:
- CVPR 2023

---

## ABLE-NeRF: Attention-Based Rendering with Learnable Embeddings for  Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-24 | Zhe Jun Tang, Tat-Jen Cham, Haiyu Zhao | cs.CV | [PDF](http://arxiv.org/pdf/2303.13817v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) is a popular method in representing 3D scenes by
optimising a continuous volumetric scene function. Its large success which lies
in applying volumetric rendering (VR) is also its Achilles' heel in producing
view-dependent effects. As a consequence, glossy and transparent surfaces often
appear murky. A remedy to reduce these artefacts is to constrain this VR
equation by excluding volumes with back-facing normal. While this approach has
some success in rendering glossy surfaces, translucent objects are still poorly
represented. In this paper, we present an alternative to the physics-based VR
approach by introducing a self-attention-based framework on volumes along a
ray. In addition, inspired by modern game engines which utilise Light Probes to
store local lighting passing through the scene, we incorporate Learnable
Embeddings to capture view dependent effects within the scene. Our method,
which we call ABLE-NeRF, significantly reduces `blurry' glossy surfaces in
rendering and produces realistic translucent surfaces which lack in prior art.
In the Blender dataset, ABLE-NeRF achieves SOTA results and surpasses Ref-NeRF
in all 3 image quality metrics PSNR, SSIM, LPIPS.

Comments:
- IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)
  2023

---

## GM-NeRF: Learning Generalizable Model-based Neural Radiance Fields from  Multi-view Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-24 | Jianchuan Chen, Wentao Yi, Liqian Ma, Xu Jia, Huchuan Lu | cs.CV | [PDF](http://arxiv.org/pdf/2303.13777v1){: .btn .btn-green } |

**Abstract**: In this work, we focus on synthesizing high-fidelity novel view images for
arbitrary human performers, given a set of sparse multi-view images. It is a
challenging task due to the large variation among articulated body poses and
heavy self-occlusions. To alleviate this, we introduce an effective
generalizable framework Generalizable Model-based Neural Radiance Fields
(GM-NeRF) to synthesize free-viewpoint images. Specifically, we propose a
geometry-guided attention mechanism to register the appearance code from
multi-view 2D images to a geometry proxy which can alleviate the misalignment
between inaccurate geometry prior and pixel space. On top of that, we further
conduct neural rendering and partial gradient backpropagation for efficient
perceptual supervision and improvement of the perceptual quality of synthesis.
To evaluate our method, we conduct experiments on synthesized datasets
THuman2.0 and Multi-garment, and real-world datasets Genebody and ZJUMocap. The
results demonstrate that our approach outperforms state-of-the-art methods in
terms of novel view synthesis and geometric reconstruction.

Comments:
- Accepted at CVPR 2023

---

## Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion  Prior



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-24 | Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, Dong Chen | cs.CV | [PDF](http://arxiv.org/pdf/2303.14184v2){: .btn .btn-green } |

**Abstract**: In this work, we investigate the problem of creating high-fidelity 3D content
from only a single image. This is inherently challenging: it essentially
involves estimating the underlying 3D geometry while simultaneously
hallucinating unseen textures. To address this challenge, we leverage prior
knowledge from a well-trained 2D diffusion model to act as 3D-aware supervision
for 3D creation. Our approach, Make-It-3D, employs a two-stage optimization
pipeline: the first stage optimizes a neural radiance field by incorporating
constraints from the reference image at the frontal view and diffusion prior at
novel views; the second stage transforms the coarse model into textured point
clouds and further elevates the realism with diffusion prior while leveraging
the high-quality textures from the reference image. Extensive experiments
demonstrate that our method outperforms prior works by a large margin,
resulting in faithful reconstructions and impressive visual quality. Our method
presents the first attempt to achieve high-quality 3D creation from a single
image for general objects and enables various applications such as text-to-3D
creation and texture editing.

Comments:
- 17 pages, 18 figures, Project page: https://make-it-3d.github.io/

---

## Set-the-Scene: Global-Local Training for Generating Controllable NeRF  Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-23 | Dana Cohen-Bar, Elad Richardson, Gal Metzer, Raja Giryes, Daniel Cohen-Or | cs.CV | [PDF](http://arxiv.org/pdf/2303.13450v1){: .btn .btn-green } |

**Abstract**: Recent breakthroughs in text-guided image generation have led to remarkable
progress in the field of 3D synthesis from text. By optimizing neural radiance
fields (NeRF) directly from text, recent methods are able to produce remarkable
results. Yet, these methods are limited in their control of each object's
placement or appearance, as they represent the scene as a whole. This can be a
major issue in scenarios that require refining or manipulating objects in the
scene. To remedy this deficit, we propose a novel GlobalLocal training
framework for synthesizing a 3D scene using object proxies. A proxy represents
the object's placement in the generated scene and optionally defines its coarse
geometry. The key to our approach is to represent each object as an independent
NeRF. We alternate between optimizing each NeRF on its own and as part of the
full scene. Thus, a complete representation of each object can be learned,
while also creating a harmonious scene with style and lighting match. We show
that using proxies allows a wide variety of editing options, such as adjusting
the placement of each independent object, removing objects from a scene, or
refining an object. Our results show that Set-the-Scene offers a powerful
solution for scene synthesis and manipulation, filling a crucial gap in
controllable text-to-3D synthesis.

Comments:
- project page at https://danacohen95.github.io/Set-the-Scene/

---

## Transforming Radiance Field with Lipschitz Network for Photorealistic 3D  Scene Stylization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-23 | Zicheng Zhang, Yinglu Liu, Congying Han, Yingwei Pan, Tiande Guo, Ting Yao | cs.CV | [PDF](http://arxiv.org/pdf/2303.13232v1){: .btn .btn-green } |

**Abstract**: Recent advances in 3D scene representation and novel view synthesis have
witnessed the rise of Neural Radiance Fields (NeRFs). Nevertheless, it is not
trivial to exploit NeRF for the photorealistic 3D scene stylization task, which
aims to generate visually consistent and photorealistic stylized scenes from
novel views. Simply coupling NeRF with photorealistic style transfer (PST) will
result in cross-view inconsistency and degradation of stylized view syntheses.
Through a thorough analysis, we demonstrate that this non-trivial task can be
simplified in a new light: When transforming the appearance representation of a
pre-trained NeRF with Lipschitz mapping, the consistency and photorealism
across source views will be seamlessly encoded into the syntheses. That
motivates us to build a concise and flexible learning framework namely LipRF,
which upgrades arbitrary 2D PST methods with Lipschitz mapping tailored for the
3D scene. Technically, LipRF first pre-trains a radiance field to reconstruct
the 3D scene, and then emulates the style on each view by 2D PST as the prior
to learn a Lipschitz network to stylize the pre-trained appearance. In view of
that Lipschitz condition highly impacts the expressivity of the neural network,
we devise an adaptive regularization to balance the reconstruction and
stylization. A gradual gradient aggregation strategy is further introduced to
optimize LipRF in a cost-efficient manner. We conduct extensive experiments to
show the high quality and robust performance of LipRF on both photorealistic 3D
stylization and object appearance editing.

Comments:
- CVPR 2023, Highlight

---

## SINE: Semantic-driven Image-based NeRF Editing with Prior-guided Editing  Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-23 | Chong Bao, Yinda Zhang, Bangbang Yang, Tianxing Fan, Zesong Yang, Hujun Bao, Guofeng Zhang, Zhaopeng Cui | cs.CV | [PDF](http://arxiv.org/pdf/2303.13277v2){: .btn .btn-green } |

**Abstract**: Despite the great success in 2D editing using user-friendly tools, such as
Photoshop, semantic strokes, or even text prompts, similar capabilities in 3D
areas are still limited, either relying on 3D modeling skills or allowing
editing within only a few categories. In this paper, we present a novel
semantic-driven NeRF editing approach, which enables users to edit a neural
radiance field with a single image, and faithfully delivers edited novel views
with high fidelity and multi-view consistency. To achieve this goal, we propose
a prior-guided editing field to encode fine-grained geometric and texture
editing in 3D space, and develop a series of techniques to aid the editing
process, including cyclic constraints with a proxy mesh to facilitate geometric
supervision, a color compositing mechanism to stabilize semantic-driven texture
editing, and a feature-cluster-based regularization to preserve the irrelevant
content unchanged. Extensive experiments and editing examples on both
real-world and synthetic data demonstrate that our method achieves
photo-realistic 3D editing using only a single edited image, pushing the bound
of semantic-driven editing in 3D real-world scenes. Our project webpage:
https://zju3dv.github.io/sine/.

Comments:
- Accepted to CVPR 2023. Project Page: https://zju3dv.github.io/sine/

---

## SCADE: NeRFs from Space Carving with Ambiguity-Aware Depth Estimates

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-23 | Mikaela Angelina Uy, Ricardo Martin-Brualla, Leonidas Guibas, Ke Li | cs.CV | [PDF](http://arxiv.org/pdf/2303.13582v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) have enabled high fidelity 3D reconstruction
from multiple 2D input views. However, a well-known drawback of NeRFs is the
less-than-ideal performance under a small number of views, due to insufficient
constraints enforced by volumetric rendering. To address this issue, we
introduce SCADE, a novel technique that improves NeRF reconstruction quality on
sparse, unconstrained input views for in-the-wild indoor scenes. To constrain
NeRF reconstruction, we leverage geometric priors in the form of per-view depth
estimates produced with state-of-the-art monocular depth estimation models,
which can generalize across scenes. A key challenge is that monocular depth
estimation is an ill-posed problem, with inherent ambiguities. To handle this
issue, we propose a new method that learns to predict, for each view, a
continuous, multimodal distribution of depth estimates using conditional
Implicit Maximum Likelihood Estimation (cIMLE). In order to disambiguate
exploiting multiple views, we introduce an original space carving loss that
guides the NeRF representation to fuse multiple hypothesized depth maps from
each view and distill from them a common geometry that is consistent with all
views. Experiments show that our approach enables higher fidelity novel view
synthesis from sparse views. Our project page can be found at
https://scade-spacecarving-nerfs.github.io .

Comments:
- CVPR 2023

---

## Plotting Behind the Scenes: Towards Learnable Game Engines

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-23 | Willi Menapace, Aliaksandr Siarohin, Stéphane Lathuilière, Panos Achlioptas, Vladislav Golyanik, Sergey Tulyakov, Elisa Ricci | cs.CV | [PDF](http://arxiv.org/pdf/2303.13472v2){: .btn .btn-green } |

**Abstract**: Neural video game simulators emerged as powerful tools to generate and edit
videos. Their idea is to represent games as the evolution of an environment's
state driven by the actions of its agents. While such a paradigm enables users
to play a game action-by-action, its rigidity precludes more semantic forms of
control. To overcome this limitation, we augment game models with prompts
specified as a set of natural language actions and desired states. The result-a
Promptable Game Model (PGM)-makes it possible for a user to play the game by
prompting it with high- and low-level action sequences. Most captivatingly, our
PGM unlocks the director's mode, where the game is played by specifying goals
for the agents in the form of a prompt. This requires learning "game AI",
encapsulated by our animation model, to navigate the scene using high-level
constraints, play against an adversary, and devise a strategy to win a point.
To render the resulting state, we use a compositional NeRF representation
encapsulated in our synthesis model. To foster future research, we present
newly collected, annotated and calibrated Tennis and Minecraft datasets. Our
method significantly outperforms existing neural video game simulators in terms
of rendering quality and unlocks applications beyond the capabilities of the
current state of the art. Our framework, data, and models are available at
https://snap-research.github.io/promptable-game-models/.

Comments:
- ACM Transactions on Graphics \c{opyright} Copyright is held by the
  owner/author(s) 2023. This is the author's version of the work. It is posted
  here for your personal use. Not for redistribution. The definitive Version of
  Record was published in ACM Transactions on Graphics,
  http://dx.doi.org/10.1145/3635705

---

## TriPlaneNet: An Encoder for EG3D Inversion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-23 | Ananta R. Bhattarai, Matthias Nießner, Artem Sevastopolsky | cs.CV | [PDF](http://arxiv.org/pdf/2303.13497v2){: .btn .btn-green } |

**Abstract**: Recent progress in NeRF-based GANs has introduced a number of approaches for
high-resolution and high-fidelity generative modeling of human heads with a
possibility for novel view rendering. At the same time, one must solve an
inverse problem to be able to re-render or modify an existing image or video.
Despite the success of universal optimization-based methods for 2D GAN
inversion, those applied to 3D GANs may fail to extrapolate the result onto the
novel view, whereas optimization-based 3D GAN inversion methods are
time-consuming and can require at least several minutes per image. Fast
encoder-based techniques, such as those developed for StyleGAN, may also be
less appealing due to the lack of identity preservation. Our work introduces a
fast technique that bridges the gap between the two approaches by directly
utilizing the tri-plane representation presented for the EG3D generative model.
In particular, we build upon a feed-forward convolutional encoder for the
latent code and extend it with a fully-convolutional predictor of tri-plane
numerical offsets. The renderings are similar in quality to the ones produced
by optimization-based techniques and outperform the ones by encoder-based
methods. As we empirically prove, this is a consequence of directly operating
in the tri-plane space, not in the GAN parameter space, while making use of an
encoder-based trainable approach. Finally, we demonstrate significantly more
correct embedding of a face image in 3D than for all the baselines, further
strengthened by a probably symmetric prior enabled during training.

Comments:
- Project page: https://anantarb.github.io/triplanenet

---

## DreamBooth3D: Subject-Driven Text-to-3D Generation



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-23 | Amit Raj, Srinivas Kaza, Ben Poole, Michael Niemeyer, Nataniel Ruiz, Ben Mildenhall, Shiran Zada, Kfir Aberman, Michael Rubinstein, Jonathan Barron, Yuanzhen Li, Varun Jampani | cs.CV | [PDF](http://arxiv.org/pdf/2303.13508v2){: .btn .btn-green } |

**Abstract**: We present DreamBooth3D, an approach to personalize text-to-3D generative
models from as few as 3-6 casually captured images of a subject. Our approach
combines recent advances in personalizing text-to-image models (DreamBooth)
with text-to-3D generation (DreamFusion). We find that naively combining these
methods fails to yield satisfactory subject-specific 3D assets due to
personalized text-to-image models overfitting to the input viewpoints of the
subject. We overcome this through a 3-stage optimization strategy where we
jointly leverage the 3D consistency of neural radiance fields together with the
personalization capability of text-to-image models. Our method can produce
high-quality, subject-specific 3D assets with text-driven modifications such as
novel poses, colors and attributes that are not seen in any of the input images
of the subject.

Comments:
- Project page at https://dreambooth3d.github.io/ Video Summary at
  https://youtu.be/kKVDrbfvOoA

---

## Semantic Ray: Learning a Generalizable Semantic Field with  Cross-Reprojection Attention

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-23 | Fangfu Liu, Chubin Zhang, Yu Zheng, Yueqi Duan | cs.CV | [PDF](http://arxiv.org/pdf/2303.13014v1){: .btn .btn-green } |

**Abstract**: In this paper, we aim to learn a semantic radiance field from multiple scenes
that is accurate, efficient and generalizable. While most existing NeRFs target
at the tasks of neural scene rendering, image synthesis and multi-view
reconstruction, there are a few attempts such as Semantic-NeRF that explore to
learn high-level semantic understanding with the NeRF structure. However,
Semantic-NeRF simultaneously learns color and semantic label from a single ray
with multiple heads, where the single ray fails to provide rich semantic
information. As a result, Semantic NeRF relies on positional encoding and needs
to train one specific model for each scene. To address this, we propose
Semantic Ray (S-Ray) to fully exploit semantic information along the ray
direction from its multi-view reprojections. As directly performing dense
attention over multi-view reprojected rays would suffer from heavy
computational cost, we design a Cross-Reprojection Attention module with
consecutive intra-view radial and cross-view sparse attentions, which
decomposes contextual information along reprojected rays and cross multiple
views and then collects dense connections by stacking the modules. Experiments
show that our S-Ray is able to learn from multiple scenes, and it presents
strong generalization ability to adapt to unseen scenes.

Comments:
- Accepted by CVPR 2023. Project page: https://liuff19.github.io/S-Ray/

---

## Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-22 | Ayaan Haque, Matthew Tancik, Alexei A. Efros, Aleksander Holynski, Angjoo Kanazawa | cs.CV | [PDF](http://arxiv.org/pdf/2303.12789v2){: .btn .btn-green } |

**Abstract**: We propose a method for editing NeRF scenes with text-instructions. Given a
NeRF of a scene and the collection of images used to reconstruct it, our method
uses an image-conditioned diffusion model (InstructPix2Pix) to iteratively edit
the input images while optimizing the underlying scene, resulting in an
optimized 3D scene that respects the edit instruction. We demonstrate that our
proposed method is able to edit large-scale, real-world scenes, and is able to
accomplish more realistic, targeted edits than prior work.

Comments:
- Project website: https://instruct-nerf2nerf.github.io; v1. Revisions
  to related work and discussion

---

## FeatureNeRF: Learning Generalizable NeRFs by Distilling Foundation  Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-22 | Jianglong Ye, Naiyan Wang, Xiaolong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2303.12786v1){: .btn .btn-green } |

**Abstract**: Recent works on generalizable NeRFs have shown promising results on novel
view synthesis from single or few images. However, such models have rarely been
applied on other downstream tasks beyond synthesis such as semantic
understanding and parsing. In this paper, we propose a novel framework named
FeatureNeRF to learn generalizable NeRFs by distilling pre-trained vision
foundation models (e.g., DINO, Latent Diffusion). FeatureNeRF leverages 2D
pre-trained foundation models to 3D space via neural rendering, and then
extract deep features for 3D query points from NeRF MLPs. Consequently, it
allows to map 2D images to continuous 3D semantic feature volumes, which can be
used for various downstream tasks. We evaluate FeatureNeRF on tasks of 2D/3D
semantic keypoint transfer and 2D/3D object part segmentation. Our extensive
experiments demonstrate the effectiveness of FeatureNeRF as a generalizable 3D
semantic feature extractor. Our project page is available at
https://jianglongye.com/featurenerf/ .

Comments:
- Project page: https://jianglongye.com/featurenerf/

---

## NLOS-NeuS: Non-line-of-sight Neural Implicit Surface



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-22 | Yuki Fujimura, Takahiro Kushida, Takuya Funatomi, Yasuhiro Mukaigawa | cs.CV | [PDF](http://arxiv.org/pdf/2303.12280v2){: .btn .btn-green } |

**Abstract**: Non-line-of-sight (NLOS) imaging is conducted to infer invisible scenes from
indirect light on visible objects. The neural transient field (NeTF) was
proposed for representing scenes as neural radiance fields in NLOS scenes. We
propose NLOS neural implicit surface (NLOS-NeuS), which extends the NeTF to
neural implicit surfaces with a signed distance function (SDF) for
reconstructing three-dimensional surfaces in NLOS scenes. We introduce two
constraints as loss functions for correctly learning an SDF to avoid non-zero
level-set surfaces. We also introduce a lower bound constraint of an SDF based
on the geometry of the first-returning photons. The experimental results
indicate that these constraints are essential for learning a correct SDF in
NLOS scenes. Compared with previous methods with discretized representation,
NLOS-NeuS with the neural continuous representation enables us to reconstruct
smooth surfaces while preserving fine details in NLOS scenes. To the best of
our knowledge, this is the first study on neural implicit surfaces with volume
rendering in NLOS scenes.

Comments:
- ICCV 2023

---

## SHERF: Generalizable Human NeRF from a Single Image

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-22 | Shoukang Hu, Fangzhou Hong, Liang Pan, Haiyi Mei, Lei Yang, Ziwei Liu | cs.CV | [PDF](http://arxiv.org/pdf/2303.12791v2){: .btn .btn-green } |

**Abstract**: Existing Human NeRF methods for reconstructing 3D humans typically rely on
multiple 2D images from multi-view cameras or monocular videos captured from
fixed camera views. However, in real-world scenarios, human images are often
captured from random camera angles, presenting challenges for high-quality 3D
human reconstruction. In this paper, we propose SHERF, the first generalizable
Human NeRF model for recovering animatable 3D humans from a single input image.
SHERF extracts and encodes 3D human representations in canonical space,
enabling rendering and animation from free views and poses. To achieve
high-fidelity novel view and pose synthesis, the encoded 3D human
representations should capture both global appearance and local fine-grained
textures. To this end, we propose a bank of 3D-aware hierarchical features,
including global, point-level, and pixel-aligned features, to facilitate
informative encoding. Global features enhance the information extracted from
the single input image and complement the information missing from the partial
2D observation. Point-level features provide strong clues of 3D human
structure, while pixel-aligned features preserve more fine-grained details. To
effectively integrate the 3D-aware hierarchical feature bank, we design a
feature fusion transformer. Extensive experiments on THuman, RenderPeople,
ZJU_MoCap, and HuMMan datasets demonstrate that SHERF achieves state-of-the-art
performance, with better generalizability for novel view and pose synthesis.

Comments:
- Accepted by ICCV2023. Project webpage:
  https://skhu101.github.io/SHERF/

---

## NeRF-GAN Distillation for Efficient 3D-Aware Generation with  Convolutions

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-22 | Mohamad Shahbazi, Evangelos Ntavelis, Alessio Tonioni, Edo Collins, Danda Pani Paudel, Martin Danelljan, Luc Van Gool | cs.CV | [PDF](http://arxiv.org/pdf/2303.12865v3){: .btn .btn-green } |

**Abstract**: Pose-conditioned convolutional generative models struggle with high-quality
3D-consistent image generation from single-view datasets, due to their lack of
sufficient 3D priors. Recently, the integration of Neural Radiance Fields
(NeRFs) and generative models, such as Generative Adversarial Networks (GANs),
has transformed 3D-aware generation from single-view images. NeRF-GANs exploit
the strong inductive bias of neural 3D representations and volumetric rendering
at the cost of higher computational complexity. This study aims at revisiting
pose-conditioned 2D GANs for efficient 3D-aware generation at inference time by
distilling 3D knowledge from pretrained NeRF-GANs. We propose a simple and
effective method, based on re-using the well-disentangled latent space of a
pre-trained NeRF-GAN in a pose-conditioned convolutional network to directly
generate 3D-consistent images corresponding to the underlying 3D
representations. Experiments on several datasets demonstrate that the proposed
method obtains results comparable with volumetric rendering in terms of quality
and 3D consistency while benefiting from the computational advantage of
convolutional networks. The code will be available at:
https://github.com/mshahbazi72/NeRF-GAN-Distillation

---

## Balanced Spherical Grid for Egocentric View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-22 | Changwoon Choi, Sang Min Kim, Young Min Kim | cs.CV | [PDF](http://arxiv.org/pdf/2303.12408v2){: .btn .btn-green } |

**Abstract**: We present EgoNeRF, a practical solution to reconstruct large-scale
real-world environments for VR assets. Given a few seconds of casually captured
360 video, EgoNeRF can efficiently build neural radiance fields which enable
high-quality rendering from novel viewpoints. Motivated by the recent
acceleration of NeRF using feature grids, we adopt spherical coordinate instead
of conventional Cartesian coordinate. Cartesian feature grid is inefficient to
represent large-scale unbounded scenes because it has a spatially uniform
resolution, regardless of distance from viewers. The spherical parameterization
better aligns with the rays of egocentric images, and yet enables factorization
for performance enhancement. However, the na\"ive spherical grid suffers from
irregularities at two poles, and also cannot represent unbounded scenes. To
avoid singularities near poles, we combine two balanced grids, which results in
a quasi-uniform angular grid. We also partition the radial grid exponentially
and place an environment map at infinity to represent unbounded scenes.
Furthermore, with our resampling technique for grid-based methods, we can
increase the number of valid samples to train NeRF volume. We extensively
evaluate our method in our newly introduced synthetic and real-world egocentric
360 video datasets, and it consistently achieves state-of-the-art performance.

Comments:
- Accepted to CVPR 2023

---

## Pre-NeRF 360: Enriching Unbounded Appearances for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-21 | Ahmad AlMughrabi, Umair Haroon, Ricardo Marques, Petia Radeva | cs.CV | [PDF](http://arxiv.org/pdf/2303.12234v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) appeared recently as a powerful tool to
generate realistic views of objects and confined areas. Still, they face
serious challenges with open scenes, where the camera has unrestricted movement
and content can appear at any distance. In such scenarios, current
NeRF-inspired models frequently yield hazy or pixelated outputs, suffer slow
training times, and might display irregularities, because of the challenging
task of reconstructing an extensive scene from a limited number of images. We
propose a new framework to boost the performance of NeRF-based architectures
yielding significantly superior outcomes compared to the prior work. Our
solution overcomes several obstacles that plagued earlier versions of NeRF,
including handling multiple video inputs, selecting keyframes, and extracting
poses from real-world frames that are ambiguous and symmetrical. Furthermore,
we applied our framework, dubbed as "Pre-NeRF 360", to enable the use of the
Nutrition5k dataset in NeRF and introduce an updated version of this dataset,
known as the N5k360 dataset.

---

## Few-shot Neural Radiance Fields Under Unconstrained Illumination

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-21 | SeokYeong Lee, JunYong Choi, Seungryong Kim, Ig-Jae Kim, Junghyun Cho | cs.CV | [PDF](http://arxiv.org/pdf/2303.11728v3){: .btn .btn-green } |

**Abstract**: In this paper, we introduce a new challenge for synthesizing novel view
images in practical environments with limited input multi-view images and
varying lighting conditions. Neural radiance fields (NeRF), one of the
pioneering works for this task, demand an extensive set of multi-view images
taken under constrained illumination, which is often unattainable in real-world
settings. While some previous works have managed to synthesize novel views
given images with different illumination, their performance still relies on a
substantial number of input multi-view images. To address this problem, we
suggest ExtremeNeRF, which utilizes multi-view albedo consistency, supported by
geometric alignment. Specifically, we extract intrinsic image components that
should be illumination-invariant across different views, enabling direct
appearance comparison between the input and novel view under unconstrained
illumination. We offer thorough experimental results for task evaluation,
employing the newly created NeRF Extreme benchmark-the first in-the-wild
benchmark for novel view synthesis under multiple viewing directions and
varying illuminations.

Comments:
- Project Page: https://seokyeong94.github.io/ExtremeNeRF/

---

## Interactive Geometry Editing of Neural Radiance Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-21 | Shaoxu Li, Ye Pan | cs.CV | [PDF](http://arxiv.org/pdf/2303.11537v2){: .btn .btn-green } |

**Abstract**: In this paper, we propose a method that enables interactive geometry editing
for neural radiance fields manipulation. We use two proxy cages(inner cage and
outer cage) to edit a scene. The inner cage defines the operation target, and
the outer cage defines the adjustment space. Various operations apply to the
two cages. After cage selection, operations on the inner cage lead to the
desired transformation of the inner cage and adjustment of the outer cage.
Users can edit the scene with translation, rotation, scaling, or combinations.
The operations on the corners and edges of the cage are also supported. Our
method does not need any explicit 3D geometry representations. The interactive
geometry editing applies directly to the implicit neural radiance fields.
Extensive experimental results demonstrate the effectiveness of our approach.

---

## 3D-CLFusion: Fast Text-to-3D Rendering with Contrastive Latent Diffusion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-21 | Yu-Jhe Li, Tao Xu, Ji Hou, Bichen Wu, Xiaoliang Dai, Albert Pumarola, Peizhao Zhang, Peter Vajda, Kris Kitani | cs.CV | [PDF](http://arxiv.org/pdf/2303.11938v2){: .btn .btn-green } |

**Abstract**: We tackle the task of text-to-3D creation with pre-trained latent-based NeRFs
(NeRFs that generate 3D objects given input latent code). Recent works such as
DreamFusion and Magic3D have shown great success in generating 3D content using
NeRFs and text prompts, but the current approach of optimizing a NeRF for every
text prompt is 1) extremely time-consuming and 2) often leads to low-resolution
outputs. To address these challenges, we propose a novel method named
3D-CLFusion which leverages the pre-trained latent-based NeRFs and performs
fast 3D content creation in less than a minute. In particular, we introduce a
latent diffusion prior network for learning the w latent from the input CLIP
text/image embeddings. This pipeline allows us to produce the w latent without
further optimization during inference and the pre-trained NeRF is able to
perform multi-view high-resolution 3D synthesis based on the latent. We note
that the novelty of our model lies in that we introduce contrastive learning
during training the diffusion prior which enables the generation of the valid
view-invariant latent code. We demonstrate through experiments the
effectiveness of our proposed view-invariant diffusion process for fast
text-to-3D creation, e.g., 100 times faster than DreamFusion. We note that our
model is able to serve as the role of a plug-and-play tool for text-to-3D with
pre-trained NeRFs.

Comments:
- 15 pages

---

## DehazeNeRF: Multiple Image Haze Removal and 3D Shape Reconstruction  using Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-20 | Wei-Ting Chen, Wang Yifan, Sy-Yen Kuo, Gordon Wetzstein | cs.CV | [PDF](http://arxiv.org/pdf/2303.11364v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) have demonstrated state-of-the-art performance
for 3D computer vision tasks, including novel view synthesis and 3D shape
reconstruction. However, these methods fail in adverse weather conditions. To
address this challenge, we introduce DehazeNeRF as a framework that robustly
operates in hazy conditions. DehazeNeRF extends the volume rendering equation
by adding physically realistic terms that model atmospheric scattering. By
parameterizing these terms using suitable networks that match the physical
properties, we introduce effective inductive biases, which, together with the
proposed regularizations, allow DehazeNeRF to demonstrate successful multi-view
haze removal, novel view synthesis, and 3D shape reconstruction where existing
approaches fail.

Comments:
- including supplemental material; project page:
  https://www.computationalimaging.org/publications/dehazenerf

---

## ContraNeRF: Generalizable Neural Radiance Fields for Synthetic-to-real  Novel View Synthesis via Contrastive Learning

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-20 | Hao Yang, Lanqing Hong, Aoxue Li, Tianyang Hu, Zhenguo Li, Gim Hee Lee, Liwei Wang | cs.CV | [PDF](http://arxiv.org/pdf/2303.11052v3){: .btn .btn-green } |

**Abstract**: Although many recent works have investigated generalizable NeRF-based novel
view synthesis for unseen scenes, they seldom consider the synthetic-to-real
generalization, which is desired in many practical applications. In this work,
we first investigate the effects of synthetic data in synthetic-to-real novel
view synthesis and surprisingly observe that models trained with synthetic data
tend to produce sharper but less accurate volume densities. For pixels where
the volume densities are correct, fine-grained details will be obtained.
Otherwise, severe artifacts will be produced. To maintain the advantages of
using synthetic data while avoiding its negative effects, we propose to
introduce geometry-aware contrastive learning to learn multi-view consistent
features with geometric constraints. Meanwhile, we adopt cross-view attention
to further enhance the geometry perception of features by querying features
across input views. Experiments demonstrate that under the synthetic-to-real
setting, our method can render images with higher quality and better
fine-grained details, outperforming existing generalizable novel view synthesis
methods in terms of PSNR, SSIM, and LPIPS. When trained on real data, our
method also achieves state-of-the-art results.

---

## SKED: Sketch-guided Text-based 3D Editing

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-19 | Aryan Mikaeili, Or Perel, Mehdi Safaee, Daniel Cohen-Or, Ali Mahdavi-Amiri | cs.CV | [PDF](http://arxiv.org/pdf/2303.10735v4){: .btn .btn-green } |

**Abstract**: Text-to-image diffusion models are gradually introduced into computer
graphics, recently enabling the development of Text-to-3D pipelines in an open
domain. However, for interactive editing purposes, local manipulations of
content through a simplistic textual interface can be arduous. Incorporating
user guided sketches with Text-to-image pipelines offers users more intuitive
control. Still, as state-of-the-art Text-to-3D pipelines rely on optimizing
Neural Radiance Fields (NeRF) through gradients from arbitrary rendering views,
conditioning on sketches is not straightforward. In this paper, we present
SKED, a technique for editing 3D shapes represented by NeRFs. Our technique
utilizes as few as two guiding sketches from different views to alter an
existing neural field. The edited region respects the prompt semantics through
a pre-trained diffusion model. To ensure the generated output adheres to the
provided sketches, we propose novel loss functions to generate the desired
edits while preserving the density and radiance of the base instance. We
demonstrate the effectiveness of our proposed method through several
qualitative and quantitative experiments. https://sked-paper.github.io/

---

## NeRF-LOAM: Neural Implicit Representation for Large-Scale Incremental  LiDAR Odometry and Mapping

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-19 | Junyuan Deng, Xieyuanli Chen, Songpengcheng Xia, Zhen Sun, Guoqing Liu, Wenxian Yu, Ling Pei | cs.CV | [PDF](http://arxiv.org/pdf/2303.10709v1){: .btn .btn-green } |

**Abstract**: Simultaneously odometry and mapping using LiDAR data is an important task for
mobile systems to achieve full autonomy in large-scale environments. However,
most existing LiDAR-based methods prioritize tracking quality over
reconstruction quality. Although the recently developed neural radiance fields
(NeRF) have shown promising advances in implicit reconstruction for indoor
environments, the problem of simultaneous odometry and mapping for large-scale
scenarios using incremental LiDAR data remains unexplored. To bridge this gap,
in this paper, we propose a novel NeRF-based LiDAR odometry and mapping
approach, NeRF-LOAM, consisting of three modules neural odometry, neural
mapping, and mesh reconstruction. All these modules utilize our proposed neural
signed distance function, which separates LiDAR points into ground and
non-ground points to reduce Z-axis drift, optimizes odometry and voxel
embeddings concurrently, and in the end generates dense smooth mesh maps of the
environment. Moreover, this joint optimization allows our NeRF-LOAM to be
pre-trained free and exhibit strong generalization abilities when applied to
different environments. Extensive evaluations on three publicly available
datasets demonstrate that our approach achieves state-of-the-art odometry and
mapping performance, as well as a strong generalization in large-scale
environments utilizing LiDAR data. Furthermore, we perform multiple ablation
studies to validate the effectiveness of our network design. The implementation
of our approach will be made available at
https://github.com/JunyuanDeng/NeRF-LOAM.

---

## StyleRF: Zero-shot 3D Style Transfer of Neural Radiance Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-19 | Kunhao Liu, Fangneng Zhan, Yiwen Chen, Jiahui Zhang, Yingchen Yu, Abdulmotaleb El Saddik, Shijian Lu, Eric Xing | cs.CV | [PDF](http://arxiv.org/pdf/2303.10598v3){: .btn .btn-green } |

**Abstract**: 3D style transfer aims to render stylized novel views of a 3D scene with
multi-view consistency. However, most existing work suffers from a three-way
dilemma over accurate geometry reconstruction, high-quality stylization, and
being generalizable to arbitrary new styles. We propose StyleRF (Style Radiance
Fields), an innovative 3D style transfer technique that resolves the three-way
dilemma by performing style transformation within the feature space of a
radiance field. StyleRF employs an explicit grid of high-level features to
represent 3D scenes, with which high-fidelity geometry can be reliably restored
via volume rendering. In addition, it transforms the grid features according to
the reference style which directly leads to high-quality zero-shot style
transfer. StyleRF consists of two innovative designs. The first is
sampling-invariant content transformation that makes the transformation
invariant to the holistic statistics of the sampled 3D points and accordingly
ensures multi-view consistency. The second is deferred style transformation of
2D feature maps which is equivalent to the transformation of 3D points but
greatly reduces memory footprint without degrading multi-view consistency.
Extensive experiments show that StyleRF achieves superior 3D stylization
quality with precise geometry reconstruction and it can generalize to various
new styles in a zero-shot manner.

Comments:
- Accepted to CVPR 2023. Project website:
  https://kunhao-liu.github.io/StyleRF/

---

## 3D Data Augmentation for Driving Scenes on Camera

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-18 | Wenwen Tong, Jiangwei Xie, Tianyu Li, Hanming Deng, Xiangwei Geng, Ruoyi Zhou, Dingchen Yang, Bo Dai, Lewei Lu, Hongyang Li | cs.CV | [PDF](http://arxiv.org/pdf/2303.10340v1){: .btn .btn-green } |

**Abstract**: Driving scenes are extremely diverse and complicated that it is impossible to
collect all cases with human effort alone. While data augmentation is an
effective technique to enrich the training data, existing methods for camera
data in autonomous driving applications are confined to the 2D image plane,
which may not optimally increase data diversity in 3D real-world scenarios. To
this end, we propose a 3D data augmentation approach termed Drive-3DAug, aiming
at augmenting the driving scenes on camera in the 3D space. We first utilize
Neural Radiance Field (NeRF) to reconstruct the 3D models of background and
foreground objects. Then, augmented driving scenes can be obtained by placing
the 3D objects with adapted location and orientation at the pre-defined valid
region of backgrounds. As such, the training database could be effectively
scaled up. However, the 3D object modeling is constrained to the image quality
and the limited viewpoints. To overcome these problems, we modify the original
NeRF by introducing a geometric rectified loss and a symmetric-aware training
strategy. We evaluate our method for the camera-only monocular 3D detection
task on the Waymo and nuScences datasets. The proposed data augmentation
approach contributes to a gain of 1.7% and 1.4% in terms of detection accuracy,
on Waymo and nuScences respectively. Furthermore, the constructed 3D models
serve as digital driving assets and could be recycled for different detectors
or other 3D perception tasks.

---

## $α$Surf: Implicit Surface Reconstruction for Semi-Transparent and  Thin Objects with Decoupled Geometry and Opacity

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-17 | Tianhao Wu, Hanxue Liang, Fangcheng Zhong, Gernot Riegler, Shimon Vainer, Cengiz Oztireli | cs.CV | [PDF](http://arxiv.org/pdf/2303.10083v1){: .btn .btn-green } |

**Abstract**: Implicit surface representations such as the signed distance function (SDF)
have emerged as a promising approach for image-based surface reconstruction.
However, existing optimization methods assume solid surfaces and are therefore
unable to properly reconstruct semi-transparent surfaces and thin structures,
which also exhibit low opacity due to the blending effect with the background.
While neural radiance field (NeRF) based methods can model semi-transparency
and achieve photo-realistic quality in synthesized novel views, their
volumetric geometry representation tightly couples geometry and opacity, and
therefore cannot be easily converted into surfaces without introducing
artifacts. We present $\alpha$Surf, a novel surface representation with
decoupled geometry and opacity for the reconstruction of semi-transparent and
thin surfaces where the colors mix. Ray-surface intersections on our
representation can be found in closed-form via analytical solutions of cubic
polynomials, avoiding Monte-Carlo sampling and is fully differentiable by
construction. Our qualitative and quantitative evaluations show that our
approach can accurately reconstruct surfaces with semi-transparent and thin
parts with fewer artifacts, achieving better reconstruction quality than
state-of-the-art SDF and NeRF methods. Website: https://alphasurf.netlify.app/

---

## Single-view Neural Radiance Fields with Depth Teacher

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-17 | Yurui Chen, Chun Gu, Feihu Zhang, Li Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2303.09952v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have been proposed for photorealistic novel
view rendering. However, it requires many different views of one scene for
training. Moreover, it has poor generalizations to new scenes and requires
retraining or fine-tuning on each scene. In this paper, we develop a new NeRF
model for novel view synthesis using only a single image as input. We propose
to combine the (coarse) planar rendering and the (fine) volume rendering to
achieve higher rendering quality and better generalizations. We also design a
depth teacher net that predicts dense pseudo depth maps to supervise the joint
rendering mechanism and boost the learning of consistent 3D geometry. We
evaluate our method on three challenging datasets. It outperforms
state-of-the-art single-view NeRFs by achieving 5$\sim$20\% improvements in
PSNR and reducing 20$\sim$50\% of the errors in the depth rendering. It also
shows excellent generalization abilities to unseen data without the need to
fine-tune on each new scene.

---

## NeRFMeshing: Distilling Neural Radiance Fields into  Geometrically-Accurate 3D Meshes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-16 | Marie-Julie Rakotosaona, Fabian Manhardt, Diego Martin Arroyo, Michael Niemeyer, Abhijit Kundu, Federico Tombari | cs.CV | [PDF](http://arxiv.org/pdf/2303.09431v1){: .btn .btn-green } |

**Abstract**: With the introduction of Neural Radiance Fields (NeRFs), novel view synthesis
has recently made a big leap forward. At the core, NeRF proposes that each 3D
point can emit radiance, allowing to conduct view synthesis using
differentiable volumetric rendering. While neural radiance fields can
accurately represent 3D scenes for computing the image rendering, 3D meshes are
still the main scene representation supported by most computer graphics and
simulation pipelines, enabling tasks such as real time rendering and
physics-based simulations. Obtaining 3D meshes from neural radiance fields
still remains an open challenge since NeRFs are optimized for view synthesis,
not enforcing an accurate underlying geometry on the radiance field. We thus
propose a novel compact and flexible architecture that enables easy 3D surface
reconstruction from any NeRF-driven approach. Upon having trained the radiance
field, we distill the volumetric 3D representation into a Signed Surface
Approximation Network, allowing easy extraction of the 3D mesh and appearance.
Our final 3D mesh is physically accurate and can be rendered in real time on an
array of devices.

---

## NeRFtrinsic Four: An End-To-End Trainable NeRF Jointly Optimizing  Diverse Intrinsic and Extrinsic Camera Parameters

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-16 | Hannah Schieber, Fabian Deuser, Bernhard Egger, Norbert Oswald, Daniel Roth | cs.CV | [PDF](http://arxiv.org/pdf/2303.09412v4){: .btn .btn-green } |

**Abstract**: Novel view synthesis using neural radiance fields (NeRF) is the
state-of-the-art technique for generating high-quality images from novel
viewpoints. Existing methods require a priori knowledge about extrinsic and
intrinsic camera parameters. This limits their applicability to synthetic
scenes, or real-world scenarios with the necessity of a preprocessing step.
Current research on the joint optimization of camera parameters and NeRF
focuses on refining noisy extrinsic camera parameters and often relies on the
preprocessing of intrinsic camera parameters. Further approaches are limited to
cover only one single camera intrinsic. To address these limitations, we
propose a novel end-to-end trainable approach called NeRFtrinsic Four. We
utilize Gaussian Fourier features to estimate extrinsic camera parameters and
dynamically predict varying intrinsic camera parameters through the supervision
of the projection error. Our approach outperforms existing joint optimization
methods on LLFF and BLEFF. In addition to these existing datasets, we introduce
a new dataset called iFF with varying intrinsic camera parameters. NeRFtrinsic
Four is a step forward in joint optimization NeRF-based view synthesis and
enables more realistic and flexible rendering in real-world scenarios with
varying camera parameters.

---

## Reliable Image Dehazing by NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-16 | Zheyan Jin, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, Yueting Chen | cs.CV | [PDF](http://arxiv.org/pdf/2303.09153v1){: .btn .btn-green } |

**Abstract**: We present an image dehazing algorithm with high quality, wide application,
and no data training or prior needed. We analyze the defects of the original
dehazing model, and propose a new and reliable dehazing reconstruction and
dehazing model based on the combination of optical scattering model and
computer graphics lighting rendering model. Based on the new haze model and the
images obtained by the cameras, we can reconstruct the three-dimensional space,
accurately calculate the objects and haze in the space, and use the
transparency relationship of haze to perform accurate haze removal. To obtain a
3D simulation dataset we used the Unreal 5 computer graphics rendering engine.
In order to obtain real shot data in different scenes, we used fog generators,
array cameras, mobile phones, underwater cameras and drones to obtain haze
data. We use formula derivation, simulation data set and real shot data set
result experimental results to prove the feasibility of the new method.
Compared with various other methods, we are far ahead in terms of calculation
indicators (4 dB higher quality average scene), color remains more natural, and
the algorithm is more robust in different scenarios and best in the subjective
perception.

Comments:
- 12pages, 8figures

---

## LERF: Language Embedded Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-16 | Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, Matthew Tancik | cs.CV | [PDF](http://arxiv.org/pdf/2303.09553v1){: .btn .btn-green } |

**Abstract**: Humans describe the physical world using natural language to refer to
specific 3D locations based on a vast range of properties: visual appearance,
semantics, abstract associations, or actionable affordances. In this work we
propose Language Embedded Radiance Fields (LERFs), a method for grounding
language embeddings from off-the-shelf models like CLIP into NeRF, which enable
these types of open-ended language queries in 3D. LERF learns a dense,
multi-scale language field inside NeRF by volume rendering CLIP embeddings
along training rays, supervising these embeddings across training views to
provide multi-view consistency and smooth the underlying language field. After
optimization, LERF can extract 3D relevancy maps for a broad range of language
prompts interactively in real-time, which has potential use cases in robotics,
understanding vision-language models, and interacting with 3D scenes. LERF
enables pixel-aligned, zero-shot queries on the distilled 3D CLIP embeddings
without relying on region proposals or masks, supporting long-tail
open-vocabulary queries hierarchically across the volume. The project website
can be found at https://lerf.io .

Comments:
- Project website can be found at https://lerf.io

---

## PartNeRF: Generating Part-Aware Editable 3D Shapes without 3D  Supervision

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-16 | Konstantinos Tertikas, Despoina Paschalidou, Boxiao Pan, Jeong Joon Park, Mikaela Angelina Uy, Ioannis Emiris, Yannis Avrithis, Leonidas Guibas | cs.CV | [PDF](http://arxiv.org/pdf/2303.09554v3){: .btn .btn-green } |

**Abstract**: Impressive progress in generative models and implicit representations gave
rise to methods that can generate 3D shapes of high quality. However, being
able to locally control and edit shapes is another essential property that can
unlock several content creation applications. Local control can be achieved
with part-aware models, but existing methods require 3D supervision and cannot
produce textures. In this work, we devise PartNeRF, a novel part-aware
generative model for editable 3D shape synthesis that does not require any
explicit 3D supervision. Our model generates objects as a set of locally
defined NeRFs, augmented with an affine transformation. This enables several
editing operations such as applying transformations on parts, mixing parts from
different objects etc. To ensure distinct, manipulable parts we enforce a hard
assignment of rays to parts that makes sure that the color of each ray is only
determined by a single NeRF. As a result, altering one part does not affect the
appearance of the others. Evaluations on various ShapeNet categories
demonstrate the ability of our model to generate editable 3D objects of
improved fidelity, compared to previous part-based generative approaches that
require 3D supervision or models relying on NeRFs.

Comments:
- To appear in CVPR 2023, Project Page:
  https://ktertikas.github.io/part_nerf

---

## Mesh Strikes Back: Fast and Efficient Human Reconstruction from RGB  videos

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-15 | Rohit Jena, Pratik Chaudhari, James Gee, Ganesh Iyer, Siddharth Choudhary, Brandon M. Smith | cs.CV | [PDF](http://arxiv.org/pdf/2303.08808v1){: .btn .btn-green } |

**Abstract**: Human reconstruction and synthesis from monocular RGB videos is a challenging
problem due to clothing, occlusion, texture discontinuities and sharpness, and
framespecific pose changes. Many methods employ deferred rendering, NeRFs and
implicit methods to represent clothed humans, on the premise that mesh-based
representations cannot capture complex clothing and textures from RGB,
silhouettes, and keypoints alone. We provide a counter viewpoint to this
fundamental premise by optimizing a SMPL+D mesh and an efficient,
multi-resolution texture representation using only RGB images, binary
silhouettes and sparse 2D keypoints. Experimental results demonstrate that our
approach is more capable of capturing geometric details compared to visual
hull, mesh-based methods. We show competitive novel view synthesis and
improvements in novel pose synthesis compared to NeRF-based methods, which
introduce noticeable, unwanted artifacts. By restricting the solution space to
the SMPL+D model combined with differentiable rendering, we obtain dramatic
speedups in compute, training times (up to 24x) and inference times (up to
192x). Our method therefore can be used as is or as a fast initialization to
NeRF-based methods.

---

## Re-ReND: Real-time Rendering of NeRFs across Devices

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-15 | Sara Rojas, Jesus Zarzar, Juan Camilo Perez, Artsiom Sanakoyeu, Ali Thabet, Albert Pumarola, Bernard Ghanem | cs.CV | [PDF](http://arxiv.org/pdf/2303.08717v1){: .btn .btn-green } |

**Abstract**: This paper proposes a novel approach for rendering a pre-trained Neural
Radiance Field (NeRF) in real-time on resource-constrained devices. We
introduce Re-ReND, a method enabling Real-time Rendering of NeRFs across
Devices. Re-ReND is designed to achieve real-time performance by converting the
NeRF into a representation that can be efficiently processed by standard
graphics pipelines. The proposed method distills the NeRF by extracting the
learned density into a mesh, while the learned color information is factorized
into a set of matrices that represent the scene's light field. Factorization
implies the field is queried via inexpensive MLP-free matrix multiplications,
while using a light field allows rendering a pixel by querying the field a
single time-as opposed to hundreds of queries when employing a radiance field.
Since the proposed representation can be implemented using a fragment shader,
it can be directly integrated with standard rasterization frameworks. Our
flexible implementation can render a NeRF in real-time with low memory
requirements and on a wide range of resource-constrained devices, including
mobiles and AR/VR headsets. Notably, we find that Re-ReND can achieve over a
2.6-fold increase in rendering speed versus the state-of-the-art without
perceptible losses in quality.

---

## RefiNeRF: Modelling dynamic neural radiance fields with inconsistent or  missing camera parameters

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-15 | Shuja Khalid, Frank Rudzicz | cs.CV | [PDF](http://arxiv.org/pdf/2303.08695v1){: .btn .btn-green } |

**Abstract**: Novel view synthesis (NVS) is a challenging task in computer vision that
involves synthesizing new views of a scene from a limited set of input images.
Neural Radiance Fields (NeRF) have emerged as a powerful approach to address
this problem, but they require accurate knowledge of camera \textit{intrinsic}
and \textit{extrinsic} parameters. Traditionally, structure-from-motion (SfM)
and multi-view stereo (MVS) approaches have been used to extract camera
parameters, but these methods can be unreliable and may fail in certain cases.
In this paper, we propose a novel technique that leverages unposed images from
dynamic datasets, such as the NVIDIA dynamic scenes dataset, to learn camera
parameters directly from data. Our approach is highly extensible and can be
integrated into existing NeRF architectures with minimal modifications. We
demonstrate the effectiveness of our method on a variety of static and dynamic
scenes and show that it outperforms traditional SfM and MVS approaches. The
code for our method is publicly available at
\href{https://github.com/redacted/refinerf}{https://github.com/redacted/refinerf}.
Our approach offers a promising new direction for improving the accuracy and
robustness of NVS using NeRF, and we anticipate that it will be a valuable tool
for a wide range of applications in computer vision and graphics.

---

## Harnessing Low-Frequency Neural Fields for Few-Shot View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-15 | Liangchen Song, Zhong Li, Xuan Gong, Lele Chen, Zhang Chen, Yi Xu, Junsong Yuan | cs.CV | [PDF](http://arxiv.org/pdf/2303.08370v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have led to breakthroughs in the novel view
synthesis problem. Positional Encoding (P.E.) is a critical factor that brings
the impressive performance of NeRF, where low-dimensional coordinates are
mapped to high-dimensional space to better recover scene details. However,
blindly increasing the frequency of P.E. leads to overfitting when the
reconstruction problem is highly underconstrained, \eg, few-shot images for
training. We harness low-frequency neural fields to regularize high-frequency
neural fields from overfitting to better address the problem of few-shot view
synthesis. We propose reconstructing with a low-frequency only field and then
finishing details with a high-frequency equipped field. Unlike most existing
solutions that regularize the output space (\ie, rendered images), our
regularization is conducted in the input space (\ie, signal frequency). We
further propose a simple-yet-effective strategy for tuning the frequency to
avoid overfitting few-shot inputs: enforcing consistency among the frequency
domain of rendered 2D images. Thanks to the input space regularizing scheme,
our method readily applies to inputs beyond spatial locations, such as the time
dimension in dynamic scenes. Comparisons with state-of-the-art on both
synthetic and natural datasets validate the effectiveness of our proposed
solution for few-shot view synthesis. Code is available at
\href{https://github.com/lsongx/halo}{https://github.com/lsongx/halo}.

---

## I$^2$-SDF: Intrinsic Indoor Scene Reconstruction and Editing via  Raytracing in Neural SDFs



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-14 | Jingsen Zhu, Yuchi Huo, Qi Ye, Fujun Luan, Jifan Li, Dianbing Xi, Lisha Wang, Rui Tang, Wei Hua, Hujun Bao, Rui Wang | cs.CV | [PDF](http://arxiv.org/pdf/2303.07634v2){: .btn .btn-green } |

**Abstract**: In this work, we present I$^2$-SDF, a new method for intrinsic indoor scene
reconstruction and editing using differentiable Monte Carlo raytracing on
neural signed distance fields (SDFs). Our holistic neural SDF-based framework
jointly recovers the underlying shapes, incident radiance and materials from
multi-view images. We introduce a novel bubble loss for fine-grained small
objects and error-guided adaptive sampling scheme to largely improve the
reconstruction quality on large-scale indoor scenes. Further, we propose to
decompose the neural radiance field into spatially-varying material of the
scene as a neural field through surface-based, differentiable Monte Carlo
raytracing and emitter semantic segmentations, which enables physically based
and photorealistic scene relighting and editing applications. Through a number
of qualitative and quantitative experiments, we demonstrate the superior
quality of our method on indoor scene reconstruction, novel view synthesis, and
scene editing compared to state-of-the-art baselines.

Comments:
- Accepted by CVPR 2023, project page:
  https://jingsenzhu.github.io/i2-sdf

---

## Frequency-Modulated Point Cloud Rendering with Easy Editing

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-14 | Yi Zhang, Xiaoyang Huang, Bingbing Ni, Teng Li, Wenjun Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2303.07596v2){: .btn .btn-green } |

**Abstract**: We develop an effective point cloud rendering pipeline for novel view
synthesis, which enables high fidelity local detail reconstruction, real-time
rendering and user-friendly editing. In the heart of our pipeline is an
adaptive frequency modulation module called Adaptive Frequency Net (AFNet),
which utilizes a hypernetwork to learn the local texture frequency encoding
that is consecutively injected into adaptive frequency activation layers to
modulate the implicit radiance signal. This mechanism improves the frequency
expressive ability of the network with richer frequency basis support, only at
a small computational budget. To further boost performance, a preprocessing
module is also proposed for point cloud geometry optimization via point opacity
estimation. In contrast to implicit rendering, our pipeline supports
high-fidelity interactive editing based on point cloud manipulation. Extensive
experimental results on NeRF-Synthetic, ScanNet, DTU and Tanks and Temples
datasets demonstrate the superior performances achieved by our method in terms
of PSNR, SSIM and LPIPS, in comparison to the state-of-the-art.

Comments:
- Accepted by CVPR 2023

---

## NEF: Neural Edge Fields for 3D Parametric Curve Reconstruction from  Multi-view Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-14 | Yunfan Ye, Renjiao Yi, Zhirui Gao, Chenyang Zhu, Zhiping Cai, Kai Xu | cs.CV | [PDF](http://arxiv.org/pdf/2303.07653v2){: .btn .btn-green } |

**Abstract**: We study the problem of reconstructing 3D feature curves of an object from a
set of calibrated multi-view images. To do so, we learn a neural implicit field
representing the density distribution of 3D edges which we refer to as Neural
Edge Field (NEF). Inspired by NeRF, NEF is optimized with a view-based
rendering loss where a 2D edge map is rendered at a given view and is compared
to the ground-truth edge map extracted from the image of that view. The
rendering-based differentiable optimization of NEF fully exploits 2D edge
detection, without needing a supervision of 3D edges, a 3D geometric operator
or cross-view edge correspondence. Several technical designs are devised to
ensure learning a range-limited and view-independent NEF for robust edge
extraction. The final parametric 3D curves are extracted from NEF with an
iterative optimization method. On our benchmark with synthetic data, we
demonstrate that NEF outperforms existing state-of-the-art methods on all
metrics. Project page: https://yunfan1202.github.io/NEF/.

Comments:
- CVPR 2023

---

## Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D  Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-14 | Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, Seungryong Kim | cs.CV | [PDF](http://arxiv.org/pdf/2303.07937v3){: .btn .btn-green } |

**Abstract**: Text-to-3D generation has shown rapid progress in recent days with the advent
of score distillation, a methodology of using pretrained text-to-2D diffusion
models to optimize neural radiance field (NeRF) in the zero-shot setting.
However, the lack of 3D awareness in the 2D diffusion models destabilizes score
distillation-based methods from reconstructing a plausible 3D scene. To address
this issue, we propose 3DFuse, a novel framework that incorporates 3D awareness
into pretrained 2D diffusion models, enhancing the robustness and 3D
consistency of score distillation-based methods. We realize this by first
constructing a coarse 3D structure of a given text prompt and then utilizing
projected, view-specific depth map as a condition for the diffusion model.
Additionally, we introduce a training strategy that enables the 2D diffusion
model learns to handle the errors and sparsity within the coarse 3D structure
for robust generation, as well as a method for ensuring semantic consistency
throughout all viewpoints of the scene. Our framework surpasses the limitations
of prior arts, and has significant implications for 3D consistent generation of
2D diffusion models.

Comments:
- Project page https://ku-cvlab.github.io/3DFuse/

---

## MELON: NeRF with Unposed Images in SO(3)

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-14 | Axel Levy, Mark Matthews, Matan Sela, Gordon Wetzstein, Dmitry Lagun | cs.CV | [PDF](http://arxiv.org/pdf/2303.08096v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields enable novel-view synthesis and scene reconstruction
with photorealistic quality from a few images, but require known and accurate
camera poses. Conventional pose estimation algorithms fail on smooth or
self-similar scenes, while methods performing inverse rendering from unposed
views require a rough initialization of the camera orientations. The main
difficulty of pose estimation lies in real-life objects being almost invariant
under certain transformations, making the photometric distance between rendered
views non-convex with respect to the camera parameters. Using an equivalence
relation that matches the distribution of local minima in camera space, we
reduce this space to its quotient set, in which pose estimation becomes a more
convex problem. Using a neural-network to regularize pose estimation, we
demonstrate that our method - MELON - can reconstruct a neural radiance field
from unposed images with state-of-the-art accuracy while requiring ten times
fewer views than adversarial approaches.

---

## FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency  Regularization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-13 | Jiawei Yang, Marco Pavone, Yue Wang | cs.CV | [PDF](http://arxiv.org/pdf/2303.07418v1){: .btn .btn-green } |

**Abstract**: Novel view synthesis with sparse inputs is a challenging problem for neural
radiance fields (NeRF). Recent efforts alleviate this challenge by introducing
external supervision, such as pre-trained models and extra depth signals, and
by non-trivial patch-based rendering. In this paper, we present Frequency
regularized NeRF (FreeNeRF), a surprisingly simple baseline that outperforms
previous methods with minimal modifications to the plain NeRF. We analyze the
key challenges in few-shot neural rendering and find that frequency plays an
important role in NeRF's training. Based on the analysis, we propose two
regularization terms. One is to regularize the frequency range of NeRF's
inputs, while the other is to penalize the near-camera density fields. Both
techniques are ``free lunches'' at no additional computational cost. We
demonstrate that even with one line of code change, the original NeRF can
achieve similar performance as other complicated methods in the few-shot
setting. FreeNeRF achieves state-of-the-art performance across diverse
datasets, including Blender, DTU, and LLFF. We hope this simple baseline will
motivate a rethinking of the fundamental role of frequency in NeRF's training
under the low-data regime and beyond.

Comments:
- Project page: https://jiawei-yang.github.io/FreeNeRF/, Code at:
  https://github.com/Jiawei-Yang/FreeNeRF

---

## NeRFLiX: High-Quality Neural View Synthesis by Learning a  Degradation-Driven Inter-viewpoint MiXer

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-13 | Kun Zhou, Wenbo Li, Yi Wang, Tao Hu, Nianjuan Jiang, Xiaoguang Han, Jiangbo Lu | cs.CV | [PDF](http://arxiv.org/pdf/2303.06919v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) show great success in novel view synthesis.
However, in real-world scenes, recovering high-quality details from the source
images is still challenging for the existing NeRF-based approaches, due to the
potential imperfect calibration information and scene representation
inaccuracy. Even with high-quality training frames, the synthetic novel views
produced by NeRF models still suffer from notable rendering artifacts, such as
noise, blur, etc. Towards to improve the synthesis quality of NeRF-based
approaches, we propose NeRFLiX, a general NeRF-agnostic restorer paradigm by
learning a degradation-driven inter-viewpoint mixer. Specially, we design a
NeRF-style degradation modeling approach and construct large-scale training
data, enabling the possibility of effectively removing NeRF-native rendering
artifacts for existing deep neural networks. Moreover, beyond the degradation
removal, we propose an inter-viewpoint aggregation framework that is able to
fuse highly related high-quality training images, pushing the performance of
cutting-edge NeRF models to entirely new levels and producing highly
photo-realistic synthetic views.

Comments:
- Accepted to CVPR 2023; Project Page: see
  https://redrock303.github.io/nerflix/

---

## Just Flip: Flipped Observation Generation and Optimization for Neural  Radiance Fields to Cover Unobserved View

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-11 | Minjae Lee, Kyeongsu Kang, Hyeonwoo Yu | cs.RO | [PDF](http://arxiv.org/pdf/2303.06335v3){: .btn .btn-green } |

**Abstract**: With the advent of Neural Radiance Field (NeRF), representing 3D scenes
through multiple observations has shown remarkable improvements in performance.
Since this cutting-edge technique is able to obtain high-resolution renderings
by interpolating dense 3D environments, various approaches have been proposed
to apply NeRF for the spatial understanding of robot perception. However,
previous works are challenging to represent unobserved scenes or views on the
unexplored robot trajectory, as these works do not take into account 3D
reconstruction without observation information. To overcome this problem, we
propose a method to generate flipped observation in order to cover unexisting
observation for unexplored robot trajectory. To achieve this, we propose a data
augmentation method for 3D reconstruction using NeRF by flipping observed
images, and estimating flipped camera 6DOF poses. Our technique exploits the
property of objects being geometrically symmetric, making it simple but fast
and powerful, thereby making it suitable for robotic applications where
real-time performance is important. We demonstrate that our method
significantly improves three representative perceptual quality measures on the
NeRF synthetic dataset.

---

## Aleth-NeRF: Low-light Condition View Synthesis with Concealing Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-10 | Ziteng Cui, Lin Gu, Xiao Sun, Xianzheng Ma, Yu Qiao, Tatsuya Harada | cs.CV | [PDF](http://arxiv.org/pdf/2303.05807v2){: .btn .btn-green } |

**Abstract**: Common capture low-light scenes are challenging for most computer vision
techniques, including Neural Radiance Fields (NeRF). Vanilla NeRF is
viewer-centred simplifies the rendering process only as light emission from 3D
locations in the viewing direction, thus failing to model the low-illumination
induced darkness. Inspired by the emission theory of ancient Greeks that visual
perception is accomplished by rays casting from eyes, we make slight
modifications on vanilla NeRF to train on multiple views of low-light scenes,
we can thus render out the well-lit scene in an unsupervised manner. We
introduce a surrogate concept, Concealing Fields, that reduces the transport of
light during the volume rendering stage. Specifically, our proposed method,
Aleth-NeRF, directly learns from the dark image to understand volumetric object
representation and concealing field under priors. By simply eliminating
Concealing Fields, we can render a single or multi-view well-lit image(s) and
gain superior performance over other 2D low-light enhancement methods.
Additionally, we collect the first paired LOw-light and normal-light Multi-view
(LOM) datasets for future research. This version is invalid, please refer to
our new AAAI version: arXiv:2312.09093

Comments:
- website page: https://cuiziteng.github.io/Aleth_NeRF_web/, refer to
  new version: arXiv:2312.09093

---

## Hardware Acceleration of Neural Graphics

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-10 | Muhammad Husnain Mubarik, Ramakrishna Kanungo, Tobias Zirr, Rakesh Kumar | cs.AR | [PDF](http://arxiv.org/pdf/2303.05735v6){: .btn .btn-green } |

**Abstract**: Rendering and inverse-rendering algorithms that drive conventional computer
graphics have recently been superseded by neural representations (NR). NRs have
recently been used to learn the geometric and the material properties of the
scenes and use the information to synthesize photorealistic imagery, thereby
promising a replacement for traditional rendering algorithms with scalable
quality and predictable performance. In this work we ask the question: Does
neural graphics (NG) need hardware support? We studied representative NG
applications showing that, if we want to render 4k res. at 60FPS there is a gap
of 1.5X-55X in the desired performance on current GPUs. For AR/VR applications,
there is an even larger gap of 2-4 OOM between the desired performance and the
required system power. We identify that the input encoding and the MLP kernels
are the performance bottlenecks, consuming 72%,60% and 59% of application time
for multi res. hashgrid, multi res. densegrid and low res. densegrid encodings,
respectively. We propose a NG processing cluster, a scalable and flexible
hardware architecture that directly accelerates the input encoding and MLP
kernels through dedicated engines and supports a wide range of NG applications.
We also accelerate the rest of the kernels by fusing them together in Vulkan,
which leads to 9.94X kernel-level performance improvement compared to un-fused
implementation of the pre-processing and the post-processing kernels. Our
results show that, NGPC gives up to 58X end-to-end application-level
performance improvement, for multi res. hashgrid encoding on average across the
four NG applications, the performance benefits are 12X,20X,33X and 39X for the
scaling factor of 8,16,32 and 64, respectively. Our results show that with
multi res. hashgrid encoding, NGPC enables the rendering of 4k res. at 30FPS
for NeRF and 8k res. at 120FPS for all our other NG applications.

---

## Self-NeRF: A Self-Training Pipeline for Few-Shot Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-10 | Jiayang Bai, Letian Huang, Wen Gong, Jie Guo, Yanwen Guo | cs.CV | [PDF](http://arxiv.org/pdf/2303.05775v1){: .btn .btn-green } |

**Abstract**: Recently, Neural Radiance Fields (NeRF) have emerged as a potent method for
synthesizing novel views from a dense set of images. Despite its impressive
performance, NeRF is plagued by its necessity for numerous calibrated views and
its accuracy diminishes significantly in a few-shot setting. To address this
challenge, we propose Self-NeRF, a self-evolved NeRF that iteratively refines
the radiance fields with very few number of input views, without incorporating
additional priors. Basically, we train our model under the supervision of
reference and unseen views simultaneously in an iterative procedure. In each
iteration, we label unseen views with the predicted colors or warped pixels
generated by the model from the preceding iteration. However, these expanded
pseudo-views are afflicted by imprecision in color and warping artifacts, which
degrades the performance of NeRF. To alleviate this issue, we construct an
uncertainty-aware NeRF with specialized embeddings. Some techniques such as
cone entropy regularization are further utilized to leverage the pseudo-views
in the most efficient manner. Through experiments under various settings, we
verified that our Self-NeRF is robust to input with uncertainty and surpasses
existing methods when trained on limited training data.

Comments:
- 11 pages, 11 figures

---

## MovingParts: Motion-based 3D Part Discovery in Dynamic Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-10 | Kaizhi Yang, Xiaoshuai Zhang, Zhiao Huang, Xuejin Chen, Zexiang Xu, Hao Su | cs.CV | [PDF](http://arxiv.org/pdf/2303.05703v2){: .btn .btn-green } |

**Abstract**: We present MovingParts, a NeRF-based method for dynamic scene reconstruction
and part discovery. We consider motion as an important cue for identifying
parts, that all particles on the same part share the common motion pattern.
From the perspective of fluid simulation, existing deformation-based methods
for dynamic NeRF can be seen as parameterizing the scene motion under the
Eulerian view, i.e., focusing on specific locations in space through which the
fluid flows as time passes. However, it is intractable to extract the motion of
constituting objects or parts using the Eulerian view representation. In this
work, we introduce the dual Lagrangian view and enforce representations under
the Eulerian/Lagrangian views to be cycle-consistent. Under the Lagrangian
view, we parameterize the scene motion by tracking the trajectory of particles
on objects. The Lagrangian view makes it convenient to discover parts by
factorizing the scene motion as a composition of part-level rigid motions.
Experimentally, our method can achieve fast and high-quality dynamic scene
reconstruction from even a single moving camera, and the induced part-based
representation allows direct applications of part tracking, animation, 3D scene
editing, etc.

Comments:
- Project Page: https://silenkzyoung.github.io/MovingParts-WebPage/

---

## You Only Train Once: Multi-Identity Free-Viewpoint Neural Human  Rendering from Monocular Videos

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-10 | Jaehyeok Kim, Dongyoon Wee, Dan Xu | cs.CV | [PDF](http://arxiv.org/pdf/2303.05835v1){: .btn .btn-green } |

**Abstract**: We introduce You Only Train Once (YOTO), a dynamic human generation
framework, which performs free-viewpoint rendering of different human
identities with distinct motions, via only one-time training from monocular
videos. Most prior works for the task require individualized optimization for
each input video that contains a distinct human identity, leading to a
significant amount of time and resources for the deployment, thereby impeding
the scalability and the overall application potential of the system. In this
paper, we tackle this problem by proposing a set of learnable identity codes to
expand the capability of the framework for multi-identity free-viewpoint
rendering, and an effective pose-conditioned code query mechanism to finely
model the pose-dependent non-rigid motions. YOTO optimizes neural radiance
fields (NeRF) by utilizing designed identity codes to condition the model for
learning various canonical T-pose appearances in a single shared volumetric
representation. Besides, our joint learning of multiple identities within a
unified model incidentally enables flexible motion transfer in high-quality
photo-realistic renderings for all learned appearances. This capability expands
its potential use in important applications, including Virtual Reality. We
present extensive experimental results on ZJU-MoCap and PeopleSnapshot to
clearly demonstrate the effectiveness of our proposed model. YOTO shows
state-of-the-art performance on all evaluation metrics while showing
significant benefits in training and inference efficiency as well as rendering
quality. The code and model will be made publicly available soon.

---

## Learning Object-Centric Neural Scattering Functions for Free-Viewpoint  Relighting and Scene Composition



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-10 | Hong-Xing Yu, Michelle Guo, Alireza Fathi, Yen-Yu Chang, Eric Ryan Chan, Ruohan Gao, Thomas Funkhouser, Jiajun Wu | cs.CV | [PDF](http://arxiv.org/pdf/2303.06138v4){: .btn .btn-green } |

**Abstract**: Photorealistic object appearance modeling from 2D images is a constant topic
in vision and graphics. While neural implicit methods (such as Neural Radiance
Fields) have shown high-fidelity view synthesis results, they cannot relight
the captured objects. More recent neural inverse rendering approaches have
enabled object relighting, but they represent surface properties as simple
BRDFs, and therefore cannot handle translucent objects. We propose
Object-Centric Neural Scattering Functions (OSFs) for learning to reconstruct
object appearance from only images. OSFs not only support free-viewpoint object
relighting, but also can model both opaque and translucent objects. While
accurately modeling subsurface light transport for translucent objects can be
highly complex and even intractable for neural methods, OSFs learn to
approximate the radiance transfer from a distant light to an outgoing direction
at any spatial location. This approximation avoids explicitly modeling complex
subsurface scattering, making learning a neural implicit model tractable.
Experiments on real and synthetic data show that OSFs accurately reconstruct
appearances for both opaque and translucent objects, allowing faithful
free-viewpoint relighting as well as scene composition.

Comments:
- Journal extension of arXiv:2012.08503 (TMLR 2023). The first two
  authors contributed equally to this work. Project page:
  https://kovenyu.com/osf/

---

## NeRFlame: FLAME-based conditioning of NeRF for 3D face rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-10 | Wojciech Zając, Joanna Waczyńska, Piotr Borycki, Jacek Tabor, Maciej Zięba, Przemysław Spurek | cs.CV | [PDF](http://arxiv.org/pdf/2303.06226v2){: .btn .btn-green } |

**Abstract**: Traditional 3D face models are based on mesh representations with texture.
One of the most important models is FLAME (Faces Learned with an Articulated
Model and Expressions), which produces meshes of human faces that are fully
controllable. Unfortunately, such models have problems with capturing geometric
and appearance details. In contrast to mesh representation, the neural radiance
field (NeRF) produces extremely sharp renders. However, implicit methods are
hard to animate and do not generalize well to unseen expressions. It is not
trivial to effectively control NeRF models to obtain face manipulation.
  The present paper proposes a novel approach, named NeRFlame, which combines
the strengths of both NeRF and FLAME methods. Our method enables high-quality
rendering capabilities of NeRF while also offering complete control over the
visual appearance, similar to FLAME. In contrast to traditional NeRF-based
structures that use neural networks for RGB color and volume density modeling,
our approach utilizes the FLAME mesh as a distinct density volume.
Consequently, color values exist only in the vicinity of the FLAME mesh. This
FLAME framework is seamlessly incorporated into the NeRF architecture for
predicting RGB colors, enabling our model to explicitly represent volume
density and implicitly capture RGB colors.

---

## PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields for  Geometry-Agnostic System Identification

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-09 | Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, Chuang Gan | cs.CV | [PDF](http://arxiv.org/pdf/2303.05512v1){: .btn .btn-green } |

**Abstract**: Existing approaches to system identification (estimating the physical
parameters of an object) from videos assume known object geometries. This
precludes their applicability in a vast majority of scenes where object
geometries are complex or unknown. In this work, we aim to identify parameters
characterizing a physical system from a set of multi-view videos without any
assumption on object geometry or topology. To this end, we propose "Physics
Augmented Continuum Neural Radiance Fields" (PAC-NeRF), to estimate both the
unknown geometry and physical parameters of highly dynamic objects from
multi-view videos. We design PAC-NeRF to only ever produce physically plausible
states by enforcing the neural radiance field to follow the conservation laws
of continuum mechanics. For this, we design a hybrid Eulerian-Lagrangian
representation of the neural radiance field, i.e., we use the Eulerian grid
representation for NeRF density and color fields, while advecting the neural
radiance fields via Lagrangian particles. This hybrid Eulerian-Lagrangian
representation seamlessly blends efficient neural rendering with the material
point method (MPM) for robust differentiable physics simulation. We validate
the effectiveness of our proposed framework on geometry and physical parameter
estimation over a vast range of materials, including elastic bodies,
plasticine, sand, Newtonian and non-Newtonian fluids, and demonstrate
significant performance gain on most tasks.

Comments:
- ICLR 2023 Spotlight. Project page:
  https://sites.google.com/view/PAC-NeRF

---

## DroNeRF: Real-time Multi-agent Drone Pose Optimization for Computing  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-08 | Dipam Patel, Phu Pham, Aniket Bera | cs.RO | [PDF](http://arxiv.org/pdf/2303.04322v2){: .btn .btn-green } |

**Abstract**: We present a novel optimization algorithm called DroNeRF for the autonomous
positioning of monocular camera drones around an object for real-time 3D
reconstruction using only a few images. Neural Radiance Fields or NeRF, is a
novel view synthesis technique used to generate new views of an object or scene
from a set of input images. Using drones in conjunction with NeRF provides a
unique and dynamic way to generate novel views of a scene, especially with
limited scene capabilities of restricted movements. Our approach focuses on
calculating optimized pose for individual drones while solely depending on the
object geometry without using any external localization system. The unique
camera positioning during the data-capturing phase significantly impacts the
quality of the 3D model. To evaluate the quality of our generated novel views,
we compute different perceptual metrics like the Peak Signal-to-Noise Ratio
(PSNR) and Structural Similarity Index Measure(SSIM). Our work demonstrates the
benefit of using an optimal placement of various drones with limited mobility
to generate perceptually better results.

Comments:
- To appear in 2023 IEEE/RSJ International Conference on Intelligent
  Robots and Systems (IROS 2023)

---

## CROSSFIRE: Camera Relocalization On Self-Supervised Features from an  Implicit Representation



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-08 | Arthur Moreau, Nathan Piasco, Moussab Bennehar, Dzmitry Tsishkou, Bogdan Stanciulescu, Arnaud de La Fortelle | cs.CV | [PDF](http://arxiv.org/pdf/2303.04869v2){: .btn .btn-green } |

**Abstract**: Beyond novel view synthesis, Neural Radiance Fields are useful for
applications that interact with the real world. In this paper, we use them as
an implicit map of a given scene and propose a camera relocalization algorithm
tailored for this representation. The proposed method enables to compute in
real-time the precise position of a device using a single RGB camera, during
its navigation. In contrast with previous work, we do not rely on pose
regression or photometric alignment but rather use dense local features
obtained through volumetric rendering which are specialized on the scene with a
self-supervised objective. As a result, our algorithm is more accurate than
competitors, able to operate in dynamic outdoor environments with changing
lightning conditions and can be readily integrated in any volumetric neural
renderer.

Comments:
- Accepted to ICCV 2023

---

## InFusionSurf: Refining Neural RGB-D Surface Reconstruction Using  Per-Frame Intrinsic Refinement and TSDF Fusion Prior Learning

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-08 | Seunghwan Lee, Gwanmo Park, Hyewon Son, Jiwon Ryu, Han Joo Chae | cs.CV | [PDF](http://arxiv.org/pdf/2303.04508v2){: .btn .btn-green } |

**Abstract**: We introduce InFusionSurf, a novel approach to enhance the fidelity of neural
radiance field (NeRF) frameworks for 3D surface reconstruction using RGB-D
video frames. Building upon previous methods that have employed feature
encoding to improve optimization speed, we further improve the reconstruction
quality with minimal impact on optimization time by refining depth information.
Our per-frame intrinsic refinement scheme addresses frame-specific blurs caused
by camera motion in each depth frame. Furthermore, InFusionSurf utilizes a
classical real-time 3D surface reconstruction method, the truncated signed
distance field (TSDF) Fusion, as prior knowledge to pretrain the feature grid
to support reconstruction details while accelerating the training. The
quantitative and qualitative experiments comparing the performances of
InFusionSurf against prior work indicate that our method is capable of
accurately reconstructing a scene without sacrificing optimization speed. We
also demonstrate the effectiveness of our per-frame intrinsic refinement and
TSDF Fusion prior learning techniques via an ablation study.

---

## NEPHELE: A Neural Platform for Highly Realistic Cloud Radiance Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-07 | Haimin Luo, Siyuan Zhang, Fuqiang Zhao, Haotian Jing, Penghao Wang, Zhenxiao Yu, Dongxue Yan, Junran Ding, Boyuan Zhang, Qiang Hu, Shu Yin, Lan Xu, JIngyi Yu | cs.GR | [PDF](http://arxiv.org/pdf/2303.04086v1){: .btn .btn-green } |

**Abstract**: We have recently seen tremendous progress in neural rendering (NR) advances,
i.e., NeRF, for photo-real free-view synthesis. Yet, as a local technique based
on a single computer/GPU, even the best-engineered Instant-NGP or i-NGP cannot
reach real-time performance when rendering at a high resolution, and often
requires huge local computing resources. In this paper, we resort to cloud
rendering and present NEPHELE, a neural platform for highly realistic cloud
radiance rendering. In stark contrast with existing NR approaches, our NEPHELE
allows for more powerful rendering capabilities by combining multiple remote
GPUs and facilitates collaboration by allowing multiple people to view the same
NeRF scene simultaneously. We introduce i-NOLF to employ opacity light fields
for ultra-fast neural radiance rendering in a one-query-per-ray manner. We
further resemble the Lumigraph with geometry proxies for fast ray querying and
subsequently employ a small MLP to model the local opacity lumishperes for
high-quality rendering. We also adopt Perfect Spatial Hashing in i-NOLF to
enhance cache coherence. As a result, our i-NOLF achieves an order of magnitude
performance gain in terms of efficiency than i-NGP, especially for the
multi-user multi-viewpoint setting under cloud rendering scenarios. We further
tailor a task scheduler accompanied by our i-NOLF representation and
demonstrate the advance of our methodological design through a comprehensive
cloud platform, consisting of a series of cooperated modules, i.e., render
farms, task assigner, frame composer, and detailed streaming strategies. Using
such a cloud platform compatible with neural rendering, we further showcase the
capabilities of our cloud radiance rendering through a series of applications,
ranging from cloud VR/AR rendering.

---

## Multiscale Tensor Decomposition and Rendering Equation Encoding for View  Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-07 | Kang Han, Wei Xiang | cs.CV | [PDF](http://arxiv.org/pdf/2303.03808v2){: .btn .btn-green } |

**Abstract**: Rendering novel views from captured multi-view images has made considerable
progress since the emergence of the neural radiance field. This paper aims to
further advance the quality of view synthesis by proposing a novel approach
dubbed the neural radiance feature field (NRFF). We first propose a multiscale
tensor decomposition scheme to organize learnable features so as to represent
scenes from coarse to fine scales. We demonstrate many benefits of the proposed
multiscale representation, including more accurate scene shape and appearance
reconstruction, and faster convergence compared with the single-scale
representation. Instead of encoding view directions to model view-dependent
effects, we further propose to encode the rendering equation in the feature
space by employing the anisotropic spherical Gaussian mixture predicted from
the proposed multiscale representation. The proposed NRFF improves
state-of-the-art rendering results by over 1 dB in PSNR on both the NeRF and
NSVF synthetic datasets. A significant improvement has also been observed on
the real-world Tanks & Temples dataset. Code can be found at
https://github.com/imkanghan/nrff.

---

## Nerflets: Local Radiance Fields for Efficient Structure-Aware 3D Scene  Representation from 2D Supervision

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-06 | Xiaoshuai Zhang, Abhijit Kundu, Thomas Funkhouser, Leonidas Guibas, Hao Su, Kyle Genova | cs.CV | [PDF](http://arxiv.org/pdf/2303.03361v2){: .btn .btn-green } |

**Abstract**: We address efficient and structure-aware 3D scene representation from images.
Nerflets are our key contribution -- a set of local neural radiance fields that
together represent a scene. Each nerflet maintains its own spatial position,
orientation, and extent, within which it contributes to panoptic, density, and
radiance reconstructions. By leveraging only photometric and inferred panoptic
image supervision, we can directly and jointly optimize the parameters of a set
of nerflets so as to form a decomposed representation of the scene, where each
object instance is represented by a group of nerflets. During experiments with
indoor and outdoor environments, we find that nerflets: (1) fit and approximate
the scene more efficiently than traditional global NeRFs, (2) allow the
extraction of panoptic and photometric renderings from arbitrary views, and (3)
enable tasks rare for NeRFs, such as 3D panoptic segmentation and interactive
editing.

Comments:
- accepted by CVPR 2023

---

## MOISST: Multimodal Optimization of Implicit Scene for SpatioTemporal  calibration

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-06 | Quentin Herau, Nathan Piasco, Moussab Bennehar, Luis Roldão, Dzmitry Tsishkou, Cyrille Migniot, Pascal Vasseur, Cédric Demonceaux | cs.CV | [PDF](http://arxiv.org/pdf/2303.03056v3){: .btn .btn-green } |

**Abstract**: With the recent advances in autonomous driving and the decreasing cost of
LiDARs, the use of multimodal sensor systems is on the rise. However, in order
to make use of the information provided by a variety of complimentary sensors,
it is necessary to accurately calibrate them. We take advantage of recent
advances in computer graphics and implicit volumetric scene representation to
tackle the problem of multi-sensor spatial and temporal calibration. Thanks to
a new formulation of the Neural Radiance Field (NeRF) optimization, we are able
to jointly optimize calibration parameters along with scene representation
based on radiometric and geometric measurements. Our method enables accurate
and robust calibration from data captured in uncontrolled and unstructured
urban environments, making our solution more scalable than existing calibration
solutions. We demonstrate the accuracy and robustness of our method in urban
scenes typically encountered in autonomous driving scenarios.

Comments:
- Accepted at IROS2023 Project site: https://qherau.github.io/MOISST/

---

## Efficient Large-scale Scene Representation with a Hybrid of  High-resolution Grid and Plane Features

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-06 | Yuqi Zhang, Guanying Chen, Shuguang Cui | cs.CV | [PDF](http://arxiv.org/pdf/2303.03003v2){: .btn .btn-green } |

**Abstract**: Existing neural radiance fields (NeRF) methods for large-scale scene modeling
require days of training using multiple GPUs, hindering their applications in
scenarios with limited computing resources. Despite fast optimization NeRF
variants have been proposed based on the explicit dense or hash grid features,
their effectivenesses are mainly demonstrated in object-scale scene
representation. In this paper, we point out that the low feature resolution in
explicit representation is the bottleneck for large-scale unbounded scene
representation. To address this problem, we introduce a new and efficient
hybrid feature representation for NeRF that fuses the 3D hash-grids and
high-resolution 2D dense plane features. Compared with the dense-grid
representation, the resolution of a dense 2D plane can be scaled up more
efficiently. Based on this hybrid representation, we propose a fast
optimization NeRF variant, called GP-NeRF, that achieves better rendering
results while maintaining a compact model size. Extensive experiments on
multiple large-scale unbounded scene datasets show that our model can converge
in 1.5 hours using a single GPU while achieving results comparable to or even
better than the existing method that requires about one day's training with 8
GPUs.

---

## Semantic-aware Occlusion Filtering Neural Radiance Fields in the Wild

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-05 | Jaewon Lee, Injae Kim, Hwan Heo, Hyunwoo J. Kim | cs.CV | [PDF](http://arxiv.org/pdf/2303.03966v1){: .btn .btn-green } |

**Abstract**: We present a learning framework for reconstructing neural scene
representations from a small number of unconstrained tourist photos. Since each
image contains transient occluders, decomposing the static and transient
components is necessary to construct radiance fields with such in-the-wild
photographs where existing methods require a lot of training data. We introduce
SF-NeRF, aiming to disentangle those two components with only a few images
given, which exploits semantic information without any supervision. The
proposed method contains an occlusion filtering module that predicts the
transient color and its opacity for each pixel, which enables the NeRF model to
solely learn the static scene representation. This filtering module learns the
transient phenomena guided by pixel-wise semantic features obtained by a
trainable image encoder that can be trained across multiple scenes to learn the
prior of transient objects. Furthermore, we present two techniques to prevent
ambiguous decomposition and noisy results of the filtering module. We
demonstrate that our method outperforms state-of-the-art novel view synthesis
methods on Phototourism dataset in a few-shot setting.

Comments:
- 11 pages, 5 figures

---

## Delicate Textured Mesh Recovery from NeRF via Adaptive Surface  Refinement

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-03 | Jiaxiang Tang, Hang Zhou, Xiaokang Chen, Tianshu Hu, Errui Ding, Jingdong Wang, Gang Zeng | cs.CV | [PDF](http://arxiv.org/pdf/2303.02091v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have constituted a remarkable breakthrough in
image-based 3D reconstruction. However, their implicit volumetric
representations differ significantly from the widely-adopted polygonal meshes
and lack support from common 3D software and hardware, making their rendering
and manipulation inefficient. To overcome this limitation, we present a novel
framework that generates textured surface meshes from images. Our approach
begins by efficiently initializing the geometry and view-dependency decomposed
appearance with a NeRF. Subsequently, a coarse mesh is extracted, and an
iterative surface refining algorithm is developed to adaptively adjust both
vertex positions and face density based on re-projected rendering errors. We
jointly refine the appearance with geometry and bake it into texture images for
real-time rendering. Extensive experiments demonstrate that our method achieves
superior mesh quality and competitive rendering quality.

Comments:
- ICCV 2023 camera-ready, Project Page: https://me.kiui.moe/nerf2mesh

---

## Multi-Plane Neural Radiance Fields for Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-03 | Youssef Abdelkareem, Shady Shehata, Fakhri Karray | cs.CV | [PDF](http://arxiv.org/pdf/2303.01736v1){: .btn .btn-green } |

**Abstract**: Novel view synthesis is a long-standing problem that revolves around
rendering frames of scenes from novel camera viewpoints. Volumetric approaches
provide a solution for modeling occlusions through the explicit 3D
representation of the camera frustum. Multi-plane Images (MPI) are volumetric
methods that represent the scene using front-parallel planes at distinct depths
but suffer from depth discretization leading to a 2.D scene representation.
Another line of approach relies on implicit 3D scene representations. Neural
Radiance Fields (NeRF) utilize neural networks for encapsulating the continuous
3D scene structure within the network weights achieving photorealistic
synthesis results, however, methods are constrained to per-scene optimization
settings which are inefficient in practice. Multi-plane Neural Radiance Fields
(MINE) open the door for combining implicit and explicit scene representations.
It enables continuous 3D scene representations, especially in the depth
dimension, while utilizing the input image features to avoid per-scene
optimization. The main drawback of the current literature work in this domain
is being constrained to single-view input, limiting the synthesis ability to
narrow viewpoint ranges. In this work, we thoroughly examine the performance,
generalization, and efficiency of single-view multi-plane neural radiance
fields. In addition, we propose a new multiplane NeRF architecture that accepts
multiple views to improve the synthesis results and expand the viewing range.
Features from the input source frames are effectively fused through a proposed
attention-aware fusion module to highlight important information from different
viewpoints. Experiments show the effectiveness of attention-based fusion and
the promising outcomes of our proposed method when compared to multi-view NeRF
and MPI techniques.

Comments:
- ICDIPV 2023

---

## S-NeRF: Neural Radiance Fields for Street Views

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-01 | Ziyang Xie, Junge Zhang, Wenye Li, Feihu Zhang, Li Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2303.00749v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) aim to synthesize novel views of objects and
scenes, given the object-centric camera views with large overlaps. However, we
conjugate that this paradigm does not fit the nature of the street views that
are collected by many self-driving cars from the large-scale unbounded scenes.
Also, the onboard cameras perceive scenes without much overlapping. Thus,
existing NeRFs often produce blurs, 'floaters' and other artifacts on
street-view synthesis. In this paper, we propose a new street-view NeRF
(S-NeRF) that considers novel view synthesis of both the large-scale background
scenes and the foreground moving vehicles jointly. Specifically, we improve the
scene parameterization function and the camera poses for learning better neural
representations from street views. We also use the the noisy and sparse LiDAR
points to boost the training and learn a robust geometry and reprojection based
confidence to address the depth outliers. Moreover, we extend our S-NeRF for
reconstructing moving vehicles that is impracticable for conventional NeRFs.
Thorough experiments on the large-scale driving datasets (e.g., nuScenes and
Waymo) demonstrate that our method beats the state-of-the-art rivals by
reducing 7% to 40% of the mean-squared error in the street-view synthesis and a
45% PSNR gain for the moving vehicles rendering.

Comments:
- ICLR 2023

---

## Renderable Neural Radiance Map for Visual Navigation



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-03-01 | Obin Kwon, Jeongho Park, Songhwai Oh | cs.CV | [PDF](http://arxiv.org/pdf/2303.00304v4){: .btn .btn-green } |

**Abstract**: We propose a novel type of map for visual navigation, a renderable neural
radiance map (RNR-Map), which is designed to contain the overall visual
information of a 3D environment. The RNR-Map has a grid form and consists of
latent codes at each pixel. These latent codes are embedded from image
observations, and can be converted to the neural radiance field which enables
image rendering given a camera pose. The recorded latent codes implicitly
contain visual information about the environment, which makes the RNR-Map
visually descriptive. This visual information in RNR-Map can be a useful
guideline for visual localization and navigation. We develop localization and
navigation frameworks that can effectively utilize the RNR-Map. We evaluate the
proposed frameworks on camera tracking, visual localization, and image-goal
navigation. Experimental results show that the RNR-Map-based localization
framework can find the target location based on a single query image with fast
speed and competitive accuracy compared to other baselines. Also, this
localization framework is robust to environmental changes, and even finds the
most visually similar places when a query image from a different environment is
given. The proposed navigation framework outperforms the existing image-goal
navigation methods in difficult scenarios, under odometry and actuation noises.
The navigation framework shows 65.7% success rate in curved scenarios of the
NRNS dataset, which is an improvement of 18.6% over the current
state-of-the-art. Project page: https://rllab-snu.github.io/projects/RNR-Map/

Comments:
- Preprint version. CVPR 2023 accepted, highlight paper. Project page:
  https://rllab-snu.github.io/projects/RNR-Map/