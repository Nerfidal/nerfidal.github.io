---
layout: default
title: May
parent: 2022
nav_order: 5
---
<!---metadata--->

## Novel View Synthesis for High-fidelity Headshot Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-31 | Satoshi Tsutsui, Weijia Mao, Sijing Lin, Yunyi Zhu, Murong Ma, Mike Zheng Shou | cs.CV | [PDF](http://arxiv.org/pdf/2205.15595v1){: .btn .btn-green } |

**Abstract**: Rendering scenes with a high-quality human face from arbitrary viewpoints is
a practical and useful technique for many real-world applications. Recently,
Neural Radiance Fields (NeRF), a rendering technique that uses neural networks
to approximate classical ray tracing, have been considered as one of the
promising approaches for synthesizing novel views from a sparse set of images.
We find that NeRF can render new views while maintaining geometric consistency,
but it does not properly maintain skin details, such as moles and pores. These
details are important particularly for faces because when we look at an image
of a face, we are much more sensitive to details than when we look at other
objects. On the other hand, 3D Morpable Models (3DMMs) based on traditional
meshes and textures can perform well in terms of skin detail despite that it
has less precise geometry and cannot cover the head and the entire scene with
background. Based on these observations, we propose a method to use both NeRF
and 3DMM to synthesize a high-fidelity novel view of a scene with a face. Our
method learns a Generative Adversarial Network (GAN) to mix a NeRF-synthesized
image and a 3DMM-rendered image and produces a photorealistic scene with a face
preserving the skin details. Experiments with various real-world scenes
demonstrate the effectiveness of our approach. The code will be available on
https://github.com/showlab/headshot .

---

## Decomposing NeRF for Editing via Feature Field Distillation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-31 | Sosuke Kobayashi, Eiichi Matsumoto, Vincent Sitzmann | cs.CV | [PDF](http://arxiv.org/pdf/2205.15585v2){: .btn .btn-green } |

**Abstract**: Emerging neural radiance fields (NeRF) are a promising scene representation
for computer graphics, enabling high-quality 3D reconstruction and novel view
synthesis from image observations. However, editing a scene represented by a
NeRF is challenging, as the underlying connectionist representations such as
MLPs or voxel grids are not object-centric or compositional. In particular, it
has been difficult to selectively edit specific regions or objects. In this
work, we tackle the problem of semantic scene decomposition of NeRFs to enable
query-based local editing of the represented 3D scenes. We propose to distill
the knowledge of off-the-shelf, self-supervised 2D image feature extractors
such as CLIP-LSeg or DINO into a 3D feature field optimized in parallel to the
radiance field. Given a user-specified query of various modalities such as
text, an image patch, or a point-and-click selection, 3D feature fields
semantically decompose 3D space without the need for re-training and enable us
to semantically select and edit regions in the radiance field. Our experiments
validate that the distilled feature fields (DFFs) can transfer recent progress
in 2D vision and language foundation models to 3D scene representations,
enabling convincing 3D segmentation and selective editing of emerging neural
graphics representations.

Comments:
- Accepted to NeurIPS 2022
  https://pfnet-research.github.io/distilled-feature-fields/

---

## SAMURAI: Shape And Material from Unconstrained Real-world Arbitrary  Image collections

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-31 | Mark Boss, Andreas Engelhardt, Abhishek Kar, Yuanzhen Li, Deqing Sun, Jonathan T. Barron, Hendrik P. A. Lensch, Varun Jampani | cs.CV | [PDF](http://arxiv.org/pdf/2205.15768v1){: .btn .btn-green } |

**Abstract**: Inverse rendering of an object under entirely unknown capture conditions is a
fundamental challenge in computer vision and graphics. Neural approaches such
as NeRF have achieved photorealistic results on novel view synthesis, but they
require known camera poses. Solving this problem with unknown camera poses is
highly challenging as it requires joint optimization over shape, radiance, and
pose. This problem is exacerbated when the input images are captured in the
wild with varying backgrounds and illuminations. Standard pose estimation
techniques fail in such image collections in the wild due to very few estimated
correspondences across images. Furthermore, NeRF cannot relight a scene under
any illumination, as it operates on radiance (the product of reflectance and
illumination). We propose a joint optimization framework to estimate the shape,
BRDF, and per-image camera pose and illumination. Our method works on
in-the-wild online image collections of an object and produces relightable 3D
assets for several use-cases such as AR/VR. To our knowledge, our method is the
first to tackle this severely unconstrained task with minimal user interaction.
Project page: https://markboss.me/publication/2022-samurai/ Video:
https://youtu.be/LlYuGDjXp-8

---

## DeVRF: Fast Deformable Voxel Radiance Fields for Dynamic Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-31 | Jia-Wei Liu, Yan-Pei Cao, Weijia Mao, Wenqiao Zhang, David Junhao Zhang, Jussi Keppo, Ying Shan, Xiaohu Qie, Mike Zheng Shou | cs.CV | [PDF](http://arxiv.org/pdf/2205.15723v2){: .btn .btn-green } |

**Abstract**: Modeling dynamic scenes is important for many applications such as virtual
reality and telepresence. Despite achieving unprecedented fidelity for novel
view synthesis in dynamic scenes, existing methods based on Neural Radiance
Fields (NeRF) suffer from slow convergence (i.e., model training time measured
in days). In this paper, we present DeVRF, a novel representation to accelerate
learning dynamic radiance fields. The core of DeVRF is to model both the 3D
canonical space and 4D deformation field of a dynamic, non-rigid scene with
explicit and discrete voxel-based representations. However, it is quite
challenging to train such a representation which has a large number of model
parameters, often resulting in overfitting issues. To overcome this challenge,
we devise a novel static-to-dynamic learning paradigm together with a new data
capture setup that is convenient to deploy in practice. This paradigm unlocks
efficient learning of deformable radiance fields via utilizing the 3D
volumetric canonical space learnt from multi-view static images to ease the
learning of 4D voxel deformation field with only few-view dynamic sequences. To
further improve the efficiency of our DeVRF and its synthesized novel view's
quality, we conduct thorough explorations and identify a set of strategies. We
evaluate DeVRF on both synthetic and real-world dynamic scenes with different
types of deformation. Experiments demonstrate that DeVRF achieves two orders of
magnitude speedup (100x faster) with on-par high-fidelity results compared to
the previous state-of-the-art approaches. The code and dataset will be released
in https://github.com/showlab/DeVRF.

Comments:
- Project page: https://jia-wei-liu.github.io/DeVRF/

---

## D$^2$NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from  a Monocular Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-31 | Tianhao Wu, Fangcheng Zhong, Andrea Tagliasacchi, Forrester Cole, Cengiz Oztireli | cs.CV | [PDF](http://arxiv.org/pdf/2205.15838v4){: .btn .btn-green } |

**Abstract**: Given a monocular video, segmenting and decoupling dynamic objects while
recovering the static environment is a widely studied problem in machine
intelligence. Existing solutions usually approach this problem in the image
domain, limiting their performance and understanding of the environment. We
introduce Decoupled Dynamic Neural Radiance Field (D$^2$NeRF), a
self-supervised approach that takes a monocular video and learns a 3D scene
representation which decouples moving objects, including their shadows, from
the static background. Our method represents the moving objects and the static
background by two separate neural radiance fields with only one allowing for
temporal changes. A naive implementation of this approach leads to the dynamic
component taking over the static one as the representation of the former is
inherently more general and prone to overfitting. To this end, we propose a
novel loss to promote correct separation of phenomena. We further propose a
shadow field network to detect and decouple dynamically moving shadows. We
introduce a new dataset containing various dynamic objects and shadows and
demonstrate that our method can achieve better performance than
state-of-the-art approaches in decoupling dynamic and static 3D objects,
occlusion and shadow removal, and image segmentation for moving objects.

---

## Fast Dynamic Radiance Fields with Time-Aware Neural Voxels

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-30 | Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias Nießner, Qi Tian | cs.CV | [PDF](http://arxiv.org/pdf/2205.15285v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) have shown great success in modeling 3D scenes
and synthesizing novel-view images. However, most previous NeRF methods take
much time to optimize one single scene. Explicit data structures, e.g. voxel
features, show great potential to accelerate the training process. However,
voxel features face two big challenges to be applied to dynamic scenes, i.e.
modeling temporal information and capturing different scales of point motions.
We propose a radiance field framework by representing scenes with time-aware
voxel features, named as TiNeuVox. A tiny coordinate deformation network is
introduced to model coarse motion trajectories and temporal information is
further enhanced in the radiance network. A multi-distance interpolation method
is proposed and applied on voxel features to model both small and large
motions. Our framework significantly accelerates the optimization of dynamic
radiance fields while maintaining high rendering quality. Empirical evaluation
is performed on both synthetic and real scenes. Our TiNeuVox completes training
with only 8 minutes and 8-MB storage cost while showing similar or even better
rendering performance than previous dynamic NeRF methods.

Comments:
- SIGGRAPH Asia 2022. Project page: https://jaminfong.cn/tineuvox

---

## Neural Volumetric Object Selection

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-30 | Zhongzheng Ren, Aseem Agarwala, Bryan Russell, Alexander G. Schwing, Oliver Wang | cs.CV | [PDF](http://arxiv.org/pdf/2205.14929v1){: .btn .btn-green } |

**Abstract**: We introduce an approach for selecting objects in neural volumetric 3D
representations, such as multi-plane images (MPI) and neural radiance fields
(NeRF). Our approach takes a set of foreground and background 2D user scribbles
in one view and automatically estimates a 3D segmentation of the desired
object, which can be rendered into novel views. To achieve this result, we
propose a novel voxel feature embedding that incorporates the neural volumetric
3D representation and multi-view image features from all input views. To
evaluate our approach, we introduce a new dataset of human-provided
segmentation masks for depicted objects in real-world multi-view scene
captures. We show that our approach out-performs strong baselines, including 2D
segmentation and 3D segmentation approaches adapted to our task.

Comments:
- CVPR 2022 camera ready

---

## Compressible-composable NeRF via Rank-residual Decomposition

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-30 | Jiaxiang Tang, Xiaokang Chen, Jingbo Wang, Gang Zeng | cs.CV | [PDF](http://arxiv.org/pdf/2205.14870v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has emerged as a compelling method to represent
3D objects and scenes for photo-realistic rendering. However, its implicit
representation causes difficulty in manipulating the models like the explicit
mesh representation. Several recent advances in NeRF manipulation are usually
restricted by a shared renderer network, or suffer from large model size. To
circumvent the hurdle, in this paper, we present an explicit neural field
representation that enables efficient and convenient manipulation of models. To
achieve this goal, we learn a hybrid tensor rank decomposition of the scene
without neural networks. Motivated by the low-rank approximation property of
the SVD algorithm, we propose a rank-residual learning strategy to encourage
the preservation of primary information in lower ranks. The model size can then
be dynamically adjusted by rank truncation to control the levels of detail,
achieving near-optimal compression without extra optimization. Furthermore,
different models can be arbitrarily transformed and composed into one scene by
concatenating along the rank dimension. The growth of storage cost can also be
mitigated by compressing the unimportant objects in the composed scene. We
demonstrate that our method is able to achieve comparable rendering quality to
state-of-the-art methods, while enabling extra capability of compression and
composition. Code will be made available at https://github.com/ashawkey/CCNeRF.

Comments:
- NeurIPS 2022 camera-ready version

---

## V4D: Voxel for 4D Novel View Synthesis



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-28 | Wanshui Gan, Hongbin Xu, Yi Huang, Shifeng Chen, Naoto Yokoya | cs.CV | [PDF](http://arxiv.org/pdf/2205.14332v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields have made a remarkable breakthrough in the novel view
synthesis task at the 3D static scene. However, for the 4D circumstance (e.g.,
dynamic scene), the performance of the existing method is still limited by the
capacity of the neural network, typically in a multilayer perceptron network
(MLP). In this paper, we utilize 3D Voxel to model the 4D neural radiance
field, short as V4D, where the 3D voxel has two formats. The first one is to
regularly model the 3D space and then use the sampled local 3D feature with the
time index to model the density field and the texture field by a tiny MLP. The
second one is in look-up tables (LUTs) format that is for the pixel-level
refinement, where the pseudo-surface produced by the volume rendering is
utilized as the guidance information to learn a 2D pixel-level refinement
mapping. The proposed LUTs-based refinement module achieves the performance
gain with little computational cost and could serve as the plug-and-play module
in the novel view synthesis task. Moreover, we propose a more effective
conditional positional encoding toward the 4D data that achieves performance
gain with negligible computational burdens. Extensive experiments demonstrate
that the proposed method achieves state-of-the-art performance at a low
computational cost.

---

## Differentiable Point-Based Radiance Fields for Efficient View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-28 | Qiang Zhang, Seung-Hwan Baek, Szymon Rusinkiewicz, Felix Heide | cs.CV | [PDF](http://arxiv.org/pdf/2205.14330v4){: .btn .btn-green } |

**Abstract**: We propose a differentiable rendering algorithm for efficient novel view
synthesis. By departing from volume-based representations in favor of a learned
point representation, we improve on existing methods more than an order of
magnitude in memory and runtime, both in training and inference. The method
begins with a uniformly-sampled random point cloud and learns per-point
position and view-dependent appearance, using a differentiable splat-based
renderer to evolve the model to match a set of input images. Our method is up
to 300x faster than NeRF in both training and inference, with only a marginal
sacrifice in quality, while using less than 10~MB of memory for a static scene.
For dynamic scenes, our method trains two orders of magnitude faster than
STNeRF and renders at near interactive rate, while maintaining high image
quality and temporal coherence even without imposing any temporal-coherency
regularizers.

---

## PREF: Phasorial Embedding Fields for Compact Neural Representations



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-26 | Binbin Huang, Xinhao Yan, Anpei Chen, Shenghua Gao, Jingyi Yu | cs.CV | [PDF](http://arxiv.org/pdf/2205.13524v3){: .btn .btn-green } |

**Abstract**: We present an efficient frequency-based neural representation termed PREF: a
shallow MLP augmented with a phasor volume that covers significant border
spectra than previous Fourier feature mapping or Positional Encoding. At the
core is our compact 3D phasor volume where frequencies distribute uniformly
along a 2D plane and dilate along a 1D axis. To this end, we develop a tailored
and efficient Fourier transform that combines both Fast Fourier transform and
local interpolation to accelerate na\"ive Fourier mapping. We also introduce a
Parsvel regularizer that stables frequency-based learning. In these ways, Our
PREF reduces the costly MLP in the frequency-based representation, thereby
significantly closing the efficiency gap between it and other hybrid
representations, and improving its interpretability. Comprehensive experiments
demonstrate that our PREF is able to capture high-frequency details while
remaining compact and robust, including 2D image generalization, 3D signed
distance function regression and 5D neural radiance field reconstruction.

---

## StylizedNeRF: Consistent 3D Scene Stylization as Stylized NeRF via 2D-3D  Mutual Learning

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-24 | Yi-Hua Huang, Yue He, Yu-Jie Yuan, Yu-Kun Lai, Lin Gao | cs.GR | [PDF](http://arxiv.org/pdf/2205.12183v2){: .btn .btn-green } |

**Abstract**: 3D scene stylization aims at generating stylized images of the scene from
arbitrary novel views following a given set of style examples, while ensuring
consistency when rendered from different views. Directly applying methods for
image or video stylization to 3D scenes cannot achieve such consistency. Thanks
to recently proposed neural radiance fields (NeRF), we are able to represent a
3D scene in a consistent way. Consistent 3D scene stylization can be
effectively achieved by stylizing the corresponding NeRF. However, there is a
significant domain gap between style examples which are 2D images and NeRF
which is an implicit volumetric representation. To address this problem, we
propose a novel mutual learning framework for 3D scene stylization that
combines a 2D image stylization network and NeRF to fuse the stylization
ability of 2D stylization network with the 3D consistency of NeRF. We first
pre-train a standard NeRF of the 3D scene to be stylized and replace its color
prediction module with a style network to obtain a stylized NeRF. It is
followed by distilling the prior knowledge of spatial consistency from NeRF to
the 2D stylization network through an introduced consistency loss. We also
introduce a mimic loss to supervise the mutual learning of the NeRF style
module and fine-tune the 2D stylization decoder. In order to further make our
model handle ambiguities of 2D stylization results, we introduce learnable
latent codes that obey the probability distributions conditioned on the style.
They are attached to training samples as conditional inputs to better learn the
style module in our novel stylized NeRF. Experimental results demonstrate that
our method is superior to existing approaches in both visual quality and
long-range consistency.

Comments:
- Accepted by CVPR 2022

---

## Mip-NeRF RGB-D: Depth Assisted Fast Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-19 | Arnab Dey, Yassine Ahmine, Andrew I. Comport | cs.CV | [PDF](http://arxiv.org/pdf/2205.09351v3){: .btn .btn-green } |

**Abstract**: Neural scene representations, such as Neural Radiance Fields (NeRF), are
based on training a multilayer perceptron (MLP) using a set of color images
with known poses. An increasing number of devices now produce RGB-D(color +
depth) information, which has been shown to be very important for a wide range
of tasks. Therefore, the aim of this paper is to investigate what improvements
can be made to these promising implicit representations by incorporating depth
information with the color images. In particular, the recently proposed
Mip-NeRF approach, which uses conical frustums instead of rays for volume
rendering, allows one to account for the varying area of a pixel with distance
from the camera center. The proposed method additionally models depth
uncertainty. This allows to address major limitations of NeRF-based approaches
including improving the accuracy of geometry, reduced artifacts, faster
training time, and shortened prediction time. Experiments are performed on
well-known benchmark scenes, and comparisons show improved accuracy in scene
geometry and photometric reconstruction, while reducing the training time by 3
- 5 times.

---

## Fast Neural Network based Solving of Partial Differential Equations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-18 | Jaroslaw Rzepecki, Daniel Bates, Chris Doran | cs.LG | [PDF](http://arxiv.org/pdf/2205.08978v2){: .btn .btn-green } |

**Abstract**: We present a novel method for using Neural Networks (NNs) for finding
solutions to a class of Partial Differential Equations (PDEs). Our method
builds on recent advances in Neural Radiance Field research (NeRFs) and allows
for a NN to converge to a PDE solution much faster than classic Physically
Informed Neural Network (PINNs) approaches.

---

## RTMV: A Ray-Traced Multi-View Synthetic Dataset for Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-14 | Jonathan Tremblay, Moustafa Meshry, Alex Evans, Jan Kautz, Alexander Keller, Sameh Khamis, Thomas Müller, Charles Loop, Nathan Morrical, Koki Nagano, Towaki Takikawa, Stan Birchfield | cs.CV | [PDF](http://arxiv.org/pdf/2205.07058v2){: .btn .btn-green } |

**Abstract**: We present a large-scale synthetic dataset for novel view synthesis
consisting of ~300k images rendered from nearly 2000 complex scenes using
high-quality ray tracing at high resolution (1600 x 1600 pixels). The dataset
is orders of magnitude larger than existing synthetic datasets for novel view
synthesis, thus providing a large unified benchmark for both training and
evaluation. Using 4 distinct sources of high-quality 3D meshes, the scenes of
our dataset exhibit challenging variations in camera views, lighting, shape,
materials, and textures. Because our dataset is too large for existing methods
to process, we propose Sparse Voxel Light Field (SVLF), an efficient
voxel-based light field approach for novel view synthesis that achieves
comparable performance to NeRF on synthetic data, while being an order of
magnitude faster to train and two orders of magnitude faster to render. SVLF
achieves this speed by relying on a sparse voxel octree, careful voxel sampling
(requiring only a handful of queries per ray), and reduced network structure;
as well as ground truth depth maps at training time. Our dataset is generated
by NViSII, a Python-based ray tracing renderer, which is designed to be simple
for non-experts to use and share, flexible and powerful through its use of
scripting, and able to create high-quality and physically-based rendered
images. Experiments with a subset of our dataset allow us to compare standard
methods like NeRF and mip-NeRF for single-scene modeling, and pixelNeRF for
category-level modeling, pointing toward the need for future improvements in
this area.

Comments:
- ECCV 2022 Workshop on Learning to Generate 3D Shapes and Scenes.
  Project page at http://www.cs.umd.edu/~mmeshry/projects/rtmv

---

## Ray Priors through Reprojection: Improving Neural Radiance Fields for  Novel View Extrapolation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-12 | Jian Zhang, Yuanqing Zhang, Huan Fu, Xiaowei Zhou, Bowen Cai, Jinchi Huang, Rongfei Jia, Binqiang Zhao, Xing Tang | cs.CV | [PDF](http://arxiv.org/pdf/2205.05922v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have emerged as a potent paradigm for
representing scenes and synthesizing photo-realistic images. A main limitation
of conventional NeRFs is that they often fail to produce high-quality
renderings under novel viewpoints that are significantly different from the
training viewpoints. In this paper, instead of exploiting few-shot image
synthesis, we study the novel view extrapolation setting that (1) the training
images can well describe an object, and (2) there is a notable discrepancy
between the training and test viewpoints' distributions. We present RapNeRF
(RAy Priors) as a solution. Our insight is that the inherent appearances of a
3D surface's arbitrary visible projections should be consistent. We thus
propose a random ray casting policy that allows training unseen views using
seen views. Furthermore, we show that a ray atlas pre-computed from the
observed rays' viewing directions could further enhance the rendering quality
for extrapolated views. A main limitation is that RapNeRF would remove the
strong view-dependent effects because it leverages the multi-view consistency
property.

---

## View Synthesis with Sculpted Neural Points

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-12 | Yiming Zuo, Jia Deng | cs.CV | [PDF](http://arxiv.org/pdf/2205.05869v2){: .btn .btn-green } |

**Abstract**: We address the task of view synthesis, generating novel views of a scene
given a set of images as input. In many recent works such as NeRF (Mildenhall
et al., 2020), the scene geometry is parameterized using neural implicit
representations (i.e., MLPs). Implicit neural representations have achieved
impressive visual quality but have drawbacks in computational efficiency. In
this work, we propose a new approach that performs view synthesis using point
clouds. It is the first point-based method that achieves better visual quality
than NeRF while being 100x faster in rendering speed. Our approach builds on
existing works on differentiable point-based rendering but introduces a novel
technique we call "Sculpted Neural Points (SNP)", which significantly improves
the robustness to errors and holes in the reconstructed point cloud. We further
propose to use view-dependent point features based on spherical harmonics to
capture non-Lambertian surfaces, and new designs in the point-based rendering
pipeline that further boost the performance. Finally, we show that our system
supports fine-grained scene editing. Code is available at
https://github.com/princeton-vl/SNP.

---

## NeRF-Editing: Geometry Editing of Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-10 | Yu-Jie Yuan, Yang-Tian Sun, Yu-Kun Lai, Yuewen Ma, Rongfei Jia, Lin Gao | cs.GR | [PDF](http://arxiv.org/pdf/2205.04978v1){: .btn .btn-green } |

**Abstract**: Implicit neural rendering, especially Neural Radiance Field (NeRF), has shown
great potential in novel view synthesis of a scene. However, current NeRF-based
methods cannot enable users to perform user-controlled shape deformation in the
scene. While existing works have proposed some approaches to modify the
radiance field according to the user's constraints, the modification is limited
to color editing or object translation and rotation. In this paper, we propose
a method that allows users to perform controllable shape deformation on the
implicit representation of the scene, and synthesizes the novel view images of
the edited scene without re-training the network. Specifically, we establish a
correspondence between the extracted explicit mesh representation and the
implicit neural representation of the target scene. Users can first utilize
well-developed mesh-based deformation methods to deform the mesh representation
of the scene. Our method then utilizes user edits from the mesh representation
to bend the camera rays by introducing a tetrahedra mesh as a proxy, obtaining
the rendering results of the edited scene. Extensive experiments demonstrate
that our framework can achieve ideal editing results not only on synthetic
data, but also on real scenes captured by users.

Comments:
- Accepted by CVPR 2022

---

## Sampling-free obstacle gradients and reactive planning in Neural  Radiance Fields (NeRF)

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-05-03 | Michael Pantic, Cesar Cadena, Roland Siegwart, Lionel Ott | cs.RO | [PDF](http://arxiv.org/pdf/2205.01389v1){: .btn .btn-green } |

**Abstract**: This work investigates the use of Neural implicit representations,
specifically Neural Radiance Fields (NeRF), for geometrical queries and motion
planning. We show that by adding the capacity to infer occupancy in a radius to
a pre-trained NeRF, we are effectively learning an approximation to a Euclidean
Signed Distance Field (ESDF). Using backward differentiation of the augmented
network, we obtain an obstacle gradient that is integrated into an obstacle
avoidance policy based on the Riemannian Motion Policies (RMP) framework. Thus,
our findings allow for very fast sampling-free obstacle avoidance planning in
the implicit representation.

Comments:
- Accepted to the "Motion Planning with Implicit Neural Representations
  of Geometry" Workshop at ICRA 2022