---
layout: default
title: February
parent: 2023
nav_order: 2
---
<!---metadata--->

## Dynamic Multi-View Scene Reconstruction Using Neural Implicit Surface



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-28 | Decai Chen, Haofei Lu, Ingo Feldmann, Oliver Schreer, Peter Eisert | cs.CV | [PDF](http://arxiv.org/pdf/2303.00050v1){: .btn .btn-green } |

**Abstract**: Reconstructing general dynamic scenes is important for many computer vision
and graphics applications. Recent works represent the dynamic scene with neural
radiance fields for photorealistic view synthesis, while their surface geometry
is under-constrained and noisy. Other works introduce surface constraints to
the implicit neural representation to disentangle the ambiguity of geometry and
appearance field for static scene reconstruction. To bridge the gap between
rendering dynamic scenes and recovering static surface geometry, we propose a
template-free method to reconstruct surface geometry and appearance using
neural implicit representations from multi-view videos. We leverage
topology-aware deformation and the signed distance field to learn complex
dynamic surfaces via differentiable volume rendering without scene-specific
prior knowledge like template models. Furthermore, we propose a novel
mask-based ray selection strategy to significantly boost the optimization on
challenging time-varying regions. Experiments on different multi-view video
datasets demonstrate that our method achieves high-fidelity surface
reconstruction as well as photorealistic novel view synthesis.

Comments:
- 5 pages, accepted by ICASSP 2023

---

## IntrinsicNGP: Intrinsic Coordinate based Hash Encoding for Human NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-28 | Bo Peng, Jun Hu, Jingtao Zhou, Xuan Gao, Juyong Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2302.14683v2){: .btn .btn-green } |

**Abstract**: Recently, many works have been proposed to utilize the neural radiance field
for novel view synthesis of human performers. However, most of these methods
require hours of training, making them difficult for practical use. To address
this challenging problem, we propose IntrinsicNGP, which can train from scratch
and achieve high-fidelity results in few minutes with videos of a human
performer. To achieve this target, we introduce a continuous and optimizable
intrinsic coordinate rather than the original explicit Euclidean coordinate in
the hash encoding module of instant-NGP. With this novel intrinsic coordinate,
IntrinsicNGP can aggregate inter-frame information for dynamic objects with the
help of proxy geometry shapes. Moreover, the results trained with the given
rough geometry shapes can be further refined with an optimizable offset field
based on the intrinsic coordinate.Extensive experimental results on several
datasets demonstrate the effectiveness and efficiency of IntrinsicNGP. We also
illustrate our approach's ability to edit the shape of reconstructed subjects.

Comments:
- Project page:https://ustc3dv.github.io/IntrinsicNGP/. arXiv admin
  note: substantial text overlap with arXiv:2210.01651

---

## BLiRF: Bandlimited Radiance Fields for Dynamic Scene Modeling

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-27 | Sameera Ramasinghe, Violetta Shevchenko, Gil Avraham, Anton Van Den Hengel | cs.CV | [PDF](http://arxiv.org/pdf/2302.13543v3){: .btn .btn-green } |

**Abstract**: Reasoning the 3D structure of a non-rigid dynamic scene from a single moving
camera is an under-constrained problem. Inspired by the remarkable progress of
neural radiance fields (NeRFs) in photo-realistic novel view synthesis of
static scenes, extensions have been proposed for dynamic settings. These
methods heavily rely on neural priors in order to regularize the problem. In
this work, we take a step back and reinvestigate how current implementations
may entail deleterious effects, including limited expressiveness, entanglement
of light and density fields, and sub-optimal motion localization. As a remedy,
we advocate for a bridge between classic non-rigid-structure-from-motion
(\nrsfm) and NeRF, enabling the well-studied priors of the former to constrain
the latter. To this end, we propose a framework that factorizes time and space
by formulating a scene as a composition of bandlimited, high-dimensional
signals. We demonstrate compelling results across complex dynamic scenes that
involve changes in lighting, texture and long-range dynamics.

---

## Efficient physics-informed neural networks using hash encoding

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-26 | Xinquan Huang, Tariq Alkhalifah | cs.LG | [PDF](http://arxiv.org/pdf/2302.13397v1){: .btn .btn-green } |

**Abstract**: Physics-informed neural networks (PINNs) have attracted a lot of attention in
scientific computing as their functional representation of partial differential
equation (PDE) solutions offers flexibility and accuracy features. However,
their training cost has limited their practical use as a real alternative to
classic numerical methods. Thus, we propose to incorporate multi-resolution
hash encoding into PINNs to improve the training efficiency, as such encoding
offers a locally-aware (at multi resolution) coordinate inputs to the neural
network. Borrowed from the neural representation field community (NeRF), we
investigate the robustness of calculating the derivatives of such hash encoded
neural networks with respect to the input coordinates, which is often needed by
the PINN loss terms. We propose to replace the automatic differentiation with
finite-difference calculations of the derivatives to address the discontinuous
nature of such derivatives. We also share the appropriate ranges for the hash
encoding hyperparameters to obtain robust derivatives. We test the proposed
method on three problems, including Burgers equation, Helmholtz equation, and
Navier-Stokes equation. The proposed method admits about a 10-fold improvement
in efficiency over the vanilla PINN implementation.

---

## CATNIPS: Collision Avoidance Through Neural Implicit Probabilistic  Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-24 | Timothy Chen, Preston Culbertson, Mac Schwager | cs.RO | [PDF](http://arxiv.org/pdf/2302.12931v2){: .btn .btn-green } |

**Abstract**: We introduce a transformation of a Neural Radiance Field (NeRF) to an
equivalent Poisson Point Process (PPP). This PPP transformation allows for
rigorous quantification of uncertainty in NeRFs, in particular, for computing
collision probabilities for a robot navigating through a NeRF environment. The
PPP is a generalization of a probabilistic occupancy grid to the continuous
volume and is fundamental to the volumetric ray-tracing model underlying
radiance fields. Building upon this PPP representation, we present a
chance-constrained trajectory optimization method for safe robot navigation in
NeRFs. Our method relies on a voxel representation called the Probabilistic
Unsafe Robot Region (PURR) that spatially fuses the chance constraint with the
NeRF model to facilitate fast trajectory optimization. We then combine a
graph-based search with a spline-based trajectory optimization to yield robot
trajectories through the NeRF that are guaranteed to satisfy a user-specific
collision probability. We validate our chance constrained planning method
through simulations and hardware experiments, showing superior performance
compared to prior works on trajectory planning in NeRF environments.

Comments:
- Under Review in IEEE Transactions on Robotics

---

## DiffusioNeRF: Regularizing Neural Radiance Fields with Denoising  Diffusion Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-23 | Jamie Wynn, Daniyar Turmukhambetov | cs.CV | [PDF](http://arxiv.org/pdf/2302.12231v3){: .btn .btn-green } |

**Abstract**: Under good conditions, Neural Radiance Fields (NeRFs) have shown impressive
results on novel view synthesis tasks. NeRFs learn a scene's color and density
fields by minimizing the photometric discrepancy between training views and
differentiable renderings of the scene. Once trained from a sufficient set of
views, NeRFs can generate novel views from arbitrary camera positions. However,
the scene geometry and color fields are severely under-constrained, which can
lead to artifacts, especially when trained with few input views.
  To alleviate this problem we learn a prior over scene geometry and color,
using a denoising diffusion model (DDM). Our DDM is trained on RGBD patches of
the synthetic Hypersim dataset and can be used to predict the gradient of the
logarithm of a joint probability distribution of color and depth patches. We
show that, these gradients of logarithms of RGBD patch priors serve to
regularize geometry and color of a scene. During NeRF training, random RGBD
patches are rendered and the estimated gradient of the log-likelihood is
backpropagated to the color and density fields. Evaluations on LLFF, the most
relevant dataset, show that our learned prior achieves improved quality in the
reconstructed geometry and improved generalization to novel views. Evaluations
on DTU show improved reconstruction quality among NeRF methods.

Comments:
- CVPR 2023. Updated LPIPS scores in Table 1

---

## MERF: Memory-Efficient Radiance Fields for Real-time View Synthesis in  Unbounded Scenes



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-23 | Christian Reiser, Richard Szeliski, Dor Verbin, Pratul P. Srinivasan, Ben Mildenhall, Andreas Geiger, Jonathan T. Barron, Peter Hedman | cs.CV | [PDF](http://arxiv.org/pdf/2302.12249v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields enable state-of-the-art photorealistic view synthesis.
However, existing radiance field representations are either too
compute-intensive for real-time rendering or require too much memory to scale
to large scenes. We present a Memory-Efficient Radiance Field (MERF)
representation that achieves real-time rendering of large-scale scenes in a
browser. MERF reduces the memory consumption of prior sparse volumetric
radiance fields using a combination of a sparse feature grid and
high-resolution 2D feature planes. To support large-scale unbounded scenes, we
introduce a novel contraction function that maps scene coordinates into a
bounded volume while still allowing for efficient ray-box intersection. We
design a lossless procedure for baking the parameterization used during
training into a model that achieves real-time rendering while still preserving
the photorealistic view synthesis quality of a volumetric radiance field.

Comments:
- Video and interactive web demo available at https://merf42.github.io

---

## Learning Neural Volumetric Representations of Dynamic Humans in Minutes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-23 | Chen Geng, Sida Peng, Zhen Xu, Hujun Bao, Xiaowei Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2302.12237v2){: .btn .btn-green } |

**Abstract**: This paper addresses the challenge of quickly reconstructing free-viewpoint
videos of dynamic humans from sparse multi-view videos. Some recent works
represent the dynamic human as a canonical neural radiance field (NeRF) and a
motion field, which are learned from videos through differentiable rendering.
But the per-scene optimization generally requires hours. Other generalizable
NeRF models leverage learned prior from datasets and reduce the optimization
time by only finetuning on new scenes at the cost of visual fidelity. In this
paper, we propose a novel method for learning neural volumetric videos of
dynamic humans from sparse view videos in minutes with competitive visual
quality. Specifically, we define a novel part-based voxelized human
representation to better distribute the representational power of the network
to different human parts. Furthermore, we propose a novel 2D motion
parameterization scheme to increase the convergence rate of deformation field
learning. Experiments demonstrate that our model can be learned 100 times
faster than prior per-scene optimization methods while being competitive in the
rendering quality. Training our model on a $512 \times 512$ video with 100
frames typically takes about 5 minutes on a single RTX 3090 GPU. The code will
be released on our project page: https://zju3dv.github.io/instant_nvr

Comments:
- Project page: https://zju3dv.github.io/instant_nvr

---

## Differentiable Rendering with Reparameterized Volume Sampling

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-21 | Nikita Morozov, Denis Rakitin, Oleg Desheulin, Dmitry Vetrov, Kirill Struminsky | cs.CV | [PDF](http://arxiv.org/pdf/2302.10970v2){: .btn .btn-green } |

**Abstract**: In view synthesis, a neural radiance field approximates underlying density
and radiance fields based on a sparse set of scene pictures. To generate a
pixel of a novel view, it marches a ray through the pixel and computes a
weighted sum of radiance emitted from a dense set of ray points. This rendering
algorithm is fully differentiable and facilitates gradient-based optimization
of the fields. However, in practice, only a tiny opaque portion of the ray
contributes most of the radiance to the sum. We propose a simple end-to-end
differentiable sampling algorithm based on inverse transform sampling. It
generates samples according to the probability distribution induced by the
density field and picks non-transparent points on the ray. We utilize the
algorithm in two ways. First, we propose a novel rendering approach based on
Monte Carlo estimates. This approach allows for evaluating and optimizing a
neural radiance field with just a few radiance field calls per ray. Second, we
use the sampling algorithm to modify the hierarchical scheme proposed in the
original NeRF work. We show that our modification improves reconstruction
quality of hierarchical models, at the same time simplifying the training
procedure by removing the need for auxiliary proposal network losses.

Comments:
- Preprint

---

## RealFusion: 360° Reconstruction of Any Object from a Single Image



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-21 | Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, Andrea Vedaldi | cs.CV | [PDF](http://arxiv.org/pdf/2302.10663v2){: .btn .btn-green } |

**Abstract**: We consider the problem of reconstructing a full 360{\deg} photographic model
of an object from a single image of it. We do so by fitting a neural radiance
field to the image, but find this problem to be severely ill-posed. We thus
take an off-the-self conditional image generator based on diffusion and
engineer a prompt that encourages it to "dream up" novel views of the object.
Using an approach inspired by DreamFields and DreamFusion, we fuse the given
input view, the conditional prior, and other regularizers in a final,
consistent reconstruction. We demonstrate state-of-the-art reconstruction
results on benchmark images when compared to prior methods for monocular 3D
reconstruction of objects. Qualitatively, our reconstructions provide a
faithful match of the input view and a plausible extrapolation of its
appearance and 3D shape, including to the side of the object not visible in the
image.

Comments:
- Project page: https://lukemelas.github.io/realfusion

---

## USR: Unsupervised Separated 3D Garment and Human Reconstruction via  Geometry and Semantic Consistency



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-21 | Yue Shi, Yuxuan Xiong, Jingyi Chai, Bingbing Ni, Wenjun Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2302.10518v3){: .btn .btn-green } |

**Abstract**: Dressed people reconstruction from images is a popular task with promising
applications in the creative media and game industry. However, most existing
methods reconstruct the human body and garments as a whole with the supervision
of 3D models, which hinders the downstream interaction tasks and requires
hard-to-obtain data. To address these issues, we propose an unsupervised
separated 3D garments and human reconstruction model (USR), which reconstructs
the human body and authentic textured clothes in layers without 3D models. More
specifically, our method proposes a generalized surface-aware neural radiance
field to learn the mapping between sparse multi-view images and geometries of
the dressed people. Based on the full geometry, we introduce a Semantic and
Confidence Guided Separation strategy (SCGS) to detect, segment, and
reconstruct the clothes layer, leveraging the consistency between 2D semantic
and 3D geometry. Moreover, we propose a Geometry Fine-tune Module to smooth
edges. Extensive experiments on our dataset show that comparing with
state-of-the-art methods, USR achieves improvements on both geometry and
appearance reconstruction while supporting generalizing to unseen people in
real time. Besides, we also introduce SMPL-D model to show the benefit of the
separated modeling of clothes and the human body that allows swapping clothes
and virtual try-on.

---

## NerfDiff: Single-image View Synthesis with NeRF-guided Distillation from  3D-aware Diffusion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-20 | Jiatao Gu, Alex Trevithick, Kai-En Lin, Josh Susskind, Christian Theobalt, Lingjie Liu, Ravi Ramamoorthi | cs.CV | [PDF](http://arxiv.org/pdf/2302.10109v1){: .btn .btn-green } |

**Abstract**: Novel view synthesis from a single image requires inferring occluded regions
of objects and scenes whilst simultaneously maintaining semantic and physical
consistency with the input. Existing approaches condition neural radiance
fields (NeRF) on local image features, projecting points to the input image
plane, and aggregating 2D features to perform volume rendering. However, under
severe occlusion, this projection fails to resolve uncertainty, resulting in
blurry renderings that lack details. In this work, we propose NerfDiff, which
addresses this issue by distilling the knowledge of a 3D-aware conditional
diffusion model (CDM) into NeRF through synthesizing and refining a set of
virtual views at test time. We further propose a novel NeRF-guided distillation
algorithm that simultaneously generates 3D consistent virtual views from the
CDM samples, and finetunes the NeRF based on the improved virtual views. Our
approach significantly outperforms existing NeRF-based and geometry-free
approaches on challenging datasets, including ShapeNet, ABO, and Clevr3D.

Comments:
- Project page: https://jiataogu.me/nerfdiff/

---

## LC-NeRF: Local Controllable Face Generation in Neural Randiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-19 | Wenyang Zhou, Lu Yuan, Shuyu Chen, Lin Gao, Shimin Hu | cs.CV | [PDF](http://arxiv.org/pdf/2302.09486v1){: .btn .btn-green } |

**Abstract**: 3D face generation has achieved high visual quality and 3D consistency thanks
to the development of neural radiance fields (NeRF). Recently, to generate and
edit 3D faces with NeRF representation, some methods are proposed and achieve
good results in decoupling geometry and texture. The latent codes of these
generative models affect the whole face, and hence modifications to these codes
cause the entire face to change. However, users usually edit a local region
when editing faces and do not want other regions to be affected. Since changes
to the latent code affect global generation results, these methods do not allow
for fine-grained control of local facial regions. To improve local
controllability in NeRF-based face editing, we propose LC-NeRF, which is
composed of a Local Region Generators Module and a Spatial-Aware Fusion Module,
allowing for local geometry and texture control of local facial regions.
Qualitative and quantitative evaluations show that our method provides better
local editing than state-of-the-art face editing methods. Our method also
performs well in downstream tasks, such as text-driven facial image editing.

---

## Temporal Interpolation Is All You Need for Dynamic Neural Radiance  Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-18 | Sungheon Park, Minjung Son, Seokhwan Jang, Young Chun Ahn, Ji-Yeon Kim, Nahyup Kang | cs.CV | [PDF](http://arxiv.org/pdf/2302.09311v2){: .btn .btn-green } |

**Abstract**: Temporal interpolation often plays a crucial role to learn meaningful
representations in dynamic scenes. In this paper, we propose a novel method to
train spatiotemporal neural radiance fields of dynamic scenes based on temporal
interpolation of feature vectors. Two feature interpolation methods are
suggested depending on underlying representations, neural networks or grids. In
the neural representation, we extract features from space-time inputs via
multiple neural network modules and interpolate them based on time frames. The
proposed multi-level feature interpolation network effectively captures
features of both short-term and long-term time ranges. In the grid
representation, space-time features are learned via four-dimensional hash
grids, which remarkably reduces training time. The grid representation shows
more than 100 times faster training speed than the previous neural-net-based
methods while maintaining the rendering quality. Concatenating static and
dynamic features and adding a simple smoothness term further improve the
performance of our proposed models. Despite the simplicity of the model
architectures, our method achieved state-of-the-art performance both in
rendering quality for the neural representation and in training speed for the
grid representation.

Comments:
- CVPR 2023. Project page:
  https://sungheonpark.github.io/tempinterpnerf

---

## MixNeRF: Modeling a Ray with Mixture Density for Novel View Synthesis  from Sparse Inputs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-17 | Seunghyeon Seo, Donghoon Han, Yeonjin Chang, Nojun Kwak | cs.CV | [PDF](http://arxiv.org/pdf/2302.08788v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has broken new ground in the novel view
synthesis due to its simple concept and state-of-the-art quality. However, it
suffers from severe performance degradation unless trained with a dense set of
images with different camera poses, which hinders its practical applications.
Although previous methods addressing this problem achieved promising results,
they relied heavily on the additional training resources, which goes against
the philosophy of sparse-input novel-view synthesis pursuing the training
efficiency. In this work, we propose MixNeRF, an effective training strategy
for novel view synthesis from sparse inputs by modeling a ray with a mixture
density model. Our MixNeRF estimates the joint distribution of RGB colors along
the ray samples by modeling it with mixture of distributions. We also propose a
new task of ray depth estimation as a useful training objective, which is
highly correlated with 3D scene geometry. Moreover, we remodel the colors with
regenerated blending weights based on the estimated ray depth and further
improves the robustness for colors and viewpoints. Our MixNeRF outperforms
other state-of-the-art methods in various standard benchmarks with superior
efficiency of training and inference.

Comments:
- CVPR 2023. Project Page: https://shawn615.github.io/mixnerf/

---

## 3D-aware Conditional Image Synthesis



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-16 | Kangle Deng, Gengshan Yang, Deva Ramanan, Jun-Yan Zhu | cs.CV | [PDF](http://arxiv.org/pdf/2302.08509v2){: .btn .btn-green } |

**Abstract**: We propose pix2pix3D, a 3D-aware conditional generative model for
controllable photorealistic image synthesis. Given a 2D label map, such as a
segmentation or edge map, our model learns to synthesize a corresponding image
from different viewpoints. To enable explicit 3D user control, we extend
conditional generative models with neural radiance fields. Given
widely-available monocular images and label map pairs, our model learns to
assign a label to every 3D point in addition to color and density, which
enables it to render the image and pixel-aligned label map simultaneously.
Finally, we build an interactive system that allows users to edit the label map
from any viewpoint and generate outputs accordingly.

Comments:
- Project Page: https://www.cs.cmu.edu/~pix2pix3D/

---

## LiveHand: Real-time and Photorealistic Neural Hand Rendering



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-15 | Akshay Mundra, Mallikarjun B R, Jiayi Wang, Marc Habermann, Christian Theobalt, Mohamed Elgharib | cs.GR | [PDF](http://arxiv.org/pdf/2302.07672v3){: .btn .btn-green } |

**Abstract**: The human hand is the main medium through which we interact with our
surroundings, making its digitization an important problem. While there are
several works modeling the geometry of hands, little attention has been paid to
capturing photo-realistic appearance. Moreover, for applications in extended
reality and gaming, real-time rendering is critical. We present the first
neural-implicit approach to photo-realistically render hands in real-time. This
is a challenging problem as hands are textured and undergo strong articulations
with pose-dependent effects. However, we show that this aim is achievable
through our carefully designed method. This includes training on a
low-resolution rendering of a neural radiance field, together with a
3D-consistent super-resolution module and mesh-guided sampling and space
canonicalization. We demonstrate a novel application of perceptual loss on the
image space, which is critical for learning details accurately. We also show a
live demo where we photo-realistically render the human hand in real-time for
the first time, while also modeling pose- and view-dependent appearance
effects. We ablate all our design choices and show that they optimize for
rendering speed and quality. Video results and our code can be accessed from
https://vcai.mpi-inf.mpg.de/projects/LiveHand/

Comments:
- Project page: https://vcai.mpi-inf.mpg.de/projects/LiveHand/ |
  Accepted at ICCV '23 | 11 pages, 7 figures

---

## VQ3D: Learning a 3D-Aware Generative Model on ImageNet

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-14 | Kyle Sargent, Jing Yu Koh, Han Zhang, Huiwen Chang, Charles Herrmann, Pratul Srinivasan, Jiajun Wu, Deqing Sun | cs.CV | [PDF](http://arxiv.org/pdf/2302.06833v1){: .btn .btn-green } |

**Abstract**: Recent work has shown the possibility of training generative models of 3D
content from 2D image collections on small datasets corresponding to a single
object class, such as human faces, animal faces, or cars. However, these models
struggle on larger, more complex datasets. To model diverse and unconstrained
image collections such as ImageNet, we present VQ3D, which introduces a
NeRF-based decoder into a two-stage vector-quantized autoencoder. Our Stage 1
allows for the reconstruction of an input image and the ability to change the
camera position around the image, and our Stage 2 allows for the generation of
new 3D scenes. VQ3D is capable of generating and reconstructing 3D-aware images
from the 1000-class ImageNet dataset of 1.2 million training images. We achieve
an ImageNet generation FID score of 16.8, compared to 69.8 for the next best
baseline method.

Comments:
- 15 pages. For visual results, please visit the project webpage at
  http://kylesargent.github.io/vq3d

---

## 3D-aware Blending with Generative NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-13 | Hyunsu Kim, Gayoung Lee, Yunjey Choi, Jin-Hwa Kim, Jun-Yan Zhu | cs.CV | [PDF](http://arxiv.org/pdf/2302.06608v3){: .btn .btn-green } |

**Abstract**: Image blending aims to combine multiple images seamlessly. It remains
challenging for existing 2D-based methods, especially when input images are
misaligned due to differences in 3D camera poses and object shapes. To tackle
these issues, we propose a 3D-aware blending method using generative Neural
Radiance Fields (NeRF), including two key components: 3D-aware alignment and
3D-aware blending. For 3D-aware alignment, we first estimate the camera pose of
the reference image with respect to generative NeRFs and then perform 3D local
alignment for each part. To further leverage 3D information of the generative
NeRF, we propose 3D-aware blending that directly blends images on the NeRF's
latent representation space, rather than raw pixel space. Collectively, our
method outperforms existing 2D baselines, as validated by extensive
quantitative and qualitative evaluations with FFHQ and AFHQ-Cat.

Comments:
- ICCV 2023, Project page: https://blandocs.github.io/blendnerf

---

## 3D Colored Shape Reconstruction from a Single RGB Image through  Diffusion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-11 | Bo Li, Xiaolin Wei, Fengwei Chen, Bin Liu | cs.CV | [PDF](http://arxiv.org/pdf/2302.05573v1){: .btn .btn-green } |

**Abstract**: We propose a novel 3d colored shape reconstruction method from a single RGB
image through diffusion model. Diffusion models have shown great development
potentials for high-quality 3D shape generation. However, most existing work
based on diffusion models only focus on geometric shape generation, they cannot
either accomplish 3D reconstruction from a single image, or produce 3D
geometric shape with color information. In this work, we propose to reconstruct
a 3D colored shape from a single RGB image through a novel conditional
diffusion model. The reverse process of the proposed diffusion model is
consisted of three modules, shape prediction module, color prediction module
and NeRF-like rendering module. In shape prediction module, the reference RGB
image is first encoded into a high-level shape feature and then the shape
feature is utilized as a condition to predict the reverse geometric noise in
diffusion model. Then the color of each 3D point updated in shape prediction
module is predicted by color prediction module. Finally, a NeRF-like rendering
module is designed to render the colored point cloud predicted by the former
two modules to 2D image space to guide the training conditioned only on a
reference image. As far as the authors know, the proposed method is the first
diffusion model for 3D colored shape reconstruction from a single RGB image.
Experimental results demonstrate that the proposed method achieves competitive
performance on colored 3D shape reconstruction, and the ablation study
validates the positive role of the color prediction module in improving the
reconstruction quality of 3D geometric point cloud.

Comments:
- 9 pages, 8 figures

---

## In-N-Out: Faithful 3D GAN Inversion with Volumetric Decomposition for  Face Editing



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-09 | Yiran Xu, Zhixin Shu, Cameron Smith, Seoung Wug Oh, Jia-Bin Huang | cs.CV | [PDF](http://arxiv.org/pdf/2302.04871v3){: .btn .btn-green } |

**Abstract**: 3D-aware GANs offer new capabilities for view synthesis while preserving the
editing functionalities of their 2D counterparts. GAN inversion is a crucial
step that seeks the latent code to reconstruct input images or videos,
subsequently enabling diverse editing tasks through manipulation of this latent
code. However, a model pre-trained on a particular dataset (e.g., FFHQ) often
has difficulty reconstructing images with out-of-distribution (OOD) objects
such as faces with heavy make-up or occluding objects. We address this issue by
explicitly modeling OOD objects from the input in 3D-aware GANs. Our core idea
is to represent the image using two individual neural radiance fields: one for
the in-distribution content and the other for the out-of-distribution object.
The final reconstruction is achieved by optimizing the composition of these two
radiance fields with carefully designed regularization. We demonstrate that our
explicit decomposition alleviates the inherent trade-off between reconstruction
fidelity and editability. We evaluate reconstruction accuracy and editability
of our method on challenging real face images and videos and showcase favorable
results against other baselines.

Comments:
- Project page: https://in-n-out-3d.github.io/

---

## Nerfstudio: A Modular Framework for Neural Radiance Field Development

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-08 | Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Justin Kerr, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, David McAllister, Angjoo Kanazawa | cs.CV | [PDF](http://arxiv.org/pdf/2302.04264v4){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) are a rapidly growing area of research with
wide-ranging applications in computer vision, graphics, robotics, and more. In
order to streamline the development and deployment of NeRF research, we propose
a modular PyTorch framework, Nerfstudio. Our framework includes plug-and-play
components for implementing NeRF-based methods, which make it easy for
researchers and practitioners to incorporate NeRF into their projects.
Additionally, the modular design enables support for extensive real-time
visualization tools, streamlined pipelines for importing captured in-the-wild
data, and tools for exporting to video, point cloud and mesh representations.
The modularity of Nerfstudio enables the development of Nerfacto, our method
that combines components from recent papers to achieve a balance between speed
and quality, while also remaining flexible to future modifications. To promote
community-driven development, all associated code and data are made publicly
available with open-source licensing at https://nerf.studio.

Comments:
- Project page at https://nerf.studio

---

## AV-NeRF: Learning Neural Fields for Real-World Audio-Visual Scene  Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-04 | Susan Liang, Chao Huang, Yapeng Tian, Anurag Kumar, Chenliang Xu | cs.CV | [PDF](http://arxiv.org/pdf/2302.02088v3){: .btn .btn-green } |

**Abstract**: Can machines recording an audio-visual scene produce realistic, matching
audio-visual experiences at novel positions and novel view directions? We
answer it by studying a new task -- real-world audio-visual scene synthesis --
and a first-of-its-kind NeRF-based approach for multimodal learning.
Concretely, given a video recording of an audio-visual scene, the task is to
synthesize new videos with spatial audios along arbitrary novel camera
trajectories in that scene. We propose an acoustic-aware audio generation
module that integrates prior knowledge of audio propagation into NeRF, in which
we implicitly associate audio generation with the 3D geometry and material
properties of a visual environment. Furthermore, we present a coordinate
transformation module that expresses a view direction relative to the sound
source, enabling the model to learn sound source-centric acoustic fields. To
facilitate the study of this new task, we collect a high-quality Real-World
Audio-Visual Scene (RWAVS) dataset. We demonstrate the advantages of our method
on this real-world dataset and the simulation-based SoundSpaces dataset.

Comments:
- NeurIPS 2023

---

## Robust Camera Pose Refinement for Multi-Resolution Hash Encoding

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-03 | Hwan Heo, Taekyung Kim, Jiyoung Lee, Jaewon Lee, Soohyun Kim, Hyunwoo J. Kim, Jin-Hwa Kim | cs.CV | [PDF](http://arxiv.org/pdf/2302.01571v1){: .btn .btn-green } |

**Abstract**: Multi-resolution hash encoding has recently been proposed to reduce the
computational cost of neural renderings, such as NeRF. This method requires
accurate camera poses for the neural renderings of given scenes. However,
contrary to previous methods jointly optimizing camera poses and 3D scenes, the
naive gradient-based camera pose refinement method using multi-resolution hash
encoding severely deteriorates performance. We propose a joint optimization
algorithm to calibrate the camera pose and learn a geometric representation
using efficient multi-resolution hash encoding. Showing that the oscillating
gradient flows of hash encoding interfere with the registration of camera
poses, our method addresses the issue by utilizing smooth interpolation
weighting to stabilize the gradient oscillation for the ray samplings across
hash grids. Moreover, the curriculum training procedure helps to learn the
level-wise hash encoding, further increasing the pose refinement. Experiments
on the novel-view synthesis datasets validate that our learning frameworks
achieve state-of-the-art performance and rapid convergence of neural rendering,
even when initial camera poses are unknown.

---

## INV: Towards Streaming Incremental Neural Videos

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-03 | Shengze Wang, Alexey Supikov, Joshua Ratcliff, Henry Fuchs, Ronald Azuma | cs.CV | [PDF](http://arxiv.org/pdf/2302.01532v1){: .btn .btn-green } |

**Abstract**: Recent works in spatiotemporal radiance fields can produce photorealistic
free-viewpoint videos. However, they are inherently unsuitable for interactive
streaming scenarios (e.g. video conferencing, telepresence) because have an
inevitable lag even if the training is instantaneous. This is because these
approaches consume videos and thus have to buffer chunks of frames (often
seconds) before processing. In this work, we take a step towards interactive
streaming via a frame-by-frame approach naturally free of lag. Conventional
wisdom believes that per-frame NeRFs are impractical due to prohibitive
training costs and storage. We break this belief by introducing Incremental
Neural Videos (INV), a per-frame NeRF that is efficiently trained and
streamable. We designed INV based on two insights: (1) Our main finding is that
MLPs naturally partition themselves into Structure and Color Layers, which
store structural and color/texture information respectively. (2) We leverage
this property to retain and improve upon knowledge from previous frames, thus
amortizing training across frames and reducing redundant learning. As a result,
with negligible changes to NeRF, INV can achieve good qualities (>28.6db) in
8min/frame. It can also outperform prior SOTA in 19% less training time.
Additionally, our Temporal Weight Compression reduces the per-frame size to
0.3MB/frame (6.6% of NeRF). More importantly, INV is free from buffer lag and
is naturally fit for streaming. While this work does not achieve real-time
training, it shows that incremental approaches like INV present new
possibilities in interactive 3D streaming. Moreover, our discovery of natural
information partition leads to a better understanding and manipulation of MLPs.
Code and dataset will be released soon.

---

## Semantic 3D-aware Portrait Synthesis and Manipulation Based on  Compositional Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-03 | Tianxiang Ma, Bingchuan Li, Qian He, Jing Dong, Tieniu Tan | cs.CV | [PDF](http://arxiv.org/pdf/2302.01579v2){: .btn .btn-green } |

**Abstract**: Recently 3D-aware GAN methods with neural radiance field have developed
rapidly. However, current methods model the whole image as an overall neural
radiance field, which limits the partial semantic editability of synthetic
results. Since NeRF renders an image pixel by pixel, it is possible to split
NeRF in the spatial dimension. We propose a Compositional Neural Radiance Field
(CNeRF) for semantic 3D-aware portrait synthesis and manipulation. CNeRF
divides the image by semantic regions and learns an independent neural radiance
field for each region, and finally fuses them and renders the complete image.
Thus we can manipulate the synthesized semantic regions independently, while
fixing the other parts unchanged. Furthermore, CNeRF is also designed to
decouple shape and texture within each semantic region. Compared to
state-of-the-art 3D-aware GAN methods, our approach enables fine-grained
semantic region manipulation, while maintaining high-quality 3D-consistent
synthesis. The ablation studies show the effectiveness of the structure and
loss function used by our method. In addition real image inversion and cartoon
portrait 3D editing experiments demonstrate the application potential of our
method.

Comments:
- Accepted by AAAI2023 Oral

---

## RobustNeRF: Ignoring Distractors with Robust Losses

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-02 | Sara Sabour, Suhani Vora, Daniel Duckworth, Ivan Krasin, David J. Fleet, Andrea Tagliasacchi | cs.CV | [PDF](http://arxiv.org/pdf/2302.00833v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) excel at synthesizing new views given
multi-view, calibrated images of a static scene. When scenes include
distractors, which are not persistent during image capture (moving objects,
lighting variations, shadows), artifacts appear as view-dependent effects or
'floaters'. To cope with distractors, we advocate a form of robust estimation
for NeRF training, modeling distractors in training data as outliers of an
optimization problem. Our method successfully removes outliers from a scene and
improves upon our baselines, on synthetic and real-world scenes. Our technique
is simple to incorporate in modern NeRF frameworks, with few hyper-parameters.
It does not assume a priori knowledge of the types of distractors, and is
instead focused on the optimization problem rather than pre-processing or
modeling transient objects. More results on our page
https://robustnerf.github.io/public.

---

## Factor Fields: A Unified Framework for Neural Fields and Beyond

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-02-02 | Anpei Chen, Zexiang Xu, Xinyue Wei, Siyu Tang, Hao Su, Andreas Geiger | cs.CV | [PDF](http://arxiv.org/pdf/2302.01226v3){: .btn .btn-green } |

**Abstract**: We present Factor Fields, a novel framework for modeling and representing
signals. Factor Fields decomposes a signal into a product of factors, each
represented by a classical or neural field representation which operates on
transformed input coordinates. This decomposition results in a unified
framework that accommodates several recent signal representations including
NeRF, Plenoxels, EG3D, Instant-NGP, and TensoRF. Additionally, our framework
allows for the creation of powerful new signal representations, such as the
"Dictionary Field" (DiF) which is a second contribution of this paper. Our
experiments show that DiF leads to improvements in approximation quality,
compactness, and training time when compared to previous fast reconstruction
methods. Experimentally, our representation achieves better image approximation
quality on 2D image regression tasks, higher geometric quality when
reconstructing 3D signed distance fields, and higher compactness for radiance
field reconstruction tasks. Furthermore, DiF enables generalization to unseen
images/3D scenes by sharing bases across signals during training which greatly
benefits use cases such as image regression from sparse observations and
few-shot radiance field reconstruction.

Comments:
- 13 pages, 7 figures; Project Page:
  https://apchenstu.github.io/FactorFields/