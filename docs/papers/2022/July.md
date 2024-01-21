---
layout: default
title: July
parent: 2022
nav_order: 7
---
<!---metadata--->

## MobileNeRF: Exploiting the Polygon Rasterization Pipeline for Efficient  Neural Field Rendering on Mobile Architectures

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-30 | Zhiqin Chen, Thomas Funkhouser, Peter Hedman, Andrea Tagliasacchi | cs.CV | [PDF](http://arxiv.org/pdf/2208.00277v5){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have demonstrated amazing ability to
synthesize images of 3D scenes from novel views. However, they rely upon
specialized volumetric rendering algorithms based on ray marching that are
mismatched to the capabilities of widely deployed graphics hardware. This paper
introduces a new NeRF representation based on textured polygons that can
synthesize novel images efficiently with standard rendering pipelines. The NeRF
is represented as a set of polygons with textures representing binary opacities
and feature vectors. Traditional rendering of the polygons with a z-buffer
yields an image with features at every pixel, which are interpreted by a small,
view-dependent MLP running in a fragment shader to produce a final pixel color.
This approach enables NeRFs to be rendered with the traditional polygon
rasterization pipeline, which provides massive pixel-level parallelism,
achieving interactive frame rates on a wide range of compute platforms,
including mobile phones.

Comments:
- CVPR 2023. Project page: https://mobile-nerf.github.io, code:
  https://github.com/google-research/jax3d/tree/main/jax3d/projects/mobilenerf

---

## Distilled Low Rank Neural Radiance Field with Quantization for Light  Field Compression

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-30 | Jinglei Shi, Christine Guillemot | cs.CV | [PDF](http://arxiv.org/pdf/2208.00164v3){: .btn .btn-green } |

**Abstract**: We propose in this paper a Quantized Distilled Low-Rank Neural Radiance Field
(QDLR-NeRF) representation for the task of light field compression. While
existing compression methods encode the set of light field sub-aperture images,
our proposed method learns an implicit scene representation in the form of a
Neural Radiance Field (NeRF), which also enables view synthesis. To reduce its
size, the model is first learned under a Low-Rank (LR) constraint using a
Tensor Train (TT) decomposition within an Alternating Direction Method of
Multipliers (ADMM) optimization framework. To further reduce the model's size,
the components of the tensor train decomposition need to be quantized. However,
simultaneously considering the optimization of the NeRF model with both the
low-rank constraint and rate-constrained weight quantization is challenging. To
address this difficulty, we introduce a network distillation operation that
separates the low-rank approximation and the weight quantization during network
training. The information from the initial LR-constrained NeRF (LR-NeRF) is
distilled into a model of much smaller dimension (DLR-NeRF) based on the TT
decomposition of the LR-NeRF. We then learn an optimized global codebook to
quantize all TT components, producing the final QDLR-NeRF. Experimental results
show that our proposed method yields better compression efficiency compared to
state-of-the-art methods, and it additionally has the advantage of allowing the
synthesis of any light field view with high quality.

---

## End-to-end View Synthesis via NeRF Attention

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-29 | Zelin Zhao, Jiaya Jia | cs.CV | [PDF](http://arxiv.org/pdf/2207.14741v3){: .btn .btn-green } |

**Abstract**: In this paper, we present a simple seq2seq formulation for view synthesis
where we take a set of ray points as input and output colors corresponding to
the rays. Directly applying a standard transformer on this seq2seq formulation
has two limitations. First, the standard attention cannot successfully fit the
volumetric rendering procedure, and therefore high-frequency components are
missing in the synthesized views. Second, applying global attention to all rays
and pixels is extremely inefficient. Inspired by the neural radiance field
(NeRF), we propose the NeRF attention (NeRFA) to address the above problems. On
the one hand, NeRFA considers the volumetric rendering equation as a soft
feature modulation procedure. In this way, the feature modulation enhances the
transformers with the NeRF-like inductive bias. On the other hand, NeRFA
performs multi-stage attention to reduce the computational overhead.
Furthermore, the NeRFA model adopts the ray and pixel transformers to learn the
interactions between rays and pixels. NeRFA demonstrates superior performance
over NeRF and NerFormer on four datasets: DeepVoxels, Blender, LLFF, and CO3D.
Besides, NeRFA establishes a new state-of-the-art under two settings: the
single-scene view synthesis and the category-centric novel view synthesis.

Comments:
- Fixed reference formatting issues

---

## Neural Density-Distance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-29 | Itsuki Ueda, Yoshihiro Fukuhara, Hirokatsu Kataoka, Hiroaki Aizawa, Hidehiko Shishido, Itaru Kitahara | cs.CV | [PDF](http://arxiv.org/pdf/2207.14455v1){: .btn .btn-green } |

**Abstract**: The success of neural fields for 3D vision tasks is now indisputable.
Following this trend, several methods aiming for visual localization (e.g.,
SLAM) have been proposed to estimate distance or density fields using neural
fields. However, it is difficult to achieve high localization performance by
only density fields-based methods such as Neural Radiance Field (NeRF) since
they do not provide density gradient in most empty regions. On the other hand,
distance field-based methods such as Neural Implicit Surface (NeuS) have
limitations in objects' surface shapes. This paper proposes Neural
Density-Distance Field (NeDDF), a novel 3D representation that reciprocally
constrains the distance and density fields. We extend distance field
formulation to shapes with no explicit boundary surface, such as fur or smoke,
which enable explicit conversion from distance field to density field.
Consistent distance and density fields realized by explicit conversion enable
both robustness to initial values and high-quality registration. Furthermore,
the consistency between fields allows fast convergence from sparse point
clouds. Experiments show that NeDDF can achieve high localization performance
while providing comparable results to NeRF on novel view synthesis. The code is
available at https://github.com/ueda0319/neddf.

Comments:
- ECCV 2022 (poster). project page: https://ueda0319.github.io/neddf/

---

## Is Attention All That NeRF Needs?

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-27 | Mukund Varma T, Peihao Wang, Xuxi Chen, Tianlong Chen, Subhashini Venugopalan, Zhangyang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2207.13298v3){: .btn .btn-green } |

**Abstract**: We present Generalizable NeRF Transformer (GNT), a transformer-based
architecture that reconstructs Neural Radiance Fields (NeRFs) and learns to
renders novel views on the fly from source views. While prior works on NeRFs
optimize a scene representation by inverting a handcrafted rendering equation,
GNT achieves neural representation and rendering that generalizes across scenes
using transformers at two stages. (1) The view transformer leverages multi-view
geometry as an inductive bias for attention-based scene representation, and
predicts coordinate-aligned features by aggregating information from epipolar
lines on the neighboring views. (2) The ray transformer renders novel views
using attention to decode the features from the view transformer along the
sampled points during ray marching. Our experiments demonstrate that when
optimized on a single scene, GNT can successfully reconstruct NeRF without an
explicit rendering formula due to the learned ray renderer. When trained on
multiple scenes, GNT consistently achieves state-of-the-art performance when
transferring to unseen scenes and outperform all other methods by ~10% on
average. Our analysis of the learned attention maps to infer depth and
occlusion indicate that attention enables learning a physically-grounded
rendering. Our results show the promise of transformers as a universal modeling
tool for graphics. Please refer to our project page for video results:
https://vita-group.github.io/GNT/.

Comments:
- International Conference on Learning Representations (ICLR), 2023

---

## Deforming Radiance Fields with Cages



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-25 | Tianhan Xu, Tatsuya Harada | cs.CV | [PDF](http://arxiv.org/pdf/2207.12298v1){: .btn .btn-green } |

**Abstract**: Recent advances in radiance fields enable photorealistic rendering of static
or dynamic 3D scenes, but still do not support explicit deformation that is
used for scene manipulation or animation. In this paper, we propose a method
that enables a new type of deformation of the radiance field: free-form
radiance field deformation. We use a triangular mesh that encloses the
foreground object called cage as an interface, and by manipulating the cage
vertices, our approach enables the free-form deformation of the radiance field.
The core of our approach is cage-based deformation which is commonly used in
mesh deformation. We propose a novel formulation to extend it to the radiance
field, which maps the position and the view direction of the sampling points
from the deformed space to the canonical space, thus enabling the rendering of
the deformed scene. The deformation results of the synthetic datasets and the
real-world datasets demonstrate the effectiveness of our approach.

Comments:
- ECCV 2022. Project page: https://xth430.github.io/deforming-nerf/

---

## Learning Generalizable Light Field Networks from Few Images



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-24 | Qian Li, Franck Multon, Adnane Boukhayma | cs.CV | [PDF](http://arxiv.org/pdf/2207.11757v1){: .btn .btn-green } |

**Abstract**: We explore a new strategy for few-shot novel view synthesis based on a neural
light field representation. Given a target camera pose, an implicit neural
network maps each ray to its target pixel's color directly. The network is
conditioned on local ray features generated by coarse volumetric rendering from
an explicit 3D feature volume. This volume is built from the input images using
a 3D ConvNet. Our method achieves competitive performances on synthetic and
real MVS data with respect to state-of-the-art neural radiance field based
competition, while offering a 100 times faster rendering.

---

## Learning Dynamic Facial Radiance Fields for Few-Shot Talking Head  Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-24 | Shuai Shen, Wanhua Li, Zheng Zhu, Yueqi Duan, Jie Zhou, Jiwen Lu | cs.CV | [PDF](http://arxiv.org/pdf/2207.11770v1){: .btn .btn-green } |

**Abstract**: Talking head synthesis is an emerging technology with wide applications in
film dubbing, virtual avatars and online education. Recent NeRF-based methods
generate more natural talking videos, as they better capture the 3D structural
information of faces. However, a specific model needs to be trained for each
identity with a large dataset. In this paper, we propose Dynamic Facial
Radiance Fields (DFRF) for few-shot talking head synthesis, which can rapidly
generalize to an unseen identity with few training data. Different from the
existing NeRF-based methods which directly encode the 3D geometry and
appearance of a specific person into the network, our DFRF conditions face
radiance field on 2D appearance images to learn the face prior. Thus the facial
radiance field can be flexibly adjusted to the new identity with few reference
images. Additionally, for better modeling of the facial deformations, we
propose a differentiable face warping module conditioned on audio signals to
deform all reference images to the query space. Extensive experiments show that
with only tens of seconds of training clip available, our proposed DFRF can
synthesize natural and high-quality audio-driven talking head videos for novel
identities with only 40k iterations. We highly recommend readers view our
supplementary video for intuitive comparisons. Code is available in
https://sstzal.github.io/DFRF/.

Comments:
- Accepted by ECCV 2022. Project page: https://sstzal.github.io/DFRF/

---

## PS-NeRF: Neural Inverse Rendering for Multi-view Photometric Stereo

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-23 | Wenqi Yang, Guanying Chen, Chaofeng Chen, Zhenfang Chen, Kwan-Yee K. Wong | cs.CV | [PDF](http://arxiv.org/pdf/2207.11406v2){: .btn .btn-green } |

**Abstract**: Traditional multi-view photometric stereo (MVPS) methods are often composed
of multiple disjoint stages, resulting in noticeable accumulated errors. In
this paper, we present a neural inverse rendering method for MVPS based on
implicit representation. Given multi-view images of a non-Lambertian object
illuminated by multiple unknown directional lights, our method jointly
estimates the geometry, materials, and lights. Our method first employs
multi-light images to estimate per-view surface normal maps, which are used to
regularize the normals derived from the neural radiance field. It then jointly
optimizes the surface normals, spatially-varying BRDFs, and lights based on a
shadow-aware differentiable rendering layer. After optimization, the
reconstructed object can be used for novel-view rendering, relighting, and
material editing. Experiments on both synthetic and real datasets demonstrate
that our method achieves far more accurate shape reconstruction than existing
MVPS and neural rendering methods. Our code and model can be found at
https://ywq.github.io/psnerf.

Comments:
- ECCV 2022, Project page: https://ywq.github.io/psnerf

---

## Neural-Sim: Learning to Generate Training Data with NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-22 | Yunhao Ge, Harkirat Behl, Jiashu Xu, Suriya Gunasekar, Neel Joshi, Yale Song, Xin Wang, Laurent Itti, Vibhav Vineet | cs.CV | [PDF](http://arxiv.org/pdf/2207.11368v1){: .btn .btn-green } |

**Abstract**: Training computer vision models usually requires collecting and labeling vast
amounts of imagery under a diverse set of scene configurations and properties.
This process is incredibly time-consuming, and it is challenging to ensure that
the captured data distribution maps well to the target domain of an application
scenario. Recently, synthetic data has emerged as a way to address both of
these issues. However, existing approaches either require human experts to
manually tune each scene property or use automatic methods that provide little
to no control; this requires rendering large amounts of random data variations,
which is slow and is often suboptimal for the target domain. We present the
first fully differentiable synthetic data pipeline that uses Neural Radiance
Fields (NeRFs) in a closed-loop with a target application's loss function. Our
approach generates data on-demand, with no human labor, to maximize accuracy
for a target task. We illustrate the effectiveness of our method on synthetic
and real-world object detection tasks. We also introduce a new
"YCB-in-the-Wild" dataset and benchmark that provides a test scenario for
object detection with varied poses in real-world environments.

Comments:
- ECCV 2022

---

## Generalizable Patch-Based Neural Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-21 | Mohammed Suhail, Carlos Esteves, Leonid Sigal, Ameesh Makadia | cs.CV | [PDF](http://arxiv.org/pdf/2207.10662v2){: .btn .btn-green } |

**Abstract**: Neural rendering has received tremendous attention since the advent of Neural
Radiance Fields (NeRF), and has pushed the state-of-the-art on novel-view
synthesis considerably. The recent focus has been on models that overfit to a
single scene, and the few attempts to learn models that can synthesize novel
views of unseen scenes mostly consist of combining deep convolutional features
with a NeRF-like model. We propose a different paradigm, where no deep features
and no NeRF-like volume rendering are needed. Our method is capable of
predicting the color of a target ray in a novel scene directly, just from a
collection of patches sampled from the scene. We first leverage epipolar
geometry to extract patches along the epipolar lines of each reference view.
Each patch is linearly projected into a 1D feature vector and a sequence of
transformers process the collection. For positional encoding, we parameterize
rays as in a light field representation, with the crucial difference that the
coordinates are canonicalized with respect to the target ray, which makes our
method independent of the reference frame and improves generalization. We show
that our approach outperforms the state-of-the-art on novel view synthesis of
unseen scenes even when being trained with considerably less data than prior
work.

Comments:
- Project Page with code and results at
  https://mohammedsuhail.net/gen_patch_neural_rendering/

---

## AdaNeRF: Adaptive Sampling for Real-time Rendering of Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-21 | Andreas Kurz, Thomas Neff, Zhaoyang Lv, Michael Zollhöfer, Markus Steinberger | cs.CV | [PDF](http://arxiv.org/pdf/2207.10312v2){: .btn .btn-green } |

**Abstract**: Novel view synthesis has recently been revolutionized by learning neural
radiance fields directly from sparse observations. However, rendering images
with this new paradigm is slow due to the fact that an accurate quadrature of
the volume rendering equation requires a large number of samples for each ray.
Previous work has mainly focused on speeding up the network evaluations that
are associated with each sample point, e.g., via caching of radiance values
into explicit spatial data structures, but this comes at the expense of model
compactness. In this paper, we propose a novel dual-network architecture that
takes an orthogonal direction by learning how to best reduce the number of
required sample points. To this end, we split our network into a sampling and
shading network that are jointly trained. Our training scheme employs fixed
sample positions along each ray, and incrementally introduces sparsity
throughout training to achieve high quality even at low sample counts. After
fine-tuning with the target number of samples, the resulting compact neural
representation can be rendered in real-time. Our experiments demonstrate that
our approach outperforms concurrent compact neural representations in terms of
quality and frame rate and performs on par with highly efficient hybrid
representations. Code and supplementary material is available at
https://thomasneff.github.io/adanerf.

Comments:
- ECCV 2022. Project page: https://thomasneff.github.io/adanerf

---

## Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for  Editable Portrait Image Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-21 | Jeong-gi Kwak, Yuanming Li, Dongsik Yoon, Donghyeon Kim, David Han, Hanseok Ko | cs.CV | [PDF](http://arxiv.org/pdf/2207.10257v2){: .btn .btn-green } |

**Abstract**: Over the years, 2D GANs have achieved great successes in photorealistic
portrait generation. However, they lack 3D understanding in the generation
process, thus they suffer from multi-view inconsistency problem. To alleviate
the issue, many 3D-aware GANs have been proposed and shown notable results, but
3D GANs struggle with editing semantic attributes. The controllability and
interpretability of 3D GANs have not been much explored. In this work, we
propose two solutions to overcome these weaknesses of 2D GANs and 3D-aware
GANs. We first introduce a novel 3D-aware GAN, SURF-GAN, which is capable of
discovering semantic attributes during training and controlling them in an
unsupervised manner. After that, we inject the prior of SURF-GAN into StyleGAN
to obtain a high-fidelity 3D-controllable generator. Unlike existing
latent-based methods allowing implicit pose control, the proposed
3D-controllable StyleGAN enables explicit pose control over portrait
generation. This distillation allows direct compatibility between 3D control
and many StyleGAN-based techniques (e.g., inversion and stylization), and also
brings an advantage in terms of computational resources. Our codes are
available at https://github.com/jgkwak95/SURF-GAN.

Comments:
- ECCV 2022, project page: https://jgkwak95.github.io/surfgan/

---

## NDF: Neural Deformable Fields for Dynamic Human Modelling



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-19 | Ruiqi Zhang, Jie Chen | cs.CV | [PDF](http://arxiv.org/pdf/2207.09193v1){: .btn .btn-green } |

**Abstract**: We propose Neural Deformable Fields (NDF), a new representation for dynamic
human digitization from a multi-view video. Recent works proposed to represent
a dynamic human body with shared canonical neural radiance fields which links
to the observation space with deformation fields estimations. However, the
learned canonical representation is static and the current design of the
deformation fields is not able to represent large movements or detailed
geometry changes. In this paper, we propose to learn a neural deformable field
wrapped around a fitted parametric body model to represent the dynamic human.
The NDF is spatially aligned by the underlying reference surface. A neural
network is then learned to map pose to the dynamics of NDF. The proposed NDF
representation can synthesize the digitized performer with novel views and
novel poses with a detailed and reasonable dynamic appearance. Experiments show
that our method significantly outperforms recent human synthesis methods.

Comments:
- 16 pages, 7 figures. Accepted by ECCV 2022

---

## Neural apparent BRDF fields for multiview photometric stereo

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-14 | Meghna Asthana, William A. P. Smith, Patrik Huber | cs.CV | [PDF](http://arxiv.org/pdf/2207.06793v1){: .btn .btn-green } |

**Abstract**: We propose to tackle the multiview photometric stereo problem using an
extension of Neural Radiance Fields (NeRFs), conditioned on light source
direction. The geometric part of our neural representation predicts surface
normal direction, allowing us to reason about local surface reflectance. The
appearance part of our neural representation is decomposed into a neural
bidirectional reflectance function (BRDF), learnt as part of the fitting
process, and a shadow prediction network (conditioned on light source
direction) allowing us to model the apparent BRDF. This balance of learnt
components with inductive biases based on physical image formation models
allows us to extrapolate far from the light source and viewer directions
observed during training. We demonstrate our approach on a multiview
photometric stereo benchmark and show that competitive performance can be
obtained with the neural density representation of a NeRF.

Comments:
- 9 pages, 6 figures, 1 table

---

## Vision Transformer for NeRF-Based View Synthesis from a Single Input  Image

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-12 | Kai-En Lin, Lin Yen-Chen, Wei-Sheng Lai, Tsung-Yi Lin, Yi-Chang Shih, Ravi Ramamoorthi | cs.CV | [PDF](http://arxiv.org/pdf/2207.05736v2){: .btn .btn-green } |

**Abstract**: Although neural radiance fields (NeRF) have shown impressive advances for
novel view synthesis, most methods typically require multiple input images of
the same scene with accurate camera poses. In this work, we seek to
substantially reduce the inputs to a single unposed image. Existing approaches
condition on local image features to reconstruct a 3D object, but often render
blurry predictions at viewpoints that are far away from the source view. To
address this issue, we propose to leverage both the global and local features
to form an expressive 3D representation. The global features are learned from a
vision transformer, while the local features are extracted from a 2D
convolutional network. To synthesize a novel view, we train a multilayer
perceptron (MLP) network conditioned on the learned 3D representation to
perform volume rendering. This novel 3D representation allows the network to
reconstruct unseen regions without enforcing constraints like symmetry or
canonical coordinate systems. Our method can render novel views from only a
single input image and generalize across multiple object categories using a
single model. Quantitative and qualitative evaluations demonstrate that the
proposed method achieves state-of-the-art performance and renders richer
details than existing approaches.

Comments:
- WACV 2023 Project website:
  https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/

---

## A Learned Radiance-Field Representation for Complex Luminaires

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-11 | Jorge Condor, Adrián Jarabo | cs.GR | [PDF](http://arxiv.org/pdf/2207.05009v1){: .btn .btn-green } |

**Abstract**: We propose an efficient method for rendering complex luminaires using a
high-quality octree-based representation of the luminaire emission. Complex
luminaires are a particularly challenging problem in rendering, due to their
caustic light paths inside the luminaire. We reduce the geometric complexity of
luminaires by using a simple proxy geometry and encode the visually-complex
emitted light field by using a neural radiance field. We tackle the multiple
challenges of using NeRFs for representing luminaires, including their high
dynamic range, high-frequency content and null-emission areas, by proposing a
specialized loss function. For rendering, we distill our luminaires' NeRF into
a Plenoctree, which we can be easily integrated into traditional rendering
systems. Our approach allows for speed-ups of up to 2 orders of magnitude in
scenes containing complex luminaires introducing minimal error.

Comments:
- 10 pages, 7 figures. Eurographics Proceedings (EGSR 2022,
  Symposium-only track) (https://diglib.eg.org/handle/10.2312/sr20221155)

---

## Progressively-connected Light Field Network for Efficient View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-10 | Peng Wang, Yuan Liu, Guying Lin, Jiatao Gu, Lingjie Liu, Taku Komura, Wenping Wang | cs.CV | [PDF](http://arxiv.org/pdf/2207.04465v1){: .btn .btn-green } |

**Abstract**: This paper presents a Progressively-connected Light Field network (ProLiF),
for the novel view synthesis of complex forward-facing scenes. ProLiF encodes a
4D light field, which allows rendering a large batch of rays in one training
step for image- or patch-level losses. Directly learning a neural light field
from images has difficulty in rendering multi-view consistent images due to its
unawareness of the underlying 3D geometry. To address this problem, we propose
a progressive training scheme and regularization losses to infer the underlying
geometry during training, both of which enforce the multi-view consistency and
thus greatly improves the rendering quality. Experiments demonstrate that our
method is able to achieve significantly better rendering quality than the
vanilla neural light fields and comparable results to NeRF-like rendering
methods on the challenging LLFF dataset and Shiny Object dataset. Moreover, we
demonstrate better compatibility with LPIPS loss to achieve robustness to
varying light conditions and CLIP loss to control the rendering style of the
scene. Project page: https://totoro97.github.io/projects/prolif.

Comments:
- Project page: https://totoro97.github.io/projects/prolif

---

## VMRF: View Matching Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-06 | Jiahui Zhang, Fangneng Zhan, Rongliang Wu, Yingchen Yu, Wenqing Zhang, Bai Song, Xiaoqin Zhang, Shijian Lu | cs.CV | [PDF](http://arxiv.org/pdf/2207.02621v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have demonstrated very impressive performance
in novel view synthesis via implicitly modelling 3D representations from
multi-view 2D images. However, most existing studies train NeRF models with
either reasonable camera pose initialization or manually-crafted camera pose
distributions which are often unavailable or hard to acquire in various
real-world data. We design VMRF, an innovative view matching NeRF that enables
effective NeRF training without requiring prior knowledge in camera poses or
camera pose distributions. VMRF introduces a view matching scheme, which
exploits unbalanced optimal transport to produce a feature transport plan for
mapping a rendered image with randomly initialized camera pose to the
corresponding real image. With the feature transport plan as the guidance, a
novel pose calibration technique is designed which rectifies the initially
randomized camera poses by predicting relative pose transformations between the
pair of rendered and real images. Extensive experiments over a number of
synthetic and real datasets show that the proposed VMRF outperforms the
state-of-the-art qualitatively and quantitatively by large margins.

Comments:
- This paper has been accepted to ACM MM 2022

---

## SNeRF: Stylized Neural Implicit Representations for 3D Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-05 | Thu Nguyen-Phuoc, Feng Liu, Lei Xiao | cs.CV | [PDF](http://arxiv.org/pdf/2207.02363v1){: .btn .btn-green } |

**Abstract**: This paper presents a stylized novel view synthesis method. Applying
state-of-the-art stylization methods to novel views frame by frame often causes
jittering artifacts due to the lack of cross-view consistency. Therefore, this
paper investigates 3D scene stylization that provides a strong inductive bias
for consistent novel view synthesis. Specifically, we adopt the emerging neural
radiance fields (NeRF) as our choice of 3D scene representation for their
capability to render high-quality novel views for a variety of scenes. However,
as rendering a novel view from a NeRF requires a large number of samples,
training a stylized NeRF requires a large amount of GPU memory that goes beyond
an off-the-shelf GPU capacity. We introduce a new training method to address
this problem by alternating the NeRF and stylization optimization steps. Such a
method enables us to make full use of our hardware memory capacity to both
generate images at higher resolution and adopt more expressive image style
transfer methods. Our experiments show that our method produces stylized NeRFs
for a wide range of content, including indoor, outdoor and dynamic scenes, and
synthesizes high-quality novel views with cross-view consistency.

Comments:
- SIGGRAPH 2022 (Journal track). Project page:
  https://research.facebook.com/publications/snerf-stylized-neural-implicit-representations-for-3d-scenes/

---

## LaTeRF: Label and Text Driven Object Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-04 | Ashkan Mirzaei, Yash Kant, Jonathan Kelly, Igor Gilitschenski | cs.CV | [PDF](http://arxiv.org/pdf/2207.01583v3){: .btn .btn-green } |

**Abstract**: Obtaining 3D object representations is important for creating photo-realistic
simulations and for collecting AR and VR assets. Neural fields have shown their
effectiveness in learning a continuous volumetric representation of a scene
from 2D images, but acquiring object representations from these models with
weak supervision remains an open challenge. In this paper we introduce LaTeRF,
a method for extracting an object of interest from a scene given 2D images of
the entire scene, known camera poses, a natural language description of the
object, and a set of point-labels of object and non-object points in the input
images. To faithfully extract the object from the scene, LaTeRF extends the
NeRF formulation with an additional `objectness' probability at each 3D point.
Additionally, we leverage the rich latent space of a pre-trained CLIP model
combined with our differentiable object renderer, to inpaint the occluded parts
of the object. We demonstrate high-fidelity object extraction on both synthetic
and real-world datasets and justify our design choices through an extensive
ablation study.

---

## Aug-NeRF: Training Stronger Neural Radiance Fields with Triple-Level  Physically-Grounded Augmentations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-07-04 | Tianlong Chen, Peihao Wang, Zhiwen Fan, Zhangyang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2207.01164v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) regresses a neural parameterized scene by
differentially rendering multi-view images with ground-truth supervision.
However, when interpolating novel views, NeRF often yields inconsistent and
visually non-smooth geometric results, which we consider as a generalization
gap between seen and unseen views. Recent advances in convolutional neural
networks have demonstrated the promise of advanced robust data augmentations,
either random or learned, in enhancing both in-distribution and
out-of-distribution generalization. Inspired by that, we propose Augmented NeRF
(Aug-NeRF), which for the first time brings the power of robust data
augmentations into regularizing the NeRF training. Particularly, our proposal
learns to seamlessly blend worst-case perturbations into three distinct levels
of the NeRF pipeline with physical grounds, including (1) the input
coordinates, to simulate imprecise camera parameters at image capture; (2)
intermediate features, to smoothen the intrinsic feature manifold; and (3)
pre-rendering output, to account for the potential degradation factors in the
multi-view image supervision. Extensive results demonstrate that Aug-NeRF
effectively boosts NeRF performance in both novel view synthesis (up to 1.5dB
PSNR gain) and underlying geometry reconstruction. Furthermore, thanks to the
implicit smooth prior injected by the triple-level augmentations, Aug-NeRF can
even recover scenes from heavily corrupted images, a highly challenging setting
untackled before. Our codes are available in
https://github.com/VITA-Group/Aug-NeRF.

Comments:
- IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
  2022