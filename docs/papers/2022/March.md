---
layout: default
title: March
parent: 2022
nav_order: 3
---
<!---metadata--->

## R2L: Distilling Neural Radiance Field to Neural Light Field for  Efficient Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-31 | Huan Wang, Jian Ren, Zeng Huang, Kyle Olszewski, Menglei Chai, Yun Fu, Sergey Tulyakov | cs.CV | [PDF](http://arxiv.org/pdf/2203.17261v2){: .btn .btn-green } |

**Abstract**: Recent research explosion on Neural Radiance Field (NeRF) shows the
encouraging potential to represent complex scenes with neural networks. One
major drawback of NeRF is its prohibitive inference time: Rendering a single
pixel requires querying the NeRF network hundreds of times. To resolve it,
existing efforts mainly attempt to reduce the number of required sampled
points. However, the problem of iterative sampling still exists. On the other
hand, Neural Light Field (NeLF) presents a more straightforward representation
over NeRF in novel view synthesis -- the rendering of a pixel amounts to one
single forward pass without ray-marching. In this work, we present a deep
residual MLP network (88 layers) to effectively learn the light field. We show
the key to successfully learning such a deep NeLF network is to have sufficient
data, for which we transfer the knowledge from a pre-trained NeRF model via
data distillation. Extensive experiments on both synthetic and real-world
scenes show the merits of our method over other counterpart algorithms. On the
synthetic scenes, we achieve 26-35x FLOPs reduction (per camera ray) and 28-31x
runtime speedup, meanwhile delivering significantly better (1.4-2.8 dB average
PSNR improvement) rendering quality than NeRF without any customized
parallelism requirement.

Comments:
- Accepted by ECCV 2022. Code: https://github.com/snap-research/R2L

---

## MPS-NeRF: Generalizable 3D Human Rendering from Multiview Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-31 | Xiangjun Gao, Jiaolong Yang, Jongyoo Kim, Sida Peng, Zicheng Liu, Xin Tong | cs.CV | [PDF](http://arxiv.org/pdf/2203.16875v2){: .btn .btn-green } |

**Abstract**: There has been rapid progress recently on 3D human rendering, including novel
view synthesis and pose animation, based on the advances of neural radiance
fields (NeRF). However, most existing methods focus on person-specific training
and their training typically requires multi-view videos. This paper deals with
a new challenging task -- rendering novel views and novel poses for a person
unseen in training, using only multiview images as input. For this task, we
propose a simple yet effective method to train a generalizable NeRF with
multiview images as conditional input. The key ingredient is a dedicated
representation combining a canonical NeRF and a volume deformation scheme.
Using a canonical space enables our method to learn shared properties of human
and easily generalize to different people. Volume deformation is used to
connect the canonical space with input and target images and query image
features for radiance and density prediction. We leverage the parametric 3D
human model fitted on the input images to derive the deformation, which works
quite well in practice when combined with our canonical NeRF. The experiments
on both real and synthetic data with the novel view synthesis and pose
animation tasks collectively demonstrate the efficacy of our method.

---

## DDNeRF: Depth Distribution Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-30 | David Dadon, Ohad Fried, Yacov Hel-Or | cs.CV | [PDF](http://arxiv.org/pdf/2203.16626v1){: .btn .btn-green } |

**Abstract**: In recent years, the field of implicit neural representation has progressed
significantly. Models such as neural radiance fields (NeRF), which uses
relatively small neural networks, can represent high-quality scenes and achieve
state-of-the-art results for novel view synthesis. Training these types of
networks, however, is still computationally very expensive. We present depth
distribution neural radiance field (DDNeRF), a new method that significantly
increases sampling efficiency along rays during training while achieving
superior results for a given sampling budget. DDNeRF achieves this by learning
a more accurate representation of the density distribution along rays. More
specifically, we train a coarse model to predict the internal distribution of
the transparency of an input volume in addition to the volume's total density.
This finer distribution then guides the sampling procedure of the fine model.
This method allows us to use fewer samples during training while reducing
computational resources.

---

## DRaCoN -- Differentiable Rasterization Conditioned Neural Radiance  Fields for Articulated Avatars

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-29 | Amit Raj, Umar Iqbal, Koki Nagano, Sameh Khamis, Pavlo Molchanov, James Hays, Jan Kautz | cs.CV | [PDF](http://arxiv.org/pdf/2203.15798v1){: .btn .btn-green } |

**Abstract**: Acquisition and creation of digital human avatars is an important problem
with applications to virtual telepresence, gaming, and human modeling. Most
contemporary approaches for avatar generation can be viewed either as 3D-based
methods, which use multi-view data to learn a 3D representation with appearance
(such as a mesh, implicit surface, or volume), or 2D-based methods which learn
photo-realistic renderings of avatars but lack accurate 3D representations. In
this work, we present, DRaCoN, a framework for learning full-body volumetric
avatars which exploits the advantages of both the 2D and 3D neural rendering
techniques. It consists of a Differentiable Rasterization module, DiffRas, that
synthesizes a low-resolution version of the target image along with additional
latent features guided by a parametric body model. The output of DiffRas is
then used as conditioning to our conditional neural 3D representation module
(c-NeRF) which generates the final high-res image along with body geometry
using volumetric rendering. While DiffRas helps in obtaining photo-realistic
image quality, c-NeRF, which employs signed distance fields (SDF) for 3D
representations, helps to obtain fine 3D geometric details. Experiments on the
challenging ZJU-MoCap and Human3.6M datasets indicate that DRaCoN outperforms
state-of-the-art methods both in terms of error metrics and visual quality.

Comments:
- Project page at https://dracon-avatars.github.io/

---

## Panoptic NeRF: 3D-to-2D Label Transfer for Panoptic Urban Scene  Segmentation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-29 | Xiao Fu, Shangzhan Zhang, Tianrun Chen, Yichong Lu, Lanyun Zhu, Xiaowei Zhou, Andreas Geiger, Yiyi Liao | cs.CV | [PDF](http://arxiv.org/pdf/2203.15224v2){: .btn .btn-green } |

**Abstract**: Large-scale training data with high-quality annotations is critical for
training semantic and instance segmentation models. Unfortunately, pixel-wise
annotation is labor-intensive and costly, raising the demand for more efficient
labeling strategies. In this work, we present a novel 3D-to-2D label transfer
method, Panoptic NeRF, which aims for obtaining per-pixel 2D semantic and
instance labels from easy-to-obtain coarse 3D bounding primitives. Our method
utilizes NeRF as a differentiable tool to unify coarse 3D annotations and 2D
semantic cues transferred from existing datasets. We demonstrate that this
combination allows for improved geometry guided by semantic information,
enabling rendering of accurate semantic maps across multiple views.
Furthermore, this fusion process resolves label ambiguity of the coarse 3D
annotations and filters noise in the 2D predictions. By inferring in 3D space
and rendering to 2D labels, our 2D semantic and instance labels are multi-view
consistent by design. Experimental results show that Panoptic NeRF outperforms
existing label transfer methods in terms of accuracy and multi-view consistency
on challenging urban scenes of the KITTI-360 dataset.

Comments:
- Project page: https://fuxiao0719.github.io/projects/panopticnerf/

---

## Towards Learning Neural Representations from Shadows

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-29 | Kushagra Tiwary, Tzofi Klinghoffer, Ramesh Raskar | cs.CV | [PDF](http://arxiv.org/pdf/2203.15946v2){: .btn .btn-green } |

**Abstract**: We present a method that learns neural shadow fields which are neural scene
representations that are only learnt from the shadows present in the scene.
While traditional shape-from-shadow (SfS) algorithms reconstruct geometry from
shadows, they assume a fixed scanning setup and fail to generalize to complex
scenes. Neural rendering algorithms, on the other hand, rely on photometric
consistency between RGB images, but largely ignore physical cues such as
shadows, which have been shown to provide valuable information about the scene.
We observe that shadows are a powerful cue that can constrain neural scene
representations to learn SfS, and even outperform NeRF to reconstruct otherwise
hidden geometry. We propose a graphics-inspired differentiable approach to
render accurate shadows with volumetric rendering, predicting a shadow map that
can be compared to the ground truth shadow. Even with just binary shadow maps,
we show that neural rendering can localize the object and estimate coarse
geometry. Our approach reveals that sparse cues in images can be used to
estimate geometry using differentiable volumetric rendering. Moreover, our
framework is highly generalizable and can work alongside existing 3D
reconstruction techniques that otherwise only use photometric consistency.

---

## RGB-D Neural Radiance Fields: Local Sampling for Faster Training

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-26 | Arnab Dey, Andrew I. Comport | cs.CV | [PDF](http://arxiv.org/pdf/2203.15587v2){: .btn .btn-green } |

**Abstract**: Learning a 3D representation of a scene has been a challenging problem for
decades in computer vision. Recent advances in implicit neural representation
from images using neural radiance fields(NeRF) have shown promising results.
Some of the limitations of previous NeRF based methods include longer training
time, and inaccurate underlying geometry. The proposed method takes advantage
of RGB-D data to reduce training time by leveraging depth sensing to improve
local sampling. This paper proposes a depth-guided local sampling strategy and
a smaller neural network architecture to achieve faster training time without
compromising quality.

---

## Continuous Dynamic-NeRF: Spline-NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-25 | Julian Knodt | cs.CV | [PDF](http://arxiv.org/pdf/2203.13800v1){: .btn .btn-green } |

**Abstract**: The problem of reconstructing continuous functions over time is important for
problems such as reconstructing moving scenes, and interpolating between time
steps. Previous approaches that use deep-learning rely on regularization to
ensure that reconstructions are approximately continuous, which works well on
short sequences. As sequence length grows, though, it becomes more difficult to
regularize, and it becomes less feasible to learn only through regularization.
We propose a new architecture for function reconstruction based on classical
Bezier splines, which ensures $C^0$ and $C^1$-continuity, where $C^0$
continuity is that $\forall c:\lim\limits_{x\to c} f(x)
  = f(c)$, or more intuitively that there are no breaks at any point in the
function. In order to demonstrate our architecture, we reconstruct dynamic
scenes using Neural Radiance Fields, but hope it is clear that our approach is
general and can be applied to a variety of problems. We recover a Bezier spline
$B(\beta, t\in[0,1])$, parametrized by the control points $\beta$. Using Bezier
splines ensures reconstructions have $C^0$ and $C^1$ continuity, allowing for
guaranteed interpolation over time. We reconstruct $\beta$ with a multi-layer
perceptron (MLP), blending machine learning with classical animation
techniques. All code is available at https://github.com/JulianKnodt/nerf_atlas,
and datasets are from prior work.

---

## NeuMan: Neural Human Radiance Field from a Single Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-23 | Wei Jiang, Kwang Moo Yi, Golnoosh Samei, Oncel Tuzel, Anurag Ranjan | cs.CV | [PDF](http://arxiv.org/pdf/2203.12575v2){: .btn .btn-green } |

**Abstract**: Photorealistic rendering and reposing of humans is important for enabling
augmented reality experiences. We propose a novel framework to reconstruct the
human and the scene that can be rendered with novel human poses and views from
just a single in-the-wild video. Given a video captured by a moving camera, we
train two NeRF models: a human NeRF model and a scene NeRF model. To train
these models, we rely on existing methods to estimate the rough geometry of the
human and the scene. Those rough geometry estimates allow us to create a
warping field from the observation space to the canonical pose-independent
space, where we train the human model in. Our method is able to learn subject
specific details, including cloth wrinkles and accessories, from just a 10
seconds video clip, and to provide high quality renderings of the human under
novel poses, from novel views, together with the background.

---

## NeRFusion: Fusing Radiance Fields for Large-Scale Scene Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-21 | Xiaoshuai Zhang, Sai Bi, Kalyan Sunkavalli, Hao Su, Zexiang Xu | cs.CV | [PDF](http://arxiv.org/pdf/2203.11283v1){: .btn .btn-green } |

**Abstract**: While NeRF has shown great success for neural reconstruction and rendering,
its limited MLP capacity and long per-scene optimization times make it
challenging to model large-scale indoor scenes. In contrast, classical 3D
reconstruction methods can handle large-scale scenes but do not produce
realistic renderings. We propose NeRFusion, a method that combines the
advantages of NeRF and TSDF-based fusion techniques to achieve efficient
large-scale reconstruction and photo-realistic rendering. We process the input
image sequence to predict per-frame local radiance fields via direct network
inference. These are then fused using a novel recurrent neural network that
incrementally reconstructs a global, sparse scene representation in real-time
at 22 fps. This global volume can be further fine-tuned to boost rendering
quality. We demonstrate that NeRFusion achieves state-of-the-art quality on
both large-scale indoor and small-scale object scenes, with substantially
faster reconstruction than NeRF and other recent methods.

Comments:
- CVPR 2022

---

## Sem2NeRF: Converting Single-View Semantic Masks to Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-21 | Yuedong Chen, Qianyi Wu, Chuanxia Zheng, Tat-Jen Cham, Jianfei Cai | cs.CV | [PDF](http://arxiv.org/pdf/2203.10821v2){: .btn .btn-green } |

**Abstract**: Image translation and manipulation have gain increasing attention along with
the rapid development of deep generative models. Although existing approaches
have brought impressive results, they mainly operated in 2D space. In light of
recent advances in NeRF-based 3D-aware generative models, we introduce a new
task, Semantic-to-NeRF translation, that aims to reconstruct a 3D scene
modelled by NeRF, conditioned on one single-view semantic mask as input. To
kick-off this novel task, we propose the Sem2NeRF framework. In particular,
Sem2NeRF addresses the highly challenging task by encoding the semantic mask
into the latent code that controls the 3D scene representation of a pre-trained
decoder. To further improve the accuracy of the mapping, we integrate a new
region-aware learning strategy into the design of both the encoder and the
decoder. We verify the efficacy of the proposed Sem2NeRF and demonstrate that
it outperforms several strong baselines on two benchmark datasets. Code and
video are available at https://donydchen.github.io/sem2nerf/

Comments:
- ECCV2022, Code: https://github.com/donydchen/sem2nerf Project Page:
  https://donydchen.github.io/sem2nerf/

---

## Conditional-Flow NeRF: Accurate 3D Modelling with Reliable Uncertainty  Quantification

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-18 | Jianxiong Shen, Antonio Agudo, Francesc Moreno-Noguer, Adria Ruiz | cs.CV | [PDF](http://arxiv.org/pdf/2203.10192v1){: .btn .btn-green } |

**Abstract**: A critical limitation of current methods based on Neural Radiance Fields
(NeRF) is that they are unable to quantify the uncertainty associated with the
learned appearance and geometry of the scene. This information is paramount in
real applications such as medical diagnosis or autonomous driving where, to
reduce potentially catastrophic failures, the confidence on the model outputs
must be included into the decision-making process. In this context, we
introduce Conditional-Flow NeRF (CF-NeRF), a novel probabilistic framework to
incorporate uncertainty quantification into NeRF-based approaches. For this
purpose, our method learns a distribution over all possible radiance fields
modelling which is used to quantify the uncertainty associated with the
modelled scene. In contrast to previous approaches enforcing strong constraints
over the radiance field distribution, CF-NeRF learns it in a flexible and fully
data-driven manner by coupling Latent Variable Modelling and Conditional
Normalizing Flows. This strategy allows to obtain reliable uncertainty
estimation while preserving model expressivity. Compared to previous
state-of-the-art methods proposed for uncertainty quantification in NeRF, our
experiments show that the proposed method achieves significantly lower
prediction errors and more reliable uncertainty values for synthetic novel view
and depth-map estimation.

---

## ViewFormer: NeRF-free Neural Rendering from Few Images Using  Transformers

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-18 | Jonáš Kulhánek, Erik Derner, Torsten Sattler, Robert Babuška | cs.CV | [PDF](http://arxiv.org/pdf/2203.10157v2){: .btn .btn-green } |

**Abstract**: Novel view synthesis is a long-standing problem. In this work, we consider a
variant of the problem where we are given only a few context views sparsely
covering a scene or an object. The goal is to predict novel viewpoints in the
scene, which requires learning priors. The current state of the art is based on
Neural Radiance Field (NeRF), and while achieving impressive results, the
methods suffer from long training times as they require evaluating millions of
3D point samples via a neural network for each image. We propose a 2D-only
method that maps multiple context views and a query pose to a new image in a
single pass of a neural network. Our model uses a two-stage architecture
consisting of a codebook and a transformer model. The codebook is used to embed
individual images into a smaller latent space, and the transformer solves the
view synthesis task in this more compact space. To train our model efficiently,
we introduce a novel branching attention mechanism that allows us to use the
same model not only for neural rendering but also for camera pose estimation.
Experimental results on real-world scenes show that our approach is competitive
compared to NeRF-based methods while not reasoning explicitly in 3D, and it is
faster to train.

Comments:
- ECCV 2022 poster

---

## Enhancement of Novel View Synthesis Using Omnidirectional Image  Completion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-18 | Takayuki Hara, Tatsuya Harada | cs.CV | [PDF](http://arxiv.org/pdf/2203.09957v4){: .btn .btn-green } |

**Abstract**: In this study, we present a method for synthesizing novel views from a single
360-degree RGB-D image based on the neural radiance field (NeRF) . Prior
studies relied on the neighborhood interpolation capability of multi-layer
perceptrons to complete missing regions caused by occlusion and zooming, which
leads to artifacts. In the method proposed in this study, the input image is
reprojected to 360-degree RGB images at other camera positions, the missing
regions of the reprojected images are completed by a 2D image generative model,
and the completed images are utilized to train the NeRF. Because multiple
completed images contain inconsistencies in 3D, we introduce a method to learn
the NeRF model using a subset of completed images that cover the target scene
with less overlap of completed regions. The selection of such a subset of
images can be attributed to the maximum weight independent set problem, which
is solved through simulated annealing. Experiments demonstrated that the
proposed method can synthesize plausible novel views while preserving the
features of the scene for both artificial and real-world data.

Comments:
- 20 pages, 19 figures

---

## TensoRF: Tensorial Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-17 | Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, Hao Su | cs.CV | [PDF](http://arxiv.org/pdf/2203.09517v2){: .btn .btn-green } |

**Abstract**: We present TensoRF, a novel approach to model and reconstruct radiance
fields. Unlike NeRF that purely uses MLPs, we model the radiance field of a
scene as a 4D tensor, which represents a 3D voxel grid with per-voxel
multi-channel features. Our central idea is to factorize the 4D scene tensor
into multiple compact low-rank tensor components. We demonstrate that applying
traditional CP decomposition -- that factorizes tensors into rank-one
components with compact vectors -- in our framework leads to improvements over
vanilla NeRF. To further boost performance, we introduce a novel vector-matrix
(VM) decomposition that relaxes the low-rank constraints for two modes of a
tensor and factorizes tensors into compact vector and matrix factors. Beyond
superior rendering quality, our models with CP and VM decompositions lead to a
significantly lower memory footprint in comparison to previous and concurrent
works that directly optimize per-voxel features. Experimentally, we demonstrate
that TensoRF with CP decomposition achieves fast reconstruction (<30 min) with
better rendering quality and even a smaller model size (<4 MB) compared to
NeRF. Moreover, TensoRF with VM decomposition further boosts rendering quality
and outperforms previous state-of-the-art methods, while reducing the
reconstruction time (<10 min) and retaining a compact model size (<75 MB).

Comments:
- Project Page: https://apchenstu.github.io/TensoRF/

---

## Sat-NeRF: Learning Multi-View Satellite Photogrammetry With Transient  Objects and Shadow Modeling Using RPC Cameras

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-16 | Roger Marí, Gabriele Facciolo, Thibaud Ehret | cs.CV | [PDF](http://arxiv.org/pdf/2203.08896v2){: .btn .btn-green } |

**Abstract**: We introduce the Satellite Neural Radiance Field (Sat-NeRF), a new end-to-end
model for learning multi-view satellite photogrammetry in the wild. Sat-NeRF
combines some of the latest trends in neural rendering with native satellite
camera models, represented by rational polynomial coefficient (RPC) functions.
The proposed method renders new views and infers surface models of similar
quality to those obtained with traditional state-of-the-art stereo pipelines.
Multi-date images exhibit significant changes in appearance, mainly due to
varying shadows and transient objects (cars, vegetation). Robustness to these
challenges is achieved by a shadow-aware irradiance model and uncertainty
weighting to deal with transient phenomena that cannot be explained by the
position of the sun. We evaluate Sat-NeRF using WorldView-3 images from
different locations and stress the advantages of applying a bundle adjustment
to the satellite camera models prior to training. This boosts the network
performance and can optionally be used to extract additional cues for depth
supervision.

Comments:
- Accepted at CVPR EarthVision Workshop 2022

---

## Animatable Implicit Neural Representations for Creating Realistic  Avatars from Videos

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-15 | Sida Peng, Zhen Xu, Junting Dong, Qianqian Wang, Shangzhan Zhang, Qing Shuai, Hujun Bao, Xiaowei Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2203.08133v4){: .btn .btn-green } |

**Abstract**: This paper addresses the challenge of reconstructing an animatable human
model from a multi-view video. Some recent works have proposed to decompose a
non-rigidly deforming scene into a canonical neural radiance field and a set of
deformation fields that map observation-space points to the canonical space,
thereby enabling them to learn the dynamic scene from images. However, they
represent the deformation field as translational vector field or SE(3) field,
which makes the optimization highly under-constrained. Moreover, these
representations cannot be explicitly controlled by input motions. Instead, we
introduce a pose-driven deformation field based on the linear blend skinning
algorithm, which combines the blend weight field and the 3D human skeleton to
produce observation-to-canonical correspondences. Since 3D human skeletons are
more observable, they can regularize the learning of the deformation field.
Moreover, the pose-driven deformation field can be controlled by input skeletal
motions to generate new deformation fields to animate the canonical human
model. Experiments show that our approach significantly outperforms recent
human modeling methods. The code is available at
https://zju3dv.github.io/animatable_nerf/.

Comments:
- Project page: https://zju3dv.github.io/animatable_nerf/. arXiv admin
  note: substantial text overlap with arXiv:2105.02872

---

## DialogueNeRF: Towards Realistic Avatar Face-to-Face Conversation Video  Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-15 | Yichao Yan, Zanwei Zhou, Zi Wang, Jingnan Gao, Xiaokang Yang | cs.CV | [PDF](http://arxiv.org/pdf/2203.07931v2){: .btn .btn-green } |

**Abstract**: Conversation is an essential component of virtual avatar activities in the
metaverse. With the development of natural language processing, textual and
vocal conversation generation has achieved a significant breakthrough. However,
face-to-face conversations account for the vast majority of daily
conversations, while most existing methods focused on single-person talking
head generation. In this work, we take a step further and consider generating
realistic face-to-face conversation videos. Conversation generation is more
challenging than single-person talking head generation, since it not only
requires generating photo-realistic individual talking heads but also demands
the listener to respond to the speaker. In this paper, we propose a novel
unified framework based on neural radiance field (NeRF) to address this task.
Specifically, we model both the speaker and listener with a NeRF framework,
with different conditions to control individual expressions. The speaker is
driven by the audio signal, while the response of the listener depends on both
visual and acoustic information. In this way, face-to-face conversation videos
are generated between human avatars, with all the interlocutors modeled within
the same network. Moreover, to facilitate future research on this task, we
collect a new human conversation dataset containing 34 clips of videos.
Quantitative and qualitative experiments evaluate our method in different
aspects, e.g., image quality, pose sequence trend, and naturalness of the
rendering videos. Experimental results demonstrate that the avatars in the
resulting videos are able to perform a realistic conversation, and maintain
individual styles. All the code, data, and models will be made publicly
available.

---

## 3D-GIF: 3D-Controllable Object Generation via Implicit Factorized  Representations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-12 | Minsoo Lee, Chaeyeon Chung, Hojun Cho, Minjung Kim, Sanghun Jung, Jaegul Choo, Minhyuk Sung | cs.CV | [PDF](http://arxiv.org/pdf/2203.06457v1){: .btn .btn-green } |

**Abstract**: While NeRF-based 3D-aware image generation methods enable viewpoint control,
limitations still remain to be adopted to various 3D applications. Due to their
view-dependent and light-entangled volume representation, the 3D geometry
presents unrealistic quality and the color should be re-rendered for every
desired viewpoint. To broaden the 3D applicability from 3D-aware image
generation to 3D-controllable object generation, we propose the factorized
representations which are view-independent and light-disentangled, and training
schemes with randomly sampled light conditions. We demonstrate the superiority
of our method by visualizing factorized representations, re-lighted images, and
albedo-textured meshes. In addition, we show that our approach improves the
quality of the generated geometry via visualization and quantitative
comparison. To the best of our knowledge, this is the first work that extracts
albedo-textured meshes with unposed 2D images without any additional labels or
assumptions.

---

## NeRFocus: Neural Radiance Field for 3D Synthetic Defocus

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-10 | Yinhuai Wang, Shuzhou Yang, Yujie Hu, Jian Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2203.05189v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) bring a new wave for 3D interactive
experiences. However, as an important part of the immersive experiences, the
defocus effects have not been fully explored within NeRF. Some recent
NeRF-based methods generate 3D defocus effects in a post-process fashion by
utilizing multiplane technology. Still, they are either time-consuming or
memory-consuming. This paper proposes a novel thin-lens-imaging-based NeRF
framework that can directly render various 3D defocus effects, dubbed NeRFocus.
Unlike the pinhole, the thin lens refracts rays of a scene point, so its
imaging on the sensor plane is scattered as a circle of confusion (CoC). A
direct solution sampling enough rays to approximate this process is
computationally expensive. Instead, we propose to inverse the thin lens imaging
to explicitly model the beam path for each point on the sensor plane and
generalize this paradigm to the beam path of each pixel, then use the
frustum-based volume rendering to render each pixel's beam path. We further
design an efficient probabilistic training (p-training) strategy to simplify
the training process vastly. Extensive experiments demonstrate that our
NeRFocus can achieve various 3D defocus effects with adjustable camera pose,
focus distance, and aperture size. Existing NeRF can be regarded as our special
case by setting aperture size as zero to render large depth-of-field images.
Despite such merits, NeRFocus does not sacrifice NeRF's original performance
(e.g., training and inference time, parameter consumption, rendering quality),
which implies its great potential for broader application and further
improvement. Code and video are available at
https://github.com/wyhuai/NeRFocus.

---

## NeRF-Pose: A First-Reconstruct-Then-Regress Approach for  Weakly-supervised 6D Object Pose Estimation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-09 | Fu Li, Hao Yu, Ivan Shugurov, Benjamin Busam, Shaowu Yang, Slobodan Ilic | cs.CV | [PDF](http://arxiv.org/pdf/2203.04802v2){: .btn .btn-green } |

**Abstract**: Pose estimation of 3D objects in monocular images is a fundamental and
long-standing problem in computer vision. Existing deep learning approaches for
6D pose estimation typically rely on the assumption of availability of 3D
object models and 6D pose annotations. However, precise annotation of 6D poses
in real data is intricate, time-consuming and not scalable, while synthetic
data scales well but lacks realism. To avoid these problems, we present a
weakly-supervised reconstruction-based pipeline, named NeRF-Pose, which needs
only 2D object segmentation and known relative camera poses during training.
Following the first-reconstruct-then-regress idea, we first reconstruct the
objects from multiple views in the form of an implicit neural representation.
Then, we train a pose regression network to predict pixel-wise 2D-3D
correspondences between images and the reconstructed model. At inference, the
approach only needs a single image as input. A NeRF-enabled PnP+RANSAC
algorithm is used to estimate stable and accurate pose from the predicted
correspondences. Experiments on LineMod and LineMod-Occlusion show that the
proposed method has state-of-the-art accuracy in comparison to the best 6D pose
estimation methods in spite of being trained only with weak labels. Besides, we
extend the Homebrewed DB dataset with more real training images to support the
weakly supervised task and achieve compelling results on this dataset. The
extended dataset and code will be released soon.

---

## NeReF: Neural Refractive Field for Fluid Surface Reconstruction and  Implicit Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-08 | Ziyu Wang, Wei Yang, Junming Cao, Lan Xu, Junqing Yu, Jingyi Yu | cs.CV | [PDF](http://arxiv.org/pdf/2203.04130v1){: .btn .btn-green } |

**Abstract**: Existing neural reconstruction schemes such as Neural Radiance Field (NeRF)
are largely focused on modeling opaque objects. We present a novel neural
refractive field(NeReF) to recover wavefront of transparent fluids by
simultaneously estimating the surface position and normal of the fluid front.
Unlike prior arts that treat the reconstruction target as a single layer of the
surface, NeReF is specifically formulated to recover a volumetric normal field
with its corresponding density field. A query ray will be refracted by NeReF
according to its accumulated refractive point and normal, and we employ the
correspondences and uniqueness of refracted ray for NeReF optimization. We show
NeReF, as a global optimization scheme, can more robustly tackle refraction
distortions detrimental to traditional methods for correspondence matching.
Furthermore, the continuous NeReF representation of wavefront enables view
synthesis as well as normal integration. We validate our approach on both
synthetic and real data and show it is particularly suitable for sparse
multi-view acquisition. We hence build a small light field array and experiment
on various surface shapes to demonstrate high fidelity NeReF reconstruction.

---

## Kubric: A scalable dataset generator

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-07 | Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J. Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, Thomas Kipf, Abhijit Kundu, Dmitry Lagun, Issam Laradji,  Hsueh-Ti,  Liu, Henning Meyer, Yishu Miao, Derek Nowrouzezahrai, Cengiz Oztireli, Etienne Pot, Noha Radwan, Daniel Rebain, Sara Sabour, Mehdi S. M. Sajjadi, Matan Sela, Vincent Sitzmann, Austin Stone, Deqing Sun, Suhani Vora, Ziyu Wang, Tianhao Wu, Kwang Moo Yi, Fangcheng Zhong, Andrea Tagliasacchi | cs.CV | [PDF](http://arxiv.org/pdf/2203.03570v1){: .btn .btn-green } |

**Abstract**: Data is the driving force of machine learning, with the amount and quality of
training data often being more important for the performance of a system than
architecture and training details. But collecting, processing and annotating
real data at scale is difficult, expensive, and frequently raises additional
privacy, fairness and legal concerns. Synthetic data is a powerful tool with
the potential to address these shortcomings: 1) it is cheap 2) supports rich
ground-truth annotations 3) offers full control over data and 4) can circumvent
or mitigate problems regarding bias, privacy and licensing. Unfortunately,
software tools for effective data generation are less mature than those for
architecture design and training, which leads to fragmented generation efforts.
To address these problems we introduce Kubric, an open-source Python framework
that interfaces with PyBullet and Blender to generate photo-realistic scenes,
with rich annotations, and seamlessly scales to large jobs distributed over
thousands of machines, and generating TBs of data. We demonstrate the
effectiveness of Kubric by presenting a series of 13 different generated
datasets for tasks ranging from studying 3D NeRF models to optical flow
estimation. We release Kubric, the used assets, all of the generation code, as
well as the rendered datasets for reuse and modification.

Comments:
- 21 pages, CVPR2022

---

## NeRF-Supervision: Learning Dense Object Descriptors from Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-03 | Lin Yen-Chen, Pete Florence, Jonathan T. Barron, Tsung-Yi Lin, Alberto Rodriguez, Phillip Isola | cs.RO | [PDF](http://arxiv.org/pdf/2203.01913v2){: .btn .btn-green } |

**Abstract**: Thin, reflective objects such as forks and whisks are common in our daily
lives, but they are particularly challenging for robot perception because it is
hard to reconstruct them using commodity RGB-D cameras or multi-view stereo
techniques. While traditional pipelines struggle with objects like these,
Neural Radiance Fields (NeRFs) have recently been shown to be remarkably
effective for performing view synthesis on objects with thin structures or
reflective materials. In this paper we explore the use of NeRF as a new source
of supervision for robust robot vision systems. In particular, we demonstrate
that a NeRF representation of a scene can be used to train dense object
descriptors. We use an optimized NeRF to extract dense correspondences between
multiple views of an object, and then use these correspondences as training
data for learning a view-invariant representation of the object. NeRF's usage
of a density field allows us to reformulate the correspondence problem with a
novel distribution-of-depths formulation, as opposed to the conventional
approach of using a depth map. Dense correspondence models supervised with our
method significantly outperform off-the-shelf learned descriptors by 106%
(PCK@3px metric, more than doubling performance) and outperform our baseline
supervised with multi-view stereo by 29%. Furthermore, we demonstrate the
learned dense descriptors enable robots to perform accurate 6-degree of freedom
(6-DoF) pick and place of thin and reflective objects.

Comments:
- ICRA 2022, Website: https://yenchenlin.me/nerf-supervision/

---

## NeuroFluid: Fluid Dynamics Grounding with Particle-Driven Neural  Radiance Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-03 | Shanyan Guan, Huayu Deng, Yunbo Wang, Xiaokang Yang | cs.LG | [PDF](http://arxiv.org/pdf/2203.01762v2){: .btn .btn-green } |

**Abstract**: Deep learning has shown great potential for modeling the physical dynamics of
complex particle systems such as fluids. Existing approaches, however, require
the supervision of consecutive particle properties, including positions and
velocities. In this paper, we consider a partially observable scenario known as
fluid dynamics grounding, that is, inferring the state transitions and
interactions within the fluid particle systems from sequential visual
observations of the fluid surface. We propose a differentiable two-stage
network named NeuroFluid. Our approach consists of (i) a particle-driven neural
renderer, which involves fluid physical properties into the volume rendering
function, and (ii) a particle transition model optimized to reduce the
differences between the rendered and the observed images. NeuroFluid provides
the first solution to unsupervised learning of particle-based fluid dynamics by
training these two models jointly. It is shown to reasonably estimate the
underlying physics of fluids with different initial shapes, viscosity, and
densities.

Comments:
- ICML 2022, the project page: https://syguan96.github.io/NeuroFluid/

---

## Playable Environments: Video Manipulation in Space and Time



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-03 | Willi Menapace, Stéphane Lathuilière, Aliaksandr Siarohin, Christian Theobalt, Sergey Tulyakov, Vladislav Golyanik, Elisa Ricci | cs.CV | [PDF](http://arxiv.org/pdf/2203.01914v2){: .btn .btn-green } |

**Abstract**: We present Playable Environments - a new representation for interactive video
generation and manipulation in space and time. With a single image at inference
time, our novel framework allows the user to move objects in 3D while
generating a video by providing a sequence of desired actions. The actions are
learnt in an unsupervised manner. The camera can be controlled to get the
desired viewpoint. Our method builds an environment state for each frame, which
can be manipulated by our proposed action module and decoded back to the image
space with volumetric rendering. To support diverse appearances of objects, we
extend neural radiance fields with style-based modulation. Our method trains on
a collection of various monocular videos requiring only the estimated camera
parameters and 2D object locations. To set a challenging benchmark, we
introduce two large scale video datasets with significant camera movements. As
evidenced by our experiments, playable environments enable several creative
applications not attainable by prior video synthesis works, including playable
3D video generation, stylization and manipulation. Further details, code and
examples are available at
https://willi-menapace.github.io/playable-environments-website

Comments:
- CVPR 2022

---

## ICARUS: A Specialized Architecture for Neural Radiance Fields Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-03-01 | Chaolin Rao, Huangjie Yu, Haochuan Wan, Jindong Zhou, Yueyang Zheng, Yu Ma, Anpei Chen, Minye Wu, Binzhe Yuan, Pingqiang Zhou, Xin Lou, Jingyi Yu | cs.AR | [PDF](http://arxiv.org/pdf/2203.01414v3){: .btn .btn-green } |

**Abstract**: The practical deployment of Neural Radiance Fields (NeRF) in rendering
applications faces several challenges, with the most critical one being low
rendering speed on even high-end graphic processing units (GPUs). In this
paper, we present ICARUS, a specialized accelerator architecture tailored for
NeRF rendering. Unlike GPUs using general purpose computing and memory
architectures for NeRF, ICARUS executes the complete NeRF pipeline using
dedicated plenoptic cores (PLCore) consisting of a positional encoding unit
(PEU), a multi-layer perceptron (MLP) engine, and a volume rendering unit
(VRU). A PLCore takes in positions \& directions and renders the corresponding
pixel colors without any intermediate data going off-chip for temporary storage
and exchange, which can be time and power consuming. To implement the most
expensive component of NeRF, i.e., the MLP, we transform the fully connected
operations to approximated reconfigurable multiple constant multiplications
(MCMs), where common subexpressions are shared across different multiplications
to improve the computation efficiency. We build a prototype ICARUS using
Synopsys HAPS-80 S104, a field programmable gate array (FPGA)-based prototyping
system for large-scale integrated circuits and systems design. We evaluate the
power-performance-area (PPA) of a PLCore using 40nm LP CMOS technology. Working
at 400 MHz, a single PLCore occupies 16.5 $mm^2$ and consumes 282.8 mW,
translating to 0.105 uJ/sample. The results are compared with those of GPU and
tensor processing unit (TPU) implementations.