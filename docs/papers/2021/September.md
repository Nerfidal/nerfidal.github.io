---
layout: default
title: September
parent: 2021
nav_order: 9
---
<!---metadata--->

## TöRF: Time-of-Flight Radiance Fields for Dynamic Scene View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-09-30 | Benjamin Attal, Eliot Laidlaw, Aaron Gokaslan, Changil Kim, Christian Richardt, James Tompkin, Matthew O'Toole | cs.CV | [PDF](http://arxiv.org/pdf/2109.15271v2){: .btn .btn-green } |

**Abstract**: Neural networks can represent and accurately reconstruct radiance fields for
static 3D scenes (e.g., NeRF). Several works extend these to dynamic scenes
captured with monocular video, with promising performance. However, the
monocular setting is known to be an under-constrained problem, and so methods
rely on data-driven priors for reconstructing dynamic content. We replace these
priors with measurements from a time-of-flight (ToF) camera, and introduce a
neural representation based on an image formation model for continuous-wave ToF
cameras. Instead of working with processed depth maps, we model the raw ToF
sensor measurements to improve reconstruction quality and avoid issues with low
reflectance regions, multi-path interference, and a sensor's limited
unambiguous depth range. We show that this approach improves robustness of
dynamic scene reconstruction to erroneous calibration and large motions, and
discuss the benefits and limitations of integrating RGB+ToF sensors that are
now available on modern smartphones.

Comments:
- Accepted to NeurIPS 2021. Web page: https://imaging.cs.cmu.edu/torf/
  NeurIPS camera ready updates -- added quantitative comparisons to new
  methods, visual side-by-side comparisons performed on larger baseline camera
  sequences

---

## Neural Human Performer: Learning Generalizable Radiance Fields for Human  Performance Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-09-15 | Youngjoong Kwon, Dahun Kim, Duygu Ceylan, Henry Fuchs | cs.CV | [PDF](http://arxiv.org/pdf/2109.07448v1){: .btn .btn-green } |

**Abstract**: In this paper, we aim at synthesizing a free-viewpoint video of an arbitrary
human performance using sparse multi-view cameras. Recently, several works have
addressed this problem by learning person-specific neural radiance fields
(NeRF) to capture the appearance of a particular human. In parallel, some work
proposed to use pixel-aligned features to generalize radiance fields to
arbitrary new scenes and objects. Adopting such generalization approaches to
humans, however, is highly challenging due to the heavy occlusions and dynamic
articulations of body parts. To tackle this, we propose Neural Human Performer,
a novel approach that learns generalizable neural radiance fields based on a
parametric human body model for robust performance capture. Specifically, we
first introduce a temporal transformer that aggregates tracked visual features
based on the skeletal body motion over time. Moreover, a multi-view transformer
is proposed to perform cross-attention between the temporally-fused features
and the pixel-aligned features at each time step to integrate observations on
the fly from multiple views. Experiments on the ZJU-MoCap and AIST datasets
show that our method significantly outperforms recent generalizable NeRF
methods on unseen identities and poses. The video results and code are
available at https://youngjoongunc.github.io/nhp.

---

## Stochastic Neural Radiance Fields: Quantifying Uncertainty in Implicit  3D Representations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-09-05 | Jianxiong Shen, Adria Ruiz, Antonio Agudo, Francesc Moreno-Noguer | cs.CV | [PDF](http://arxiv.org/pdf/2109.02123v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) has become a popular framework for learning
implicit 3D representations and addressing different tasks such as novel-view
synthesis or depth-map estimation. However, in downstream applications where
decisions need to be made based on automatic predictions, it is critical to
leverage the confidence associated with the model estimations. Whereas
uncertainty quantification is a long-standing problem in Machine Learning, it
has been largely overlooked in the recent NeRF literature. In this context, we
propose Stochastic Neural Radiance Fields (S-NeRF), a generalization of
standard NeRF that learns a probability distribution over all the possible
radiance fields modeling the scene. This distribution allows to quantify the
uncertainty associated with the scene information provided by the model. S-NeRF
optimization is posed as a Bayesian learning problem which is efficiently
addressed using the Variational Inference framework. Exhaustive experiments
over benchmark datasets demonstrate that S-NeRF is able to provide more
reliable predictions and confidence values than generic approaches previously
proposed for uncertainty estimation in other domains.

---

## Learning Object-Compositional Neural Radiance Field for Editable Scene  Rendering



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-09-04 | Bangbang Yang, Yinda Zhang, Yinghao Xu, Yijin Li, Han Zhou, Hujun Bao, Guofeng Zhang, Zhaopeng Cui | cs.CV | [PDF](http://arxiv.org/pdf/2109.01847v1){: .btn .btn-green } |

**Abstract**: Implicit neural rendering techniques have shown promising results for novel
view synthesis. However, existing methods usually encode the entire scene as a
whole, which is generally not aware of the object identity and limits the
ability to the high-level editing tasks such as moving or adding furniture. In
this paper, we present a novel neural scene rendering system, which learns an
object-compositional neural radiance field and produces realistic rendering
with editing capability for a clustered and real-world scene. Specifically, we
design a novel two-pathway architecture, in which the scene branch encodes the
scene geometry and appearance, and the object branch encodes each standalone
object conditioned on learnable object activation codes. To survive the
training in heavily cluttered scenes, we propose a scene-guided training
strategy to solve the 3D space ambiguity in the occluded regions and learn
sharp boundaries for each object. Extensive experiments demonstrate that our
system not only achieves competitive performance for static scene novel-view
synthesis, but also produces realistic rendering for object-level editing.

Comments:
- Accepted to ICCV 2021. Project Page:
  https://zju3dv.github.io/object_nerf

---

## CodeNeRF: Disentangled Neural Radiance Fields for Object Categories

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-09-03 | Wonbong Jang, Lourdes Agapito | cs.GR | [PDF](http://arxiv.org/pdf/2109.01750v1){: .btn .btn-green } |

**Abstract**: CodeNeRF is an implicit 3D neural representation that learns the variation of
object shapes and textures across a category and can be trained, from a set of
posed images, to synthesize novel views of unseen objects. Unlike the original
NeRF, which is scene specific, CodeNeRF learns to disentangle shape and texture
by learning separate embeddings. At test time, given a single unposed image of
an unseen object, CodeNeRF jointly estimates camera viewpoint, and shape and
appearance codes via optimization. Unseen objects can be reconstructed from a
single image, and then rendered from new viewpoints or their shape and texture
edited by varying the latent codes. We conduct experiments on the SRN
benchmark, which show that CodeNeRF generalises well to unseen objects and
achieves on-par performance with methods that require known camera pose at test
time. Our results on real-world images demonstrate that CodeNeRF can bridge the
sim-to-real gap. Project page: \url{https://github.com/wayne1123/code-nerf}

Comments:
- 10 pages, 15 figures, ICCV 2021

---

## NerfingMVS: Guided Optimization of Neural Radiance Fields for Indoor  Multi-view Stereo

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-09-02 | Yi Wei, Shaohui Liu, Yongming Rao, Wang Zhao, Jiwen Lu, Jie Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2109.01129v3){: .btn .btn-green } |

**Abstract**: In this work, we present a new multi-view depth estimation method that
utilizes both conventional reconstruction and learning-based priors over the
recently proposed neural radiance fields (NeRF). Unlike existing neural network
based optimization method that relies on estimated correspondences, our method
directly optimizes over implicit volumes, eliminating the challenging step of
matching pixels in indoor scenes. The key to our approach is to utilize the
learning-based priors to guide the optimization process of NeRF. Our system
firstly adapts a monocular depth network over the target scene by finetuning on
its sparse SfM+MVS reconstruction from COLMAP. Then, we show that the
shape-radiance ambiguity of NeRF still exists in indoor environments and
propose to address the issue by employing the adapted depth priors to monitor
the sampling process of volume rendering. Finally, a per-pixel confidence map
acquired by error computation on the rendered image can be used to further
improve the depth quality. Experiments show that our proposed framework
significantly outperforms state-of-the-art methods on indoor scenes, with
surprising findings presented on the effectiveness of correspondence-based
optimization and NeRF-based optimization over the adapted depth priors. In
addition, we show that the guided optimization scheme does not sacrifice the
original synthesis capability of neural radiance fields, improving the
rendering quality on both seen and novel views. Code is available at
https://github.com/weiyithu/NerfingMVS.

Comments:
- To appear in ICCV 2021 (Oral). Project page:
  https://weiyithu.github.io/NerfingMVS/