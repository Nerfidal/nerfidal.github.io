---
layout: default
title: January
parent: 2022
nav_order: 1
---
<!---metadata--->

## From data to functa: Your data point is a function and you can treat it  like one

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-01-28 | Emilien Dupont, Hyunjik Kim, S. M. Ali Eslami, Danilo Rezende, Dan Rosenbaum | cs.LG | [PDF](http://arxiv.org/pdf/2201.12204v3){: .btn .btn-green } |

**Abstract**: It is common practice in deep learning to represent a measurement of the
world on a discrete grid, e.g. a 2D grid of pixels. However, the underlying
signal represented by these measurements is often continuous, e.g. the scene
depicted in an image. A powerful continuous alternative is then to represent
these measurements using an implicit neural representation, a neural function
trained to output the appropriate measurement value for any input spatial
location. In this paper, we take this idea to its next level: what would it
take to perform deep learning on these functions instead, treating them as
data? In this context we refer to the data as functa, and propose a framework
for deep learning on functa. This view presents a number of challenges around
efficient conversion from data to functa, compact representation of functa, and
effectively solving downstream tasks on functa. We outline a recipe to overcome
these challenges and apply it to a wide range of data modalities including
images, 3D shapes, neural radiance fields (NeRF) and data on manifolds. We
demonstrate that this approach has various compelling properties across data
modalities, in particular on the canonical tasks of generative modeling, data
imputation, novel view synthesis and classification. Code:
https://github.com/deepmind/functa

---

## Point-NeRF: Point-based Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-01-21 | Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, Ulrich Neumann | cs.CV | [PDF](http://arxiv.org/pdf/2201.08845v7){: .btn .btn-green } |

**Abstract**: Volumetric neural rendering methods like NeRF generate high-quality view
synthesis results but are optimized per-scene leading to prohibitive
reconstruction time. On the other hand, deep multi-view stereo methods can
quickly reconstruct scene geometry via direct network inference. Point-NeRF
combines the advantages of these two approaches by using neural 3D point
clouds, with associated neural features, to model a radiance field. Point-NeRF
can be rendered efficiently by aggregating neural point features near scene
surfaces, in a ray marching-based rendering pipeline. Moreover, Point-NeRF can
be initialized via direct inference of a pre-trained deep network to produce a
neural point cloud; this point cloud can be finetuned to surpass the visual
quality of NeRF with 30X faster training time. Point-NeRF can be combined with
other 3D reconstruction methods and handles the errors and outliers in such
methods via a novel pruning and growing mechanism. The experiments on the DTU,
the NeRF Synthetics , the ScanNet and the Tanks and Temples datasets
demonstrate Point-NeRF can surpass the existing methods and achieve the
state-of-the-art results.

Comments:
- Accepted to CVPR 2022 (Oral)

---

## Semantic-Aware Implicit Neural Audio-Driven Video Portrait Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-01-19 | Xian Liu, Yinghao Xu, Qianyi Wu, Hang Zhou, Wayne Wu, Bolei Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2201.07786v1){: .btn .btn-green } |

**Abstract**: Animating high-fidelity video portrait with speech audio is crucial for
virtual reality and digital entertainment. While most previous studies rely on
accurate explicit structural information, recent works explore the implicit
scene representation of Neural Radiance Fields (NeRF) for realistic generation.
In order to capture the inconsistent motions as well as the semantic difference
between human head and torso, some work models them via two individual sets of
NeRF, leading to unnatural results. In this work, we propose Semantic-aware
Speaking Portrait NeRF (SSP-NeRF), which creates delicate audio-driven
portraits using one unified set of NeRF. The proposed model can handle the
detailed local facial semantics and the global head-torso relationship through
two semantic-aware modules. Specifically, we first propose a Semantic-Aware
Dynamic Ray Sampling module with an additional parsing branch that facilitates
audio-driven volume rendering. Moreover, to enable portrait rendering in one
unified neural radiance field, a Torso Deformation module is designed to
stabilize the large-scale non-rigid torso motions. Extensive evaluations
demonstrate that our proposed approach renders more realistic video portraits
compared to previous methods. Project page:
https://alvinliu0.github.io/projects/SSP-NeRF

Comments:
- 12 pages, 3 figures. Project page:
  https://alvinliu0.github.io/projects/SSP-NeRF

---

## Virtual Elastic Objects



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-01-12 | Hsiao-yu Chen, Edgar Tretschk, Tuur Stuyck, Petr Kadlecek, Ladislav Kavan, Etienne Vouga, Christoph Lassner | cs.CV | [PDF](http://arxiv.org/pdf/2201.04623v1){: .btn .btn-green } |

**Abstract**: We present Virtual Elastic Objects (VEOs): virtual objects that not only look
like their real-world counterparts but also behave like them, even when subject
to novel interactions. Achieving this presents multiple challenges: not only do
objects have to be captured including the physical forces acting on them, then
faithfully reconstructed and rendered, but also plausible material parameters
found and simulated. To create VEOs, we built a multi-view capture system that
captures objects under the influence of a compressed air stream. Building on
recent advances in model-free, dynamic Neural Radiance Fields, we reconstruct
the objects and corresponding deformation fields. We propose to use a
differentiable, particle-based simulator to use these deformation fields to
find representative material parameters, which enable us to run new
simulations. To render simulated objects, we devise a method for integrating
the simulation results with Neural Radiance Fields. The resulting method is
applicable to a wide range of scenarios: it can handle objects composed of
inhomogeneous material, with very different shapes, and it can simulate
interactions with other virtual objects. We present our results using a newly
collected dataset of 12 objects under a variety of force fields, which will be
shared with the community.

---

## NeROIC: Neural Rendering of Objects from Online Image Collections



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-01-07 | Zhengfei Kuang, Kyle Olszewski, Menglei Chai, Zeng Huang, Panos Achlioptas, Sergey Tulyakov | cs.CV | [PDF](http://arxiv.org/pdf/2201.02533v2){: .btn .btn-green } |

**Abstract**: We present a novel method to acquire object representations from online image
collections, capturing high-quality geometry and material properties of
arbitrary objects from photographs with varying cameras, illumination, and
backgrounds. This enables various object-centric rendering applications such as
novel-view synthesis, relighting, and harmonized background composition from
challenging in-the-wild input. Using a multi-stage approach extending neural
radiance fields, we first infer the surface geometry and refine the coarsely
estimated initial camera parameters, while leveraging coarse foreground object
masks to improve the training efficiency and geometry quality. We also
introduce a robust normal estimation technique which eliminates the effect of
geometric noise while retaining crucial details. Lastly, we extract surface
material properties and ambient illumination, represented in spherical
harmonics with extensions that handle transient elements, e.g. sharp shadows.
The union of these components results in a highly modular and efficient object
acquisition framework. Extensive evaluations and comparisons demonstrate the
advantages of our approach in capturing high-quality geometry and appearance
properties useful for rendering applications.

Comments:
- SIGGRAPH 2022 (Journal Track). Project page:
  https://formyfamily.github.io/NeROIC/ Code repository:
  https://github.com/snap-research/NeROIC/

---

## Surface-Aligned Neural Radiance Fields for Controllable 3D Human  Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-01-05 | Tianhan Xu, Yasuhiro Fujita, Eiichi Matsumoto | cs.CV | [PDF](http://arxiv.org/pdf/2201.01683v2){: .btn .btn-green } |

**Abstract**: We propose a new method for reconstructing controllable implicit 3D human
models from sparse multi-view RGB videos. Our method defines the neural scene
representation on the mesh surface points and signed distances from the surface
of a human body mesh. We identify an indistinguishability issue that arises
when a point in 3D space is mapped to its nearest surface point on a mesh for
learning surface-aligned neural scene representation. To address this issue, we
propose projecting a point onto a mesh surface using a barycentric
interpolation with modified vertex normals. Experiments with the ZJU-MoCap and
Human3.6M datasets show that our approach achieves a higher quality in a
novel-view and novel-pose synthesis than existing methods. We also demonstrate
that our method easily supports the control of body shape and clothes. Project
page: https://pfnet-research.github.io/surface-aligned-nerf/.

Comments:
- CVPR 2022. Project page:
  https://pfnet-research.github.io/surface-aligned-nerf/

---

## DFA-NeRF: Personalized Talking Head Generation via Disentangled Face  Attributes Neural Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-01-03 | Shunyu Yao, RuiZhe Zhong, Yichao Yan, Guangtao Zhai, Xiaokang Yang | cs.CV | [PDF](http://arxiv.org/pdf/2201.00791v1){: .btn .btn-green } |

**Abstract**: While recent advances in deep neural networks have made it possible to render
high-quality images, generating photo-realistic and personalized talking head
remains challenging. With given audio, the key to tackling this task is
synchronizing lip movement and simultaneously generating personalized
attributes like head movement and eye blink. In this work, we observe that the
input audio is highly correlated to lip motion while less correlated to other
personalized attributes (e.g., head movements). Inspired by this, we propose a
novel framework based on neural radiance field to pursue high-fidelity and
personalized talking head generation. Specifically, neural radiance field takes
lip movements features and personalized attributes as two disentangled
conditions, where lip movements are directly predicted from the audio inputs to
achieve lip-synchronized generation. In the meanwhile, personalized attributes
are sampled from a probabilistic model, where we design a Transformer-based
variational autoencoder sampled from Gaussian Process to learn plausible and
natural-looking head pose and eye blink. Experiments on several benchmarks
demonstrate that our method achieves significantly better results than
state-of-the-art methods.