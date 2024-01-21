---
layout: default
title: December
parent: 2020
nav_order: 12
---
<!---metadata--->

## STaR: Self-supervised Tracking and Reconstruction of Rigid Objects in  Motion with Neural Rendering



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-12-22 | Wentao Yuan, Zhaoyang Lv, Tanner Schmidt, Steven Lovegrove | cs.CV | [PDF](http://arxiv.org/pdf/2101.01602v1){: .btn .btn-green } |

**Abstract**: We present STaR, a novel method that performs Self-supervised Tracking and
Reconstruction of dynamic scenes with rigid motion from multi-view RGB videos
without any manual annotation. Recent work has shown that neural networks are
surprisingly effective at the task of compressing many views of a scene into a
learned function which maps from a viewing ray to an observed radiance value
via volume rendering. Unfortunately, these methods lose all their predictive
power once any object in the scene has moved. In this work, we explicitly model
rigid motion of objects in the context of neural representations of radiance
fields. We show that without any additional human specified supervision, we can
reconstruct a dynamic scene with a single rigid object in motion by
simultaneously decomposing it into its two constituent parts and encoding each
with its own neural representation. We achieve this by jointly optimizing the
parameters of two neural radiance fields and a set of rigid poses which align
the two fields at each frame. On both synthetic and real world datasets, we
demonstrate that our method can render photorealistic novel views, where
novelty is measured on both spatial and temporal axes. Our factored
representation furthermore enables animation of unseen object motion.

---

## Non-Rigid Neural Radiance Fields: Reconstruction and Novel View  Synthesis of a Dynamic Scene From Monocular Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-12-22 | Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhöfer, Christoph Lassner, Christian Theobalt | cs.CV | [PDF](http://arxiv.org/pdf/2012.12247v4){: .btn .btn-green } |

**Abstract**: We present Non-Rigid Neural Radiance Fields (NR-NeRF), a reconstruction and
novel view synthesis approach for general non-rigid dynamic scenes. Our
approach takes RGB images of a dynamic scene as input (e.g., from a monocular
video recording), and creates a high-quality space-time geometry and appearance
representation. We show that a single handheld consumer-grade camera is
sufficient to synthesize sophisticated renderings of a dynamic scene from novel
virtual camera views, e.g. a `bullet-time' video effect. NR-NeRF disentangles
the dynamic scene into a canonical volume and its deformation. Scene
deformation is implemented as ray bending, where straight rays are deformed
non-rigidly. We also propose a novel rigidity network to better constrain rigid
regions of the scene, leading to more stable results. The ray bending and
rigidity network are trained without explicit supervision. Our formulation
enables dense correspondence estimation across views and time, and compelling
video editing applications such as motion exaggeration. Our code will be open
sourced.

Comments:
- Project page (incl. supplemental videos and code):
  https://vcai.mpi-inf.mpg.de/projects/nonrigid_nerf/ or
  https://gvv.mpi-inf.mpg.de/projects/nonrigid_nerf/

---

## Learning Compositional Radiance Fields of Dynamic Human Heads

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-12-17 | Ziyan Wang, Timur Bagautdinov, Stephen Lombardi, Tomas Simon, Jason Saragih, Jessica Hodgins, Michael Zollhöfer | cs.CV | [PDF](http://arxiv.org/pdf/2012.09955v1){: .btn .btn-green } |

**Abstract**: Photorealistic rendering of dynamic humans is an important ability for
telepresence systems, virtual shopping, synthetic data generation, and more.
Recently, neural rendering methods, which combine techniques from computer
graphics and machine learning, have created high-fidelity models of humans and
objects. Some of these methods do not produce results with high-enough fidelity
for driveable human models (Neural Volumes) whereas others have extremely long
rendering times (NeRF). We propose a novel compositional 3D representation that
combines the best of previous methods to produce both higher-resolution and
faster results. Our representation bridges the gap between discrete and
continuous volumetric representations by combining a coarse 3D-structure-aware
grid of animation codes with a continuous learned scene function that maps
every position and its corresponding local animation code to its view-dependent
emitted radiance and local volume density. Differentiable volume rendering is
employed to compute photo-realistic novel views of the human head and upper
body as well as to train our novel representation end-to-end using only 2D
supervision. In addition, we show that the learned dynamic radiance field can
be used to synthesize novel unseen expressions based on a global animation
code. Our approach achieves state-of-the-art results for synthesizing novel
views of dynamic human heads and the upper body.

---

## Neural Volume Rendering: NeRF And Beyond

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-12-17 | Frank Dellaert, Lin Yen-Chen | cs.CV | [PDF](http://arxiv.org/pdf/2101.05204v2){: .btn .btn-green } |

**Abstract**: Besides the COVID-19 pandemic and political upheaval in the US, 2020 was also
the year in which neural volume rendering exploded onto the scene, triggered by
the impressive NeRF paper by Mildenhall et al. (2020). Both of us have tried to
capture this excitement, Frank on a blog post (Dellaert, 2020) and Yen-Chen in
a Github collection (Yen-Chen, 2020). This note is an annotated bibliography of
the relevant papers, and we posted the associated bibtex file on the
repository.

Comments:
- Blog: https://dellaert.github.io/NeRF/ Bibtex:
  https://github.com/yenchenlin/awesome-NeRF

---

## Object-Centric Neural Scene Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-12-15 | Michelle Guo, Alireza Fathi, Jiajun Wu, Thomas Funkhouser | cs.CV | [PDF](http://arxiv.org/pdf/2012.08503v1){: .btn .btn-green } |

**Abstract**: We present a method for composing photorealistic scenes from captured images
of objects. Our work builds upon neural radiance fields (NeRFs), which
implicitly model the volumetric density and directionally-emitted radiance of a
scene. While NeRFs synthesize realistic pictures, they only model static scenes
and are closely tied to specific imaging conditions. This property makes NeRFs
hard to generalize to new scenarios, including new lighting or new arrangements
of objects. Instead of learning a scene radiance field as a NeRF does, we
propose to learn object-centric neural scattering functions (OSFs), a
representation that models per-object light transport implicitly using a
lighting- and view-dependent neural network. This enables rendering scenes even
when objects or lights move, without retraining. Combined with a volumetric
path tracing procedure, our framework is capable of rendering both intra- and
inter-object light transport effects including occlusions, specularities,
shadows, and indirect illumination. We evaluate our approach on scene
composition and show that it generalizes to novel illumination conditions,
producing photorealistic, physically accurate renderings of multi-object
scenes.

Comments:
- Summary Video: https://youtu.be/NtR7xgxSL1U Project Webpage:
  https://shellguo.com/osf

---

## Portrait Neural Radiance Fields from a Single Image

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-12-10 | Chen Gao, Yichang Shih, Wei-Sheng Lai, Chia-Kai Liang, Jia-Bin Huang | cs.CV | [PDF](http://arxiv.org/pdf/2012.05903v2){: .btn .btn-green } |

**Abstract**: We present a method for estimating Neural Radiance Fields (NeRF) from a
single headshot portrait. While NeRF has demonstrated high-quality view
synthesis, it requires multiple images of static scenes and thus impractical
for casual captures and moving subjects. In this work, we propose to pretrain
the weights of a multilayer perceptron (MLP), which implicitly models the
volumetric density and colors, with a meta-learning framework using a light
stage portrait dataset. To improve the generalization to unseen faces, we train
the MLP in the canonical coordinate space approximated by 3D face morphable
models. We quantitatively evaluate the method using controlled captures and
demonstrate the generalization to real portrait images, showing favorable
results against state-of-the-arts.

Comments:
- Project webpage: https://portrait-nerf.github.io/

---

## INeRF: Inverting Neural Radiance Fields for Pose Estimation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-12-10 | Lin Yen-Chen, Pete Florence, Jonathan T. Barron, Alberto Rodriguez, Phillip Isola, Tsung-Yi Lin | cs.CV | [PDF](http://arxiv.org/pdf/2012.05877v3){: .btn .btn-green } |

**Abstract**: We present iNeRF, a framework that performs mesh-free pose estimation by
"inverting" a Neural RadianceField (NeRF). NeRFs have been shown to be
remarkably effective for the task of view synthesis - synthesizing
photorealistic novel views of real-world scenes or objects. In this work, we
investigate whether we can apply analysis-by-synthesis via NeRF for mesh-free,
RGB-only 6DoF pose estimation - given an image, find the translation and
rotation of a camera relative to a 3D object or scene. Our method assumes that
no object mesh models are available during either training or test time.
Starting from an initial pose estimate, we use gradient descent to minimize the
residual between pixels rendered from a NeRF and pixels in an observed image.
In our experiments, we first study 1) how to sample rays during pose refinement
for iNeRF to collect informative gradients and 2) how different batch sizes of
rays affect iNeRF on a synthetic dataset. We then show that for complex
real-world scenes from the LLFF dataset, iNeRF can improve NeRF by estimating
the camera poses of novel images and using these images as additional training
data for NeRF. Finally, we show iNeRF can perform category-level object pose
estimation, including object instances not seen during training, with RGB
images by inverting a NeRF model inferred from a single view.

Comments:
- IROS 2021, Website: http://yenchenlin.me/inerf/

---

## Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar  Reconstruction



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-12-05 | Guy Gafni, Justus Thies, Michael Zollhöfer, Matthias Nießner | cs.CV | [PDF](http://arxiv.org/pdf/2012.03065v1){: .btn .btn-green } |

**Abstract**: We present dynamic neural radiance fields for modeling the appearance and
dynamics of a human face. Digitally modeling and reconstructing a talking human
is a key building-block for a variety of applications. Especially, for
telepresence applications in AR or VR, a faithful reproduction of the
appearance including novel viewpoints or head-poses is required. In contrast to
state-of-the-art approaches that model the geometry and material properties
explicitly, or are purely image-based, we introduce an implicit representation
of the head based on scene representation networks. To handle the dynamics of
the face, we combine our scene representation network with a low-dimensional
morphable model which provides explicit control over pose and expressions. We
use volumetric rendering to generate images from this hybrid representation and
demonstrate that such a dynamic neural scene representation can be learned from
monocular input data only, without the need of a specialized capture setup. In
our experiments, we show that this learned volumetric representation allows for
photo-realistic image generation that surpasses the quality of state-of-the-art
video-based reenactment methods.

Comments:
- Video: https://youtu.be/m7oROLdQnjk | Project page:
  https://gafniguy.github.io/4D-Facial-Avatars/

---

## pixelNeRF: Neural Radiance Fields from One or Few Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-12-03 | Alex Yu, Vickie Ye, Matthew Tancik, Angjoo Kanazawa | cs.CV | [PDF](http://arxiv.org/pdf/2012.02190v3){: .btn .btn-green } |

**Abstract**: We propose pixelNeRF, a learning framework that predicts a continuous neural
scene representation conditioned on one or few input images. The existing
approach for constructing neural radiance fields involves optimizing the
representation to every scene independently, requiring many calibrated views
and significant compute time. We take a step towards resolving these
shortcomings by introducing an architecture that conditions a NeRF on image
inputs in a fully convolutional manner. This allows the network to be trained
across multiple scenes to learn a scene prior, enabling it to perform novel
view synthesis in a feed-forward manner from a sparse set of views (as few as
one). Leveraging the volume rendering approach of NeRF, our model can be
trained directly from images with no explicit 3D supervision. We conduct
extensive experiments on ShapeNet benchmarks for single image novel view
synthesis tasks with held-out objects as well as entire unseen categories. We
further demonstrate the flexibility of pixelNeRF by demonstrating it on
multi-object ShapeNet scenes and real scenes from the DTU dataset. In all
cases, pixelNeRF outperforms current state-of-the-art baselines for novel view
synthesis and single image 3D reconstruction. For the video and code, please
visit the project website: https://alexyu.net/pixelnerf

Comments:
- CVPR 2021