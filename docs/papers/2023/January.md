---
layout: default
title: January
parent: 2023
nav_order: 1
---
<!---metadata--->

## GeneFace: Generalized and High-Fidelity Audio-Driven 3D Talking Face  Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-31 | Zhenhui Ye, Ziyue Jiang, Yi Ren, Jinglin Liu, JinZheng He, Zhou Zhao | cs.CV | [PDF](http://arxiv.org/pdf/2301.13430v1){: .btn .btn-green } |

**Abstract**: Generating photo-realistic video portrait with arbitrary speech audio is a
crucial problem in film-making and virtual reality. Recently, several works
explore the usage of neural radiance field in this task to improve 3D realness
and image fidelity. However, the generalizability of previous NeRF-based
methods to out-of-domain audio is limited by the small scale of training data.
In this work, we propose GeneFace, a generalized and high-fidelity NeRF-based
talking face generation method, which can generate natural results
corresponding to various out-of-domain audio. Specifically, we learn a
variaitional motion generator on a large lip-reading corpus, and introduce a
domain adaptative post-net to calibrate the result. Moreover, we learn a
NeRF-based renderer conditioned on the predicted facial motion. A head-aware
torso-NeRF is proposed to eliminate the head-torso separation problem.
Extensive experiments show that our method achieves more generalized and
high-fidelity talking face generation compared to previous methods.

Comments:
- Accepted by ICLR2023. Project page: https://geneface.github.io/

---

## Equivariant Architectures for Learning in Deep Weight Spaces

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-30 | Aviv Navon, Aviv Shamsian, Idan Achituve, Ethan Fetaya, Gal Chechik, Haggai Maron | cs.LG | [PDF](http://arxiv.org/pdf/2301.12780v2){: .btn .btn-green } |

**Abstract**: Designing machine learning architectures for processing neural networks in
their raw weight matrix form is a newly introduced research direction.
Unfortunately, the unique symmetry structure of deep weight spaces makes this
design very challenging. If successful, such architectures would be capable of
performing a wide range of intriguing tasks, from adapting a pre-trained
network to a new domain to editing objects represented as functions (INRs or
NeRFs). As a first step towards this goal, we present here a novel network
architecture for learning in deep weight spaces. It takes as input a
concatenation of weights and biases of a pre-trained MLP and processes it using
a composition of layers that are equivariant to the natural permutation
symmetry of the MLP's weights: Changing the order of neurons in intermediate
layers of the MLP does not affect the function it represents. We provide a full
characterization of all affine equivariant and invariant layers for these
symmetries and show how these layers can be implemented using three basic
operations: pooling, broadcasting, and fully connected layers applied to the
input in an appropriate manner. We demonstrate the effectiveness of our
architecture and its advantages over natural baselines in a variety of learning
tasks.

Comments:
- ICML 2023

---

## HyperNeRFGAN: Hypernetwork approach to 3D NeRF GAN

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-27 | Adam Kania, Artur Kasymov, Maciej Zięba, Przemysław Spurek | cs.CV | [PDF](http://arxiv.org/pdf/2301.11631v1){: .btn .btn-green } |

**Abstract**: Recently, generative models for 3D objects are gaining much popularity in VR
and augmented reality applications. Training such models using standard 3D
representations, like voxels or point clouds, is challenging and requires
complex tools for proper color rendering. In order to overcome this limitation,
Neural Radiance Fields (NeRFs) offer a state-of-the-art quality in synthesizing
novel views of complex 3D scenes from a small subset of 2D images.
  In the paper, we propose a generative model called HyperNeRFGAN, which uses
hypernetworks paradigm to produce 3D objects represented by NeRF. Our GAN
architecture leverages a hypernetwork paradigm to transfer gaussian noise into
weights of NeRF model. The model is further used to render 2D novel views, and
a classical 2D discriminator is utilized for training the entire GAN-based
structure. Our architecture produces 2D images, but we use 3D-aware NeRF
representation, which forces the model to produce correct 3D objects. The
advantage of the model over existing approaches is that it produces a dedicated
NeRF representation for the object without sharing some global parameters of
the rendering component. We show the superiority of our approach compared to
reference baselines on three challenging datasets from various domains.

---

## A Comparison of Tiny-nerf versus Spatial Representations for 3d  Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-27 | Saulo Abraham Gante, Juan Irving Vasquez, Marco Antonio Valencia, Mauricio Olguín Carbajal | cs.AI | [PDF](http://arxiv.org/pdf/2301.11522v1){: .btn .btn-green } |

**Abstract**: Neural rendering has emerged as a powerful paradigm for synthesizing images,
offering many benefits over classical rendering by using neural networks to
reconstruct surfaces, represent shapes, and synthesize novel views, either for
objects or scenes. In this neural rendering, the environment is encoded into a
neural network. We believe that these new representations can be used to codify
the scene for a mobile robot. Therefore, in this work, we perform a comparison
between a trending neural rendering, called tiny-NeRF, and other volume
representations that are commonly used as maps in robotics, such as voxel maps,
point clouds, and triangular meshes. The target is to know the advantages and
disadvantages of neural representations in the robotics context. The comparison
is made in terms of spatial complexity and processing time to obtain a model.
Experiments show that tiny-NeRF requires three times less memory space compared
to other representations. In terms of processing time, tiny-NeRF takes about
six times more to compute the model.

---

## SNeRL: Semantic-aware Neural Radiance Fields for Reinforcement Learning

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-27 | Dongseok Shim, Seungjae Lee, H. Jin Kim | cs.LG | [PDF](http://arxiv.org/pdf/2301.11520v3){: .btn .btn-green } |

**Abstract**: As previous representations for reinforcement learning cannot effectively
incorporate a human-intuitive understanding of the 3D environment, they usually
suffer from sub-optimal performances. In this paper, we present Semantic-aware
Neural Radiance Fields for Reinforcement Learning (SNeRL), which jointly
optimizes semantic-aware neural radiance fields (NeRF) with a convolutional
encoder to learn 3D-aware neural implicit representation from multi-view
images. We introduce 3D semantic and distilled feature fields in parallel to
the RGB radiance fields in NeRF to learn semantic and object-centric
representation for reinforcement learning. SNeRL outperforms not only previous
pixel-based representations but also recent 3D-aware representations both in
model-free and model-based reinforcement learning.

Comments:
- ICML 2023. First two authors contributed equally. Order was
  determined by coin flip

---

## Text-To-4D Dynamic Scene Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-26 | Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, Yaniv Taigman | cs.CV | [PDF](http://arxiv.org/pdf/2301.11280v1){: .btn .btn-green } |

**Abstract**: We present MAV3D (Make-A-Video3D), a method for generating three-dimensional
dynamic scenes from text descriptions. Our approach uses a 4D dynamic Neural
Radiance Field (NeRF), which is optimized for scene appearance, density, and
motion consistency by querying a Text-to-Video (T2V) diffusion-based model. The
dynamic video output generated from the provided text can be viewed from any
camera location and angle, and can be composited into any 3D environment. MAV3D
does not require any 3D or 4D data and the T2V model is trained only on
Text-Image pairs and unlabeled videos. We demonstrate the effectiveness of our
approach using comprehensive quantitative and qualitative experiments and show
an improvement over previously established internal baselines. To the best of
our knowledge, our method is the first to generate 3D dynamic scenes given a
text description.

---

## GeCoNeRF: Few-shot Neural Radiance Fields via Geometric Consistency

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-26 | Min-seop Kwak, Jiuhn Song, Seungryong Kim | cs.CV | [PDF](http://arxiv.org/pdf/2301.10941v3){: .btn .btn-green } |

**Abstract**: We present a novel framework to regularize Neural Radiance Field (NeRF) in a
few-shot setting with a geometry-aware consistency regularization. The proposed
approach leverages a rendered depth map at unobserved viewpoint to warp sparse
input images to the unobserved viewpoint and impose them as pseudo ground
truths to facilitate learning of NeRF. By encouraging such geometry-aware
consistency at a feature-level instead of using pixel-level reconstruction
loss, we regularize the NeRF at semantic and structural levels while allowing
for modeling view dependent radiance to account for color variations across
viewpoints. We also propose an effective method to filter out erroneous warped
solutions, along with training strategies to stabilize training during
optimization. We show that our model achieves competitive results compared to
state-of-the-art few-shot NeRF models. Project page is available at
https://ku-cvlab.github.io/GeCoNeRF/.

Comments:
- ICML 2023

---

## Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-25 | Magdalena Wysocki, Mohammad Farid Azampour, Christine Eilers, Benjamin Busam, Mehrdad Salehi, Nassir Navab | eess.IV | [PDF](http://arxiv.org/pdf/2301.10520v2){: .btn .btn-green } |

**Abstract**: We present a physics-enhanced implicit neural representation (INR) for
ultrasound (US) imaging that learns tissue properties from overlapping US
sweeps. Our proposed method leverages a ray-tracing-based neural rendering for
novel view US synthesis. Recent publications demonstrated that INR models could
encode a representation of a three-dimensional scene from a set of
two-dimensional US frames. However, these models fail to consider the
view-dependent changes in appearance and geometry intrinsic to US imaging. In
our work, we discuss direction-dependent changes in the scene and show that a
physics-inspired rendering improves the fidelity of US image synthesis. In
particular, we demonstrate experimentally that our proposed method generates
geometrically accurate B-mode images for regions with ambiguous representation
owing to view-dependent differences of the US images. We conduct our
experiments using simulated B-mode US sweeps of the liver and acquired US
sweeps of a spine phantom tracked with a robotic arm. The experiments
corroborate that our method generates US frames that enable consistent volume
compounding from previously unseen views. To the best of our knowledge, the
presented work is the first to address view-dependent US image synthesis using
INR.

Comments:
- accepted for oral presentation at MIDL 2023
  (https://openreview.net/forum?id=x4McMBwVyi)

---

## HexPlane: A Fast Representation for Dynamic Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-23 | Ang Cao, Justin Johnson | cs.CV | [PDF](http://arxiv.org/pdf/2301.09632v2){: .btn .btn-green } |

**Abstract**: Modeling and re-rendering dynamic 3D scenes is a challenging task in 3D
vision. Prior approaches build on NeRF and rely on implicit representations.
This is slow since it requires many MLP evaluations, constraining real-world
applications. We show that dynamic 3D scenes can be explicitly represented by
six planes of learned features, leading to an elegant solution we call
HexPlane. A HexPlane computes features for points in spacetime by fusing
vectors extracted from each plane, which is highly efficient. Pairing a
HexPlane with a tiny MLP to regress output colors and training via volume
rendering gives impressive results for novel view synthesis on dynamic scenes,
matching the image quality of prior work but reducing training time by more
than $100\times$. Extensive ablations confirm our HexPlane design and show that
it is robust to different feature fusion mechanisms, coordinate systems, and
decoding mechanisms. HexPlane is a simple and effective solution for
representing 4D volumes, and we hope they can broadly contribute to modeling
spacetime for dynamic 3D scenes.

Comments:
- CVPR 2023, Camera Ready Project page:
  https://caoang327.github.io/HexPlane

---

## 3D Reconstruction of Non-cooperative Resident Space Objects using  Instant NGP-accelerated NeRF and D-NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-22 | Basilio Caruso, Trupti Mahendrakar, Van Minh Nguyen, Ryan T. White, Todd Steffen | cs.CV | [PDF](http://arxiv.org/pdf/2301.09060v3){: .btn .btn-green } |

**Abstract**: The proliferation of non-cooperative resident space objects (RSOs) in orbit
has spurred the demand for active space debris removal, on-orbit servicing
(OOS), classification, and functionality identification of these RSOs. Recent
advances in computer vision have enabled high-definition 3D modeling of objects
based on a set of 2D images captured from different viewing angles. This work
adapts Instant NeRF and D-NeRF, variations of the neural radiance field (NeRF)
algorithm to the problem of mapping RSOs in orbit for the purposes of
functionality identification and assisting with OOS. The algorithms are
evaluated for 3D reconstruction quality and hardware requirements using
datasets of images of a spacecraft mock-up taken under two different lighting
and motion conditions at the Orbital Robotic Interaction, On-Orbit Servicing
and Navigation (ORION) Laboratory at Florida Institute of Technology. Instant
NeRF is shown to learn high-fidelity 3D models with a computational cost that
could feasibly be trained on on-board computers.

Comments:
- Presented at AAS/AIAA Spaceflight Mechanics Conference 2023, 14
  pages, 10 figures, 2 tables

---

## RecolorNeRF: Layer Decomposed Radiance Fields for Efficient Color  Editing of 3D Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-19 | Bingchen Gong, Yuehao Wang, Xiaoguang Han, Qi Dou | cs.CV | [PDF](http://arxiv.org/pdf/2301.07958v3){: .btn .btn-green } |

**Abstract**: Radiance fields have gradually become a main representation of media.
Although its appearance editing has been studied, how to achieve
view-consistent recoloring in an efficient manner is still under explored. We
present RecolorNeRF, a novel user-friendly color editing approach for the
neural radiance fields. Our key idea is to decompose the scene into a set of
pure-colored layers, forming a palette. By this means, color manipulation can
be conducted by altering the color components of the palette directly. To
support efficient palette-based editing, the color of each layer needs to be as
representative as possible. In the end, the problem is formulated as an
optimization problem, where the layers and their blending weights are jointly
optimized with the NeRF itself. Extensive experiments show that our
jointly-optimized layer decomposition can be used against multiple backbones
and produce photo-realistic recolored novel-view renderings. We demonstrate
that RecolorNeRF outperforms baseline methods both quantitatively and
qualitatively for color editing even in complex real-world scenes.

Comments:
- To appear in ACM Multimedia 2023. Project website is accessible at
  https://sites.google.com/view/recolornerf

---

## NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via  Novel-View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-18 | Allan Zhou, Moo Jin Kim, Lirui Wang, Pete Florence, Chelsea Finn | cs.LG | [PDF](http://arxiv.org/pdf/2301.08556v1){: .btn .btn-green } |

**Abstract**: Expert demonstrations are a rich source of supervision for training visual
robotic manipulation policies, but imitation learning methods often require
either a large number of demonstrations or expensive online expert supervision
to learn reactive closed-loop behaviors. In this work, we introduce SPARTN
(Synthetic Perturbations for Augmenting Robot Trajectories via NeRF): a
fully-offline data augmentation scheme for improving robot policies that use
eye-in-hand cameras. Our approach leverages neural radiance fields (NeRFs) to
synthetically inject corrective noise into visual demonstrations, using NeRFs
to generate perturbed viewpoints while simultaneously calculating the
corrective actions. This requires no additional expert supervision or
environment interaction, and distills the geometric information in NeRFs into a
real-time reactive RGB-only policy. In a simulated 6-DoF visual grasping
benchmark, SPARTN improves success rates by 2.8$\times$ over imitation learning
without the corrective augmentations and even outperforms some methods that use
online supervision. It additionally closes the gap between RGB-only and RGB-D
success rates, eliminating the previous need for depth sensors. In real-world
6-DoF robotic grasping experiments from limited human demonstrations, our
method improves absolute success rates by $22.5\%$ on average, including
objects that are traditionally challenging for depth-based methods. See video
results at \url{https://bland.website/spartn}.

---

## Behind the Scenes: Density Fields for Single View Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-18 | Felix Wimbauer, Nan Yang, Christian Rupprecht, Daniel Cremers | cs.CV | [PDF](http://arxiv.org/pdf/2301.07668v3){: .btn .btn-green } |

**Abstract**: Inferring a meaningful geometric scene representation from a single image is
a fundamental problem in computer vision. Approaches based on traditional depth
map prediction can only reason about areas that are visible in the image.
Currently, neural radiance fields (NeRFs) can capture true 3D including color,
but are too complex to be generated from a single image. As an alternative, we
propose to predict implicit density fields. A density field maps every location
in the frustum of the input image to volumetric density. By directly sampling
color from the available views instead of storing color in the density field,
our scene representation becomes significantly less complex compared to NeRFs,
and a neural network can predict it in a single forward pass. The prediction
network is trained through self-supervision from only video data. Our
formulation allows volume rendering to perform both depth prediction and novel
view synthesis. Through experiments, we show that our method is able to predict
meaningful geometry for regions that are occluded in the input image.
Additionally, we demonstrate the potential of our approach on three datasets
for depth prediction and novel-view synthesis.

Comments:
- Project Page: https://fwmb.github.io/bts/

---

## A Large-Scale Outdoor Multi-modal Dataset and Benchmark for Novel View  Synthesis and Implicit Scene Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-17 | Chongshan Lu, Fukun Yin, Xin Chen, Tao Chen, Gang YU, Jiayuan Fan | cs.CV | [PDF](http://arxiv.org/pdf/2301.06782v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) has achieved impressive results in single
object scene reconstruction and novel view synthesis, which have been
demonstrated on many single modality and single object focused indoor scene
datasets like DTU, BMVS, and NeRF Synthetic.However, the study of NeRF on
large-scale outdoor scene reconstruction is still limited, as there is no
unified outdoor scene dataset for large-scale NeRF evaluation due to expensive
data acquisition and calibration costs. In this paper, we propose a large-scale
outdoor multi-modal dataset, OMMO dataset, containing complex land objects and
scenes with calibrated images, point clouds and prompt annotations. Meanwhile,
a new benchmark for several outdoor NeRF-based tasks is established, such as
novel view synthesis, surface reconstruction, and multi-modal NeRF. To create
the dataset, we capture and collect a large number of real fly-view videos and
select high-quality and high-resolution clips from them. Then we design a
quality review module to refine images, remove low-quality frames and
fail-to-calibrate scenes through a learning-based automatic evaluation plus
manual review. Finally, a number of volunteers are employed to add the text
descriptions for each scene and key-frame to meet the potential multi-modal
requirements in the future. Compared with existing NeRF datasets, our dataset
contains abundant real-world urban and natural scenes with various scales,
camera trajectories, and lighting conditions. Experiments show that our dataset
can benchmark most state-of-the-art NeRF methods on different tasks. We will
release the dataset and model weights very soon.

---

## Laser: Latent Set Representations for 3D Generative Modeling

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-13 | Pol Moreno, Adam R. Kosiorek, Heiko Strathmann, Daniel Zoran, Rosalia G. Schneider, Björn Winckler, Larisa Markeeva, Théophane Weber, Danilo J. Rezende | cs.CV | [PDF](http://arxiv.org/pdf/2301.05747v1){: .btn .btn-green } |

**Abstract**: NeRF provides unparalleled fidelity of novel view synthesis: rendering a 3D
scene from an arbitrary viewpoint. NeRF requires training on a large number of
views that fully cover a scene, which limits its applicability. While these
issues can be addressed by learning a prior over scenes in various forms,
previous approaches have been either applied to overly simple scenes or
struggling to render unobserved parts. We introduce Laser-NV: a generative
model which achieves high modelling capacity, and which is based on a
set-valued latent representation modelled by normalizing flows. Similarly to
previous amortized approaches, Laser-NV learns structure from multiple scenes
and is capable of fast, feed-forward inference from few views. To encourage
higher rendering fidelity and consistency with observed views, Laser-NV further
incorporates a geometry-informed attention mechanism over the observed views.
Laser-NV further produces diverse and plausible completions of occluded parts
of a scene while remaining consistent with observations. Laser-NV shows
state-of-the-art novel-view synthesis quality when evaluated on ShapeNet and on
a novel simulated City dataset, which features high uncertainty in the
unobserved regions of the scene.

Comments:
- See https://laser-nv-paper.github.io/ for video results

---

## Neural Radiance Field Codebooks



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-10 | Matthew Wallingford, Aditya Kusupati, Alex Fang, Vivek Ramanujan, Aniruddha Kembhavi, Roozbeh Mottaghi, Ali Farhadi | cs.CV | [PDF](http://arxiv.org/pdf/2301.04101v2){: .btn .btn-green } |

**Abstract**: Compositional representations of the world are a promising step towards
enabling high-level scene understanding and efficient transfer to downstream
tasks. Learning such representations for complex scenes and tasks remains an
open challenge. Towards this goal, we introduce Neural Radiance Field Codebooks
(NRC), a scalable method for learning object-centric representations through
novel view reconstruction. NRC learns to reconstruct scenes from novel views
using a dictionary of object codes which are decoded through a volumetric
renderer. This enables the discovery of reoccurring visual and geometric
patterns across scenes which are transferable to downstream tasks. We show that
NRC representations transfer well to object navigation in THOR, outperforming
2D and 3D representation learning methods by 3.1% success rate. We demonstrate
that our approach is able to perform unsupervised segmentation for more complex
synthetic (THOR) and real scenes (NYU Depth) better than prior methods (29%
relative improvement). Finally, we show that NRC improves on the task of depth
ordering by 5.5% accuracy in THOR.

Comments:
- 19 pages, 8 figures, 9 tables

---

## Benchmarking Robustness in Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-10 | Chen Wang, Angtian Wang, Junbo Li, Alan Yuille, Cihang Xie | cs.CV | [PDF](http://arxiv.org/pdf/2301.04075v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has demonstrated excellent quality in novel view
synthesis, thanks to its ability to model 3D object geometries in a concise
formulation. However, current approaches to NeRF-based models rely on clean
images with accurate camera calibration, which can be difficult to obtain in
the real world, where data is often subject to corruption and distortion. In
this work, we provide the first comprehensive analysis of the robustness of
NeRF-based novel view synthesis algorithms in the presence of different types
of corruptions.
  We find that NeRF-based models are significantly degraded in the presence of
corruption, and are more sensitive to a different set of corruptions than image
recognition models. Furthermore, we analyze the robustness of the feature
encoder in generalizable methods, which synthesize images using neural features
extracted via convolutional neural networks or transformers, and find that it
only contributes marginally to robustness. Finally, we reveal that standard
data augmentation techniques, which can significantly improve the robustness of
recognition models, do not help the robustness of NeRF-based models. We hope
that our findings will attract more researchers to study the robustness of
NeRF-based approaches and help to improve their performance in the real world.

---

## Traditional Readability Formulas Compared for English

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-08 | Bruce W. Lee, Jason Hyung-Jong Lee | cs.CL | [PDF](http://arxiv.org/pdf/2301.02975v2){: .btn .btn-green } |

**Abstract**: Traditional English readability formulas, or equations, were largely
developed in the 20th century. Nonetheless, many researchers still rely on them
for various NLP applications. This phenomenon is presumably due to the
convenience and straightforwardness of readability formulas. In this work, we
contribute to the NLP community by 1. introducing New English Readability
Formula (NERF), 2. recalibrating the coefficients of old readability formulas
(Flesch-Kincaid Grade Level, Fog Index, SMOG Index, Coleman-Liau Index, and
Automated Readability Index), 3. evaluating the readability formulas, for use
in text simplification studies and medical texts, and 4. developing a
Python-based program for the wide application to various NLP projects.

Comments:
- Submitted to EMNLP 2022

---

## Towards Open World NeRF-Based SLAM

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-08 | Daniil Lisus, Connor Holmes, Steven Waslander | cs.RO | [PDF](http://arxiv.org/pdf/2301.03102v4){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) offer versatility and robustness in map
representations for Simultaneous Localization and Mapping (SLAM) tasks. This
paper extends NICE-SLAM, a recent state-of-the-art NeRF-based SLAM algorithm
capable of producing high quality NeRF maps. However, depending on the hardware
used, the required number of iterations to produce these maps often makes
NICE-SLAM run at less than real time. Additionally, the estimated trajectories
fail to be competitive with classical SLAM approaches. Finally, NICE-SLAM
requires a grid covering the considered environment to be defined prior to
runtime, making it difficult to extend into previously unseen scenes. This
paper seeks to make NICE-SLAM more open-world-capable by improving the
robustness and tracking accuracy, and generalizing the map representation to
handle unconstrained environments. This is done by improving measurement
uncertainty handling, incorporating motion information, and modelling the map
as having an explicit foreground and background. It is shown that these changes
are able to improve tracking accuracy by 85% to 97% depending on the available
resources, while also improving mapping in environments with visual information
extending outside of the predefined grid.

Comments:
- Presented at Conference on Robots and Vision (CRV) 2023. 8 pages, 2
  figures, 2 tables

---

## WIRE: Wavelet Implicit Neural Representations



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-05 | Vishwanath Saragadam, Daniel LeJeune, Jasper Tan, Guha Balakrishnan, Ashok Veeraraghavan, Richard G. Baraniuk | cs.CV | [PDF](http://arxiv.org/pdf/2301.05187v1){: .btn .btn-green } |

**Abstract**: Implicit neural representations (INRs) have recently advanced numerous
vision-related areas. INR performance depends strongly on the choice of the
nonlinear activation function employed in its multilayer perceptron (MLP)
network. A wide range of nonlinearities have been explored, but, unfortunately,
current INRs designed to have high accuracy also suffer from poor robustness
(to signal noise, parameter variation, etc.). Inspired by harmonic analysis, we
develop a new, highly accurate and robust INR that does not exhibit this
tradeoff. Wavelet Implicit neural REpresentation (WIRE) uses a continuous
complex Gabor wavelet activation function that is well-known to be optimally
concentrated in space-frequency and to have excellent biases for representing
images. A wide range of experiments (image denoising, image inpainting,
super-resolution, computed tomography reconstruction, image overfitting, and
novel view synthesis with neural radiance fields) demonstrate that WIRE defines
the new state of the art in INR accuracy, training time, and robustness.

---

## Class-Continuous Conditional Generative Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-03 | Jiwook Kim, Minhyeok Lee | cs.CV | [PDF](http://arxiv.org/pdf/2301.00950v3){: .btn .btn-green } |

**Abstract**: The 3D-aware image synthesis focuses on conserving spatial consistency
besides generating high-resolution images with fine details. Recently, Neural
Radiance Field (NeRF) has been introduced for synthesizing novel views with low
computational cost and superior performance. While several works investigate a
generative NeRF and show remarkable achievement, they cannot handle conditional
and continuous feature manipulation in the generation procedure. In this work,
we introduce a novel model, called Class-Continuous Conditional Generative NeRF
($\text{C}^{3}$G-NeRF), which can synthesize conditionally manipulated
photorealistic 3D-consistent images by projecting conditional features to the
generator and the discriminator. The proposed $\text{C}^{3}$G-NeRF is evaluated
with three image datasets, AFHQ, CelebA, and Cars. As a result, our model shows
strong 3D-consistency with fine details and smooth interpolation in conditional
feature manipulation. For instance, $\text{C}^{3}$G-NeRF exhibits a Fr\'echet
Inception Distance (FID) of 7.64 in 3D-aware face image synthesis with a
$\text{128}^{2}$ resolution. Additionally, we provide FIDs of generated
3D-aware images of each class of the datasets as it is possible to synthesize
class-conditional images with $\text{C}^{3}$G-NeRF.

Comments:
- BMVC 2023 (Accepted)

---

## Detachable Novel Views Synthesis of Dynamic Scenes Using  Distribution-Driven Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-01-01 | Boyu Zhang, Wenbo Xu, Zheng Zhu, Guan Huang | cs.CV | [PDF](http://arxiv.org/pdf/2301.00411v2){: .btn .btn-green } |

**Abstract**: Representing and synthesizing novel views in real-world dynamic scenes from
casual monocular videos is a long-standing problem. Existing solutions
typically approach dynamic scenes by applying geometry techniques or utilizing
temporal information between several adjacent frames without considering the
underlying background distribution in the entire scene or the transmittance
over the ray dimension, limiting their performance on static and occlusion
areas. Our approach $\textbf{D}$istribution-$\textbf{D}$riven neural radiance
fields offers high-quality view synthesis and a 3D solution to
$\textbf{D}$etach the background from the entire $\textbf{D}$ynamic scene,
which is called $\text{D}^4$NeRF. Specifically, it employs a neural
representation to capture the scene distribution in the static background and a
6D-input NeRF to represent dynamic objects, respectively. Each ray sample is
given an additional occlusion weight to indicate the transmittance lying in the
static and dynamic components. We evaluate $\text{D}^4$NeRF on public dynamic
scenes and our urban driving scenes acquired from an autonomous-driving
dataset. Extensive experiments demonstrate that our approach outperforms
previous methods in rendering texture details and motion areas while also
producing a clean static background. Our code will be released at
https://github.com/Luciferbobo/D4NeRF.