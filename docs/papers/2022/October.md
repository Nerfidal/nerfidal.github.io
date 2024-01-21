---
layout: default
title: October
parent: 2022
nav_order: 10
---
<!---metadata--->

## Mixed Reality Interface for Digital Twin of Plant Factory



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-29 | Byunghyun Ban | cs.HC | [PDF](http://arxiv.org/pdf/2211.00597v1){: .btn .btn-green } |

**Abstract**: An easier and intuitive interface architecture is necessary for digital twin
of plant factory. I suggest an immersive and interactive mixed reality
interface for digital twin models of smart farming, for remote work rather than
simulation of components. The environment is constructed with UI display and a
streaming background scene, which is a real time scene taken from camera device
located in the plant factory, processed with deformable neural radiance fields.
User can monitor and control the remote plant factory facilities with HMD or 2D
display based mixed reality environment. This paper also introduces detailed
concept and describes the system architecture to implement suggested mixed
reality interface.

Comments:
- 5 pages, 7 figures

---

## NeRFPlayer: A Streamable Dynamic Scene Representation with Decomposed  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-28 | Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele Chen, Junsong Yuan, Yi Xu, Andreas Geiger | cs.CV | [PDF](http://arxiv.org/pdf/2210.15947v2){: .btn .btn-green } |

**Abstract**: Visually exploring in a real-world 4D spatiotemporal space freely in VR has
been a long-term quest. The task is especially appealing when only a few or
even single RGB cameras are used for capturing the dynamic scene. To this end,
we present an efficient framework capable of fast reconstruction, compact
modeling, and streamable rendering. First, we propose to decompose the 4D
spatiotemporal space according to temporal characteristics. Points in the 4D
space are associated with probabilities of belonging to three categories:
static, deforming, and new areas. Each area is represented and regularized by a
separate neural field. Second, we propose a hybrid representations based
feature streaming scheme for efficiently modeling the neural fields. Our
approach, coined NeRFPlayer, is evaluated on dynamic scenes captured by single
hand-held cameras and multi-camera arrays, achieving comparable or superior
rendering performance in terms of quality and speed comparable to recent
state-of-the-art methods, achieving reconstruction in 10 seconds per frame and
interactive rendering.

Comments:
- Project page: https://lsongx.github.io/projects/nerfplayer.html

---

## ProbNeRF: Uncertainty-Aware Inference of 3D Shapes from 2D Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-27 | Matthew D. Hoffman, Tuan Anh Le, Pavel Sountsov, Christopher Suter, Ben Lee, Vikash K. Mansinghka, Rif A. Saurous | cs.CV | [PDF](http://arxiv.org/pdf/2210.17415v1){: .btn .btn-green } |

**Abstract**: The problem of inferring object shape from a single 2D image is
underconstrained. Prior knowledge about what objects are plausible can help,
but even given such prior knowledge there may still be uncertainty about the
shapes of occluded parts of objects. Recently, conditional neural radiance
field (NeRF) models have been developed that can learn to infer good point
estimates of 3D models from single 2D images. The problem of inferring
uncertainty estimates for these models has received less attention. In this
work, we propose probabilistic NeRF (ProbNeRF), a model and inference strategy
for learning probabilistic generative models of 3D objects' shapes and
appearances, and for doing posterior inference to recover those properties from
2D images. ProbNeRF is trained as a variational autoencoder, but at test time
we use Hamiltonian Monte Carlo (HMC) for inference. Given one or a few 2D
images of an object (which may be partially occluded), ProbNeRF is able not
only to accurately model the parts it sees, but also to propose realistic and
diverse hypotheses about the parts it does not see. We show that key to the
success of ProbNeRF are (i) a deterministic rendering scheme, (ii) an
annealed-HMC strategy, (iii) a hypernetwork-based decoder architecture, and
(iv) doing inference over a full set of NeRF weights, rather than just a
low-dimensional code.

Comments:
- 18 pages, 18 figures, 1 table; submitted to the 26th International
  Conference on Artificial Intelligence and Statistics (AISTATS 2023)

---

## Boosting Point Clouds Rendering via Radiance Mapping

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-27 | Xiaoyang Huang, Yi Zhang, Bingbing Ni, Teng Li, Kai Chen, Wenjun Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2210.15107v2){: .btn .btn-green } |

**Abstract**: Recent years we have witnessed rapid development in NeRF-based image
rendering due to its high quality. However, point clouds rendering is somehow
less explored. Compared to NeRF-based rendering which suffers from dense
spatial sampling, point clouds rendering is naturally less computation
intensive, which enables its deployment in mobile computing device. In this
work, we focus on boosting the image quality of point clouds rendering with a
compact model design. We first analyze the adaption of the volume rendering
formulation on point clouds. Based on the analysis, we simplify the NeRF
representation to a spatial mapping function which only requires single
evaluation per pixel. Further, motivated by ray marching, we rectify the the
noisy raw point clouds to the estimated intersection between rays and surfaces
as queried coordinates, which could avoid \textit{spatial frequency collapse}
and neighbor point disturbance. Composed of rasterization, spatial mapping and
the refinement stages, our method achieves the state-of-the-art performance on
point clouds rendering, outperforming prior works by notable margins, with a
smaller model size. We obtain a PSNR of 31.74 on NeRF-Synthetic, 25.88 on
ScanNet and 30.81 on DTU. Code and data are publicly available at
https://github.com/seanywang0408/RadianceMapping.

Comments:
- Accepted by Thirty-Seventh AAAI Conference on Artificial Intelligence
  (AAAI 2023)

---

## Learning Neural Radiance Fields from Multi-View Geometry

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-24 | Marco Orsingher, Paolo Zani, Paolo Medici, Massimo Bertozzi | cs.CV | [PDF](http://arxiv.org/pdf/2210.13041v1){: .btn .btn-green } |

**Abstract**: We present a framework, called MVG-NeRF, that combines classical Multi-View
Geometry algorithms and Neural Radiance Fields (NeRF) for image-based 3D
reconstruction. NeRF has revolutionized the field of implicit 3D
representations, mainly due to a differentiable volumetric rendering
formulation that enables high-quality and geometry-aware novel view synthesis.
However, the underlying geometry of the scene is not explicitly constrained
during training, thus leading to noisy and incorrect results when extracting a
mesh with marching cubes. To this end, we propose to leverage pixelwise depths
and normals from a classical 3D reconstruction pipeline as geometric priors to
guide NeRF optimization. Such priors are used as pseudo-ground truth during
training in order to improve the quality of the estimated underlying surface.
Moreover, each pixel is weighted by a confidence value based on the
forward-backward reprojection error for additional robustness. Experimental
results on real-world data demonstrate the effectiveness of this approach in
obtaining clean 3D meshes from images, while maintaining competitive
performances in novel view synthesis.

Comments:
- ECCV 2022 Workshop on "Learning to Generate 3D Shapes and Scenes"

---

## NeRF-SLAM: Real-Time Dense Monocular SLAM with Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-24 | Antoni Rosinol, John J. Leonard, Luca Carlone | cs.CV | [PDF](http://arxiv.org/pdf/2210.13641v1){: .btn .btn-green } |

**Abstract**: We propose a novel geometric and photometric 3D mapping pipeline for accurate
and real-time scene reconstruction from monocular images. To achieve this, we
leverage recent advances in dense monocular SLAM and real-time hierarchical
volumetric neural radiance fields. Our insight is that dense monocular SLAM
provides the right information to fit a neural radiance field of the scene in
real-time, by providing accurate pose estimates and depth-maps with associated
uncertainty. With our proposed uncertainty-based depth loss, we achieve not
only good photometric accuracy, but also great geometric accuracy. In fact, our
proposed pipeline achieves better geometric and photometric accuracy than
competing approaches (up to 179% better PSNR and 86% better L1 depth), while
working in real-time and using only monocular images.

Comments:
- 10 pages, 6 figures

---

## Compressing Explicit Voxel Grid Representations: fast NeRFs become also  small

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-23 | Chenxi Lola Deng, Enzo Tartaglione | cs.CV | [PDF](http://arxiv.org/pdf/2210.12782v1){: .btn .btn-green } |

**Abstract**: NeRFs have revolutionized the world of per-scene radiance field
reconstruction because of their intrinsic compactness. One of the main
limitations of NeRFs is their slow rendering speed, both at training and
inference time. Recent research focuses on the optimization of an explicit
voxel grid (EVG) that represents the scene, which can be paired with neural
networks to learn radiance fields. This approach significantly enhances the
speed both at train and inference time, but at the cost of large memory
occupation. In this work we propose Re:NeRF, an approach that specifically
targets EVG-NeRFs compressibility, aiming to reduce memory storage of NeRF
models while maintaining comparable performance. We benchmark our approach with
three different EVG-NeRF architectures on four popular benchmarks, showing
Re:NeRF's broad usability and effectiveness.

---

## Joint Rigid Motion Correction and Sparse-View CT via Self-Calibrating  Neural Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-23 | Qing Wu, Xin Li, Hongjiang Wei, Jingyi Yu, Yuyao Zhang | eess.IV | [PDF](http://arxiv.org/pdf/2210.12731v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has widely received attention in Sparse-View
Computed Tomography (SVCT) reconstruction tasks as a self-supervised deep
learning framework. NeRF-based SVCT methods represent the desired CT image as a
continuous function of spatial coordinates and train a Multi-Layer Perceptron
(MLP) to learn the function by minimizing loss on the SV sinogram. Benefiting
from the continuous representation provided by NeRF, the high-quality CT image
can be reconstructed. However, existing NeRF-based SVCT methods strictly
suppose there is completely no relative motion during the CT acquisition
because they require \textit{accurate} projection poses to model the X-rays
that scan the SV sinogram. Therefore, these methods suffer from severe
performance drops for real SVCT imaging with motion. In this work, we propose a
self-calibrating neural field to recover the artifacts-free image from the
rigid motion-corrupted SV sinogram without using any external data.
Specifically, we parametrize the inaccurate projection poses caused by rigid
motion as trainable variables and then jointly optimize these pose variables
and the MLP. We conduct numerical experiments on a public CT image dataset. The
results indicate our model significantly outperforms two representative
NeRF-based methods for SVCT reconstruction tasks with four different levels of
rigid motion.

Comments:
- 5 pages

---

## An Exploration of Neural Radiance Field Scene Reconstruction: Synthetic,  Real-world and Dynamic Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-21 | Benedict Quartey, Tuluhan Akbulut, Wasiwasi Mgonzo, Zheng Xin Yong | cs.CV | [PDF](http://arxiv.org/pdf/2210.12268v1){: .btn .btn-green } |

**Abstract**: This project presents an exploration into 3D scene reconstruction of
synthetic and real-world scenes using Neural Radiance Field (NeRF) approaches.
We primarily take advantage of the reduction in training and rendering time of
neural graphic primitives multi-resolution hash encoding, to reconstruct static
video game scenes and real-world scenes, comparing and observing reconstruction
detail and limitations. Additionally, we explore dynamic scene reconstruction
using Neural Radiance Fields for Dynamic Scenes(D-NeRF). Finally, we extend the
implementation of D-NeRF, originally constrained to handle synthetic scenes to
also handle real-world dynamic scenes.

---

## One-Shot Neural Fields for 3D Object Understanding

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-21 | Valts Blukis, Taeyeop Lee, Jonathan Tremblay, Bowen Wen, In So Kweon, Kuk-Jin Yoon, Dieter Fox, Stan Birchfield | cs.RO | [PDF](http://arxiv.org/pdf/2210.12126v3){: .btn .btn-green } |

**Abstract**: We present a unified and compact scene representation for robotics, where
each object in the scene is depicted by a latent code capturing geometry and
appearance. This representation can be decoded for various tasks such as novel
view rendering, 3D reconstruction (e.g. recovering depth, point clouds, or
voxel maps), collision checking, and stable grasp prediction. We build our
representation from a single RGB input image at test time by leveraging recent
advances in Neural Radiance Fields (NeRF) that learn category-level priors on
large multiview datasets, then fine-tune on novel objects from one or few
views. We expand the NeRF model for additional grasp outputs and explore ways
to leverage this representation for robotics. At test-time, we build the
representation from a single RGB input image observing the scene from only one
viewpoint. We find that the recovered representation allows rendering from
novel views, including of occluded object parts, and also for predicting
successful stable grasps. Grasp poses can be directly decoded from our latent
representation with an implicit grasp decoder. We experimented in both
simulation and real world and demonstrated the capability for robust robotic
grasping using such compact representation. Website:
https://nerfgrasp.github.io

Comments:
- IEEE/CVF Conference on Computer Vision and Pattern Recognition
  Workshop (CVPRW) on XRNeRF: Advances in NeRF for the Metaverse 2023

---

## HDHumans: A Hybrid Approach for High-fidelity Digital Humans

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-21 | Marc Habermann, Lingjie Liu, Weipeng Xu, Gerard Pons-Moll, Michael Zollhoefer, Christian Theobalt | cs.CV | [PDF](http://arxiv.org/pdf/2210.12003v2){: .btn .btn-green } |

**Abstract**: Photo-real digital human avatars are of enormous importance in graphics, as
they enable immersive communication over the globe, improve gaming and
entertainment experiences, and can be particularly beneficial for AR and VR
settings. However, current avatar generation approaches either fall short in
high-fidelity novel view synthesis, generalization to novel motions,
reproduction of loose clothing, or they cannot render characters at the high
resolution offered by modern displays. To this end, we propose HDHumans, which
is the first method for HD human character synthesis that jointly produces an
accurate and temporally coherent 3D deforming surface and highly
photo-realistic images of arbitrary novel views and of motions not seen at
training time. At the technical core, our method tightly integrates a classical
deforming character template with neural radiance fields (NeRF). Our method is
carefully designed to achieve a synergy between classical surface deformation
and NeRF. First, the template guides the NeRF, which allows synthesizing novel
views of a highly dynamic and articulated character and even enables the
synthesis of novel motions. Second, we also leverage the dense pointclouds
resulting from NeRF to further improve the deforming surface via 3D-to-3D
supervision. We outperform the state of the art quantitatively and
qualitatively in terms of synthesis quality and resolution, as well as the
quality of 3D surface reconstruction.

---

## RGB-Only Reconstruction of Tabletop Scenes for Collision-Free  Manipulator Control

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-21 | Zhenggang Tang, Balakumar Sundaralingam, Jonathan Tremblay, Bowen Wen, Ye Yuan, Stephen Tyree, Charles Loop, Alexander Schwing, Stan Birchfield | cs.RO | [PDF](http://arxiv.org/pdf/2210.11668v2){: .btn .btn-green } |

**Abstract**: We present a system for collision-free control of a robot manipulator that
uses only RGB views of the world. Perceptual input of a tabletop scene is
provided by multiple images of an RGB camera (without depth) that is either
handheld or mounted on the robot end effector. A NeRF-like process is used to
reconstruct the 3D geometry of the scene, from which the Euclidean full signed
distance function (ESDF) is computed. A model predictive control algorithm is
then used to control the manipulator to reach a desired pose while avoiding
obstacles in the ESDF. We show results on a real dataset collected and
annotated in our lab.

Comments:
- ICRA 2023. Project page at https://ngp-mpc.github.io/

---

## Coordinates Are NOT Lonely -- Codebook Prior Helps Implicit Neural 3D  Representations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-20 | Fukun Yin, Wen Liu, Zilong Huang, Pei Cheng, Tao Chen, Gang YU | cs.CV | [PDF](http://arxiv.org/pdf/2210.11170v2){: .btn .btn-green } |

**Abstract**: Implicit neural 3D representation has achieved impressive results in surface
or scene reconstruction and novel view synthesis, which typically uses the
coordinate-based multi-layer perceptrons (MLPs) to learn a continuous scene
representation. However, existing approaches, such as Neural Radiance Field
(NeRF) and its variants, usually require dense input views (i.e. 50-150) to
obtain decent results. To relive the over-dependence on massive calibrated
images and enrich the coordinate-based feature representation, we explore
injecting the prior information into the coordinate-based network and introduce
a novel coordinate-based model, CoCo-INR, for implicit neural 3D
representation. The cores of our method are two attention modules: codebook
attention and coordinate attention. The former extracts the useful prototypes
containing rich geometry and appearance information from the prior codebook,
and the latter propagates such prior information into each coordinate and
enriches its feature representation for a scene or object surface. With the
help of the prior information, our method can render 3D views with more
photo-realistic appearance and geometries than the current methods using fewer
calibrated images available. Experiments on various scene reconstruction
datasets, including DTU and BlendedMVS, and the full 3D head reconstruction
dataset, H3DS, demonstrate the robustness under fewer input views and fine
detail-preserving capability of our proposed method.

Comments:
- NeurIPS 2022

---

## ARAH: Animatable Volume Rendering of Articulated Human SDFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-18 | Shaofei Wang, Katja Schwarz, Andreas Geiger, Siyu Tang | cs.CV | [PDF](http://arxiv.org/pdf/2210.10036v1){: .btn .btn-green } |

**Abstract**: Combining human body models with differentiable rendering has recently
enabled animatable avatars of clothed humans from sparse sets of multi-view RGB
videos. While state-of-the-art approaches achieve realistic appearance with
neural radiance fields (NeRF), the inferred geometry often lacks detail due to
missing geometric constraints. Further, animating avatars in
out-of-distribution poses is not yet possible because the mapping from
observation space to canonical space does not generalize faithfully to unseen
poses. In this work, we address these shortcomings and propose a model to
create animatable clothed human avatars with detailed geometry that generalize
well to out-of-distribution poses. To achieve detailed geometry, we combine an
articulated implicit surface representation with volume rendering. For
generalization, we propose a novel joint root-finding algorithm for
simultaneous ray-surface intersection search and correspondence search. Our
algorithm enables efficient point sampling and accurate point canonicalization
while generalizing well to unseen poses. We demonstrate that our proposed
pipeline can generate clothed avatars with high-quality pose-dependent geometry
and appearance from a sparse set of multi-view RGB videos. Our method achieves
state-of-the-art performance on geometry and appearance reconstruction while
creating animatable avatars that generalize well to out-of-distribution poses
beyond the small number of training poses.

Comments:
- Accepted to ECCV 2022. Project page:
  https://neuralbodies.github.io/arah/

---

## Parallel Inversion of Neural Radiance Fields for Robust Pose Estimation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-18 | Yunzhi Lin, Thomas Müller, Jonathan Tremblay, Bowen Wen, Stephen Tyree, Alex Evans, Patricio A. Vela, Stan Birchfield | cs.CV | [PDF](http://arxiv.org/pdf/2210.10108v2){: .btn .btn-green } |

**Abstract**: We present a parallelized optimization method based on fast Neural Radiance
Fields (NeRF) for estimating 6-DoF pose of a camera with respect to an object
or scene. Given a single observed RGB image of the target, we can predict the
translation and rotation of the camera by minimizing the residual between
pixels rendered from a fast NeRF model and pixels in the observed image. We
integrate a momentum-based camera extrinsic optimization procedure into Instant
Neural Graphics Primitives, a recent exceptionally fast NeRF implementation. By
introducing parallel Monte Carlo sampling into the pose estimation task, our
method overcomes local minima and improves efficiency in a more extensive
search space. We also show the importance of adopting a more robust pixel-based
loss function to reduce error. Experiments demonstrate that our method can
achieve improved generalization and robustness on both synthetic and real-world
benchmarks.

Comments:
- ICRA 2023. Project page at https://pnerfp.github.io/

---

## Differentiable Physics Simulation of Dynamics-Augmented Neural Objects

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-17 | Simon Le Cleac'h, Hong-Xing Yu, Michelle Guo, Taylor A. Howell, Ruohan Gao, Jiajun Wu, Zachary Manchester, Mac Schwager | cs.RO | [PDF](http://arxiv.org/pdf/2210.09420v3){: .btn .btn-green } |

**Abstract**: We present a differentiable pipeline for simulating the motion of objects
that represent their geometry as a continuous density field parameterized as a
deep network. This includes Neural Radiance Fields (NeRFs), and other related
models. From the density field, we estimate the dynamical properties of the
object, including its mass, center of mass, and inertia matrix. We then
introduce a differentiable contact model based on the density field for
computing normal and friction forces resulting from collisions. This allows a
robot to autonomously build object models that are visually and
\emph{dynamically} accurate from still images and videos of objects in motion.
The resulting Dynamics-Augmented Neural Objects (DANOs) are simulated with an
existing differentiable simulation engine, Dojo, interacting with other
standard simulation objects, such as spheres, planes, and robots specified as
URDFs. A robot can use this simulation to optimize grasps and manipulation
trajectories of neural objects, or to improve the neural object models through
gradient-based real-to-simulation transfer. We demonstrate the pipeline to
learn the coefficient of friction of a bar of soap from a real video of the
soap sliding on a table. We also learn the coefficient of friction and mass of
a Stanford bunny through interactions with a Panda robot arm from synthetic
data, and we optimize trajectories in simulation for the Panda arm to push the
bunny to a goal location.

---

## SPIDR: SDF-based Neural Point Fields for Illumination and Deformation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-15 | Ruofan Liang, Jiahao Zhang, Haoda Li, Chen Yang, Yushi Guan, Nandita Vijaykumar | cs.CV | [PDF](http://arxiv.org/pdf/2210.08398v3){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) have recently emerged as a promising approach
for 3D reconstruction and novel view synthesis. However, NeRF-based methods
encode shape, reflectance, and illumination implicitly and this makes it
challenging for users to manipulate these properties in the rendered images
explicitly. Existing approaches only enable limited editing of the scene and
deformation of the geometry. Furthermore, no existing work enables accurate
scene illumination after object deformation. In this work, we introduce SPIDR,
a new hybrid neural SDF representation. SPIDR combines point cloud and neural
implicit representations to enable the reconstruction of higher quality object
surfaces for geometry deformation and lighting estimation. meshes and surfaces
for object deformation and lighting estimation. To more accurately capture
environment illumination for scene relighting, we propose a novel neural
implicit model to learn environment light. To enable more accurate illumination
updates after deformation, we use the shadow mapping technique to approximate
the light visibility updates caused by geometry editing. We demonstrate the
effectiveness of SPIDR in enabling high quality geometry editing with more
accurate updates to the illumination of the scene.

Comments:
- Project page: https://nexuslrf.github.io/SPIDR_webpage/

---

## IBL-NeRF: Image-Based Lighting Formulation of Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-15 | Changwoon Choi, Juhyeon Kim, Young Min Kim | cs.CV | [PDF](http://arxiv.org/pdf/2210.08202v2){: .btn .btn-green } |

**Abstract**: We propose IBL-NeRF, which decomposes the neural radiance fields (NeRF) of
large-scale indoor scenes into intrinsic components. Recent approaches further
decompose the baked radiance of the implicit volume into intrinsic components
such that one can partially approximate the rendering equation. However, they
are limited to representing isolated objects with a shared environment
lighting, and suffer from computational burden to aggregate rays with Monte
Carlo integration. In contrast, our prefiltered radiance field extends the
original NeRF formulation to capture the spatial variation of lighting within
the scene volume, in addition to surface properties. Specifically, the scenes
of diverse materials are decomposed into intrinsic components for rendering,
namely, albedo, roughness, surface normal, irradiance, and prefiltered
radiance. All of the components are inferred as neural images from MLP, which
can model large-scale general scenes. Especially the prefiltered radiance
effectively models the volumetric light field, and captures spatial variation
beyond a single environment light. The prefiltering aggregates rays in a set of
predefined neighborhood sizes such that we can replace the costly Monte Carlo
integration of global illumination with a simple query from a neural image. By
adopting NeRF, our approach inherits superior visual quality and multi-view
consistency for synthesized images as well as the intrinsic components. We
demonstrate the performance on scenes with complex object layouts and light
configurations, which could not be processed in any of the previous works.

Comments:
- Computer Graphics Forum (Pacific Graphics 2023)

---

## 3D GAN Inversion with Pose Optimization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-13 | Jaehoon Ko, Kyusun Cho, Daewon Choi, Kwangrok Ryoo, Seungryong Kim | cs.CV | [PDF](http://arxiv.org/pdf/2210.07301v2){: .btn .btn-green } |

**Abstract**: With the recent advances in NeRF-based 3D aware GANs quality, projecting an
image into the latent space of these 3D-aware GANs has a natural advantage over
2D GAN inversion: not only does it allow multi-view consistent editing of the
projected image, but it also enables 3D reconstruction and novel view synthesis
when given only a single image. However, the explicit viewpoint control acts as
a main hindrance in the 3D GAN inversion process, as both camera pose and
latent code have to be optimized simultaneously to reconstruct the given image.
Most works that explore the latent space of the 3D-aware GANs rely on
ground-truth camera viewpoint or deformable 3D model, thus limiting their
applicability. In this work, we introduce a generalizable 3D GAN inversion
method that infers camera viewpoint and latent code simultaneously to enable
multi-view consistent semantic image editing. The key to our approach is to
leverage pre-trained estimators for better initialization and utilize the
pixel-wise depth calculated from NeRF parameters to better reconstruct the
given image. We conduct extensive experiments on image reconstruction and
editing both quantitatively and qualitatively, and further compare our results
with 2D GAN-based editing to demonstrate the advantages of utilizing the latent
space of 3D GANs. Additional results and visualizations are available at
https://3dgan-inversion.github.io .

Comments:
- Project Page: https://3dgan-inversion.github.io

---

## MonoNeRF: Learning Generalizable NeRFs from Monocular Videos without  Camera Pose

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-13 | Yang Fu, Ishan Misra, Xiaolong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2210.07181v2){: .btn .btn-green } |

**Abstract**: We propose a generalizable neural radiance fields - MonoNeRF, that can be
trained on large-scale monocular videos of moving in static scenes without any
ground-truth annotations of depth and camera poses. MonoNeRF follows an
Autoencoder-based architecture, where the encoder estimates the monocular depth
and the camera pose, and the decoder constructs a Multiplane NeRF
representation based on the depth encoder feature, and renders the input frames
with the estimated camera. The learning is supervised by the reconstruction
error. Once the model is learned, it can be applied to multiple applications
including depth estimation, camera pose estimation, and single-image novel view
synthesis. More qualitative results are available at:
https://oasisyang.github.io/mononerf .

Comments:
- ICML 2023 camera ready version. Project page:
  https://oasisyang.github.io/mononerf

---

## GraspNeRF: Multiview-based 6-DoF Grasp Detection for Transparent and  Specular Objects Using Generalizable NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-12 | Qiyu Dai, Yan Zhu, Yiran Geng, Ciyu Ruan, Jiazhao Zhang, He Wang | cs.RO | [PDF](http://arxiv.org/pdf/2210.06575v3){: .btn .btn-green } |

**Abstract**: In this work, we tackle 6-DoF grasp detection for transparent and specular
objects, which is an important yet challenging problem in vision-based robotic
systems, due to the failure of depth cameras in sensing their geometry. We, for
the first time, propose a multiview RGB-based 6-DoF grasp detection network,
GraspNeRF, that leverages the generalizable neural radiance field (NeRF) to
achieve material-agnostic object grasping in clutter. Compared to the existing
NeRF-based 3-DoF grasp detection methods that rely on densely captured input
images and time-consuming per-scene optimization, our system can perform
zero-shot NeRF construction with sparse RGB inputs and reliably detect 6-DoF
grasps, both in real-time. The proposed framework jointly learns generalizable
NeRF and grasp detection in an end-to-end manner, optimizing the scene
representation construction for the grasping. For training data, we generate a
large-scale photorealistic domain-randomized synthetic dataset of grasping in
cluttered tabletop scenes that enables direct transfer to the real world. Our
extensive experiments in synthetic and real-world environments demonstrate that
our method significantly outperforms all the baselines in all the experiments
while remaining in real-time. Project page can be found at
https://pku-epic.github.io/GraspNeRF

Comments:
- IEEE International Conference on Robotics and Automation (ICRA), 2023

---

## Reconstructing Personalized Semantic Facial NeRF Models From Monocular  Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-12 | Xuan Gao, Chenglai Zhong, Jun Xiang, Yang Hong, Yudong Guo, Juyong Zhang | cs.GR | [PDF](http://arxiv.org/pdf/2210.06108v1){: .btn .btn-green } |

**Abstract**: We present a novel semantic model for human head defined with neural radiance
field. The 3D-consistent head model consist of a set of disentangled and
interpretable bases, and can be driven by low-dimensional expression
coefficients. Thanks to the powerful representation ability of neural radiance
field, the constructed model can represent complex facial attributes including
hair, wearings, which can not be represented by traditional mesh blendshape. To
construct the personalized semantic facial model, we propose to define the
bases as several multi-level voxel fields. With a short monocular RGB video as
input, our method can construct the subject's semantic facial NeRF model with
only ten to twenty minutes, and can render a photo-realistic human head image
in tens of miliseconds with a given expression coefficient and view direction.
With this novel representation, we apply it to many tasks like facial
retargeting and expression editing. Experimental results demonstrate its strong
representation ability and training/inference speed. Demo videos and released
code are provided in our project page:
https://ustc3dv.github.io/NeRFBlendShape/

Comments:
- Accepted by SIGGRAPH Asia 2022 (Journal Track). Project page:
  https://ustc3dv.github.io/NeRFBlendShape/

---

## X-NeRF: Explicit Neural Radiance Field for Multi-Scene 360$^{\circ} $  Insufficient RGB-D Views

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-11 | Haoyi Zhu, Hao-Shu Fang, Cewu Lu | cs.CV | [PDF](http://arxiv.org/pdf/2210.05135v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs), despite their outstanding performance on
novel view synthesis, often need dense input views. Many papers train one model
for each scene respectively and few of them explore incorporating multi-modal
data into this problem. In this paper, we focus on a rarely discussed but
important setting: can we train one model that can represent multiple scenes,
with 360$^\circ $ insufficient views and RGB-D images? We refer insufficient
views to few extremely sparse and almost non-overlapping views. To deal with
it, X-NeRF, a fully explicit approach which learns a general scene completion
process instead of a coordinate-based mapping, is proposed. Given a few
insufficient RGB-D input views, X-NeRF first transforms them to a sparse point
cloud tensor and then applies a 3D sparse generative Convolutional Neural
Network (CNN) to complete it to an explicit radiance field whose volumetric
rendering can be conducted fast without running networks during inference. To
avoid overfitting, besides common rendering loss, we apply perceptual loss as
well as view augmentation through random rotation on point clouds. The proposed
methodology significantly out-performs previous implicit methods in our
setting, indicating the great potential of proposed problem and approach. Codes
and data are available at https://github.com/HaoyiZhu/XNeRF.

---

## NerfAcc: A General NeRF Acceleration Toolbox

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-10 | Ruilong Li, Matthew Tancik, Angjoo Kanazawa | cs.CV | [PDF](http://arxiv.org/pdf/2210.04847v3){: .btn .btn-green } |

**Abstract**: We propose NerfAcc, a toolbox for efficient volumetric rendering of radiance
fields. We build on the techniques proposed in Instant-NGP, and extend these
techniques to not only support bounded static scenes, but also for dynamic
scenes and unbounded scenes. NerfAcc comes with a user-friendly Python API, and
is ready for plug-and-play acceleration of most NeRFs. Various examples are
provided to show how to use this toolbox. Code can be found here:
https://github.com/KAIR-BAIR/nerfacc. Note this write-up matches with NerfAcc
v0.3.5. For the latest features in NerfAcc, please check out our more recent
write-up at arXiv:2305.04966

Comments:
- Webpage: https://www.nerfacc.com/; Updated Write-up: arXiv:2305.04966

---

## SiNeRF: Sinusoidal Neural Radiance Fields for Joint Pose Estimation and  Scene Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-10 | Yitong Xia, Hao Tang, Radu Timofte, Luc Van Gool | cs.CV | [PDF](http://arxiv.org/pdf/2210.04553v1){: .btn .btn-green } |

**Abstract**: NeRFmm is the Neural Radiance Fields (NeRF) that deal with Joint Optimization
tasks, i.e., reconstructing real-world scenes and registering camera parameters
simultaneously. Despite NeRFmm producing precise scene synthesis and pose
estimations, it still struggles to outperform the full-annotated baseline on
challenging scenes. In this work, we identify that there exists a systematic
sub-optimality in joint optimization and further identify multiple potential
sources for it. To diminish the impacts of potential sources, we propose
Sinusoidal Neural Radiance Fields (SiNeRF) that leverage sinusoidal activations
for radiance mapping and a novel Mixed Region Sampling (MRS) for selecting ray
batch efficiently. Quantitative and qualitative results show that compared to
NeRFmm, SiNeRF achieves comprehensive significant improvements in image
synthesis quality and pose estimation accuracy. Codes are available at
https://github.com/yitongx/sinerf.

Comments:
- Accepted yet not published by BMVC2022

---

## NeRF2Real: Sim2real Transfer of Vision-guided Bipedal Motion Skills  using Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-10 | Arunkumar Byravan, Jan Humplik, Leonard Hasenclever, Arthur Brussee, Francesco Nori, Tuomas Haarnoja, Ben Moran, Steven Bohez, Fereshteh Sadeghi, Bojan Vujatovic, Nicolas Heess | cs.RO | [PDF](http://arxiv.org/pdf/2210.04932v1){: .btn .btn-green } |

**Abstract**: We present a system for applying sim2real approaches to "in the wild" scenes
with realistic visuals, and to policies which rely on active perception using
RGB cameras. Given a short video of a static scene collected using a generic
phone, we learn the scene's contact geometry and a function for novel view
synthesis using a Neural Radiance Field (NeRF). We augment the NeRF rendering
of the static scene by overlaying the rendering of other dynamic objects (e.g.
the robot's own body, a ball). A simulation is then created using the rendering
engine in a physics simulator which computes contact dynamics from the static
scene geometry (estimated from the NeRF volume density) and the dynamic
objects' geometry and physical properties (assumed known). We demonstrate that
we can use this simulation to learn vision-based whole body navigation and ball
pushing policies for a 20 degrees of freedom humanoid robot with an actuated
head-mounted RGB camera, and we successfully transfer these policies to a real
robot. Project video is available at
https://sites.google.com/view/nerf2real/home

---

## EVA3D: Compositional 3D Human Generation from 2D Image Collections

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-10 | Fangzhou Hong, Zhaoxi Chen, Yushi Lan, Liang Pan, Ziwei Liu | cs.CV | [PDF](http://arxiv.org/pdf/2210.04888v1){: .btn .btn-green } |

**Abstract**: Inverse graphics aims to recover 3D models from 2D observations. Utilizing
differentiable rendering, recent 3D-aware generative models have shown
impressive results of rigid object generation using 2D images. However, it
remains challenging to generate articulated objects, like human bodies, due to
their complexity and diversity in poses and appearances. In this work, we
propose, EVA3D, an unconditional 3D human generative model learned from 2D
image collections only. EVA3D can sample 3D humans with detailed geometry and
render high-quality images (up to 512x256) without bells and whistles (e.g.
super resolution). At the core of EVA3D is a compositional human NeRF
representation, which divides the human body into local parts. Each part is
represented by an individual volume. This compositional representation enables
1) inherent human priors, 2) adaptive allocation of network parameters, 3)
efficient training and rendering. Moreover, to accommodate for the
characteristics of sparse 2D human image collections (e.g. imbalanced pose
distribution), we propose a pose-guided sampling strategy for better GAN
learning. Extensive experiments validate that EVA3D achieves state-of-the-art
3D human generation performance regarding both geometry and texture quality.
Notably, EVA3D demonstrates great potential and scalability to
"inverse-graphics" diverse human bodies with a clean framework.

Comments:
- Project Page at https://hongfz16.github.io/projects/EVA3D.html

---

## Robustifying the Multi-Scale Representation of Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-09 | Nishant Jain, Suryansh Kumar, Luc Van Gool | cs.CV | [PDF](http://arxiv.org/pdf/2210.04233v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) recently emerged as a new paradigm for object
representation from multi-view (MV) images. Yet, it cannot handle multi-scale
(MS) images and camera pose estimation errors, which generally is the case with
multi-view images captured from a day-to-day commodity camera. Although
recently proposed Mip-NeRF could handle multi-scale imaging problems with NeRF,
it cannot handle camera pose estimation error. On the other hand, the newly
proposed BARF can solve the camera pose problem with NeRF but fails if the
images are multi-scale in nature. This paper presents a robust multi-scale
neural radiance fields representation approach to simultaneously overcome both
real-world imaging issues. Our method handles multi-scale imaging effects and
camera-pose estimation problems with NeRF-inspired approaches by leveraging the
fundamentals of scene rigidity. To reduce unpleasant aliasing artifacts due to
multi-scale images in the ray space, we leverage Mip-NeRF multi-scale
representation. For joint estimation of robust camera pose, we propose
graph-neural network-based multiple motion averaging in the neural volume
rendering framework. We demonstrate, with examples, that for an accurate neural
representation of an object from day-to-day acquired multi-view images, it is
crucial to have precise camera-pose estimates. Without considering robustness
measures in the camera pose estimation, modeling for multi-scale aliasing
artifacts via conical frustum can be counterproductive. We present extensive
experiments on the benchmark datasets to demonstrate that our approach provides
better results than the recent NeRF-inspired approaches for such realistic
settings.

Comments:
- Accepted for publication at British Machine Vision Conference (BMVC)
  2022. Draft info: 13 pages, 3 Figures, and 4 Tables

---

## Estimating Neural Reflectance Field from Radiance Field using Tree  Structures

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-09 | Xiu Li, Xiao Li, Yan Lu | cs.CV | [PDF](http://arxiv.org/pdf/2210.04217v1){: .btn .btn-green } |

**Abstract**: We present a new method for estimating the Neural Reflectance Field (NReF) of
an object from a set of posed multi-view images under unknown lighting. NReF
represents 3D geometry and appearance of objects in a disentangled manner, and
are hard to be estimated from images only. Our method solves this problem by
exploiting the Neural Radiance Field (NeRF) as a proxy representation, from
which we perform further decomposition. A high-quality NeRF decomposition
relies on good geometry information extraction as well as good prior terms to
properly resolve ambiguities between different components. To extract
high-quality geometry information from radiance fields, we re-design a new
ray-casting based method for surface point extraction. To efficiently compute
and apply prior terms, we convert different prior terms into different type of
filter operations on the surface extracted from radiance field. We then employ
two type of auxiliary data structures, namely Gaussian KD-tree and octree, to
support fast querying of surface points and efficient computation of surface
filters during training. Based on this, we design a multi-stage decomposition
optimization pipeline for estimating neural reflectance field from neural
radiance fields. Extensive experiments show our method outperforms other
state-of-the-art methods on different data, and enable high-quality free-view
relighting as well as material editing tasks.

---

## VM-NeRF: Tackling Sparsity in NeRF with View Morphing

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-09 | Matteo Bortolon, Alessio Del Bue, Fabio Poiesi | cs.CV | [PDF](http://arxiv.org/pdf/2210.04214v2){: .btn .btn-green } |

**Abstract**: NeRF aims to learn a continuous neural scene representation by using a finite
set of input images taken from various viewpoints. A well-known limitation of
NeRF methods is their reliance on data: the fewer the viewpoints, the higher
the likelihood of overfitting. This paper addresses this issue by introducing a
novel method to generate geometrically consistent image transitions between
viewpoints using View Morphing. Our VM-NeRF approach requires no prior
knowledge about the scene structure, as View Morphing is based on the
fundamental principles of projective geometry. VM-NeRF tightly integrates this
geometric view generation process during the training procedure of standard
NeRF approaches. Notably, our method significantly improves novel view
synthesis, particularly when only a few views are available. Experimental
evaluation reveals consistent improvement over current methods that handle
sparse viewpoints in NeRF models. We report an increase in PSNR of up to 1.8dB
and 1.0dB when training uses eight and four views, respectively. Source code:
\url{https://github.com/mbortolon97/VM-NeRF}

Comments:
- ICIAP 2023

---

## Towards Efficient Neural Scene Graphs by Learning Consistency Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-09 | Yeji Song, Chaerin Kong, Seoyoung Lee, Nojun Kwak, Joonseok Lee | cs.CV | [PDF](http://arxiv.org/pdf/2210.04127v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) achieves photo-realistic image rendering from
novel views, and the Neural Scene Graphs (NSG) \cite{ost2021neural} extends it
to dynamic scenes (video) with multiple objects. Nevertheless, computationally
heavy ray marching for every image frame becomes a huge burden. In this paper,
taking advantage of significant redundancy across adjacent frames in videos, we
propose a feature-reusing framework. From the first try of naively reusing the
NSG features, however, we learn that it is crucial to disentangle
object-intrinsic properties consistent across frames from transient ones. Our
proposed method, \textit{Consistency-Field-based NSG (CF-NSG)}, reformulates
neural radiance fields to additionally consider \textit{consistency fields}.
With disentangled representations, CF-NSG takes full advantage of the
feature-reusing scheme and performs an extended degree of scene manipulation in
a more controllable manner. We empirically verify that CF-NSG greatly improves
the inference efficiency by using 85\% less queries than NSG without notable
degradation in rendering quality. Code will be available at:
https://github.com/ldynx/CF-NSG

Comments:
- BMVC 2022, 22 pages

---

## ViewFool: Evaluating the Robustness of Visual Recognition to Adversarial  Viewpoints

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-08 | Yinpeng Dong, Shouwei Ruan, Hang Su, Caixin Kang, Xingxing Wei, Jun Zhu | cs.CV | [PDF](http://arxiv.org/pdf/2210.03895v1){: .btn .btn-green } |

**Abstract**: Recent studies have demonstrated that visual recognition models lack
robustness to distribution shift. However, current work mainly considers model
robustness to 2D image transformations, leaving viewpoint changes in the 3D
world less explored. In general, viewpoint changes are prevalent in various
real-world applications (e.g., autonomous driving), making it imperative to
evaluate viewpoint robustness. In this paper, we propose a novel method called
ViewFool to find adversarial viewpoints that mislead visual recognition models.
By encoding real-world objects as neural radiance fields (NeRF), ViewFool
characterizes a distribution of diverse adversarial viewpoints under an
entropic regularizer, which helps to handle the fluctuations of the real camera
pose and mitigate the reality gap between the real objects and their neural
representations. Experiments validate that the common image classifiers are
extremely vulnerable to the generated adversarial viewpoints, which also
exhibit high cross-model transferability. Based on ViewFool, we introduce
ImageNet-V, a new out-of-distribution dataset for benchmarking viewpoint
robustness of image classifiers. Evaluation results on 40 classifiers with
diverse architectures, objective functions, and data augmentations reveal a
significant drop in model performance when tested on ImageNet-V, which provides
a possibility to leverage ViewFool as an effective data augmentation strategy
to improve viewpoint robustness.

Comments:
- NeurIPS 2022

---

## SelfNeRF: Fast Training NeRF for Human from Monocular Self-rotating  Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-04 | Bo Peng, Jun Hu, Jingtao Zhou, Juyong Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2210.01651v1){: .btn .btn-green } |

**Abstract**: In this paper, we propose SelfNeRF, an efficient neural radiance field based
novel view synthesis method for human performance. Given monocular
self-rotating videos of human performers, SelfNeRF can train from scratch and
achieve high-fidelity results in about twenty minutes. Some recent works have
utilized the neural radiance field for dynamic human reconstruction. However,
most of these methods need multi-view inputs and require hours of training,
making it still difficult for practical use. To address this challenging
problem, we introduce a surface-relative representation based on
multi-resolution hash encoding that can greatly improve the training speed and
aggregate inter-frame information. Extensive experimental results on several
different datasets demonstrate the effectiveness and efficiency of SelfNeRF to
challenging monocular videos.

Comments:
- Project page: https://ustc3dv.github.io/SelfNeRF

---

## Capturing and Animation of Body and Clothing from Monocular Video



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-04 | Yao Feng, Jinlong Yang, Marc Pollefeys, Michael J. Black, Timo Bolkart | cs.CV | [PDF](http://arxiv.org/pdf/2210.01868v1){: .btn .btn-green } |

**Abstract**: While recent work has shown progress on extracting clothed 3D human avatars
from a single image, video, or a set of 3D scans, several limitations remain.
Most methods use a holistic representation to jointly model the body and
clothing, which means that the clothing and body cannot be separated for
applications like virtual try-on. Other methods separately model the body and
clothing, but they require training from a large set of 3D clothed human meshes
obtained from 3D/4D scanners or physics simulations. Our insight is that the
body and clothing have different modeling requirements. While the body is well
represented by a mesh-based parametric 3D model, implicit representations and
neural radiance fields are better suited to capturing the large variety in
shape and appearance present in clothing. Building on this insight, we propose
SCARF (Segmented Clothed Avatar Radiance Field), a hybrid model combining a
mesh-based body with a neural radiance field. Integrating the mesh into the
volumetric rendering in combination with a differentiable rasterizer enables us
to optimize SCARF directly from monocular videos, without any 3D supervision.
The hybrid modeling enables SCARF to (i) animate the clothed body avatar by
changing body poses (including hand articulation and facial expressions), (ii)
synthesize novel views of the avatar, and (iii) transfer clothing between
avatars in virtual try-on applications. We demonstrate that SCARF reconstructs
clothing with higher visual quality than existing methods, that the clothing
deforms with changing body pose and body shape, and that clothing can be
successfully transferred between avatars of different subjects. The code and
models are available at https://github.com/YadiraF/SCARF.

Comments:
- 7 pages main paper, 2 pages supp. mat

---

## NARF22: Neural Articulated Radiance Fields for Configuration-Aware  Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-03 | Stanley Lewis, Jana Pavlasek, Odest Chadwicke Jenkins | cs.RO | [PDF](http://arxiv.org/pdf/2210.01166v1){: .btn .btn-green } |

**Abstract**: Articulated objects pose a unique challenge for robotic perception and
manipulation. Their increased number of degrees-of-freedom makes tasks such as
localization computationally difficult, while also making the process of
real-world dataset collection unscalable. With the aim of addressing these
scalability issues, we propose Neural Articulated Radiance Fields (NARF22), a
pipeline which uses a fully-differentiable, configuration-parameterized Neural
Radiance Field (NeRF) as a means of providing high quality renderings of
articulated objects. NARF22 requires no explicit knowledge of the object
structure at inference time. We propose a two-stage parts-based training
mechanism which allows the object rendering models to generalize well across
the configuration space even if the underlying training data has as few as one
configuration represented. We demonstrate the efficacy of NARF22 by training
configurable renderers on a real-world articulated tool dataset collected via a
Fetch mobile manipulation robot. We show the applicability of the model to
gradient-based inference methods through a configuration estimation and 6
degree-of-freedom pose refinement task. The project webpage is available at:
https://progress.eecs.umich.edu/projects/narf/.

Comments:
- Accepted to the 2022 IEEE/RSJ International Conference on Intelligent
  Robots and Systems (IROS). Contact: Stanley Lewis, stanlew@umich.edu

---

## Unsupervised Multi-View Object Segmentation Using Radiance Field  Propagation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-02 | Xinhang Liu, Jiaben Chen, Huai Yu, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2210.00489v2){: .btn .btn-green } |

**Abstract**: We present radiance field propagation (RFP), a novel approach to segmenting
objects in 3D during reconstruction given only unlabeled multi-view images of a
scene. RFP is derived from emerging neural radiance field-based techniques,
which jointly encodes semantics with appearance and geometry. The core of our
method is a novel propagation strategy for individual objects' radiance fields
with a bidirectional photometric loss, enabling an unsupervised partitioning of
a scene into salient or meaningful regions corresponding to different object
instances. To better handle complex scenes with multiple objects and
occlusions, we further propose an iterative expectation-maximization algorithm
to refine object masks. RFP is one of the first unsupervised approach for
tackling 3D real scene object segmentation for neural radiance field (NeRF)
without any supervision, annotations, or other cues such as 3D bounding boxes
and prior knowledge of object class. Experiments demonstrate that RFP achieves
feasible segmentation results that are more accurate than previous unsupervised
image/scene segmentation approaches, and are comparable to existing supervised
NeRF-based methods. The segmented object representations enable individual 3D
object editing operations.

Comments:
- 23 pages, 14 figures, NeurIPS 2022

---

## IntrinsicNeRF: Learning Intrinsic Neural Radiance Fields for Editable  Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-02 | Weicai Ye, Shuo Chen, Chong Bao, Hujun Bao, Marc Pollefeys, Zhaopeng Cui, Guofeng Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2210.00647v3){: .btn .btn-green } |

**Abstract**: Existing inverse rendering combined with neural rendering methods can only
perform editable novel view synthesis on object-specific scenes, while we
present intrinsic neural radiance fields, dubbed IntrinsicNeRF, which introduce
intrinsic decomposition into the NeRF-based neural rendering method and can
extend its application to room-scale scenes. Since intrinsic decomposition is a
fundamentally under-constrained inverse problem, we propose a novel
distance-aware point sampling and adaptive reflectance iterative clustering
optimization method, which enables IntrinsicNeRF with traditional intrinsic
decomposition constraints to be trained in an unsupervised manner, resulting in
multi-view consistent intrinsic decomposition results. To cope with the problem
that different adjacent instances of similar reflectance in a scene are
incorrectly clustered together, we further propose a hierarchical clustering
method with coarse-to-fine optimization to obtain a fast hierarchical indexing
representation. It supports compelling real-time augmented applications such as
recoloring and illumination variation. Extensive experiments and editing
samples on both object-specific/room-scale scenes and synthetic/real-word data
demonstrate that we can obtain consistent intrinsic decomposition results and
high-fidelity novel view synthesis even for challenging sequences.

Comments:
- Accepted to ICCV2023, Project webpage:
  https://zju3dv.github.io/intrinsic_nerf/, code:
  https://github.com/zju3dv/IntrinsicNeRF

---

## Neural Implicit Surface Reconstruction from Noisy Camera Observations



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-02 | Sarthak Gupta, Patrik Huber | cs.CV | [PDF](http://arxiv.org/pdf/2210.01548v1){: .btn .btn-green } |

**Abstract**: Representing 3D objects and scenes with neural radiance fields has become
very popular over the last years. Recently, surface-based representations have
been proposed, that allow to reconstruct 3D objects from simple photographs.
However, most current techniques require an accurate camera calibration, i.e.
camera parameters corresponding to each image, which is often a difficult task
to do in real-life situations. To this end, we propose a method for learning 3D
surfaces from noisy camera parameters. We show that we can learn camera
parameters together with learning the surface representation, and demonstrate
good quality 3D surface reconstruction even with noisy camera observations.

Comments:
- 4 pages - 2 for paper, 2 for supplementary

---

## Structure-Aware NeRF without Posed Camera via Epipolar Constraint

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-01 | Shu Chen, Yang Zhang, Yaxin Xu, Beiji Zou | cs.CV | [PDF](http://arxiv.org/pdf/2210.00183v1){: .btn .btn-green } |

**Abstract**: The neural radiance field (NeRF) for realistic novel view synthesis requires
camera poses to be pre-acquired by a structure-from-motion (SfM) approach. This
two-stage strategy is not convenient to use and degrades the performance
because the error in the pose extraction can propagate to the view synthesis.
We integrate the pose extraction and view synthesis into a single end-to-end
procedure so they can benefit from each other. For training NeRF models, only
RGB images are given, without pre-known camera poses. The camera poses are
obtained by the epipolar constraint in which the identical feature in different
views has the same world coordinates transformed from the local camera
coordinates according to the extracted poses. The epipolar constraint is
jointly optimized with pixel color constraint. The poses are represented by a
CNN-based deep network, whose input is the related frames. This joint
optimization enables NeRF to be aware of the scene's structure that has an
improved generalization performance. Extensive experiments on a variety of
scenes demonstrate the effectiveness of the proposed approach. Code is
available at https://github.com/XTU-PR-LAB/SaNerf.

---

## NeRF: Neural Radiance Field in 3D Vision, A Comprehensive Review

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-10-01 | Kyle Gao, Yina Gao, Hongjie He, Dening Lu, Linlin Xu, Jonathan Li | cs.CV | [PDF](http://arxiv.org/pdf/2210.00379v5){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has recently become a significant development in
the field of Computer Vision, allowing for implicit, neural network-based scene
representation and novel view synthesis. NeRF models have found diverse
applications in robotics, urban mapping, autonomous navigation, virtual
reality/augmented reality, and more. Due to the growing popularity of NeRF and
its expanding research area, we present a comprehensive survey of NeRF papers
from the past two years. Our survey is organized into architecture and
application-based taxonomies and provides an introduction to the theory of NeRF
and its training via differentiable volume rendering. We also present a
benchmark comparison of the performance and speed of key NeRF models. By
creating this survey, we hope to introduce new researchers to NeRF, provide a
helpful reference for influential works in this field, as well as motivate
future research directions with our discussion section.

Comments:
- Fixed some typos from previous version