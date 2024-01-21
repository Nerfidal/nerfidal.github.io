---
layout: default
title: April
parent: 2021
nav_order: 4
---
<!---metadata--->

## Editable Free-viewpoint Video Using a Layered Neural Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-30 | Jiakai Zhang, Xinhang Liu, Xinyi Ye, Fuqiang Zhao, Yanshun Zhang, Minye Wu, Yingliang Zhang, Lan Xu, Jingyi Yu | cs.CV | [PDF](http://arxiv.org/pdf/2104.14786v1){: .btn .btn-green } |

**Abstract**: Generating free-viewpoint videos is critical for immersive VR/AR experience
but recent neural advances still lack the editing ability to manipulate the
visual perception for large dynamic scenes. To fill this gap, in this paper we
propose the first approach for editable photo-realistic free-viewpoint video
generation for large-scale dynamic scenes using only sparse 16 cameras. The
core of our approach is a new layered neural representation, where each dynamic
entity including the environment itself is formulated into a space-time
coherent neural layered radiance representation called ST-NeRF. Such layered
representation supports fully perception and realistic manipulation of the
dynamic scene whilst still supporting a free viewing experience in a wide
range. In our ST-NeRF, the dynamic entity/layer is represented as continuous
functions, which achieves the disentanglement of location, deformation as well
as the appearance of the dynamic entity in a continuous and self-supervised
manner. We propose a scene parsing 4D label map tracking to disentangle the
spatial information explicitly, and a continuous deform module to disentangle
the temporal motion implicitly. An object-aware volume rendering scheme is
further introduced for the re-assembling of all the neural layers. We adopt a
novel layered loss and motion-aware ray sampling strategy to enable efficient
training for a large dynamic scene with multiple performers, Our framework
further enables a variety of editing functions, i.e., manipulating the scale
and location, duplicating or retiming individual neural layers to create
numerous visual effects while preserving high realism. Extensive experiments
demonstrate the effectiveness of our approach to achieve high-quality,
photo-realistic, and editable free-viewpoint video generation for dynamic
scenes.

---

## UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for  Multi-View Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-20 | Michael Oechsle, Songyou Peng, Andreas Geiger | cs.CV | [PDF](http://arxiv.org/pdf/2104.10078v2){: .btn .btn-green } |

**Abstract**: Neural implicit 3D representations have emerged as a powerful paradigm for
reconstructing surfaces from multi-view images and synthesizing novel views.
Unfortunately, existing methods such as DVR or IDR require accurate per-pixel
object masks as supervision. At the same time, neural radiance fields have
revolutionized novel view synthesis. However, NeRF's estimated volume density
does not admit accurate surface reconstruction. Our key insight is that
implicit surface models and radiance fields can be formulated in a unified way,
enabling both surface and volume rendering using the same model. This unified
perspective enables novel, more efficient sampling procedures and the ability
to reconstruct accurate surfaces without input masks. We compare our method on
the DTU, BlendedMVS, and a synthetic indoor dataset. Our experiments
demonstrate that we outperform NeRF in terms of reconstruction quality while
performing on par with IDR without requiring masks.

Comments:
- ICCV 2021 oral

---

## Shadow Neural Radiance Fields for Multi-view Satellite Photogrammetry

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-20 | Dawa Derksen, Dario Izzo | cs.CV | [PDF](http://arxiv.org/pdf/2104.09877v1){: .btn .btn-green } |

**Abstract**: We present a new generic method for shadow-aware multi-view satellite
photogrammetry of Earth Observation scenes. Our proposed method, the Shadow
Neural Radiance Field (S-NeRF) follows recent advances in implicit volumetric
representation learning. For each scene, we train S-NeRF using very high
spatial resolution optical images taken from known viewing angles. The learning
requires no labels or shape priors: it is self-supervised by an image
reconstruction loss. To accommodate for changing light source conditions both
from a directional light source (the Sun) and a diffuse light source (the sky),
we extend the NeRF approach in two ways. First, direct illumination from the
Sun is modeled via a local light source visibility field. Second, indirect
illumination from a diffuse light source is learned as a non-local color field
as a function of the position of the Sun. Quantitatively, the combination of
these factors reduces the altitude and color errors in shaded areas, compared
to NeRF. The S-NeRF methodology not only performs novel view synthesis and full
3D shape estimation, it also enables shadow detection, albedo synthesis, and
transient object filtering, without any explicit shape supervision.

Comments:
- Accepted to CVPR2021 - EarthVision

---

## FiG-NeRF: Figure-Ground Neural Radiance Fields for 3D Object Category  Modelling

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-17 | Christopher Xie, Keunhong Park, Ricardo Martin-Brualla, Matthew Brown | cs.CV | [PDF](http://arxiv.org/pdf/2104.08418v1){: .btn .btn-green } |

**Abstract**: We investigate the use of Neural Radiance Fields (NeRF) to learn high quality
3D object category models from collections of input images. In contrast to
previous work, we are able to do this whilst simultaneously separating
foreground objects from their varying backgrounds. We achieve this via a
2-component NeRF model, FiG-NeRF, that prefers explanation of the scene as a
geometrically constant background and a deformable foreground that represents
the object category. We show that this method can learn accurate 3D object
category models using only photometric supervision and casually captured images
of the objects. Additionally, our 2-part decomposition allows the model to
perform accurate and crisp amodal segmentation. We quantitatively evaluate our
method with view synthesis and image fidelity metrics, using synthetic,
lab-captured, and in-the-wild data. Our results demonstrate convincing 3D
object category modelling that exceed the performance of existing methods.

---

## Stereo Radiance Fields (SRF): Learning View Synthesis for Sparse Views  of Novel Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-14 | Julian Chibane, Aayush Bansal, Verica Lazova, Gerard Pons-Moll | cs.CV | [PDF](http://arxiv.org/pdf/2104.06935v1){: .btn .btn-green } |

**Abstract**: Recent neural view synthesis methods have achieved impressive quality and
realism, surpassing classical pipelines which rely on multi-view
reconstruction. State-of-the-Art methods, such as NeRF, are designed to learn a
single scene with a neural network and require dense multi-view inputs. Testing
on a new scene requires re-training from scratch, which takes 2-3 days. In this
work, we introduce Stereo Radiance Fields (SRF), a neural view synthesis
approach that is trained end-to-end, generalizes to new scenes, and requires
only sparse views at test time. The core idea is a neural architecture inspired
by classical multi-view stereo methods, which estimates surface points by
finding similar image regions in stereo images. In SRF, we predict color and
density for each 3D point given an encoding of its stereo correspondence in the
input images. The encoding is implicitly learned by an ensemble of pair-wise
similarities -- emulating classical stereo. Experiments show that SRF learns
structure instead of overfitting on a scene. We train on multiple scenes of the
DTU dataset and generalize to new ones without re-training, requiring only 10
sparse and spread-out views as input. We show that 10-15 minutes of fine-tuning
further improve the results, achieving significantly sharper, more detailed
results than scene-specific models. The code, model, and videos are available
at https://virtualhumans.mpi-inf.mpg.de/srf/.

Comments:
- IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
  2021

---

## BARF: Bundle-Adjusting Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-13 | Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, Simon Lucey | cs.CV | [PDF](http://arxiv.org/pdf/2104.06405v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have recently gained a surge of interest within
the computer vision community for its power to synthesize photorealistic novel
views of real-world scenes. One limitation of NeRF, however, is its requirement
of accurate camera poses to learn the scene representations. In this paper, we
propose Bundle-Adjusting Neural Radiance Fields (BARF) for training NeRF from
imperfect (or even unknown) camera poses -- the joint problem of learning
neural 3D representations and registering camera frames. We establish a
theoretical connection to classical image alignment and show that
coarse-to-fine registration is also applicable to NeRF. Furthermore, we show
that na\"ively applying positional encoding in NeRF has a negative impact on
registration with a synthesis-based objective. Experiments on synthetic and
real-world data show that BARF can effectively optimize the neural scene
representations and resolve large camera pose misalignment at the same time.
This enables view synthesis and localization of video sequences from unknown
camera poses, opening up new avenues for visual localization systems (e.g.
SLAM) and potential applications for dense 3D mapping and reconstruction.

Comments:
- Accepted to ICCV 2021 as oral presentation (project page & code:
  https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF)

---

## Neural RGB-D Surface Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-09 | Dejan Azinović, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, Justus Thies | cs.CV | [PDF](http://arxiv.org/pdf/2104.04532v3){: .btn .btn-green } |

**Abstract**: Obtaining high-quality 3D reconstructions of room-scale scenes is of
paramount importance for upcoming applications in AR or VR. These range from
mixed reality applications for teleconferencing, virtual measuring, virtual
room planing, to robotic applications. While current volume-based view
synthesis methods that use neural radiance fields (NeRFs) show promising
results in reproducing the appearance of an object or scene, they do not
reconstruct an actual surface. The volumetric representation of the surface
based on densities leads to artifacts when a surface is extracted using
Marching Cubes, since during optimization, densities are accumulated along the
ray and are not used at a single sample point in isolation. Instead of this
volumetric representation of the surface, we propose to represent the surface
using an implicit function (truncated signed distance function). We show how to
incorporate this representation in the NeRF framework, and extend it to use
depth measurements from a commodity RGB-D sensor, such as a Kinect. In
addition, we propose a pose and camera refinement technique which improves the
overall reconstruction quality. In contrast to concurrent work on integrating
depth priors in NeRF which concentrates on novel view synthesis, our approach
is able to reconstruct high-quality, metrical 3D reconstructions.

Comments:
- CVPR'22; Project page:
  https://dazinovic.github.io/neural-rgbd-surface-reconstruction/ Video:
  https://youtu.be/iWuSowPsC3g

---

## MirrorNeRF: One-shot Neural Portrait Radiance Field from Multi-mirror  Catadioptric Imaging

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-06 | Ziyu Wang, Liao Wang, Fuqiang Zhao, Minye Wu, Lan Xu, Jingyi Yu | cs.CV | [PDF](http://arxiv.org/pdf/2104.02607v2){: .btn .btn-green } |

**Abstract**: Photo-realistic neural reconstruction and rendering of the human portrait are
critical for numerous VR/AR applications. Still, existing solutions inherently
rely on multi-view capture settings, and the one-shot solution to get rid of
the tedious multi-view synchronization and calibration remains extremely
challenging. In this paper, we propose MirrorNeRF - a one-shot neural portrait
free-viewpoint rendering approach using a catadioptric imaging system with
multiple sphere mirrors and a single high-resolution digital camera, which is
the first to combine neural radiance field with catadioptric imaging so as to
enable one-shot photo-realistic human portrait reconstruction and rendering, in
a low-cost and casual capture setting. More specifically, we propose a
light-weight catadioptric system design with a sphere mirror array to enable
diverse ray sampling in the continuous 3D space as well as an effective online
calibration for the camera and the mirror array. Our catadioptric imaging
system can be easily deployed with a low budget and the casual capture ability
for convenient daily usages. We introduce a novel neural warping radiance field
representation to learn a continuous displacement field that implicitly
compensates for the misalignment due to our flexible system setting. We further
propose a density regularization scheme to leverage the inherent geometry
information from the catadioptric data in a self-supervision manner, which not
only improves the training efficiency but also provides more effective density
supervision for higher rendering quality. Extensive experiments demonstrate the
effectiveness and robustness of our scheme to achieve one-shot photo-realistic
and high-quality appearance free-viewpoint rendering for human portrait scenes.

---

## Convolutional Neural Opacity Radiance Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-05 | Haimin Luo, Anpei Chen, Qixuan Zhang, Bai Pang, Minye Wu, Lan Xu, Jingyi Yu | cs.CV | [PDF](http://arxiv.org/pdf/2104.01772v1){: .btn .btn-green } |

**Abstract**: Photo-realistic modeling and rendering of fuzzy objects with complex opacity
are critical for numerous immersive VR/AR applications, but it suffers from
strong view-dependent brightness, color. In this paper, we propose a novel
scheme to generate opacity radiance fields with a convolutional neural renderer
for fuzzy objects, which is the first to combine both explicit opacity
supervision and convolutional mechanism into the neural radiance field
framework so as to enable high-quality appearance and global consistent alpha
mattes generation in arbitrary novel views. More specifically, we propose an
efficient sampling strategy along with both the camera rays and image plane,
which enables efficient radiance field sampling and learning in a patch-wise
manner, as well as a novel volumetric feature integration scheme that generates
per-patch hybrid feature embeddings to reconstruct the view-consistent
fine-detailed appearance and opacity output. We further adopt a patch-wise
adversarial training scheme to preserve both high-frequency appearance and
opacity details in a self-supervised framework. We also introduce an effective
multi-view image capture system to capture high-quality color and alpha maps
for challenging fuzzy objects. Extensive experiments on existing and our new
challenging fuzzy object dataset demonstrate that our method achieves
photo-realistic, globally consistent, and fined detailed appearance and opacity
free-viewpoint rendering for various fuzzy objects.

---

## Decomposing 3D Scenes into Objects via Unsupervised Volume Segmentation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-02 | Karl Stelzner, Kristian Kersting, Adam R. Kosiorek | cs.CV | [PDF](http://arxiv.org/pdf/2104.01148v1){: .btn .btn-green } |

**Abstract**: We present ObSuRF, a method which turns a single image of a scene into a 3D
model represented as a set of Neural Radiance Fields (NeRFs), with each NeRF
corresponding to a different object. A single forward pass of an encoder
network outputs a set of latent vectors describing the objects in the scene.
These vectors are used independently to condition a NeRF decoder, defining the
geometry and appearance of each object. We make learning more computationally
efficient by deriving a novel loss, which allows training NeRFs on RGB-D inputs
without explicit ray marching. After confirming that the model performs equal
or better than state of the art on three 2D image segmentation benchmarks, we
apply it to two multi-object 3D datasets: A multiview version of CLEVR, and a
novel dataset in which scenes are populated by ShapeNet models. We find that
after training ObSuRF on RGB-D views of training scenes, it is capable of not
only recovering the 3D geometry of a scene depicted in a single input image,
but also to segment it into objects, despite receiving no supervision in that
regard.

Comments:
- 15 pages, 3 figures. For project page with videos, see
  http://stelzner.github.io/obsurf/

---

## Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-01 | Ajay Jain, Matthew Tancik, Pieter Abbeel | cs.CV | [PDF](http://arxiv.org/pdf/2104.00677v1){: .btn .btn-green } |

**Abstract**: We present DietNeRF, a 3D neural scene representation estimated from a few
images. Neural Radiance Fields (NeRF) learn a continuous volumetric
representation of a scene through multi-view consistency, and can be rendered
from novel viewpoints by ray casting. While NeRF has an impressive ability to
reconstruct geometry and fine details given many images, up to 100 for
challenging 360{\deg} scenes, it often finds a degenerate solution to its image
reconstruction objective when only a few input views are available. To improve
few-shot quality, we propose DietNeRF. We introduce an auxiliary semantic
consistency loss that encourages realistic renderings at novel poses. DietNeRF
is trained on individual scenes to (1) correctly render given input views from
the same pose, and (2) match high-level semantic attributes across different,
random poses. Our semantic loss allows us to supervise DietNeRF from arbitrary
poses. We extract these semantics using a pre-trained visual encoder such as
CLIP, a Vision Transformer trained on hundreds of millions of diverse
single-view, 2D photographs mined from the web with natural language
supervision. In experiments, DietNeRF improves the perceptual quality of
few-shot view synthesis when learned from scratch, can render novel views with
as few as one observed image when pre-trained on a multi-view dataset, and
produces plausible completions of completely unobserved regions.

Comments:
- Project website: https://www.ajayj.com/dietnerf

---

## NeRF-VAE: A Geometry Aware 3D Scene Generative Model

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-04-01 | Adam R. Kosiorek, Heiko Strathmann, Daniel Zoran, Pol Moreno, Rosalia Schneider, Soňa Mokrá, Danilo J. Rezende | stat.ML | [PDF](http://arxiv.org/pdf/2104.00587v1){: .btn .btn-green } |

**Abstract**: We propose NeRF-VAE, a 3D scene generative model that incorporates geometric
structure via NeRF and differentiable volume rendering. In contrast to NeRF,
our model takes into account shared structure across scenes, and is able to
infer the structure of a novel scene -- without the need to re-train -- using
amortized inference. NeRF-VAE's explicit 3D rendering process further contrasts
previous generative models with convolution-based rendering which lacks
geometric structure. Our model is a VAE that learns a distribution over
radiance fields by conditioning them on a latent scene representation. We show
that, once trained, NeRF-VAE is able to infer and render
geometrically-consistent scenes from previously unseen 3D environments using
very few input images. We further demonstrate that NeRF-VAE generalizes well to
out-of-distribution cameras, while convolutional models do not. Finally, we
introduce and study an attention-based conditioning mechanism of NeRF-VAE's
decoder, which improves model performance.

Comments:
- 17 pages, 15 figures, under review