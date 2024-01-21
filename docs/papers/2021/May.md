---
layout: default
title: May
parent: 2021
nav_order: 5
---
<!---metadata--->

## Stylizing 3D Scene via Implicit Representation and HyperNetwork

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-05-27 | Pei-Ze Chiang, Meng-Shiun Tsai, Hung-Yu Tseng, Wei-sheng Lai, Wei-Chen Chiu | cs.CV | [PDF](http://arxiv.org/pdf/2105.13016v3){: .btn .btn-green } |

**Abstract**: In this work, we aim to address the 3D scene stylization problem - generating
stylized images of the scene at arbitrary novel view angles. A straightforward
solution is to combine existing novel view synthesis and image/video style
transfer approaches, which often leads to blurry results or inconsistent
appearance. Inspired by the high-quality results of the neural radiance fields
(NeRF) method, we propose a joint framework to directly render novel views with
the desired style. Our framework consists of two components: an implicit
representation of the 3D scene with the neural radiance fields model, and a
hypernetwork to transfer the style information into the scene representation.
In particular, our implicit representation model disentangles the scene into
the geometry and appearance branches, and the hypernetwork learns to predict
the parameters of the appearance branch from the reference style image. To
alleviate the training difficulties and memory burden, we propose a two-stage
training procedure and a patch sub-sampling approach to optimize the style and
content losses with the neural radiance fields model. After optimization, our
model is able to render consistent novel views at arbitrary view angles with
arbitrary style. Both quantitative evaluation and human subject study have
demonstrated that the proposed method generates faithful stylization results
with consistent appearance across different views.

Comments:
- Accepted to WACV2022; Project page:
  https://ztex08010518.github.io/3dstyletransfer/

---

## Recursive-NeRF: An Efficient and Dynamically Growing NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-05-19 | Guo-Wei Yang, Wen-Yang Zhou, Hao-Yang Peng, Dun Liang, Tai-Jiang Mu, Shi-Min Hu | cs.CV | [PDF](http://arxiv.org/pdf/2105.09103v1){: .btn .btn-green } |

**Abstract**: View synthesis methods using implicit continuous shape representations
learned from a set of images, such as the Neural Radiance Field (NeRF) method,
have gained increasing attention due to their high quality imagery and
scalability to high resolution. However, the heavy computation required by its
volumetric approach prevents NeRF from being useful in practice; minutes are
taken to render a single image of a few megapixels. Now, an image of a scene
can be rendered in a level-of-detail manner, so we posit that a complicated
region of the scene should be represented by a large neural network while a
small neural network is capable of encoding a simple region, enabling a balance
between efficiency and quality. Recursive-NeRF is our embodiment of this idea,
providing an efficient and adaptive rendering and training approach for NeRF.
The core of Recursive-NeRF learns uncertainties for query coordinates,
representing the quality of the predicted color and volumetric intensity at
each level. Only query coordinates with high uncertainties are forwarded to the
next level to a bigger neural network with a more powerful representational
capability. The final rendered image is a composition of results from neural
networks of all levels. Our evaluation on three public datasets shows that
Recursive-NeRF is more efficient than NeRF while providing state-of-the-art
quality. The code will be available at https://github.com/Gword/Recursive-NeRF.

Comments:
- 11 pages, 12 figures

---

## Editing Conditional Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-05-13 | Steven Liu, Xiuming Zhang, Zhoutong Zhang, Richard Zhang, Jun-Yan Zhu, Bryan Russell | cs.CV | [PDF](http://arxiv.org/pdf/2105.06466v2){: .btn .btn-green } |

**Abstract**: A neural radiance field (NeRF) is a scene model supporting high-quality view
synthesis, optimized per scene. In this paper, we explore enabling user editing
of a category-level NeRF - also known as a conditional radiance field - trained
on a shape category. Specifically, we introduce a method for propagating coarse
2D user scribbles to the 3D space, to modify the color or shape of a local
region. First, we propose a conditional radiance field that incorporates new
modular network components, including a shape branch that is shared across
object instances. Observing multiple instances of the same category, our model
learns underlying part semantics without any supervision, thereby allowing the
propagation of coarse 2D user scribbles to the entire 3D region (e.g., chair
seat). Next, we propose a hybrid network update strategy that targets specific
network components, which balances efficiency and accuracy. During user
interaction, we formulate an optimization problem that both satisfies the
user's constraints and preserves the original object structure. We demonstrate
our approach on various editing tasks over three shape datasets and show that
it outperforms prior neural editing approaches. Finally, we edit the appearance
and shape of a real photograph and show that the edit propagates to
extrapolated novel views.

Comments:
- Code: https://github.com/stevliu/editnerf Website:
  http://editnerf.csail.mit.edu/, v2 updated figure 8 and included additional
  details

---

## Dynamic View Synthesis from Dynamic Monocular Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-05-13 | Chen Gao, Ayush Saraf, Johannes Kopf, Jia-Bin Huang | cs.CV | [PDF](http://arxiv.org/pdf/2105.06468v1){: .btn .btn-green } |

**Abstract**: We present an algorithm for generating novel views at arbitrary viewpoints
and any input time step given a monocular video of a dynamic scene. Our work
builds upon recent advances in neural implicit representation and uses
continuous and differentiable functions for modeling the time-varying structure
and the appearance of the scene. We jointly train a time-invariant static NeRF
and a time-varying dynamic NeRF, and learn how to blend the results in an
unsupervised manner. However, learning this implicit function from a single
video is highly ill-posed (with infinitely many solutions that match the input
video). To resolve the ambiguity, we introduce regularization losses to
encourage a more physically plausible solution. We show extensive quantitative
and qualitative results of dynamic view synthesis from casually captured
videos.

Comments:
- Project webpage: https://free-view-video.github.io/

---

## Neural Trajectory Fields for Dynamic Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-05-12 | Chaoyang Wang, Ben Eckart, Simon Lucey, Orazio Gallo | cs.CV | [PDF](http://arxiv.org/pdf/2105.05994v1){: .btn .btn-green } |

**Abstract**: Recent approaches to render photorealistic views from a limited set of
photographs have pushed the boundaries of our interactions with pictures of
static scenes. The ability to recreate moments, that is, time-varying
sequences, is perhaps an even more interesting scenario, but it remains largely
unsolved. We introduce DCT-NeRF, a coordinatebased neural representation for
dynamic scenes. DCTNeRF learns smooth and stable trajectories over the input
sequence for each point in space. This allows us to enforce consistency between
any two frames in the sequence, which results in high quality reconstruction,
particularly in dynamic regions.

---

## Vision-based Neural Scene Representations for Spacecraft

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-05-11 | Anne Mergy, Gurvan Lecuyer, Dawa Derksen, Dario Izzo | cs.CV | [PDF](http://arxiv.org/pdf/2105.06405v1){: .btn .btn-green } |

**Abstract**: In advanced mission concepts with high levels of autonomy, spacecraft need to
internally model the pose and shape of nearby orbiting objects. Recent works in
neural scene representations show promising results for inferring generic
three-dimensional scenes from optical images. Neural Radiance Fields (NeRF)
have shown success in rendering highly specular surfaces using a large number
of images and their pose. More recently, Generative Radiance Fields (GRAF)
achieved full volumetric reconstruction of a scene from unposed images only,
thanks to the use of an adversarial framework to train a NeRF. In this paper,
we compare and evaluate the potential of NeRF and GRAF to render novel views
and extract the 3D shape of two different spacecraft, the Soil Moisture and
Ocean Salinity satellite of ESA's Living Planet Programme and a generic cube
sat. Considering the best performances of both models, we observe that NeRF has
the ability to render more accurate images regarding the material specularity
of the spacecraft and its pose. For its part, GRAF generates precise novel
views with accurate details even when parts of the satellites are shadowed
while having the significant advantage of not needing any information about the
relative pose.

---

## Neural 3D Scene Compression via Model Compression



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-05-07 | Berivan Isik | cs.CV | [PDF](http://arxiv.org/pdf/2105.03120v1){: .btn .btn-green } |

**Abstract**: Rendering 3D scenes requires access to arbitrary viewpoints from the scene.
Storage of such a 3D scene can be done in two ways; (1) storing 2D images taken
from the 3D scene that can reconstruct the scene back through interpolations,
or (2) storing a representation of the 3D scene itself that already encodes
views from all directions. So far, traditional 3D compression methods have
focused on the first type of storage and compressed the original 2D images with
image compression techniques. With this approach, the user first decodes the
stored 2D images and then renders the 3D scene. However, this separated
procedure is inefficient since a large amount of 2D images have to be stored.
In this work, we take a different approach and compress a functional
representation of 3D scenes. In particular, we introduce a method to compress
3D scenes by compressing the neural networks that represent the scenes as
neural radiance fields. Our method provides more efficient storage of 3D scenes
since it does not store 2D images -- which are redundant when we render the
scene from the neural functional representation.

Comments:
- Stanford CS 231A Final Project, 2021. WiCV at CVPR 2021

---

## Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-05-06 | Sida Peng, Junting Dong, Qianqian Wang, Shangzhan Zhang, Qing Shuai, Xiaowei Zhou, Hujun Bao | cs.CV | [PDF](http://arxiv.org/pdf/2105.02872v2){: .btn .btn-green } |

**Abstract**: This paper addresses the challenge of reconstructing an animatable human
model from a multi-view video. Some recent works have proposed to decompose a
non-rigidly deforming scene into a canonical neural radiance field and a set of
deformation fields that map observation-space points to the canonical space,
thereby enabling them to learn the dynamic scene from images. However, they
represent the deformation field as translational vector field or SE(3) field,
which makes the optimization highly under-constrained. Moreover, these
representations cannot be explicitly controlled by input motions. Instead, we
introduce neural blend weight fields to produce the deformation fields. Based
on the skeleton-driven deformation, blend weight fields are used with 3D human
skeletons to generate observation-to-canonical and canonical-to-observation
correspondences. Since 3D human skeletons are more observable, they can
regularize the learning of deformation fields. Moreover, the learned blend
weight fields can be combined with input skeletal motions to generate new
deformation fields to animate the human model. Experiments show that our
approach significantly outperforms recent human synthesis methods. The code and
supplementary materials are available at
https://zju3dv.github.io/animatable_nerf/.

Comments:
- Accepted to ICCV 2021. The first two authors contributed equally to
  this paper. Project page: https://zju3dv.github.io/animatable_nerf/