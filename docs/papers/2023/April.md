---
layout: default
title: April
parent: 2023
nav_order: 4
---
<!---metadata--->

## Unsupervised Object-Centric Voxelization for Dynamic Scene Understanding

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-30 | Siyu Gao, Yanpeng Zhao, Yunbo Wang, Xiaokang Yang | cs.CV | [PDF](http://arxiv.org/pdf/2305.00393v3){: .btn .btn-green } |

**Abstract**: Understanding the compositional dynamics of multiple objects in unsupervised
visual environments is challenging, and existing object-centric representation
learning methods often ignore 3D consistency in scene decomposition. We propose
DynaVol, an inverse graphics approach that learns object-centric volumetric
representations in a neural rendering framework. DynaVol maintains time-varying
3D voxel grids that explicitly represent the probability of each spatial
location belonging to different objects, and decouple temporal dynamics and
spatial information by learning a canonical-space deformation field. To
optimize the volumetric features, we embed them into a fully differentiable
neural network, binding them to object-centric global features and then driving
a compositional NeRF for scene reconstruction. DynaVol outperforms existing
methods in novel view synthesis and unsupervised scene decomposition and allows
for the editing of dynamic scenes, such as adding, deleting, replacing objects,
and modifying their trajectories.

---

## Neural Radiance Fields (NeRFs): A Review and Some Recent Developments

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-30 | Mohamed Debbagh | cs.CV | [PDF](http://arxiv.org/pdf/2305.00375v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) is a framework that represents a 3D scene in the
weights of a fully connected neural network, known as the Multi-Layer
Perception(MLP). The method was introduced for the task of novel view synthesis
and is able to achieve state-of-the-art photorealistic image renderings from a
given continuous viewpoint. NeRFs have become a popular field of research as
recent developments have been made that expand the performance and capabilities
of the base framework. Recent developments include methods that require less
images to train the model for view synthesis as well as methods that are able
to generate views from unconstrained and dynamic scene representations.

Comments:
- volume rendering, view synthesis, scene representation, deep learning

---

## ViP-NeRF: Visibility Prior for Sparse Input Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-28 | Nagabhushan Somraj, Rajiv Soundararajan | cs.CV | [PDF](http://arxiv.org/pdf/2305.00041v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) have achieved impressive performances in view
synthesis by encoding neural representations of a scene. However, NeRFs require
hundreds of images per scene to synthesize photo-realistic novel views.
Training them on sparse input views leads to overfitting and incorrect scene
depth estimation resulting in artifacts in the rendered novel views. Sparse
input NeRFs were recently regularized by providing dense depth estimated from
pre-trained networks as supervision, to achieve improved performance over
sparse depth constraints. However, we find that such depth priors may be
inaccurate due to generalization issues. Instead, we hypothesize that the
visibility of pixels in different input views can be more reliably estimated to
provide dense supervision. In this regard, we compute a visibility prior
through the use of plane sweep volumes, which does not require any
pre-training. By regularizing the NeRF training with the visibility prior, we
successfully train the NeRF with few input views. We reformulate the NeRF to
also directly output the visibility of a 3D point from a given viewpoint to
reduce the training time with the visibility constraint. On multiple datasets,
our model outperforms the competing sparse input NeRF models including those
that use learned priors. The source code for our model can be found on our
project page:
https://nagabhushansn95.github.io/publications/2023/ViP-NeRF.html.

Comments:
- SIGGRAPH 2023

---

## NeRF-LiDAR: Generating Realistic LiDAR Point Clouds with Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-28 | Junge Zhang, Feihu Zhang, Shaochen Kuang, Li Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2304.14811v2){: .btn .btn-green } |

**Abstract**: Labeling LiDAR point clouds for training autonomous driving is extremely
expensive and difficult. LiDAR simulation aims at generating realistic LiDAR
data with labels for training and verifying self-driving algorithms more
efficiently. Recently, Neural Radiance Fields (NeRF) have been proposed for
novel view synthesis using implicit reconstruction of 3D scenes. Inspired by
this, we present NeRF-LIDAR, a novel LiDAR simulation method that leverages
real-world information to generate realistic LIDAR point clouds. Different from
existing LiDAR simulators, we use real images and point cloud data collected by
self-driving cars to learn the 3D scene representation, point cloud generation
and label rendering. We verify the effectiveness of our NeRF-LiDAR by training
different 3D segmentation models on the generated LiDAR point clouds. It
reveals that the trained models are able to achieve similar accuracy when
compared with the same model trained on the real LiDAR data. Besides, the
generated data is capable of boosting the accuracy through pre-training which
helps reduce the requirements of the real labeled data.

---

## Compositional 3D Human-Object Neural Animation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-27 | Zhi Hou, Baosheng Yu, Dacheng Tao | cs.CV | [PDF](http://arxiv.org/pdf/2304.14070v1){: .btn .btn-green } |

**Abstract**: Human-object interactions (HOIs) are crucial for human-centric scene
understanding applications such as human-centric visual generation, AR/VR, and
robotics. Since existing methods mainly explore capturing HOIs, rendering HOI
remains less investigated. In this paper, we address this challenge in HOI
animation from a compositional perspective, i.e., animating novel HOIs
including novel interaction, novel human and/or novel object driven by a novel
pose sequence. Specifically, we adopt neural human-object deformation to model
and render HOI dynamics based on implicit neural representations. To enable the
interaction pose transferring among different persons and objects, we then
devise a new compositional conditional neural radiance field (or CC-NeRF),
which decomposes the interdependence between human and object using latent
codes to enable compositionally animation control of novel HOIs. Experiments
show that the proposed method can generalize well to various novel HOI
animation settings. Our project page is https://zhihou7.github.io/CHONA/

Comments:
- 14 pages, 6 figures

---

## ContraNeRF: 3D-Aware Generative Model via Contrastive Learning with  Unsupervised Implicit Pose Embedding

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-27 | Mijeong Kim, Hyunjoon Lee, Bohyung Han | cs.CV | [PDF](http://arxiv.org/pdf/2304.14005v2){: .btn .btn-green } |

**Abstract**: Although 3D-aware GANs based on neural radiance fields have achieved
competitive performance, their applicability is still limited to objects or
scenes with the ground-truths or prediction models for clearly defined
canonical camera poses. To extend the scope of applicable datasets, we propose
a novel 3D-aware GAN optimization technique through contrastive learning with
implicit pose embeddings. To this end, we first revise the discriminator design
and remove dependency on ground-truth camera poses. Then, to capture complex
and challenging 3D scene structures more effectively, we make the discriminator
estimate a high-dimensional implicit pose embedding from a given image and
perform contrastive learning on the pose embedding. The proposed approach can
be employed for the dataset, where the canonical camera pose is ill-defined
because it does not look up or estimate camera poses. Experimental results show
that our algorithm outperforms existing methods by large margins on the
datasets with multiple object categories and inconsistent canonical camera
poses.

Comments:
- 20 pages. For the project page, see
  https://cv.snu.ac.kr/research/ContraNeRF/

---

## ActorsNeRF: Animatable Few-shot Human Rendering with Generalizable NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-27 | Jiteng Mu, Shen Sang, Nuno Vasconcelos, Xiaolong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2304.14401v1){: .btn .btn-green } |

**Abstract**: While NeRF-based human representations have shown impressive novel view
synthesis results, most methods still rely on a large number of images / views
for training. In this work, we propose a novel animatable NeRF called
ActorsNeRF. It is first pre-trained on diverse human subjects, and then adapted
with few-shot monocular video frames for a new actor with unseen poses.
Building on previous generalizable NeRFs with parameter sharing using a ConvNet
encoder, ActorsNeRF further adopts two human priors to capture the large human
appearance, shape, and pose variations. Specifically, in the encoded feature
space, we will first align different human subjects in a category-level
canonical space, and then align the same human from different frames in an
instance-level canonical space for rendering. We quantitatively and
qualitatively demonstrate that ActorsNeRF significantly outperforms the
existing state-of-the-art on few-shot generalization to new people and poses on
multiple datasets. Project Page: https://jitengmu.github.io/ActorsNeRF/

Comments:
- Project Page : https://jitengmu.github.io/ActorsNeRF/

---

## Combining HoloLens with Instant-NeRFs: Advanced Real-Time 3D Mobile  Mapping

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-27 | Dennis Haitz, Boris Jutzi, Markus Ulrich, Miriam Jaeger, Patrick Huebner | cs.CV | [PDF](http://arxiv.org/pdf/2304.14301v2){: .btn .btn-green } |

**Abstract**: This work represents a large step into modern ways of fast 3D reconstruction
based on RGB camera images. Utilizing a Microsoft HoloLens 2 as a multisensor
platform that includes an RGB camera and an inertial measurement unit for
SLAM-based camera-pose determination, we train a Neural Radiance Field (NeRF)
as a neural scene representation in real-time with the acquired data from the
HoloLens. The HoloLens is connected via Wifi to a high-performance PC that is
responsible for the training and 3D reconstruction. After the data stream ends,
the training is stopped and the 3D reconstruction is initiated, which extracts
a point cloud of the scene. With our specialized inference algorithm, five
million scene points can be extracted within 1 second. In addition, the point
cloud also includes radiometry per point. Our method of 3D reconstruction
outperforms grid point sampling with NeRFs by multiple orders of magnitude and
can be regarded as a complete real-time 3D reconstruction method in a mobile
mapping setup.

Comments:
- 8 pages, 6 figures

---

## Learning a Diffusion Prior for NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-27 | Guandao Yang, Abhijit Kundu, Leonidas J. Guibas, Jonathan T. Barron, Ben Poole | cs.CV | [PDF](http://arxiv.org/pdf/2304.14473v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have emerged as a powerful neural 3D
representation for objects and scenes derived from 2D data. Generating NeRFs,
however, remains difficult in many scenarios. For instance, training a NeRF
with only a small number of views as supervision remains challenging since it
is an under-constrained problem. In such settings, it calls for some inductive
prior to filter out bad local minima. One way to introduce such inductive
priors is to learn a generative model for NeRFs modeling a certain class of
scenes. In this paper, we propose to use a diffusion model to generate NeRFs
encoded on a regularized grid. We show that our model can sample realistic
NeRFs, while at the same time allowing conditional generations, given a certain
observation as guidance.

---

## Super-NeRF: View-consistent Detail Generation for NeRF super-resolution

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-26 | Yuqi Han, Tao Yu, Xiaohang Yu, Yuwang Wang, Qionghai Dai | cs.CV | [PDF](http://arxiv.org/pdf/2304.13518v1){: .btn .btn-green } |

**Abstract**: The neural radiance field (NeRF) achieved remarkable success in modeling 3D
scenes and synthesizing high-fidelity novel views. However, existing NeRF-based
methods focus more on the make full use of the image resolution to generate
novel views, but less considering the generation of details under the limited
input resolution. In analogy to the extensive usage of image super-resolution,
NeRF super-resolution is an effective way to generate the high-resolution
implicit representation of 3D scenes and holds great potential applications. Up
to now, such an important topic is still under-explored. In this paper, we
propose a NeRF super-resolution method, named Super-NeRF, to generate
high-resolution NeRF from only low-resolution inputs. Given multi-view
low-resolution images, Super-NeRF constructs a consistency-controlling
super-resolution module to generate view-consistent high-resolution details for
NeRF. Specifically, an optimizable latent code is introduced for each
low-resolution input image to control the 2D super-resolution images to
converge to the view-consistent output. The latent codes of each low-resolution
image are optimized synergistically with the target Super-NeRF representation
to fully utilize the view consistency constraint inherent in NeRF construction.
We verify the effectiveness of Super-NeRF on synthetic, real-world, and
AI-generated NeRF datasets. Super-NeRF achieves state-of-the-art NeRF
super-resolution performance on high-resolution detail generation and
cross-view consistency.

---

## VGOS: Voxel Grid Optimization for View Synthesis from Sparse Inputs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-26 | Jiakai Sun, Zhanjie Zhang, Jiafu Chen, Guangyuan Li, Boyan Ji, Lei Zhao, Wei Xing, Huaizhong Lin | cs.CV | [PDF](http://arxiv.org/pdf/2304.13386v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) has shown great success in novel view synthesis
due to its state-of-the-art quality and flexibility. However, NeRF requires
dense input views (tens to hundreds) and a long training time (hours to days)
for a single scene to generate high-fidelity images. Although using the voxel
grids to represent the radiance field can significantly accelerate the
optimization process, we observe that for sparse inputs, the voxel grids are
more prone to overfitting to the training views and will have holes and
floaters, which leads to artifacts. In this paper, we propose VGOS, an approach
for fast (3-5 minutes) radiance field reconstruction from sparse inputs (3-10
views) to address these issues. To improve the performance of voxel-based
radiance field in sparse input scenarios, we propose two methods: (a) We
introduce an incremental voxel training strategy, which prevents overfitting by
suppressing the optimization of peripheral voxels in the early stage of
reconstruction. (b) We use several regularization techniques to smooth the
voxels, which avoids degenerate solutions. Experiments demonstrate that VGOS
achieves state-of-the-art performance for sparse inputs with super-fast
convergence. Code will be available at https://github.com/SJoJoK/VGOS.

Comments:
- IJCAI 2023 Accepted (Main Track)

---

## Local Implicit Ray Function for Generalizable Radiance Field  Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-25 | Xin Huang, Qi Zhang, Ying Feng, Xiaoyu Li, Xuan Wang, Qing Wang | cs.CV | [PDF](http://arxiv.org/pdf/2304.12746v1){: .btn .btn-green } |

**Abstract**: We propose LIRF (Local Implicit Ray Function), a generalizable neural
rendering approach for novel view rendering. Current generalizable neural
radiance fields (NeRF) methods sample a scene with a single ray per pixel and
may therefore render blurred or aliased views when the input views and rendered
views capture scene content with different resolutions. To solve this problem,
we propose LIRF to aggregate the information from conical frustums to construct
a ray. Given 3D positions within conical frustums, LIRF takes 3D coordinates
and the features of conical frustums as inputs and predicts a local volumetric
radiance field. Since the coordinates are continuous, LIRF renders high-quality
novel views at a continuously-valued scale via volume rendering. Besides, we
predict the visible weights for each input view via transformer-based feature
matching to improve the performance in occluded areas. Experimental results on
real-world scenes validate that our method outperforms state-of-the-art methods
on novel view rendering of unseen scenes at arbitrary scales.

Comments:
- Accepted to CVPR 2023. Project page: https://xhuangcv.github.io/lirf/

---

## MF-NeRF: Memory Efficient NeRF with Mixed-Feature Hash Table

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-25 | Yongjae Lee, Li Yang, Deliang Fan | cs.CV | [PDF](http://arxiv.org/pdf/2304.12587v4){: .btn .btn-green } |

**Abstract**: Neural radiance field (NeRF) has shown remarkable performance in generating
photo-realistic novel views. Among recent NeRF related research, the approaches
that involve the utilization of explicit structures like grids to manage
features achieve exceptionally fast training by reducing the complexity of
multilayer perceptron (MLP) networks. However, storing features in dense grids
demands a substantial amount of memory space, resulting in a notable memory
bottleneck within computer system. Consequently, it leads to a significant
increase in training times without prior hyper-parameter tuning. To address
this issue, in this work, we are the first to propose MF-NeRF, a
memory-efficient NeRF framework that employs a Mixed-Feature hash table to
improve memory efficiency and reduce training time while maintaining
reconstruction quality. Specifically, we first design a mixed-feature hash
encoding to adaptively mix part of multi-level feature grids and map it to a
single hash table. Following that, in order to obtain the correct index of a
grid point, we further develop an index transformation method that transforms
indices of an arbitrary level grid to those of a canonical grid. Extensive
experiments benchmarking with state-of-the-art Instant-NGP, TensoRF, and DVGO,
indicate our MF-NeRF could achieve the fastest training time on the same GPU
hardware with similar or even higher reconstruction quality.

---

## Explicit Correspondence Matching for Generalizable Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-24 | Yuedong Chen, Haofei Xu, Qianyi Wu, Chuanxia Zheng, Tat-Jen Cham, Jianfei Cai | cs.CV | [PDF](http://arxiv.org/pdf/2304.12294v1){: .btn .btn-green } |

**Abstract**: We present a new generalizable NeRF method that is able to directly
generalize to new unseen scenarios and perform novel view synthesis with as few
as two source views. The key to our approach lies in the explicitly modeled
correspondence matching information, so as to provide the geometry prior to the
prediction of NeRF color and density for volume rendering. The explicit
correspondence matching is quantified with the cosine similarity between image
features sampled at the 2D projections of a 3D point on different views, which
is able to provide reliable cues about the surface geometry. Unlike previous
methods where image features are extracted independently for each view, we
consider modeling the cross-view interactions via Transformer cross-attention,
which greatly improves the feature matching quality. Our method achieves
state-of-the-art results on different evaluation settings, with the experiments
showing a strong correlation between our learned cosine feature similarity and
volume density, demonstrating the effectiveness and superiority of our proposed
method. Code is at https://github.com/donydchen/matchnerf

Comments:
- Code and pre-trained models: https://github.com/donydchen/matchnerf
  Project Page: https://donydchen.github.io/matchnerf/

---

## Gen-NeRF: Efficient and Generalizable Neural Radiance Fields via  Algorithm-Hardware Co-Design

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-24 | Yonggan Fu, Zhifan Ye, Jiayi Yuan, Shunyao Zhang, Sixu Li, Haoran You, Yingyan Lin | cs.CV | [PDF](http://arxiv.org/pdf/2304.11842v3){: .btn .btn-green } |

**Abstract**: Novel view synthesis is an essential functionality for enabling immersive
experiences in various Augmented- and Virtual-Reality (AR/VR) applications, for
which generalizable Neural Radiance Fields (NeRFs) have gained increasing
popularity thanks to their cross-scene generalization capability. Despite their
promise, the real-device deployment of generalizable NeRFs is bottlenecked by
their prohibitive complexity due to the required massive memory accesses to
acquire scene features, causing their ray marching process to be
memory-bounded. To this end, we propose Gen-NeRF, an algorithm-hardware
co-design framework dedicated to generalizable NeRF acceleration, which for the
first time enables real-time generalizable NeRFs. On the algorithm side,
Gen-NeRF integrates a coarse-then-focus sampling strategy, leveraging the fact
that different regions of a 3D scene contribute differently to the rendered
pixel, to enable sparse yet effective sampling. On the hardware side, Gen-NeRF
highlights an accelerator micro-architecture to maximize the data reuse
opportunities among different rays by making use of their epipolar geometric
relationship. Furthermore, our Gen-NeRF accelerator features a customized
dataflow to enhance data locality during point-to-hardware mapping and an
optimized scene feature storage strategy to minimize memory bank conflicts.
Extensive experiments validate the effectiveness of our proposed Gen-NeRF
framework in enabling real-time and generalizable novel view synthesis.

Comments:
- Accepted by ISCA 2023

---

## HOSNeRF: Dynamic Human-Object-Scene Neural Radiance Fields from a Single  Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-24 | Jia-Wei Liu, Yan-Pei Cao, Tianyuan Yang, Eric Zhongcong Xu, Jussi Keppo, Ying Shan, Xiaohu Qie, Mike Zheng Shou | cs.CV | [PDF](http://arxiv.org/pdf/2304.12281v1){: .btn .btn-green } |

**Abstract**: We introduce HOSNeRF, a novel 360{\deg} free-viewpoint rendering method that
reconstructs neural radiance fields for dynamic human-object-scene from a
single monocular in-the-wild video. Our method enables pausing the video at any
frame and rendering all scene details (dynamic humans, objects, and
backgrounds) from arbitrary viewpoints. The first challenge in this task is the
complex object motions in human-object interactions, which we tackle by
introducing the new object bones into the conventional human skeleton hierarchy
to effectively estimate large object deformations in our dynamic human-object
model. The second challenge is that humans interact with different objects at
different times, for which we introduce two new learnable object state
embeddings that can be used as conditions for learning our human-object
representation and scene representation, respectively. Extensive experiments
show that HOSNeRF significantly outperforms SOTA approaches on two challenging
datasets by a large margin of 40% ~ 50% in terms of LPIPS. The code, data, and
compelling examples of 360{\deg} free-viewpoint renderings from single videos
will be released in https://showlab.github.io/HOSNeRF.

Comments:
- Project page: https://showlab.github.io/HOSNeRF

---

## TextMesh: Generation of Realistic 3D Meshes From Text Prompts

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-24 | Christina Tsalicoglou, Fabian Manhardt, Alessio Tonioni, Michael Niemeyer, Federico Tombari | cs.CV | [PDF](http://arxiv.org/pdf/2304.12439v1){: .btn .btn-green } |

**Abstract**: The ability to generate highly realistic 2D images from mere text prompts has
recently made huge progress in terms of speed and quality, thanks to the advent
of image diffusion models. Naturally, the question arises if this can be also
achieved in the generation of 3D content from such text prompts. To this end, a
new line of methods recently emerged trying to harness diffusion models,
trained on 2D images, for supervision of 3D model generation using view
dependent prompts. While achieving impressive results, these methods, however,
have two major drawbacks. First, rather than commonly used 3D meshes, they
instead generate neural radiance fields (NeRFs), making them impractical for
most real applications. Second, these approaches tend to produce over-saturated
models, giving the output a cartoonish looking effect. Therefore, in this work
we propose a novel method for generation of highly realistic-looking 3D meshes.
To this end, we extend NeRF to employ an SDF backbone, leading to improved 3D
mesh extraction. In addition, we propose a novel way to finetune the mesh
texture, removing the effect of high saturation and improving the details of
the output 3D mesh.

Comments:
- Project Website: https://fabi92.github.io/textmesh/

---

## Segment Anything in 3D with NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-24 | Jiazhong Cen, Zanwei Zhou, Jiemin Fang, Chen Yang, Wei Shen, Lingxi Xie, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian | cs.CV | [PDF](http://arxiv.org/pdf/2304.12308v4){: .btn .btn-green } |

**Abstract**: Recently, the Segment Anything Model (SAM) emerged as a powerful vision
foundation model which is capable to segment anything in 2D images. This paper
aims to generalize SAM to segment 3D objects. Rather than replicating the data
acquisition and annotation procedure which is costly in 3D, we design an
efficient solution, leveraging the Neural Radiance Field (NeRF) as a cheap and
off-the-shelf prior that connects multi-view 2D images to the 3D space. We
refer to the proposed solution as SA3D, for Segment Anything in 3D. It is only
required to provide a manual segmentation prompt (e.g., rough points) for the
target object in a single view, which is used to generate its 2D mask in this
view with SAM. Next, SA3D alternately performs mask inverse rendering and
cross-view self-prompting across various views to iteratively complete the 3D
mask of the target object constructed with voxel grids. The former projects the
2D mask obtained by SAM in the current view onto 3D mask with guidance of the
density distribution learned by the NeRF; The latter extracts reliable prompts
automatically as the input to SAM from the NeRF-rendered 2D mask in another
view. We show in experiments that SA3D adapts to various scenes and achieves 3D
segmentation within minutes. Our research reveals a potential methodology to
lift the ability of a 2D vision foundation model to 3D, as long as the 2D model
can steadily address promptable segmentation across multiple views. Our code is
available at https://github.com/Jumpat/SegmentAnythingin3D.

Comments:
- NeurIPS 2023. Project page: https://jumpat.github.io/SA3D/

---

## Instant-3D: Instant Neural Radiance Field Training Towards On-Device  AR/VR 3D Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-24 | Sixu Li, Chaojian Li, Wenbo Zhu,  Boyang,  Yu,  Yang,  Zhao, Cheng Wan, Haoran You, Huihong Shi,  Yingyan,  Lin | cs.AR | [PDF](http://arxiv.org/pdf/2304.12467v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) based 3D reconstruction is highly desirable for
immersive Augmented and Virtual Reality (AR/VR) applications, but achieving
instant (i.e., < 5 seconds) on-device NeRF training remains a challenge. In
this work, we first identify the inefficiency bottleneck: the need to
interpolate NeRF embeddings up to 200,000 times from a 3D embedding grid during
each training iteration. To alleviate this, we propose Instant-3D, an
algorithm-hardware co-design acceleration framework that achieves instant
on-device NeRF training. Our algorithm decomposes the embedding grid
representation in terms of color and density, enabling computational redundancy
to be squeezed out by adopting different (1) grid sizes and (2) update
frequencies for the color and density branches. Our hardware accelerator
further reduces the dominant memory accesses for embedding grid interpolation
by (1) mapping multiple nearby points' memory read requests into one during the
feed-forward process, (2) merging embedding grid updates from the same sliding
time window during back-propagation, and (3) fusing different computation cores
to support the different grid sizes needed by the color and density branches of
Instant-3D algorithm. Extensive experiments validate the effectiveness of
Instant-3D, achieving a large training time reduction of 41x - 248x while
maintaining the same reconstruction quality. Excitingly, Instant-3D has enabled
instant 3D reconstruction for AR/VR, requiring a reconstruction time of only
1.6 seconds per scene and meeting the AR/VR power consumption constraint of 1.9
W.

Comments:
- Accepted by ISCA'23

---

## 3D-IntPhys: Towards More Generalized 3D-grounded Visual Intuitive  Physics under Challenging Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-22 | Haotian Xue, Antonio Torralba, Joshua B. Tenenbaum, Daniel LK Yamins, Yunzhu Li, Hsiao-Yu Tung | cs.CV | [PDF](http://arxiv.org/pdf/2304.11470v1){: .btn .btn-green } |

**Abstract**: Given a visual scene, humans have strong intuitions about how a scene can
evolve over time under given actions. The intuition, often termed visual
intuitive physics, is a critical ability that allows us to make effective plans
to manipulate the scene to achieve desired outcomes without relying on
extensive trial and error. In this paper, we present a framework capable of
learning 3D-grounded visual intuitive physics models from videos of complex
scenes with fluids. Our method is composed of a conditional Neural Radiance
Field (NeRF)-style visual frontend and a 3D point-based dynamics prediction
backend, using which we can impose strong relational and structural inductive
bias to capture the structure of the underlying environment. Unlike existing
intuitive point-based dynamics works that rely on the supervision of dense
point trajectory from simulators, we relax the requirements and only assume
access to multi-view RGB images and (imperfect) instance masks acquired using
color prior. This enables the proposed model to handle scenarios where accurate
point estimation and tracking are hard or impossible. We generate datasets
including three challenging scenarios involving fluid, granular materials, and
rigid objects in the simulation. The datasets do not include any dense particle
information so most previous 3D-based intuitive physics pipelines can barely
deal with that. We show our model can make long-horizon future predictions by
learning from raw images and significantly outperforms models that do not
employ an explicit 3D representation space. We also show that once trained, our
model can achieve strong generalization in complex scenarios under extrapolate
settings.

---

## Dehazing-NeRF: Neural Radiance Fields from Hazy Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-22 | Tian Li, LU Li, Wei Wang, Zhangchi Feng | cs.CV | [PDF](http://arxiv.org/pdf/2304.11448v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has received much attention in recent years due
to the impressively high quality in 3D scene reconstruction and novel view
synthesis. However, image degradation caused by the scattering of atmospheric
light and object light by particles in the atmosphere can significantly
decrease the reconstruction quality when shooting scenes in hazy conditions. To
address this issue, we propose Dehazing-NeRF, a method that can recover clear
NeRF from hazy image inputs. Our method simulates the physical imaging process
of hazy images using an atmospheric scattering model, and jointly learns the
atmospheric scattering model and a clean NeRF model for both image dehazing and
novel view synthesis. Different from previous approaches, Dehazing-NeRF is an
unsupervised method with only hazy images as the input, and also does not rely
on hand-designed dehazing priors. By jointly combining the depth estimated from
the NeRF 3D scene with the atmospheric scattering model, our proposed model
breaks through the ill-posed problem of single-image dehazing while maintaining
geometric consistency. Besides, to alleviate the degradation of image quality
caused by information loss, soft margin consistency regularization, as well as
atmospheric consistency and contrast discriminative loss, are addressed during
the model training process. Extensive experiments demonstrate that our method
outperforms the simple combination of single-image dehazing and NeRF on both
image dehazing and novel view image synthesis.

---

## NaviNeRF: NeRF-based 3D Representation Disentanglement by Latent  Semantic Navigation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-22 | Baao Xie, Bohan Li, Zequn Zhang, Junting Dong, Xin Jin, Jingyu Yang, Wenjun Zeng | cs.CV | [PDF](http://arxiv.org/pdf/2304.11342v1){: .btn .btn-green } |

**Abstract**: 3D representation disentanglement aims to identify, decompose, and manipulate
the underlying explanatory factors of 3D data, which helps AI fundamentally
understand our 3D world. This task is currently under-explored and poses great
challenges: (i) the 3D representations are complex and in general contains much
more information than 2D image; (ii) many 3D representations are not well
suited for gradient-based optimization, let alone disentanglement. To address
these challenges, we use NeRF as a differentiable 3D representation, and
introduce a self-supervised Navigation to identify interpretable semantic
directions in the latent space. To our best knowledge, this novel method,
dubbed NaviNeRF, is the first work to achieve fine-grained 3D disentanglement
without any priors or supervisions. Specifically, NaviNeRF is built upon the
generative NeRF pipeline, and equipped with an Outer Navigation Branch and an
Inner Refinement Branch. They are complementary -- the outer navigation is to
identify global-view semantic directions, and the inner refinement dedicates to
fine-grained attributes. A synergistic loss is further devised to coordinate
two branches. Extensive experiments demonstrate that NaviNeRF has a superior
fine-grained 3D disentanglement ability than the previous 3D-aware models. Its
performance is also comparable to editing-oriented models relying on semantic
or geometry priors.

---

## AutoNeRF: Training Implicit Scene Representations with Autonomous Agents

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-21 | Pierre Marza, Laetitia Matignon, Olivier Simonin, Dhruv Batra, Christian Wolf, Devendra Singh Chaplot | cs.CV | [PDF](http://arxiv.org/pdf/2304.11241v2){: .btn .btn-green } |

**Abstract**: Implicit representations such as Neural Radiance Fields (NeRF) have been
shown to be very effective at novel view synthesis. However, these models
typically require manual and careful human data collection for training. In
this paper, we present AutoNeRF, a method to collect data required to train
NeRFs using autonomous embodied agents. Our method allows an agent to explore
an unseen environment efficiently and use the experience to build an implicit
map representation autonomously. We compare the impact of different exploration
strategies including handcrafted frontier-based exploration, end-to-end and
modular approaches composed of trained high-level planners and classical
low-level path followers. We train these models with different reward functions
tailored to this problem and evaluate the quality of the learned
representations on four different downstream tasks: classical viewpoint
rendering, map reconstruction, planning, and pose refinement. Empirical results
show that NeRFs can be trained on actively collected data using just a single
episode of experience in an unseen environment, and can be used for several
downstream robotic tasks, and that modular trained exploration models
outperform other classical and end-to-end baselines. Finally, we show that
AutoNeRF can reconstruct large-scale scenes, and is thus a useful tool to
perform scene-specific adaptation as the produced 3D environment models can be
loaded into a simulator to fine-tune a policy of interest.

---

## Omni-Line-of-Sight Imaging for Holistic Shape Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-21 | Binbin Huang, Xingyue Peng, Siyuan Shen, Suan Xia, Ruiqian Li, Yanhua Yu, Yuehan Wang, Shenghua Gao, Wenzheng Chen, Shiying Li, Jingyi Yu | cs.CV | [PDF](http://arxiv.org/pdf/2304.10780v1){: .btn .btn-green } |

**Abstract**: We introduce Omni-LOS, a neural computational imaging method for conducting
holistic shape reconstruction (HSR) of complex objects utilizing a
Single-Photon Avalanche Diode (SPAD)-based time-of-flight sensor. As
illustrated in Fig. 1, our method enables new capabilities to reconstruct
near-$360^\circ$ surrounding geometry of an object from a single scan spot. In
such a scenario, traditional line-of-sight (LOS) imaging methods only see the
front part of the object and typically fail to recover the occluded back
regions. Inspired by recent advances of non-line-of-sight (NLOS) imaging
techniques which have demonstrated great power to reconstruct occluded objects,
Omni-LOS marries LOS and NLOS together, leveraging their complementary
advantages to jointly recover the holistic shape of the object from a single
scan position. The core of our method is to put the object nearby diffuse walls
and augment the LOS scan in the front view with the NLOS scans from the
surrounding walls, which serve as virtual ``mirrors'' to trap lights toward the
object. Instead of separately recovering the LOS and NLOS signals, we adopt an
implicit neural network to represent the object, analogous to NeRF and NeTF.
While transients are measured along straight rays in LOS but over the spherical
wavefronts in NLOS, we derive differentiable ray propagation models to
simultaneously model both types of transient measurements so that the NLOS
reconstruction also takes into account the direct LOS measurements and vice
versa. We further develop a proof-of-concept Omni-LOS hardware prototype for
real-world validation. Comprehensive experiments on various wall settings
demonstrate that Omni-LOS successfully resolves shape ambiguities caused by
occlusions, achieves high-fidelity 3D scan quality, and manages to recover
objects of various scales and complexity.

---

## A Comparative Neural Radiance Field (NeRF) 3D Analysis of Camera Poses  from HoloLens Trajectories and Structure from Motion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-20 | Miriam Jäger, Patrick Hübner, Dennis Haitz, Boris Jutzi | cs.CV | [PDF](http://arxiv.org/pdf/2304.10664v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) are trained using a set of camera poses and
associated images as input to estimate density and color values for each
position. The position-dependent density learning is of particular interest for
photogrammetry, enabling 3D reconstruction by querying and filtering the NeRF
coordinate system based on the object density. While traditional methods like
Structure from Motion are commonly used for camera pose calculation in
pre-processing for NeRFs, the HoloLens offers an interesting interface for
extracting the required input data directly. We present a workflow for
high-resolution 3D reconstructions almost directly from HoloLens data using
NeRFs. Thereby, different investigations are considered: Internal camera poses
from the HoloLens trajectory via a server application, and external camera
poses from Structure from Motion, both with an enhanced variant applied through
pose refinement. Results show that the internal camera poses lead to NeRF
convergence with a PSNR of 25\,dB with a simple rotation around the x-axis and
enable a 3D reconstruction. Pose refinement enables comparable quality compared
to external camera poses, resulting in improved training process with a PSNR of
27\,dB and a better 3D reconstruction. Overall, NeRF reconstructions outperform
the conventional photogrammetric dense reconstruction using Multi-View Stereo
in terms of completeness and level of detail.

Comments:
- 7 pages, 5 figures. Will be published in the ISPRS The International
  Archives of Photogrammetry, Remote Sensing and Spatial Information Sciences

---

## Learning Neural Duplex Radiance Fields for Real-Time View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-20 | Ziyu Wan, Christian Richardt, Aljaž Božič, Chao Li, Vijay Rengarajan, Seonghyeon Nam, Xiaoyu Xiang, Tuotuo Li, Bo Zhu, Rakesh Ranjan, Jing Liao | cs.CV | [PDF](http://arxiv.org/pdf/2304.10537v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) enable novel view synthesis with unprecedented
visual quality. However, to render photorealistic images, NeRFs require
hundreds of deep multilayer perceptron (MLP) evaluations - for each pixel. This
is prohibitively expensive and makes real-time rendering infeasible, even on
powerful modern GPUs. In this paper, we propose a novel approach to distill and
bake NeRFs into highly efficient mesh-based neural representations that are
fully compatible with the massively parallel graphics rendering pipeline. We
represent scenes as neural radiance features encoded on a two-layer duplex
mesh, which effectively overcomes the inherent inaccuracies in 3D surface
reconstruction by learning the aggregated radiance information from a reliable
interval of ray-surface intersections. To exploit local geometric relationships
of nearby pixels, we leverage screen-space convolutions instead of the MLPs
used in NeRFs to achieve high-quality appearance. Finally, the performance of
the whole framework is further boosted by a novel multi-view distillation
optimization strategy. We demonstrate the effectiveness and superiority of our
approach via extensive experiments on a range of standard datasets.

Comments:
- CVPR 2023. Project page: http://raywzy.com/NDRF

---

## Nerfbusters: Removing Ghostly Artifacts from Casually Captured NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-20 | Frederik Warburg, Ethan Weber, Matthew Tancik, Aleksander Holynski, Angjoo Kanazawa | cs.CV | [PDF](http://arxiv.org/pdf/2304.10532v3){: .btn .btn-green } |

**Abstract**: Casually captured Neural Radiance Fields (NeRFs) suffer from artifacts such
as floaters or flawed geometry when rendered outside the camera trajectory.
Existing evaluation protocols often do not capture these effects, since they
usually only assess image quality at every 8th frame of the training capture.
To push forward progress in novel-view synthesis, we propose a new dataset and
evaluation procedure, where two camera trajectories are recorded of the scene:
one used for training, and the other for evaluation. In this more challenging
in-the-wild setting, we find that existing hand-crafted regularizers do not
remove floaters nor improve scene geometry. Thus, we propose a 3D
diffusion-based method that leverages local 3D priors and a novel density-based
score distillation sampling loss to discourage artifacts during NeRF
optimization. We show that this data-driven prior removes floaters and improves
scene geometry for casual captures.

Comments:
- ICCV 2023, project page: https://ethanweber.me/nerfbusters

---

## ReLight My NeRF: A Dataset for Novel View Synthesis and Relighting of  Real World Objects

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-20 | Marco Toschi, Riccardo De Matteo, Riccardo Spezialetti, Daniele De Gregorio, Luigi Di Stefano, Samuele Salti | cs.CV | [PDF](http://arxiv.org/pdf/2304.10448v1){: .btn .btn-green } |

**Abstract**: In this paper, we focus on the problem of rendering novel views from a Neural
Radiance Field (NeRF) under unobserved light conditions. To this end, we
introduce a novel dataset, dubbed ReNe (Relighting NeRF), framing real world
objects under one-light-at-time (OLAT) conditions, annotated with accurate
ground-truth camera and light poses. Our acquisition pipeline leverages two
robotic arms holding, respectively, a camera and an omni-directional point-wise
light source. We release a total of 20 scenes depicting a variety of objects
with complex geometry and challenging materials. Each scene includes 2000
images, acquired from 50 different points of views under 40 different OLAT
conditions. By leveraging the dataset, we perform an ablation study on the
relighting capability of variants of the vanilla NeRF architecture and identify
a lightweight architecture that can render novel views of an object under novel
light conditions, which we use to establish a non-trivial baseline for the
dataset. Dataset and benchmark are available at
https://eyecan-ai.github.io/rene.

Comments:
- Accepted at CVPR 2023 as a highlight

---

## LiDAR-NeRF: Novel LiDAR View Synthesis via Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-20 | Tang Tao, Longfei Gao, Guangrun Wang, Yixing Lao, Peng Chen, Hengshuang Zhao, Dayang Hao, Xiaodan Liang, Mathieu Salzmann, Kaicheng Yu | cs.CV | [PDF](http://arxiv.org/pdf/2304.10406v2){: .btn .btn-green } |

**Abstract**: We introduce a new task, novel view synthesis for LiDAR sensors. While
traditional model-based LiDAR simulators with style-transfer neural networks
can be applied to render novel views, they fall short of producing accurate and
realistic LiDAR patterns because the renderers rely on explicit 3D
reconstruction and exploit game engines, that ignore important attributes of
LiDAR points. We address this challenge by formulating, to the best of our
knowledge, the first differentiable end-to-end LiDAR rendering framework,
LiDAR-NeRF, leveraging a neural radiance field (NeRF) to facilitate the joint
learning of geometry and the attributes of 3D points. However, simply employing
NeRF cannot achieve satisfactory results, as it only focuses on learning
individual pixels while ignoring local information, especially at low texture
areas, resulting in poor geometry. To this end, we have taken steps to address
this issue by introducing a structural regularization method to preserve local
structural details. To evaluate the effectiveness of our approach, we establish
an object-centric multi-view LiDAR dataset, dubbed NeRF-MVL. It contains
observations of objects from 9 categories seen from 360-degree viewpoints
captured with multiple LiDAR sensors. Our extensive experiments on the
scene-level KITTI-360 dataset, and on our object-level NeRF-MVL show that our
LiDAR-NeRF surpasses the model-based algorithms significantly.

Comments:
- This paper introduces a new task of novel LiDAR view synthesis, and
  proposes a differentiable framework called LiDAR-NeRF with a structural
  regularization, as well as an object-centric multi-view LiDAR dataset called
  NeRF-MVL

---

## Revisiting Implicit Neural Representations in Low-Level Vision

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-20 | Wentian Xu, Jianbo Jiao | cs.CV | [PDF](http://arxiv.org/pdf/2304.10250v1){: .btn .btn-green } |

**Abstract**: Implicit Neural Representation (INR) has been emerging in computer vision in
recent years. It has been shown to be effective in parameterising continuous
signals such as dense 3D models from discrete image data, e.g. the neural
radius field (NeRF). However, INR is under-explored in 2D image processing
tasks. Considering the basic definition and the structure of INR, we are
interested in its effectiveness in low-level vision problems such as image
restoration. In this work, we revisit INR and investigate its application in
low-level image restoration tasks including image denoising, super-resolution,
inpainting, and deblurring. Extensive experimental evaluations suggest the
superior performance of INR in several low-level vision tasks with limited
resources, outperforming its counterparts by over 2dB. Code and models are
available at https://github.com/WenTXuL/LINR

Comments:
- Published at the ICLR 2023 Neural Fields workshop. Project Webpage:
  https://wentxul.github.io/LINR-projectpage

---

## Multiscale Representation for Real-Time Anti-Aliasing Neural Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-20 | Dongting Hu, Zhenkai Zhang, Tingbo Hou, Tongliang Liu, Huan Fu, Mingming Gong | cs.CV | [PDF](http://arxiv.org/pdf/2304.10075v2){: .btn .btn-green } |

**Abstract**: The rendering scheme in neural radiance field (NeRF) is effective in
rendering a pixel by casting a ray into the scene. However, NeRF yields blurred
rendering results when the training images are captured at non-uniform scales,
and produces aliasing artifacts if the test images are taken in distant views.
To address this issue, Mip-NeRF proposes a multiscale representation as a
conical frustum to encode scale information. Nevertheless, this approach is
only suitable for offline rendering since it relies on integrated positional
encoding (IPE) to query a multilayer perceptron (MLP). To overcome this
limitation, we propose mip voxel grids (Mip-VoG), an explicit multiscale
representation with a deferred architecture for real-time anti-aliasing
rendering. Our approach includes a density Mip-VoG for scene geometry and a
feature Mip-VoG with a small MLP for view-dependent color. Mip-VoG encodes
scene scale using the level of detail (LOD) derived from ray differentials and
uses quadrilinear interpolation to map a queried 3D location to its features
and density from two neighboring downsampled voxel grids. To our knowledge, our
approach is the first to offer multiscale training and real-time anti-aliasing
rendering simultaneously. We conducted experiments on multiscale datasets, and
the results show that our approach outperforms state-of-the-art real-time
rendering baselines.

---

## Neural Radiance Fields: Past, Present, and Future

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-20 | Ansh Mittal | cs.CV | [PDF](http://arxiv.org/pdf/2304.10050v2){: .btn .btn-green } |

**Abstract**: The various aspects like modeling and interpreting 3D environments and
surroundings have enticed humans to progress their research in 3D Computer
Vision, Computer Graphics, and Machine Learning. An attempt made by Mildenhall
et al in their paper about NeRFs (Neural Radiance Fields) led to a boom in
Computer Graphics, Robotics, Computer Vision, and the possible scope of
High-Resolution Low Storage Augmented Reality and Virtual Reality-based 3D
models have gained traction from res with more than 1000 preprints related to
NeRFs published. This paper serves as a bridge for people starting to study
these fields by building on the basics of Mathematics, Geometry, Computer
Vision, and Computer Graphics to the difficulties encountered in Implicit
Representations at the intersection of all these disciplines. This survey
provides the history of rendering, Implicit Learning, and NeRFs, the
progression of research on NeRFs, and the potential applications and
implications of NeRFs in today's world. In doing so, this survey categorizes
all the NeRF-related research in terms of the datasets used, objective
functions, applications solved, and evaluation criteria for these applications.

Comments:
- 413 pages, 9 figures, 277 citations

---

## Reference-guided Controllable Inpainting of Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-19 | Ashkan Mirzaei, Tristan Aumentado-Armstrong, Marcus A. Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G. Derpanis, Igor Gilitschenski | cs.CV | [PDF](http://arxiv.org/pdf/2304.09677v2){: .btn .btn-green } |

**Abstract**: The popularity of Neural Radiance Fields (NeRFs) for view synthesis has led
to a desire for NeRF editing tools. Here, we focus on inpainting regions in a
view-consistent and controllable manner. In addition to the typical NeRF inputs
and masks delineating the unwanted region in each view, we require only a
single inpainted view of the scene, i.e., a reference view. We use monocular
depth estimators to back-project the inpainted view to the correct 3D
positions. Then, via a novel rendering technique, a bilateral solver can
construct view-dependent effects in non-reference views, making the inpainted
region appear consistent from any view. For non-reference disoccluded regions,
which cannot be supervised by the single reference view, we devise a method
based on image inpainters to guide both the geometry and appearance. Our
approach shows superior performance to NeRF inpainting baselines, with the
additional advantage that a user can control the generated scene via a single
inpainted image. Project page: https://ashmrz.github.io/reference-guided-3d

Comments:
- Project Page: https://ashmrz.github.io/reference-guided-3d

---

## Tetra-NeRF: Representing Neural Radiance Fields Using Tetrahedra

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-19 | Jonas Kulhanek, Torsten Sattler | cs.CV | [PDF](http://arxiv.org/pdf/2304.09987v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) are a very recent and very popular approach
for the problems of novel view synthesis and 3D reconstruction. A popular scene
representation used by NeRFs is to combine a uniform, voxel-based subdivision
of the scene with an MLP. Based on the observation that a (sparse) point cloud
of the scene is often available, this paper proposes to use an adaptive
representation based on tetrahedra obtained by Delaunay triangulation instead
of uniform subdivision or point-based representations. We show that such a
representation enables efficient training and leads to state-of-the-art
results. Our approach elegantly combines concepts from 3D geometry processing,
triangle-based rendering, and modern neural radiance fields. Compared to
voxel-based representations, ours provides more detail around parts of the
scene likely to be close to the surface. Compared to point-based
representations, our approach achieves better performance. The source code is
publicly available at: https://jkulhanek.com/tetra-nerf.

Comments:
- ICCV 2023, Web: https://jkulhanek.com/tetra-nerf

---

## Anything-3D: Towards Single-view Anything Reconstruction in the Wild



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-19 | Qiuhong Shen, Xingyi Yang, Xinchao Wang | cs.CV | [PDF](http://arxiv.org/pdf/2304.10261v1){: .btn .btn-green } |

**Abstract**: 3D reconstruction from a single-RGB image in unconstrained real-world
scenarios presents numerous challenges due to the inherent diversity and
complexity of objects and environments. In this paper, we introduce
Anything-3D, a methodical framework that ingeniously combines a series of
visual-language models and the Segment-Anything object segmentation model to
elevate objects to 3D, yielding a reliable and versatile system for single-view
conditioned 3D reconstruction task. Our approach employs a BLIP model to
generate textural descriptions, utilizes the Segment-Anything model for the
effective extraction of objects of interest, and leverages a text-to-image
diffusion model to lift object into a neural radiance field. Demonstrating its
ability to produce accurate and detailed 3D reconstructions for a wide array of
objects, \emph{Anything-3D\footnotemark[2]} shows promise in addressing the
limitations of existing methodologies. Through comprehensive experiments and
evaluations on various datasets, we showcase the merits of our approach,
underscoring its potential to contribute meaningfully to the field of 3D
reconstruction. Demos and code will be available at
\href{https://github.com/Anything-of-anything/Anything-3D}{https://github.com/Anything-of-anything/Anything-3D}.

---

## NeAI: A Pre-convoluted Representation for Plug-and-Play Neural Ambient  Illumination

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-18 | Yiyu Zhuang, Qi Zhang, Xuan Wang, Hao Zhu, Ying Feng, Xiaoyu Li, Ying Shan, Xun Cao | cs.CV | [PDF](http://arxiv.org/pdf/2304.08757v1){: .btn .btn-green } |

**Abstract**: Recent advances in implicit neural representation have demonstrated the
ability to recover detailed geometry and material from multi-view images.
However, the use of simplified lighting models such as environment maps to
represent non-distant illumination, or using a network to fit indirect light
modeling without a solid basis, can lead to an undesirable decomposition
between lighting and material. To address this, we propose a fully
differentiable framework named neural ambient illumination (NeAI) that uses
Neural Radiance Fields (NeRF) as a lighting model to handle complex lighting in
a physically based way. Together with integral lobe encoding for
roughness-adaptive specular lobe and leveraging the pre-convoluted background
for accurate decomposition, the proposed method represents a significant step
towards integrating physically based rendering into the NeRF representation.
The experiments demonstrate the superior performance of novel-view rendering
compared to previous works, and the capability to re-render objects under
arbitrary NeRF-style environments opens up exciting possibilities for bridging
the gap between virtual and real-world scenes. The project and supplementary
materials are available at https://yiyuzhuang.github.io/NeAI/.

Comments:
- Project page: <a class="link-external link-https"
  href="https://yiyuzhuang.github.io/NeAI/" rel="external noopener
  nofollow">https://yiyuzhuang.github.io/NeAI/</a>

---

## SurfelNeRF: Neural Surfel Radiance Fields for Online Photorealistic  Reconstruction of Indoor Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-18 | Yiming Gao, Yan-Pei Cao, Ying Shan | cs.CV | [PDF](http://arxiv.org/pdf/2304.08971v1){: .btn .btn-green } |

**Abstract**: Online reconstructing and rendering of large-scale indoor scenes is a
long-standing challenge. SLAM-based methods can reconstruct 3D scene geometry
progressively in real time but can not render photorealistic results. While
NeRF-based methods produce promising novel view synthesis results, their long
offline optimization time and lack of geometric constraints pose challenges to
efficiently handling online input. Inspired by the complementary advantages of
classical 3D reconstruction and NeRF, we thus investigate marrying explicit
geometric representation with NeRF rendering to achieve efficient online
reconstruction and high-quality rendering. We introduce SurfelNeRF, a variant
of neural radiance field which employs a flexible and scalable neural surfel
representation to store geometric attributes and extracted appearance features
from input images. We further extend the conventional surfel-based fusion
scheme to progressively integrate incoming input frames into the reconstructed
global neural scene representation. In addition, we propose a highly-efficient
differentiable rasterization scheme for rendering neural surfel radiance
fields, which helps SurfelNeRF achieve $10\times$ speedups in training and
inference time, respectively. Experimental results show that our method
achieves the state-of-the-art 23.82 PSNR and 29.58 PSNR on ScanNet in
feedforward inference and per-scene optimization settings, respectively.

Comments:
- To appear in CVPR 2023

---

## MoDA: Modeling Deformable 3D Objects from Casual Videos

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-17 | Chaoyue Song, Tianyi Chen, Yiwen Chen, Jiacheng Wei, Chuan Sheng Foo, Fayao Liu, Guosheng Lin | cs.CV | [PDF](http://arxiv.org/pdf/2304.08279v2){: .btn .btn-green } |

**Abstract**: In this paper, we focus on the challenges of modeling deformable 3D objects
from casual videos. With the popularity of neural radiance fields (NeRF), many
works extend it to dynamic scenes with a canonical NeRF and a deformation model
that achieves 3D point transformation between the observation space and the
canonical space. Recent works rely on linear blend skinning (LBS) to achieve
the canonical-observation transformation. However, the linearly weighted
combination of rigid transformation matrices is not guaranteed to be rigid. As
a matter of fact, unexpected scale and shear factors often appear. In practice,
using LBS as the deformation model can always lead to skin-collapsing artifacts
for bending or twisting motions. To solve this problem, we propose neural dual
quaternion blend skinning (NeuDBS) to achieve 3D point deformation, which can
perform rigid transformation without skin-collapsing artifacts. In the endeavor
to register 2D pixels across different frames, we establish a correspondence
between canonical feature embeddings that encodes 3D points within the
canonical space, and 2D image features by solving an optimal transport problem.
Besides, we introduce a texture filtering approach for texture rendering that
effectively minimizes the impact of noisy colors outside target deformable
objects. Extensive experiments on real and synthetic datasets show that our
approach can reconstruct 3D models for humans and animals with better
qualitative and quantitative performance than state-of-the-art methods.

---

## NeRF-Loc: Visual Localization with Conditional Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-17 | Jianlin Liu, Qiang Nie, Yong Liu, Chengjie Wang | cs.CV | [PDF](http://arxiv.org/pdf/2304.07979v1){: .btn .btn-green } |

**Abstract**: We propose a novel visual re-localization method based on direct matching
between the implicit 3D descriptors and the 2D image with transformer. A
conditional neural radiance field(NeRF) is chosen as the 3D scene
representation in our pipeline, which supports continuous 3D descriptors
generation and neural rendering. By unifying the feature matching and the scene
coordinate regression to the same framework, our model learns both
generalizable knowledge and scene prior respectively during two training
stages. Furthermore, to improve the localization robustness when domain gap
exists between training and testing phases, we propose an appearance adaptation
layer to explicitly align styles between the 3D model and the query image.
Experiments show that our method achieves higher localization accuracy than
other learning-based approaches on multiple benchmarks. Code is available at
\url{https://github.com/JenningsL/nerf-loc}.

Comments:
- accepted by ICRA 2023

---

## Likelihood-Based Generative Radiance Field with Latent Space  Energy-Based Model for 3D-Aware Disentangled Image Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-16 | Yaxuan Zhu, Jianwen Xie, Ping Li | cs.CV | [PDF](http://arxiv.org/pdf/2304.07918v1){: .btn .btn-green } |

**Abstract**: We propose the NeRF-LEBM, a likelihood-based top-down 3D-aware 2D image
generative model that incorporates 3D representation via Neural Radiance Fields
(NeRF) and 2D imaging process via differentiable volume rendering. The model
represents an image as a rendering process from 3D object to 2D image and is
conditioned on some latent variables that account for object characteristics
and are assumed to follow informative trainable energy-based prior models. We
propose two likelihood-based learning frameworks to train the NeRF-LEBM: (i)
maximum likelihood estimation with Markov chain Monte Carlo-based inference and
(ii) variational inference with the reparameterization trick. We study our
models in the scenarios with both known and unknown camera poses. Experiments
on several benchmark datasets demonstrate that the NeRF-LEBM can infer 3D
object structures from 2D images, generate 2D images with novel views and
objects, learn from incomplete 2D images, and learn from 2D images with known
or unknown camera poses.

---

## SeaThru-NeRF: Neural Radiance Fields in Scattering Media

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-16 | Deborah Levy, Amit Peleg, Naama Pearl, Dan Rosenbaum, Derya Akkaynak, Simon Korman, Tali Treibitz | cs.CV | [PDF](http://arxiv.org/pdf/2304.07743v1){: .btn .btn-green } |

**Abstract**: Research on neural radiance fields (NeRFs) for novel view generation is
exploding with new models and extensions. However, a question that remains
unanswered is what happens in underwater or foggy scenes where the medium
strongly influences the appearance of objects. Thus far, NeRF and its variants
have ignored these cases. However, since the NeRF framework is based on
volumetric rendering, it has inherent capability to account for the medium's
effects, once modeled appropriately. We develop a new rendering model for NeRFs
in scattering media, which is based on the SeaThru image formation model, and
suggest a suitable architecture for learning both scene information and medium
parameters. We demonstrate the strength of our method using simulated and
real-world scenes, correctly rendering novel photorealistic views underwater.
Even more excitingly, we can render clear views of these scenes, removing the
medium between the camera and the scene and reconstructing the appearance and
depth of far objects, which are severely occluded by the medium. Our code and
unique datasets are available on the project's website.

---

## CAT-NeRF: Constancy-Aware Tx$^2$Former for Dynamic Body Modeling

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-16 | Haidong Zhu, Zhaoheng Zheng, Wanrong Zheng, Ram Nevatia | cs.CV | [PDF](http://arxiv.org/pdf/2304.07915v1){: .btn .btn-green } |

**Abstract**: This paper addresses the problem of human rendering in the video with
temporal appearance constancy. Reconstructing dynamic body shapes with
volumetric neural rendering methods, such as NeRF, requires finding the
correspondence of the points in the canonical and observation space, which
demands understanding human body shape and motion. Some methods use rigid
transformation, such as SE(3), which cannot precisely model each frame's unique
motion and muscle movements. Others generate the transformation for each frame
with a trainable network, such as neural blend weight field or translation
vector field, which does not consider the appearance constancy of general body
shape. In this paper, we propose CAT-NeRF for self-awareness of appearance
constancy with Tx$^2$Former, a novel way to combine two Transformer layers, to
separate appearance constancy and uniqueness. Appearance constancy models the
general shape across the video, and uniqueness models the unique patterns for
each frame. We further introduce a novel Covariance Loss to limit the
correlation between each pair of appearance uniquenesses to ensure the
frame-unique pattern is maximally captured in appearance uniqueness. We assess
our method on H36M and ZJU-MoCap and show state-of-the-art performance.

---

## UVA: Towards Unified Volumetric Avatar for View Synthesis, Pose  rendering, Geometry and Texture Editing

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-14 | Jinlong Fan, Jing Zhang, Dacheng Tao | cs.CV | [PDF](http://arxiv.org/pdf/2304.06969v1){: .btn .btn-green } |

**Abstract**: Neural radiance field (NeRF) has become a popular 3D representation method
for human avatar reconstruction due to its high-quality rendering capabilities,
e.g., regarding novel views and poses. However, previous methods for editing
the geometry and appearance of the avatar only allow for global editing through
body shape parameters and 2D texture maps. In this paper, we propose a new
approach named \textbf{U}nified \textbf{V}olumetric \textbf{A}vatar
(\textbf{UVA}) that enables local and independent editing of both geometry and
texture, while retaining the ability to render novel views and poses. UVA
transforms each observation point to a canonical space using a skinning motion
field and represents geometry and texture in separate neural fields. Each field
is composed of a set of structured latent codes that are attached to anchor
nodes on a deformable mesh in canonical space and diffused into the entire
space via interpolation, allowing for local editing. To address spatial
ambiguity in code interpolation, we use a local signed height indicator. We
also replace the view-dependent radiance color with a pose-dependent shading
factor to better represent surface illumination in different poses. Experiments
on multiple human avatars demonstrate that our UVA achieves competitive results
in novel view synthesis and novel pose rendering while enabling local and
independent editing of geometry and appearance. The source code will be
released.

---

## Single-Stage Diffusion NeRF: A Unified Approach to 3D Generation and  Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-13 | Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen Tu, Lingjie Liu, Hao Su | cs.CV | [PDF](http://arxiv.org/pdf/2304.06714v4){: .btn .btn-green } |

**Abstract**: 3D-aware image synthesis encompasses a variety of tasks, such as scene
generation and novel view synthesis from images. Despite numerous task-specific
methods, developing a comprehensive model remains challenging. In this paper,
we present SSDNeRF, a unified approach that employs an expressive diffusion
model to learn a generalizable prior of neural radiance fields (NeRF) from
multi-view images of diverse objects. Previous studies have used two-stage
approaches that rely on pretrained NeRFs as real data to train diffusion
models. In contrast, we propose a new single-stage training paradigm with an
end-to-end objective that jointly optimizes a NeRF auto-decoder and a latent
diffusion model, enabling simultaneous 3D reconstruction and prior learning,
even from sparsely available views. At test time, we can directly sample the
diffusion prior for unconditional generation, or combine it with arbitrary
observations of unseen objects for NeRF reconstruction. SSDNeRF demonstrates
robust results comparable to or better than leading task-specific methods in
unconditional generation and single/sparse-view 3D reconstruction.

Comments:
- ICCV 2023 final version. Project page:
  https://lakonik.github.io/ssdnerf

---

## Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-13 | Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, Peter Hedman | cs.CV | [PDF](http://arxiv.org/pdf/2304.06706v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Field training can be accelerated through the use of
grid-based representations in NeRF's learned mapping from spatial coordinates
to colors and volumetric density. However, these grid-based approaches lack an
explicit understanding of scale and therefore often introduce aliasing, usually
in the form of jaggies or missing scene content. Anti-aliasing has previously
been addressed by mip-NeRF 360, which reasons about sub-volumes along a cone
rather than points along a ray, but this approach is not natively compatible
with current grid-based techniques. We show how ideas from rendering and signal
processing can be used to construct a technique that combines mip-NeRF 360 and
grid-based models such as Instant NGP to yield error rates that are 8% - 77%
lower than either prior technique, and that trains 24x faster than mip-NeRF
360.

Comments:
- Project page: https://jonbarron.info/zipnerf/

---

## NeRFVS: Neural Radiance Fields for Free View Synthesis via Geometry  Scaffolds

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-13 | Chen Yang, Peihao Li, Zanwei Zhou, Shanxin Yuan, Bingbing Liu, Xiaokang Yang, Weichao Qiu, Wei Shen | cs.CV | [PDF](http://arxiv.org/pdf/2304.06287v2){: .btn .btn-green } |

**Abstract**: We present NeRFVS, a novel neural radiance fields (NeRF) based method to
enable free navigation in a room. NeRF achieves impressive performance in
rendering images for novel views similar to the input views while suffering for
novel views that are significantly different from the training views. To
address this issue, we utilize the holistic priors, including pseudo depth maps
and view coverage information, from neural reconstruction to guide the learning
of implicit neural representations of 3D indoor scenes. Concretely, an
off-the-shelf neural reconstruction method is leveraged to generate a geometry
scaffold. Then, two loss functions based on the holistic priors are proposed to
improve the learning of NeRF: 1) A robust depth loss that can tolerate the
error of the pseudo depth map to guide the geometry learning of NeRF; 2) A
variance loss to regularize the variance of implicit neural representations to
reduce the geometry and color ambiguity in the learning procedure. These two
loss functions are modulated during NeRF optimization according to the view
coverage information to reduce the negative influence brought by the view
coverage imbalance. Extensive results demonstrate that our NeRFVS outperforms
state-of-the-art view synthesis methods quantitatively and qualitatively on
indoor scenes, achieving high-fidelity free navigation results.

Comments:
- 10 pages, 7 figures

---

## RO-MAP: Real-Time Multi-Object Mapping with Neural Radiance Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-12 | Xiao Han, Houxuan Liu, Yunchao Ding, Lu Yang | cs.RO | [PDF](http://arxiv.org/pdf/2304.05735v2){: .btn .btn-green } |

**Abstract**: Accurate perception of objects in the environment is important for improving
the scene understanding capability of SLAM systems. In robotic and augmented
reality applications, object maps with semantic and metric information show
attractive advantages. In this paper, we present RO-MAP, a novel multi-object
mapping pipeline that does not rely on 3D priors. Given only monocular input,
we use neural radiance fields to represent objects and couple them with a
lightweight object SLAM based on multi-view geometry, to simultaneously
localize objects and implicitly learn their dense geometry. We create separate
implicit models for each detected object and train them dynamically and in
parallel as new observations are added. Experiments on synthetic and real-world
datasets demonstrate that our method can generate semantic object map with
shape reconstruction, and be competitive with offline methods while achieving
real-time performance (25Hz). The code and dataset will be available at:
https://github.com/XiaoHan-Git/RO-MAP

Comments:
- The code and dataset are available at:
  https://github.com/XiaoHan-Git/RO-MAP

---

## NutritionVerse-Thin: An Optimized Strategy for Enabling Improved  Rendering of 3D Thin Food Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-12 | Chi-en Amy Tai, Jason Li, Sriram Kumar, Saeejith Nair, Yuhao Chen, Pengcheng Xi, Alexander Wong | cs.CV | [PDF](http://arxiv.org/pdf/2304.05620v1){: .btn .btn-green } |

**Abstract**: With the growth in capabilities of generative models, there has been growing
interest in using photo-realistic renders of common 3D food items to improve
downstream tasks such as food printing, nutrition prediction, or management of
food wastage. Despite 3D modelling capabilities being more accessible than ever
due to the success of NeRF based view-synthesis, such rendering methods still
struggle to correctly capture thin food objects, often generating meshes with
significant holes. In this study, we present an optimized strategy for enabling
improved rendering of thin 3D food models, and demonstrate qualitative
improvements in rendering quality. Our method generates the 3D model mesh via a
proposed thin-object-optimized differentiable reconstruction method and tailors
the strategy at both the data collection and training stages to better handle
thin objects. While simple, we find that this technique can be employed for
quick and highly consistent capturing of thin 3D objects.

---

## Improving Neural Radiance Fields with Depth-aware Optimization for Novel  View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-11 | Shu Chen, Junyao Li, Yang Zhang, Beiji Zou | cs.CV | [PDF](http://arxiv.org/pdf/2304.05218v1){: .btn .btn-green } |

**Abstract**: With dense inputs, Neural Radiance Fields (NeRF) is able to render
photo-realistic novel views under static conditions. Although the synthesis
quality is excellent, existing NeRF-based methods fail to obtain moderate
three-dimensional (3D) structures. The novel view synthesis quality drops
dramatically given sparse input due to the implicitly reconstructed inaccurate
3D-scene structure. We propose SfMNeRF, a method to better synthesize novel
views as well as reconstruct the 3D-scene geometry. SfMNeRF leverages the
knowledge from the self-supervised depth estimation methods to constrain the
3D-scene geometry during view synthesis training. Specifically, SfMNeRF employs
the epipolar, photometric consistency, depth smoothness, and
position-of-matches constraints to explicitly reconstruct the 3D-scene
structure. Through these explicit constraints and the implicit constraint from
NeRF, our method improves the view synthesis as well as the 3D-scene geometry
performance of NeRF at the same time. In addition, SfMNeRF synthesizes novel
sub-pixels in which the ground truth is obtained by image interpolation. This
strategy enables SfMNeRF to include more samples to improve generalization
performance. Experiments on two public datasets demonstrate that SfMNeRF
surpasses state-of-the-art approaches. Code is available at
https://github.com/XTU-PR-LAB/SfMNeRF

---

## One-Shot High-Fidelity Talking-Head Synthesis with Deformable Neural  Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-11 | Weichuang Li, Longhao Zhang, Dong Wang, Bin Zhao, Zhigang Wang, Mulin Chen, Bang Zhang, Zhongjian Wang, Liefeng Bo, Xuelong Li | cs.CV | [PDF](http://arxiv.org/pdf/2304.05097v1){: .btn .btn-green } |

**Abstract**: Talking head generation aims to generate faces that maintain the identity
information of the source image and imitate the motion of the driving image.
Most pioneering methods rely primarily on 2D representations and thus will
inevitably suffer from face distortion when large head rotations are
encountered. Recent works instead employ explicit 3D structural representations
or implicit neural rendering to improve performance under large pose changes.
Nevertheless, the fidelity of identity and expression is not so desirable,
especially for novel-view synthesis. In this paper, we propose HiDe-NeRF, which
achieves high-fidelity and free-view talking-head synthesis. Drawing on the
recently proposed Deformable Neural Radiance Fields, HiDe-NeRF represents the
3D dynamic scene into a canonical appearance field and an implicit deformation
field, where the former comprises the canonical source face and the latter
models the driving pose and expression. In particular, we improve fidelity from
two aspects: (i) to enhance identity expressiveness, we design a generalized
appearance module that leverages multi-scale volume features to preserve face
shape and details; (ii) to improve expression preciseness, we propose a
lightweight deformation module that explicitly decouples the pose and
expression to enable precise expression modeling. Extensive experiments
demonstrate that our proposed approach can generate better results than
previous works. Project page: https://www.waytron.net/hidenerf/

Comments:
- Accepted by CVPR 2023

---

## MRVM-NeRF: Mask-Based Pretraining for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-11 | Ganlin Yang, Guoqiang Wei, Zhizheng Zhang, Yan Lu, Dong Liu | cs.CV | [PDF](http://arxiv.org/pdf/2304.04962v1){: .btn .btn-green } |

**Abstract**: Most Neural Radiance Fields (NeRFs) have poor generalization ability,
limiting their application when representing multiple scenes by a single model.
To ameliorate this problem, existing methods simply condition NeRF models on
image features, lacking the global understanding and modeling of the entire 3D
scene. Inspired by the significant success of mask-based modeling in other
research fields, we propose a masked ray and view modeling method for
generalizable NeRF (MRVM-NeRF), the first attempt to incorporate mask-based
pretraining into 3D implicit representations. Specifically, considering that
the core of NeRFs lies in modeling 3D representations along the rays and across
the views, we randomly mask a proportion of sampled points along the ray at
fine stage by discarding partial information obtained from multi-viewpoints,
targeting at predicting the corresponding features produced in the coarse
branch. In this way, the learned prior knowledge of 3D scenes during
pretraining helps the model generalize better to novel scenarios after
finetuning. Extensive experiments demonstrate the superiority of our proposed
MRVM-NeRF under various synthetic and real-world settings, both qualitatively
and quantitatively. Our empirical studies reveal the effectiveness of our
proposed innovative MRVM which is specifically designed for NeRF models.

---

## Neural Image-based Avatars: Generalizable Radiance Fields for Human  Avatar Modeling

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-10 | Youngjoong Kwon, Dahun Kim, Duygu Ceylan, Henry Fuchs | cs.CV | [PDF](http://arxiv.org/pdf/2304.04897v1){: .btn .btn-green } |

**Abstract**: We present a method that enables synthesizing novel views and novel poses of
arbitrary human performers from sparse multi-view images. A key ingredient of
our method is a hybrid appearance blending module that combines the advantages
of the implicit body NeRF representation and image-based rendering. Existing
generalizable human NeRF methods that are conditioned on the body model have
shown robustness against the geometric variation of arbitrary human performers.
Yet they often exhibit blurry results when generalized onto unseen identities.
Meanwhile, image-based rendering shows high-quality results when sufficient
observations are available, whereas it suffers artifacts in sparse-view
settings. We propose Neural Image-based Avatars (NIA) that exploits the best of
those two methods: to maintain robustness under new articulations and
self-occlusions while directly leveraging the available (sparse) source view
colors to preserve appearance details of new subject identities. Our hybrid
design outperforms recent methods on both in-domain identity generalization as
well as challenging cross-dataset generalization settings. Also, in terms of
the pose generalization, our method outperforms even the per-subject optimized
animatable NeRF methods. The video results are available at
https://youngjoongunc.github.io/nia

---

## Neural Residual Radiance Fields for Streamably Free-Viewpoint Videos

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-10 | Liao Wang, Qiang Hu, Qihan He, Ziyu Wang, Jingyi Yu, Tinne Tuytelaars, Lan Xu, Minye Wu | cs.CV | [PDF](http://arxiv.org/pdf/2304.04452v2){: .btn .btn-green } |

**Abstract**: The success of the Neural Radiance Fields (NeRFs) for modeling and free-view
rendering static objects has inspired numerous attempts on dynamic scenes.
Current techniques that utilize neural rendering for facilitating free-view
videos (FVVs) are restricted to either offline rendering or are capable of
processing only brief sequences with minimal motion. In this paper, we present
a novel technique, Residual Radiance Field or ReRF, as a highly compact neural
representation to achieve real-time FVV rendering on long-duration dynamic
scenes. ReRF explicitly models the residual information between adjacent
timestamps in the spatial-temporal feature space, with a global
coordinate-based tiny MLP as the feature decoder. Specifically, ReRF employs a
compact motion grid along with a residual feature grid to exploit inter-frame
feature similarities. We show such a strategy can handle large motions without
sacrificing quality. We further present a sequential training scheme to
maintain the smoothness and the sparsity of the motion/residual grids. Based on
ReRF, we design a special FVV codec that achieves three orders of magnitudes
compression rate and provides a companion ReRF player to support online
streaming of long-duration FVVs of dynamic scenes. Extensive experiments
demonstrate the effectiveness of ReRF for compactly representing dynamic
radiance fields, enabling an unprecedented free-viewpoint viewing experience in
speed and quality.

Comments:
- Accepted by CVPR 2023. Project page, see
  https://aoliao12138.github.io/ReRF/

---

## Inferring Fluid Dynamics via Inverse Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-10 | Jinxian Liu, Ye Chen, Bingbing Ni, Jiyao Mao, Zhenbo Yu | cs.CV | [PDF](http://arxiv.org/pdf/2304.04446v1){: .btn .btn-green } |

**Abstract**: Humans have a strong intuitive understanding of physical processes such as
fluid falling by just a glimpse of such a scene picture, i.e., quickly derived
from our immersive visual experiences in memory. This work achieves such a
photo-to-fluid-dynamics reconstruction functionality learned from unannotated
videos, without any supervision of ground-truth fluid dynamics. In a nutshell,
a differentiable Euler simulator modeled with a ConvNet-based pressure
projection solver, is integrated with a volumetric renderer, supporting
end-to-end/coherent differentiable dynamic simulation and rendering. By
endowing each sampled point with a fluid volume value, we derive a NeRF-like
differentiable renderer dedicated from fluid data; and thanks to this
volume-augmented representation, fluid dynamics could be inversely inferred
from the error signal between the rendered result and ground-truth video frame
(i.e., inverse rendering). Experiments on our generated Fluid Fall datasets and
DPI Dam Break dataset are conducted to demonstrate both effectiveness and
generalization ability of our method.

---

## Instance Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-10 | Yichen Liu, Benran Hu, Junkai Huang, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2304.04395v3){: .btn .btn-green } |

**Abstract**: This paper presents one of the first learning-based NeRF 3D instance
segmentation pipelines, dubbed as Instance Neural Radiance Field, or Instance
NeRF. Taking a NeRF pretrained from multi-view RGB images as input, Instance
NeRF can learn 3D instance segmentation of a given scene, represented as an
instance field component of the NeRF model. To this end, we adopt a 3D
proposal-based mask prediction network on the sampled volumetric features from
NeRF, which generates discrete 3D instance masks. The coarse 3D mask prediction
is then projected to image space to match 2D segmentation masks from different
views generated by existing panoptic segmentation models, which are used to
supervise the training of the instance field. Notably, beyond generating
consistent 2D segmentation maps from novel views, Instance NeRF can query
instance information at any 3D point, which greatly enhances NeRF object
segmentation and manipulation. Our method is also one of the first to achieve
such results in pure inference. Experimented on synthetic and real-world NeRF
datasets with complex indoor scenes, Instance NeRF surpasses previous NeRF
segmentation works and competitive 2D segmentation methods in segmentation
performance on unseen views. Watch the demo video at
https://youtu.be/wW9Bme73coI. Code and data are available at
https://github.com/lyclyc52/Instance_NeRF.

Comments:
- International Conference on Computer Vision (ICCV) 2023

---

## NeRF applied to satellite imagery for surface reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-09 | Federico Semeraro, Yi Zhang, Wenying Wu, Patrick Carroll | cs.CV | [PDF](http://arxiv.org/pdf/2304.04133v4){: .btn .btn-green } |

**Abstract**: We present Surf-NeRF, a modified implementation of the recently introduced
Shadow Neural Radiance Field (S-NeRF) model. This method is able to synthesize
novel views from a sparse set of satellite images of a scene, while accounting
for the variation in lighting present in the pictures. The trained model can
also be used to accurately estimate the surface elevation of the scene, which
is often a desirable quantity for satellite observation applications. S-NeRF
improves on the standard Neural Radiance Field (NeRF) method by considering the
radiance as a function of the albedo and the irradiance. Both these quantities
are output by fully connected neural network branches of the model, and the
latter is considered as a function of the direct light from the sun and the
diffuse color from the sky. The implementations were run on a dataset of
satellite images, augmented using a zoom-and-crop technique. A hyperparameter
study for NeRF was carried out, leading to intriguing observations on the
model's convergence. Finally, both NeRF and S-NeRF were run until 100k epochs
in order to fully fit the data and produce their best possible predictions. The
code related to this article can be found at
https://github.com/fsemerar/surfnerf.

---

## PVD-AL: Progressive Volume Distillation with Active Learning for  Efficient Conversion Between Different NeRF Architectures

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-08 | Shuangkang Fang, Yufeng Wang, Yi Yang, Weixin Xu, Heng Wang, Wenrui Ding, Shuchang Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2304.04012v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have been widely adopted as practical and
versatile representations for 3D scenes, facilitating various downstream tasks.
However, different architectures, including plain Multi-Layer Perceptron (MLP),
Tensors, low-rank Tensors, Hashtables, and their compositions, have their
trade-offs. For instance, Hashtables-based representations allow for faster
rendering but lack clear geometric meaning, making spatial-relation-aware
editing challenging. To address this limitation and maximize the potential of
each architecture, we propose Progressive Volume Distillation with Active
Learning (PVD-AL), a systematic distillation method that enables any-to-any
conversions between different architectures. PVD-AL decomposes each structure
into two parts and progressively performs distillation from shallower to deeper
volume representation, leveraging effective information retrieved from the
rendering process. Additionally, a Three-Levels of active learning technique
provides continuous feedback during the distillation process, resulting in
high-performance results. Empirical evidence is presented to validate our
method on multiple benchmark datasets. For example, PVD-AL can distill an
MLP-based model from a Hashtables-based model at a 10~20X faster speed and
0.8dB~2dB higher PSNR than training the NeRF model from scratch. Moreover,
PVD-AL permits the fusion of diverse features among distinct structures,
enabling models with multiple editing properties and providing a more efficient
model to meet real-time requirements. Project website:http://sk-fun.fun/PVD-AL.

Comments:
- Project website: http://sk-fun.fun/PVD-AL. arXiv admin note:
  substantial text overlap with arXiv:2211.15977

---

## Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative  Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-07 | Leheng Li, Qing Lian, Luozhou Wang, Ningning Ma, Ying-Cong Chen | cs.CV | [PDF](http://arxiv.org/pdf/2304.03526v1){: .btn .btn-green } |

**Abstract**: This work explores the use of 3D generative models to synthesize training
data for 3D vision tasks. The key requirements of the generative models are
that the generated data should be photorealistic to match the real-world
scenarios, and the corresponding 3D attributes should be aligned with given
sampling labels. However, we find that the recent NeRF-based 3D GANs hardly
meet the above requirements due to their designed generation pipeline and the
lack of explicit 3D supervision. In this work, we propose Lift3D, an inverted
2D-to-3D generation framework to achieve the data generation objectives. Lift3D
has several merits compared to prior methods: (1) Unlike previous 3D GANs that
the output resolution is fixed after training, Lift3D can generalize to any
camera intrinsic with higher resolution and photorealistic output. (2) By
lifting well-disentangled 2D GAN to 3D object NeRF, Lift3D provides explicit 3D
information of generated objects, thus offering accurate 3D annotations for
downstream tasks. We evaluate the effectiveness of our framework by augmenting
autonomous driving datasets. Experimental results demonstrate that our data
generation framework can effectively improve the performance of 3D object
detectors. Project page: https://len-li.github.io/lift3d-web.

Comments:
- CVPR 2023

---

## Event-based Camera Tracker by $\nabla$t NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-07 | Mana Masuda, Yusuke Sekikawa, Hideo Saito | cs.CV | [PDF](http://arxiv.org/pdf/2304.04559v1){: .btn .btn-green } |

**Abstract**: When a camera travels across a 3D world, only a fraction of pixel value
changes; an event-based camera observes the change as sparse events. How can we
utilize sparse events for efficient recovery of the camera pose? We show that
we can recover the camera pose by minimizing the error between sparse events
and the temporal gradient of the scene represented as a neural radiance field
(NeRF). To enable the computation of the temporal gradient of the scene, we
augment NeRF's camera pose as a time function. When the input pose to the NeRF
coincides with the actual pose, the output of the temporal gradient of NeRF
equals the observed intensity changes on the event's points. Using this
principle, we propose an event-based camera pose tracking framework called
TeGRA which realizes the pose update by using the sparse event's observation.
To the best of our knowledge, this is the first camera pose estimation
algorithm using the scene's implicit representation and the sparse intensity
change from events.

---

## Beyond NeRF Underwater: Learning Neural Reflectance Fields for True  Color Correction of Marine Imagery

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-06 | Tianyi Zhang, Matthew Johnson-Roberson | cs.CV | [PDF](http://arxiv.org/pdf/2304.03384v2){: .btn .btn-green } |

**Abstract**: Underwater imagery often exhibits distorted coloration as a result of
light-water interactions, which complicates the study of benthic environments
in marine biology and geography. In this research, we propose an algorithm to
restore the true color (albedo) in underwater imagery by jointly learning the
effects of the medium and neural scene representations. Our approach models
water effects as a combination of light attenuation with distance and
backscattered light. The proposed neural scene representation is based on a
neural reflectance field model, which learns albedos, normals, and volume
densities of the underwater environment. We introduce a logistic regression
model to separate water from the scene and apply distinct light physics during
training. Our method avoids the need to estimate complex backscatter effects in
water by employing several approximations, enhancing sampling efficiency and
numerical stability during training. The proposed technique integrates
underwater light effects into a volume rendering framework with end-to-end
differentiability. Experimental results on both synthetic and real-world data
demonstrate that our method effectively restores true color from underwater
imagery, outperforming existing approaches in terms of color consistency.

Comments:
- Robotics and Automation Letters (RA-L) VOL. 8, NO. 10, OCTOBER 2023

---

## LANe: Lighting-Aware Neural Fields for Compositional Scene Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-06 | Akshay Krishnan, Amit Raj, Xianling Zhang, Alexandra Carlson, Nathan Tseng, Sandhya Sridhar, Nikita Jaipuria, James Hays | cs.CV | [PDF](http://arxiv.org/pdf/2304.03280v1){: .btn .btn-green } |

**Abstract**: Neural fields have recently enjoyed great success in representing and
rendering 3D scenes. However, most state-of-the-art implicit representations
model static or dynamic scenes as a whole, with minor variations. Existing work
on learning disentangled world and object neural fields do not consider the
problem of composing objects into different world neural fields in a
lighting-aware manner. We present Lighting-Aware Neural Field (LANe) for the
compositional synthesis of driving scenes in a physically consistent manner.
Specifically, we learn a scene representation that disentangles the static
background and transient elements into a world-NeRF and class-specific
object-NeRFs to allow compositional synthesis of multiple objects in the scene.
Furthermore, we explicitly designed both the world and object models to handle
lighting variation, which allows us to compose objects into scenes with
spatially varying lighting. This is achieved by constructing a light field of
the scene and using it in conjunction with a learned shader to modulate the
appearance of the object NeRFs. We demonstrate the performance of our model on
a synthetic dataset of diverse lighting conditions rendered with the CARLA
simulator, as well as a novel real-world dataset of cars collected at different
times of the day. Our approach shows that it outperforms state-of-the-art
compositional scene synthesis on the challenging dataset setup, via composing
object-NeRFs learned from one scene into an entirely different scene whilst
still respecting the lighting variations in the novel scene. For more results,
please visit our project website https://lane-composition.github.io/.

Comments:
- Project website: https://lane-composition.github.io

---

## Neural Fields meet Explicit Geometric Representation for Inverse  Rendering of Urban Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-06 | Zian Wang, Tianchang Shen, Jun Gao, Shengyu Huang, Jacob Munkberg, Jon Hasselgren, Zan Gojcic, Wenzheng Chen, Sanja Fidler | cs.CV | [PDF](http://arxiv.org/pdf/2304.03266v1){: .btn .btn-green } |

**Abstract**: Reconstruction and intrinsic decomposition of scenes from captured imagery
would enable many applications such as relighting and virtual object insertion.
Recent NeRF based methods achieve impressive fidelity of 3D reconstruction, but
bake the lighting and shadows into the radiance field, while mesh-based methods
that facilitate intrinsic decomposition through differentiable rendering have
not yet scaled to the complexity and scale of outdoor scenes. We present a
novel inverse rendering framework for large urban scenes capable of jointly
reconstructing the scene geometry, spatially-varying materials, and HDR
lighting from a set of posed RGB images with optional depth. Specifically, we
use a neural field to account for the primary rays, and use an explicit mesh
(reconstructed from the underlying neural field) for modeling secondary rays
that produce higher-order lighting effects such as cast shadows. By faithfully
disentangling complex geometry and materials from lighting effects, our method
enables photorealistic relighting with specular and shadow effects on several
outdoor datasets. Moreover, it supports physics-based scene manipulations such
as virtual object insertion with ray-traced shadow casting.

Comments:
- CVPR 2023. Project page: https://nv-tlabs.github.io/fegr/

---

## DITTO-NeRF: Diffusion-based Iterative Text To Omni-directional 3D Model

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-06 | Hoigi Seo, Hayeon Kim, Gwanghyun Kim, Se Young Chun | cs.CV | [PDF](http://arxiv.org/pdf/2304.02827v1){: .btn .btn-green } |

**Abstract**: The increasing demand for high-quality 3D content creation has motivated the
development of automated methods for creating 3D object models from a single
image and/or from a text prompt. However, the reconstructed 3D objects using
state-of-the-art image-to-3D methods still exhibit low correspondence to the
given image and low multi-view consistency. Recent state-of-the-art text-to-3D
methods are also limited, yielding 3D samples with low diversity per prompt
with long synthesis time. To address these challenges, we propose DITTO-NeRF, a
novel pipeline to generate a high-quality 3D NeRF model from a text prompt or a
single image. Our DITTO-NeRF consists of constructing high-quality partial 3D
object for limited in-boundary (IB) angles using the given or text-generated 2D
image from the frontal view and then iteratively reconstructing the remaining
3D NeRF using inpainting latent diffusion model. We propose progressive 3D
object reconstruction schemes in terms of scales (low to high resolution),
angles (IB angles initially to outer-boundary (OB) later), and masks (object to
background boundary) in our DITTO-NeRF so that high-quality information on IB
can be propagated into OB. Our DITTO-NeRF outperforms state-of-the-art methods
in terms of fidelity and diversity qualitatively and quantitatively with much
faster training times than prior arts on image/text-to-3D such as DreamFusion,
and NeuralLift-360.

Comments:
- Project page: https://janeyeon.github.io/ditto-nerf/

---

## Image Stabilization for Hololens Camera in Remote Collaboration

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-05 | Gowtham Senthil, Siva Vignesh Krishnan, Annamalai Lakshmanan, Florence Kissling | cs.CV | [PDF](http://arxiv.org/pdf/2304.02736v1){: .btn .btn-green } |

**Abstract**: With the advent of new technologies, Augmented Reality (AR) has become an
effective tool in remote collaboration. Narrow field-of-view (FoV) and motion
blur can offer an unpleasant experience with limited cognition for remote
viewers of AR headsets. In this article, we propose a two-stage pipeline to
tackle this issue and ensure a stable viewing experience with a larger FoV. The
solution involves an offline 3D reconstruction of the indoor environment,
followed by enhanced rendering using only the live poses of AR device. We
experiment with and evaluate the two different 3D reconstruction methods, RGB-D
geometric approach and Neural Radiance Fields (NeRF), based on their data
requirements, reconstruction quality, rendering, and training times. The
generated sequences from these methods had smoother transitions and provided a
better perspective of the environment. The geometry-based enhanced FoV method
had better renderings as it lacked blurry outputs making it better than the
other attempted approaches. Structural Similarity Index (SSIM) and Peak Signal
to Noise Ratio (PSNR) metrics were used to quantitatively show that the
rendering quality using the geometry-based enhanced FoV method is better. Link
to the code repository -
https://github.com/MixedRealityETHZ/ImageStabilization.

---

## Generating Continual Human Motion in Diverse 3D Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-04 | Aymen Mir, Xavier Puig, Angjoo Kanazawa, Gerard Pons-Moll | cs.CV | [PDF](http://arxiv.org/pdf/2304.02061v3){: .btn .btn-green } |

**Abstract**: We introduce a method to synthesize animator guided human motion across 3D
scenes. Given a set of sparse (3 or 4) joint locations (such as the location of
a person's hand and two feet) and a seed motion sequence in a 3D scene, our
method generates a plausible motion sequence starting from the seed motion
while satisfying the constraints imposed by the provided keypoints. We
decompose the continual motion synthesis problem into walking along paths and
transitioning in and out of the actions specified by the keypoints, which
enables long generation of motions that satisfy scene constraints without
explicitly incorporating scene information. Our method is trained only using
scene agnostic mocap data. As a result, our approach is deployable across 3D
scenes with various geometries. For achieving plausible continual motion
synthesis without drift, our key contribution is to generate motion in a
goal-centric canonical coordinate frame where the next immediate target is
situated at the origin. Our model can generate long sequences of diverse
actions such as grabbing, sitting and leaning chained together in arbitrary
order, demonstrated on scenes of varying geometry: HPS, Replica, Matterport,
ScanNet and scenes represented using NeRFs. Several experiments demonstrate
that our method outperforms existing methods that navigate paths in 3D scenes.

---

## MonoHuman: Animatable Human Neural Field from Monocular Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-04 | Zhengming Yu, Wei Cheng, Xian Liu, Wayne Wu, Kwan-Yee Lin | cs.CV | [PDF](http://arxiv.org/pdf/2304.02001v1){: .btn .btn-green } |

**Abstract**: Animating virtual avatars with free-view control is crucial for various
applications like virtual reality and digital entertainment. Previous studies
have attempted to utilize the representation power of the neural radiance field
(NeRF) to reconstruct the human body from monocular videos. Recent works
propose to graft a deformation network into the NeRF to further model the
dynamics of the human neural field for animating vivid human motions. However,
such pipelines either rely on pose-dependent representations or fall short of
motion coherency due to frame-independent optimization, making it difficult to
generalize to unseen pose sequences realistically. In this paper, we propose a
novel framework MonoHuman, which robustly renders view-consistent and
high-fidelity avatars under arbitrary novel poses. Our key insight is to model
the deformation field with bi-directional constraints and explicitly leverage
the off-the-peg keyframe information to reason the feature correlations for
coherent results. Specifically, we first propose a Shared Bidirectional
Deformation module, which creates a pose-independent generalizable deformation
field by disentangling backward and forward deformation correspondences into
shared skeletal motion weight and separate non-rigid motions. Then, we devise a
Forward Correspondence Search module, which queries the correspondence feature
of keyframes to guide the rendering network. The rendered results are thus
multi-view consistent with high fidelity, even under challenging novel pose
settings. Extensive experiments demonstrate the superiority of our proposed
MonoHuman over state-of-the-art methods.

Comments:
- 15 pages, 14 figures. Accepted to CVPR 2023. Project page:
  https://yzmblog.github.io/projects/MonoHuman/

---

## Learning Personalized High Quality Volumetric Head Avatars from  Monocular RGB Videos



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-04 | Ziqian Bai, Feitong Tan, Zeng Huang, Kripasindhu Sarkar, Danhang Tang, Di Qiu, Abhimitra Meka, Ruofei Du, Mingsong Dou, Sergio Orts-Escolano, Rohit Pandey, Ping Tan, Thabo Beeler, Sean Fanello, Yinda Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2304.01436v1){: .btn .btn-green } |

**Abstract**: We propose a method to learn a high-quality implicit 3D head avatar from a
monocular RGB video captured in the wild. The learnt avatar is driven by a
parametric face model to achieve user-controlled facial expressions and head
poses. Our hybrid pipeline combines the geometry prior and dynamic tracking of
a 3DMM with a neural radiance field to achieve fine-grained control and
photorealism. To reduce over-smoothing and improve out-of-model expressions
synthesis, we propose to predict local features anchored on the 3DMM geometry.
These learnt features are driven by 3DMM deformation and interpolated in 3D
space to yield the volumetric radiance at a designated query point. We further
show that using a Convolutional Neural Network in the UV space is critical in
incorporating spatial context and producing representative local features.
Extensive experiments show that we are able to reconstruct high-quality
avatars, with more accurate expression-dependent details, good generalization
to out-of-training expressions, and quantitatively superior renderings compared
to other state-of-the-art approaches.

Comments:
- In CVPR2023. Project page:
  https://augmentedperception.github.io/monoavatar/

---

## DreamAvatar: Text-and-Shape Guided 3D Human Avatar Generation via  Diffusion Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-03 | Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, Kwan-Yee K. Wong | cs.CV | [PDF](http://arxiv.org/pdf/2304.00916v3){: .btn .btn-green } |

**Abstract**: We present DreamAvatar, a text-and-shape guided framework for generating
high-quality 3D human avatars with controllable poses. While encouraging
results have been reported by recent methods on text-guided 3D common object
generation, generating high-quality human avatars remains an open challenge due
to the complexity of the human body's shape, pose, and appearance. We propose
DreamAvatar to tackle this challenge, which utilizes a trainable NeRF for
predicting density and color for 3D points and pretrained text-to-image
diffusion models for providing 2D self-supervision. Specifically, we leverage
the SMPL model to provide shape and pose guidance for the generation. We
introduce a dual-observation-space design that involves the joint optimization
of a canonical space and a posed space that are related by a learnable
deformation field. This facilitates the generation of more complete textures
and geometry faithful to the target pose. We also jointly optimize the losses
computed from the full body and from the zoomed-in 3D head to alleviate the
common multi-face ''Janus'' problem and improve facial details in the generated
avatars. Extensive evaluations demonstrate that DreamAvatar significantly
outperforms existing methods, establishing a new state-of-the-art for
text-and-shape guided 3D human avatar generation.

Comments:
- Project page: https://yukangcao.github.io/DreamAvatar/

---

## Disorder-invariant Implicit Neural Representation



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-03 | Hao Zhu, Shaowen Xie, Zhen Liu, Fengyi Liu, Qi Zhang, You Zhou, Yi Lin, Zhan Ma, Xun Cao | cs.CV | [PDF](http://arxiv.org/pdf/2304.00837v1){: .btn .btn-green } |

**Abstract**: Implicit neural representation (INR) characterizes the attributes of a signal
as a function of corresponding coordinates which emerges as a sharp weapon for
solving inverse problems. However, the expressive power of INR is limited by
the spectral bias in the network training. In this paper, we find that such a
frequency-related problem could be greatly solved by re-arranging the
coordinates of the input signal, for which we propose the disorder-invariant
implicit neural representation (DINER) by augmenting a hash-table to a
traditional INR backbone. Given discrete signals sharing the same histogram of
attributes and different arrangement orders, the hash-table could project the
coordinates into the same distribution for which the mapped signal can be
better modeled using the subsequent INR network, leading to significantly
alleviated spectral bias. Furthermore, the expressive power of the DINER is
determined by the width of the hash-table. Different width corresponds to
different geometrical elements in the attribute space, \textit{e.g.}, 1D curve,
2D curved-plane and 3D curved-volume when the width is set as $1$, $2$ and $3$,
respectively. More covered areas of the geometrical elements result in stronger
expressive power. Experiments not only reveal the generalization of the DINER
for different INR backbones (MLP vs. SIREN) and various tasks (image/video
representation, phase retrieval, refractive index recovery, and neural radiance
field optimization) but also show the superiority over the state-of-the-art
algorithms both in quality and speed. \textit{Project page:}
\url{https://ezio77.github.io/DINER-website/}

Comments:
- Journal extension of the CVPR'23 highlight paper "DINER:
  Disorder-invariant Implicit Neural Representation". In the extension, we
  model the expressive power of the DINER using parametric functions in the
  attribute space. As a result, better results are achieved than the conference
  version. arXiv admin note: substantial text overlap with arXiv:2211.07871

---

## JacobiNeRF: NeRF Shaping with Mutual Information Gradients

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-04-01 | Xiaomeng Xu, Yanchao Yang, Kaichun Mo, Boxiao Pan, Li Yi, Leonidas Guibas | cs.CV | [PDF](http://arxiv.org/pdf/2304.00341v1){: .btn .btn-green } |

**Abstract**: We propose a method that trains a neural radiance field (NeRF) to encode not
only the appearance of the scene but also semantic correlations between scene
points, regions, or entities -- aiming to capture their mutual co-variation
patterns. In contrast to the traditional first-order photometric reconstruction
objective, our method explicitly regularizes the learning dynamics to align the
Jacobians of highly-correlated entities, which proves to maximize the mutual
information between them under random scene perturbations. By paying attention
to this second-order information, we can shape a NeRF to express semantically
meaningful synergies when the network weights are changed by a delta along the
gradient of a single entity, region, or even a point. To demonstrate the merit
of this mutual information modeling, we leverage the coordinated behavior of
scene entities that emerges from our shaping to perform label propagation for
semantic and instance segmentation. Our experiments show that a JacobiNeRF is
more efficient in propagating annotations among 2D pixels and 3D points
compared to NeRFs without mutual information shaping, especially in extremely
sparse label regimes -- thus reducing annotation burden. The same machinery can
further be used for entity selection or scene modifications.