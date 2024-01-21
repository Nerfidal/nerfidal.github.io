---
layout: default
title: September
parent: 2022
nav_order: 9
---
<!---metadata--->

## Improving 3D-aware Image Synthesis with A Geometry-aware Discriminator

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-30 | Zifan Shi, Yinghao Xu, Yujun Shen, Deli Zhao, Qifeng Chen, Dit-Yan Yeung | cs.CV | [PDF](http://arxiv.org/pdf/2209.15637v1){: .btn .btn-green } |

**Abstract**: 3D-aware image synthesis aims at learning a generative model that can render
photo-realistic 2D images while capturing decent underlying 3D shapes. A
popular solution is to adopt the generative adversarial network (GAN) and
replace the generator with a 3D renderer, where volume rendering with neural
radiance field (NeRF) is commonly used. Despite the advancement of synthesis
quality, existing methods fail to obtain moderate 3D shapes. We argue that,
considering the two-player game in the formulation of GANs, only making the
generator 3D-aware is not enough. In other words, displacing the generative
mechanism only offers the capability, but not the guarantee, of producing
3D-aware images, because the supervision of the generator primarily comes from
the discriminator. To address this issue, we propose GeoD through learning a
geometry-aware discriminator to improve 3D-aware GANs. Concretely, besides
differentiating real and fake samples from the 2D image space, the
discriminator is additionally asked to derive the geometry information from the
inputs, which is then applied as the guidance of the generator. Such a simple
yet effective design facilitates learning substantially more accurate 3D
shapes. Extensive experiments on various generator architectures and training
datasets verify the superiority of GeoD over state-of-the-art alternatives.
Moreover, our approach is registered as a general framework such that a more
capable discriminator (i.e., with a third task of novel view synthesis beyond
domain classification and geometry extraction) can further assist the generator
with a better multi-view consistency.

Comments:
- Accepted by NeurIPS 2022. Project page:
  https://vivianszf.github.io/geod

---

## TT-NF: Tensor Train Neural Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-30 | Anton Obukhov, Mikhail Usvyatsov, Christos Sakaridis, Konrad Schindler, Luc Van Gool | cs.LG | [PDF](http://arxiv.org/pdf/2209.15529v1){: .btn .btn-green } |

**Abstract**: Learning neural fields has been an active topic in deep learning research,
focusing, among other issues, on finding more compact and easy-to-fit
representations. In this paper, we introduce a novel low-rank representation
termed Tensor Train Neural Fields (TT-NF) for learning neural fields on dense
regular grids and efficient methods for sampling from them. Our representation
is a TT parameterization of the neural field, trained with backpropagation to
minimize a non-convex objective. We analyze the effect of low-rank compression
on the downstream task quality metrics in two settings. First, we demonstrate
the efficiency of our method in a sandbox task of tensor denoising, which
admits comparison with SVD-based schemes designed to minimize reconstruction
error. Furthermore, we apply the proposed approach to Neural Radiance Fields,
where the low-rank structure of the field corresponding to the best quality can
be discovered only through learning.

Comments:
- Preprint, under review

---

## Understanding Pure CLIP Guidance for Voxel Grid NeRF Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-30 | Han-Hung Lee, Angel X. Chang | cs.CV | [PDF](http://arxiv.org/pdf/2209.15172v1){: .btn .btn-green } |

**Abstract**: We explore the task of text to 3D object generation using CLIP. Specifically,
we use CLIP for guidance without access to any datasets, a setting we refer to
as pure CLIP guidance. While prior work has adopted this setting, there is no
systematic study of mechanics for preventing adversarial generations within
CLIP. We illustrate how different image-based augmentations prevent the
adversarial generation problem, and how the generated results are impacted. We
test different CLIP model architectures and show that ensembling different
models for guidance can prevent adversarial generations within bigger models
and generate sharper results. Furthermore, we implement an implicit voxel grid
model to show how neural networks provide an additional layer of
regularization, resulting in better geometrical structure and coherency of
generated objects. Compared to prior work, we achieve more coherent results
with higher memory efficiency and faster training speeds.

---

## DreamFusion: Text-to-3D using 2D Diffusion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-29 | Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall | cs.CV | [PDF](http://arxiv.org/pdf/2209.14988v1){: .btn .btn-green } |

**Abstract**: Recent breakthroughs in text-to-image synthesis have been driven by diffusion
models trained on billions of image-text pairs. Adapting this approach to 3D
synthesis would require large-scale datasets of labeled 3D data and efficient
architectures for denoising 3D data, neither of which currently exist. In this
work, we circumvent these limitations by using a pretrained 2D text-to-image
diffusion model to perform text-to-3D synthesis. We introduce a loss based on
probability density distillation that enables the use of a 2D diffusion model
as a prior for optimization of a parametric image generator. Using this loss in
a DeepDream-like procedure, we optimize a randomly-initialized 3D model (a
Neural Radiance Field, or NeRF) via gradient descent such that its 2D
renderings from random angles achieve a low loss. The resulting 3D model of the
given text can be viewed from any angle, relit by arbitrary illumination, or
composited into any 3D environment. Our approach requires no 3D training data
and no modifications to the image diffusion model, demonstrating the
effectiveness of pretrained image diffusion models as priors.

Comments:
- see project page at https://dreamfusion3d.github.io/

---

## SymmNeRF: Learning to Explore Symmetry Prior for Single-View View  Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-29 | Xingyi Li, Chaoyi Hong, Yiran Wang, Zhiguo Cao, Ke Xian, Guosheng Lin | cs.CV | [PDF](http://arxiv.org/pdf/2209.14819v2){: .btn .btn-green } |

**Abstract**: We study the problem of novel view synthesis of objects from a single image.
Existing methods have demonstrated the potential in single-view view synthesis.
However, they still fail to recover the fine appearance details, especially in
self-occluded areas. This is because a single view only provides limited
information. We observe that manmade objects usually exhibit symmetric
appearances, which introduce additional prior knowledge. Motivated by this, we
investigate the potential performance gains of explicitly embedding symmetry
into the scene representation. In this paper, we propose SymmNeRF, a neural
radiance field (NeRF) based framework that combines local and global
conditioning under the introduction of symmetry priors. In particular, SymmNeRF
takes the pixel-aligned image features and the corresponding symmetric features
as extra inputs to the NeRF, whose parameters are generated by a hypernetwork.
As the parameters are conditioned on the image-encoded latent codes, SymmNeRF
is thus scene-independent and can generalize to new scenes. Experiments on
synthetic and real-world datasets show that SymmNeRF synthesizes novel views
with more details regardless of the pose transformation, and demonstrates good
generalization when applied to unseen objects. Code is available at:
https://github.com/xingyi-li/SymmNeRF.

Comments:
- Accepted by ACCV 2022

---

## 360FusionNeRF: Panoramic Neural Radiance Fields with Joint Guidance

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-28 | Shreyas Kulkarni, Peng Yin, Sebastian Scherer | cs.CV | [PDF](http://arxiv.org/pdf/2209.14265v2){: .btn .btn-green } |

**Abstract**: We present a method to synthesize novel views from a single $360^\circ$
panorama image based on the neural radiance field (NeRF). Prior studies in a
similar setting rely on the neighborhood interpolation capability of
multi-layer perceptions to complete missing regions caused by occlusion, which
leads to artifacts in their predictions. We propose 360FusionNeRF, a
semi-supervised learning framework where we introduce geometric supervision and
semantic consistency to guide the progressive training process. Firstly, the
input image is re-projected to $360^\circ$ images, and auxiliary depth maps are
extracted at other camera positions. The depth supervision, in addition to the
NeRF color guidance, improves the geometry of the synthesized views.
Additionally, we introduce a semantic consistency loss that encourages
realistic renderings of novel views. We extract these semantic features using a
pre-trained visual encoder such as CLIP, a Vision Transformer trained on
hundreds of millions of diverse 2D photographs mined from the web with natural
language supervision. Experiments indicate that our proposed method can produce
plausible completions of unobserved regions while preserving the features of
the scene. When trained across various scenes, 360FusionNeRF consistently
achieves the state-of-the-art performance when transferring to synthetic
Structured3D dataset (PSNR~5%, SSIM~3% LPIPS~13%), real-world Matterport3D
dataset (PSNR~3%, SSIM~3% LPIPS~9%) and Replica360 dataset (PSNR~8%, SSIM~2%
LPIPS~18%).

Comments:
- 8 pages, Fig 3, Submitted to IEEE RAL. arXiv admin note: text overlap
  with arXiv:2106.10859, arXiv:2104.00677, arXiv:2203.09957, arXiv:2204.00928
  by other authors

---

## OmniNeRF: Hybriding Omnidirectional Distance and Radiance fields for  Neural Surface Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-27 | Jiaming Shen, Bolin Song, Zirui Wu, Yi Xu | cs.CV | [PDF](http://arxiv.org/pdf/2209.13433v1){: .btn .btn-green } |

**Abstract**: 3D reconstruction from images has wide applications in Virtual Reality and
Automatic Driving, where the precision requirement is very high.
Ground-breaking research in the neural radiance field (NeRF) by utilizing
Multi-Layer Perceptions has dramatically improved the representation quality of
3D objects. Some later studies improved NeRF by building truncated signed
distance fields (TSDFs) but still suffer from the problem of blurred surfaces
in 3D reconstruction. In this work, this surface ambiguity is addressed by
proposing a novel way of 3D shape representation, OmniNeRF. It is based on
training a hybrid implicit field of Omni-directional Distance Field (ODF) and
neural radiance field, replacing the apparent density in NeRF with
omnidirectional information. Moreover, we introduce additional supervision on
the depth map to further improve reconstruction quality. The proposed method
has been proven to effectively deal with NeRF defects at the edges of the
surface reconstruction, providing higher quality 3D scene reconstruction
results.

Comments:
- Accepted by CMSDA 2022

---

## Orbeez-SLAM: A Real-time Monocular Visual SLAM with ORB Features and  NeRF-realized Mapping

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-27 | Chi-Ming Chung, Yang-Che Tseng, Ya-Ching Hsu, Xiang-Qian Shi, Yun-Hung Hua, Jia-Fong Yeh, Wen-Chin Chen, Yi-Ting Chen, Winston H. Hsu | cs.RO | [PDF](http://arxiv.org/pdf/2209.13274v2){: .btn .btn-green } |

**Abstract**: A spatial AI that can perform complex tasks through visual signals and
cooperate with humans is highly anticipated. To achieve this, we need a visual
SLAM that easily adapts to new scenes without pre-training and generates dense
maps for downstream tasks in real-time. None of the previous learning-based and
non-learning-based visual SLAMs satisfy all needs due to the intrinsic
limitations of their components. In this work, we develop a visual SLAM named
Orbeez-SLAM, which successfully collaborates with implicit neural
representation and visual odometry to achieve our goals. Moreover, Orbeez-SLAM
can work with the monocular camera since it only needs RGB inputs, making it
widely applicable to the real world. Results show that our SLAM is up to 800x
faster than the strong baseline with superior rendering outcomes. Code link:
https://github.com/MarvinChung/Orbeez-SLAM.

---

## WaterNeRF: Neural Radiance Fields for Underwater Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-27 | Advaith Venkatramanan Sethuraman, Manikandasriram Srinivasan Ramanagopal, Katherine A. Skinner | cs.RO | [PDF](http://arxiv.org/pdf/2209.13091v2){: .btn .btn-green } |

**Abstract**: Underwater imaging is a critical task performed by marine robots for a wide
range of applications including aquaculture, marine infrastructure inspection,
and environmental monitoring. However, water column effects, such as
attenuation and backscattering, drastically change the color and quality of
imagery captured underwater. Due to varying water conditions and
range-dependency of these effects, restoring underwater imagery is a
challenging problem. This impacts downstream perception tasks including depth
estimation and 3D reconstruction. In this paper, we advance state-of-the-art in
neural radiance fields (NeRFs) to enable physics-informed dense depth
estimation and color correction. Our proposed method, WaterNeRF, estimates
parameters of a physics-based model for underwater image formation, leading to
a hybrid data-driven and model-based solution. After determining the scene
structure and radiance field, we can produce novel views of degraded as well as
corrected underwater images, along with dense depth of the scene. We evaluate
the proposed method qualitatively and quantitatively on a real underwater
dataset.

---

## Baking in the Feature: Accelerating Volumetric Segmentation by Rendering  Feature Maps

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-26 | Kenneth Blomqvist, Lionel Ott, Jen Jen Chung, Roland Siegwart | cs.CV | [PDF](http://arxiv.org/pdf/2209.12744v1){: .btn .btn-green } |

**Abstract**: Methods have recently been proposed that densely segment 3D volumes into
classes using only color images and expert supervision in the form of sparse
semantically annotated pixels. While impressive, these methods still require a
relatively large amount of supervision and segmenting an object can take
several minutes in practice. Such systems typically only optimize their
representation on the particular scene they are fitting, without leveraging any
prior information from previously seen images. In this paper, we propose to use
features extracted with models trained on large existing datasets to improve
segmentation performance. We bake this feature representation into a Neural
Radiance Field (NeRF) by volumetrically rendering feature maps and supervising
on features extracted from each input image. We show that by baking this
representation into the NeRF, we make the subsequent classification task much
easier. Our experiments show that our method achieves higher segmentation
accuracy with fewer semantic annotations than existing methods over a wide
range of scenes.

---

## Enforcing safety for vision-based controllers via Control Barrier  Functions and Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-25 | Mukun Tong, Charles Dawson, Chuchu Fan | cs.RO | [PDF](http://arxiv.org/pdf/2209.12266v3){: .btn .btn-green } |

**Abstract**: To navigate complex environments, robots must increasingly use
high-dimensional visual feedback (e.g. images) for control. However, relying on
high-dimensional image data to make control decisions raises important
questions; particularly, how might we prove the safety of a visual-feedback
controller? Control barrier functions (CBFs) are powerful tools for certifying
the safety of feedback controllers in the state-feedback setting, but CBFs have
traditionally been poorly-suited to visual feedback control due to the need to
predict future observations in order to evaluate the barrier function. In this
work, we solve this issue by leveraging recent advances in neural radiance
fields (NeRFs), which learn implicit representations of 3D scenes and can
render images from previously-unseen camera perspectives, to provide
single-step visual foresight for a CBF-based controller. This novel combination
is able to filter out unsafe actions and intervene to preserve safety. We
demonstrate the effect of our controller in real-time simulation experiments
where it successfully prevents the robot from taking dangerous actions.

Comments:
- Accepted to ICRA 2023

---

## NeRF-Loc: Transformer-Based Object Localization Within Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-24 | Jiankai Sun, Yan Xu, Mingyu Ding, Hongwei Yi, Chen Wang, Jingdong Wang, Liangjun Zhang, Mac Schwager | cs.CV | [PDF](http://arxiv.org/pdf/2209.12068v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have become a widely-applied scene
representation technique in recent years, showing advantages for robot
navigation and manipulation tasks. To further advance the utility of NeRFs for
robotics, we propose a transformer-based framework, NeRF-Loc, to extract 3D
bounding boxes of objects in NeRF scenes. NeRF-Loc takes a pre-trained NeRF
model and camera view as input and produces labeled, oriented 3D bounding boxes
of objects as output. Using current NeRF training tools, a robot can train a
NeRF environment model in real-time and, using our algorithm, identify 3D
bounding boxes of objects of interest within the NeRF for downstream navigation
or manipulation tasks. Concretely, we design a pair of paralleled transformer
encoder branches, namely the coarse stream and the fine stream, to encode both
the context and details of target objects. The encoded features are then fused
together with attention layers to alleviate ambiguities for accurate object
localization. We have compared our method with conventional RGB(-D) based
methods that take rendered RGB images and depths from NeRFs as inputs. Our
method is better than the baselines.

---

## NeRF-SOS: Any-View Self-supervised Object Segmentation on Complex Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-19 | Zhiwen Fan, Peihao Wang, Yifan Jiang, Xinyu Gong, Dejia Xu, Zhangyang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2209.08776v6){: .btn .btn-green } |

**Abstract**: Neural volumetric representations have shown the potential that Multi-layer
Perceptrons (MLPs) can be optimized with multi-view calibrated images to
represent scene geometry and appearance, without explicit 3D supervision.
Object segmentation can enrich many downstream applications based on the
learned radiance field. However, introducing hand-crafted segmentation to
define regions of interest in a complex real-world scene is non-trivial and
expensive as it acquires per view annotation. This paper carries out the
exploration of self-supervised learning for object segmentation using NeRF for
complex real-world scenes. Our framework, called NeRF with Self-supervised
Object Segmentation NeRF-SOS, couples object segmentation and neural radiance
field to segment objects in any view within a scene. By proposing a novel
collaborative contrastive loss in both appearance and geometry levels, NeRF-SOS
encourages NeRF models to distill compact geometry-aware segmentation clusters
from their density fields and the self-supervised pre-trained 2D visual
features. The self-supervised object segmentation framework can be applied to
various NeRF models that both lead to photo-realistic rendering results and
convincing segmentation maps for both indoor and outdoor scenarios. Extensive
results on the LLFF, Tank & Temple, and BlendedMVS datasets validate the
effectiveness of NeRF-SOS. It consistently surpasses other 2D-based
self-supervised baselines and predicts finer semantics masks than existing
supervised counterparts. Please refer to the video on our project page for more
details:https://zhiwenfan.github.io/NeRF-SOS.

---

## Density-aware NeRF Ensembles: Quantifying Predictive Uncertainty in  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-19 | Niko Sünderhauf, Jad Abou-Chakra, Dimity Miller | cs.CV | [PDF](http://arxiv.org/pdf/2209.08718v1){: .btn .btn-green } |

**Abstract**: We show that ensembling effectively quantifies model uncertainty in Neural
Radiance Fields (NeRFs) if a density-aware epistemic uncertainty term is
considered. The naive ensembles investigated in prior work simply average
rendered RGB images to quantify the model uncertainty caused by conflicting
explanations of the observed scene. In contrast, we additionally consider the
termination probabilities along individual rays to identify epistemic model
uncertainty due to a lack of knowledge about the parts of a scene unobserved
during training. We achieve new state-of-the-art performance across established
uncertainty quantification benchmarks for NeRFs, outperforming methods that
require complex changes to the NeRF architecture and training regime. We
furthermore demonstrate that NeRF uncertainty can be utilised for next-best
view selection and model refinement.

---

## Loc-NeRF: Monte Carlo Localization using Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-19 | Dominic Maggio, Marcus Abate, Jingnan Shi, Courtney Mario, Luca Carlone | cs.RO | [PDF](http://arxiv.org/pdf/2209.09050v1){: .btn .btn-green } |

**Abstract**: We present Loc-NeRF, a real-time vision-based robot localization approach
that combines Monte Carlo localization and Neural Radiance Fields (NeRF). Our
system uses a pre-trained NeRF model as the map of an environment and can
localize itself in real-time using an RGB camera as the only exteroceptive
sensor onboard the robot. While neural radiance fields have seen significant
applications for visual rendering in computer vision and graphics, they have
found limited use in robotics. Existing approaches for NeRF-based localization
require both a good initial pose guess and significant computation, making them
impractical for real-time robotics applications. By using Monte Carlo
localization as a workhorse to estimate poses using a NeRF map model, Loc-NeRF
is able to perform localization faster than the state of the art and without
relying on an initial pose estimate. In addition to testing on synthetic data,
we also run our system using real data collected by a Clearpath Jackal UGV and
demonstrate for the first time the ability to perform real-time global
localization with neural radiance fields. We make our code publicly available
at https://github.com/MIT-SPARK/Loc-NeRF.

---

## ActiveNeRF: Learning where to See with Uncertainty Estimation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-18 | Xuran Pan, Zihang Lai, Shiji Song, Gao Huang | cs.CV | [PDF](http://arxiv.org/pdf/2209.08546v1){: .btn .btn-green } |

**Abstract**: Recently, Neural Radiance Fields (NeRF) has shown promising performances on
reconstructing 3D scenes and synthesizing novel views from a sparse set of 2D
images. Albeit effective, the performance of NeRF is highly influenced by the
quality of training samples. With limited posed images from the scene, NeRF
fails to generalize well to novel views and may collapse to trivial solutions
in unobserved regions. This makes NeRF impractical under resource-constrained
scenarios. In this paper, we present a novel learning framework, ActiveNeRF,
aiming to model a 3D scene with a constrained input budget. Specifically, we
first incorporate uncertainty estimation into a NeRF model, which ensures
robustness under few observations and provides an interpretation of how NeRF
understands the scene. On this basis, we propose to supplement the existing
training set with newly captured samples based on an active learning scheme. By
evaluating the reduction of uncertainty given new inputs, we select the samples
that bring the most information gain. In this way, the quality of novel view
synthesis can be improved with minimal additional resources. Extensive
experiments validate the performance of our model on both realistic and
synthetic scenes, especially with scarcer training data. Code will be released
at \url{https://github.com/LeapLabTHU/ActiveNeRF}.

Comments:
- Accepted by ECCV2022

---

## LATITUDE: Robotic Global Localization with Truncated Dynamic Low-pass  Filter in City-scale NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-18 | Zhenxin Zhu, Yuantao Chen, Zirui Wu, Chao Hou, Yongliang Shi, Chuxuan Li, Pengfei Li, Hao Zhao, Guyue Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2209.08498v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have made great success in representing
complex 3D scenes with high-resolution details and efficient memory.
Nevertheless, current NeRF-based pose estimators have no initial pose
prediction and are prone to local optima during optimization. In this paper, we
present LATITUDE: Global Localization with Truncated Dynamic Low-pass Filter,
which introduces a two-stage localization mechanism in city-scale NeRF. In
place recognition stage, we train a regressor through images generated from
trained NeRFs, which provides an initial value for global localization. In pose
optimization stage, we minimize the residual between the observed image and
rendered image by directly optimizing the pose on tangent plane. To avoid
convergence to local optimum, we introduce a Truncated Dynamic Low-pass Filter
(TDLF) for coarse-to-fine pose registration. We evaluate our method on both
synthetic and real-world data and show its potential applications for
high-precision navigation in large-scale city scenes. Codes and data will be
publicly available at https://github.com/jike5/LATITUDE.

Comments:
- 7 pages, 6 figures, ICRA 2023

---

## Uncertainty Guided Policy for Active Robotic 3D Reconstruction using  Neural Radiance Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-17 | Soomin Lee, Le Chen, Jiahao Wang, Alexander Liniger, Suryansh Kumar, Fisher Yu | cs.CV | [PDF](http://arxiv.org/pdf/2209.08409v1){: .btn .btn-green } |

**Abstract**: In this paper, we tackle the problem of active robotic 3D reconstruction of
an object. In particular, we study how a mobile robot with an arm-held camera
can select a favorable number of views to recover an object's 3D shape
efficiently. Contrary to the existing solution to this problem, we leverage the
popular neural radiance fields-based object representation, which has recently
shown impressive results for various computer vision tasks. However, it is not
straightforward to directly reason about an object's explicit 3D geometric
details using such a representation, making the next-best-view selection
problem for dense 3D reconstruction challenging. This paper introduces a
ray-based volumetric uncertainty estimator, which computes the entropy of the
weight distribution of the color samples along each ray of the object's
implicit neural representation. We show that it is possible to infer the
uncertainty of the underlying 3D geometry given a novel view with the proposed
estimator. We then present a next-best-view selection policy guided by the
ray-based volumetric uncertainty in neural radiance fields-based
representations. Encouraging experimental results on synthetic and real-world
data suggest that the approach presented in this paper can enable a new
research direction of using an implicit 3D object representation for the
next-best-view problem in robot vision applications, distinguishing our
approach from the existing approaches that rely on explicit 3D geometric
modeling.

Comments:
- 8 pages, 9 figure; Accepted for publication at IEEE Robotics and
  Automation Letters (RA-L) 2022

---

## iDF-SLAM: End-to-End RGB-D SLAM with Neural Implicit Mapping and Deep  Feature Tracking

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-16 | Yuhang Ming, Weicai Ye, Andrew Calway | cs.RO | [PDF](http://arxiv.org/pdf/2209.07919v1){: .btn .btn-green } |

**Abstract**: We propose a novel end-to-end RGB-D SLAM, iDF-SLAM, which adopts a
feature-based deep neural tracker as the front-end and a NeRF-style neural
implicit mapper as the back-end. The neural implicit mapper is trained
on-the-fly, while though the neural tracker is pretrained on the ScanNet
dataset, it is also finetuned along with the training of the neural implicit
mapper. Under such a design, our iDF-SLAM is capable of learning to use
scene-specific features for camera tracking, thus enabling lifelong learning of
the SLAM system. Both the training for the tracker and the mapper are
self-supervised without introducing ground truth poses. We test the performance
of our iDF-SLAM on the Replica and ScanNet datasets and compare the results to
the two recent NeRF-based neural SLAM systems. The proposed iDF-SLAM
demonstrates state-of-the-art results in terms of scene reconstruction and
competitive performance in camera tracking.

Comments:
- 7 pages, 6 figures, 3 tables

---

## 3DMM-RF: Convolutional Radiance Fields for 3D Face Modeling



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-15 | Stathis Galanakis, Baris Gecer, Alexandros Lattas, Stefanos Zafeiriou | cs.CV | [PDF](http://arxiv.org/pdf/2209.07366v1){: .btn .btn-green } |

**Abstract**: Facial 3D Morphable Models are a main computer vision subject with countless
applications and have been highly optimized in the last two decades. The
tremendous improvements of deep generative networks have created various
possibilities for improving such models and have attracted wide interest.
Moreover, the recent advances in neural radiance fields, are revolutionising
novel-view synthesis of known scenes. In this work, we present a facial 3D
Morphable Model, which exploits both of the above, and can accurately model a
subject's identity, pose and expression and render it in arbitrary
illumination. This is achieved by utilizing a powerful deep style-based
generator to overcome two main weaknesses of neural radiance fields, their
rigidity and rendering speed. We introduce a style-based generative network
that synthesizes in one pass all and only the required rendering samples of a
neural radiance field. We create a vast labelled synthetic dataset of facial
renders, and train the network on these data, so that it can accurately model
and generalize on facial identity, pose and appearance. Finally, we show that
this model can accurately be fit to "in-the-wild" facial images of arbitrary
pose and illumination, extract the facial characteristics, and be used to
re-render the face in controllable conditions.

---

## StructNeRF: Neural Radiance Fields for Indoor Scenes with Structural  Hints

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-12 | Zheng Chen, Chen Wang, Yuan-Chen Guo, Song-Hai Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2209.05277v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) achieve photo-realistic view synthesis with
densely captured input images. However, the geometry of NeRF is extremely
under-constrained given sparse views, resulting in significant degradation of
novel view synthesis quality. Inspired by self-supervised depth estimation
methods, we propose StructNeRF, a solution to novel view synthesis for indoor
scenes with sparse inputs. StructNeRF leverages the structural hints naturally
embedded in multi-view inputs to handle the unconstrained geometry issue in
NeRF. Specifically, it tackles the texture and non-texture regions
respectively: a patch-based multi-view consistent photometric loss is proposed
to constrain the geometry of textured regions; for non-textured ones, we
explicitly restrict them to be 3D consistent planes. Through the dense
self-supervised depth constraints, our method improves both the geometry and
the view synthesis performance of NeRF without any additional training on
external data. Extensive experiments on several real-world datasets demonstrate
that StructNeRF surpasses state-of-the-art methods for indoor scenes with
sparse inputs both quantitatively and qualitatively.

---

## Generative Deformable Radiance Fields for Disentangled Image Synthesis  of Topology-Varying Objects

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-09 | Ziyu Wang, Yu Deng, Jiaolong Yang, Jingyi Yu, Xin Tong | cs.CV | [PDF](http://arxiv.org/pdf/2209.04183v1){: .btn .btn-green } |

**Abstract**: 3D-aware generative models have demonstrated their superb performance to
generate 3D neural radiance fields (NeRF) from a collection of monocular 2D
images even for topology-varying object categories. However, these methods
still lack the capability to separately control the shape and appearance of the
objects in the generated radiance fields. In this paper, we propose a
generative model for synthesizing radiance fields of topology-varying objects
with disentangled shape and appearance variations. Our method generates
deformable radiance fields, which builds the dense correspondence between the
density fields of the objects and encodes their appearances in a shared
template field. Our disentanglement is achieved in an unsupervised manner
without introducing extra labels to previous 3D-aware GAN training. We also
develop an effective image inversion scheme for reconstructing the radiance
field of an object in a real monocular image and manipulating its shape and
appearance. Experiments show that our method can successfully learn the
generative model from unstructured monocular images and well disentangle the
shape and appearance for objects (e.g., chairs) with large topological
variance. The model trained on synthetic data can faithfully reconstruct the
real object in a given single image and achieve high-quality texture and shape
editing results.

Comments:
- Accepted at Pacific Graphics 2022 & COMPUTER GRAPHICS Forum, Project
  Page: https://ziyuwang98.github.io/GDRF/

---

## PixTrack: Precise 6DoF Object Pose Tracking using NeRF Templates and  Feature-metric Alignment

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-08 | Prajwal Chidananda, Saurabh Nair, Douglas Lee, Adrian Kaehler | cs.CV | [PDF](http://arxiv.org/pdf/2209.03910v1){: .btn .btn-green } |

**Abstract**: We present PixTrack, a vision based object pose tracking framework using
novel view synthesis and deep feature-metric alignment. Our evaluations
demonstrate that our method produces highly accurate, robust, and jitter-free
6DoF pose estimates of objects in RGB images without the need of any data
annotation or trajectory smoothing. Our method is also computationally
efficient making it easy to have multi-object tracking with no alteration to
our method and just using CPU multiprocessing.

---

## im2nerf: Image to Neural Radiance Field in the Wild

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-08 | Lu Mi, Abhijit Kundu, David Ross, Frank Dellaert, Noah Snavely, Alireza Fathi | cs.CV | [PDF](http://arxiv.org/pdf/2209.04061v1){: .btn .btn-green } |

**Abstract**: We propose im2nerf, a learning framework that predicts a continuous neural
object representation given a single input image in the wild, supervised by
only segmentation output from off-the-shelf recognition methods. The standard
approach to constructing neural radiance fields takes advantage of multi-view
consistency and requires many calibrated views of a scene, a requirement that
cannot be satisfied when learning on large-scale image data in the wild. We
take a step towards addressing this shortcoming by introducing a model that
encodes the input image into a disentangled object representation that contains
a code for object shape, a code for object appearance, and an estimated camera
pose from which the object image is captured. Our model conditions a NeRF on
the predicted object representation and uses volume rendering to generate
images from novel views. We train the model end-to-end on a large collection of
input images. As the model is only provided with single-view images, the
problem is highly under-constrained. Therefore, in addition to using a
reconstruction loss on the synthesized input view, we use an auxiliary
adversarial loss on the novel rendered views. Furthermore, we leverage object
symmetry and cycle camera pose consistency. We conduct extensive quantitative
and qualitative experiments on the ShapeNet dataset as well as qualitative
experiments on Open Images dataset. We show that in all cases, im2nerf achieves
the state-of-the-art performance for novel view synthesis from a single-view
unposed image in the wild.

Comments:
- 12 pages, 8 figures, 4 tables

---

## Neural Feature Fusion Fields: 3D Distillation of Self-Supervised 2D  Image Representations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-07 | Vadim Tschernezki, Iro Laina, Diane Larlus, Andrea Vedaldi | cs.CV | [PDF](http://arxiv.org/pdf/2209.03494v1){: .btn .btn-green } |

**Abstract**: We present Neural Feature Fusion Fields (N3F), a method that improves dense
2D image feature extractors when the latter are applied to the analysis of
multiple images reconstructible as a 3D scene. Given an image feature
extractor, for example pre-trained using self-supervision, N3F uses it as a
teacher to learn a student network defined in 3D space. The 3D student network
is similar to a neural radiance field that distills said features and can be
trained with the usual differentiable rendering machinery. As a consequence,
N3F is readily applicable to most neural rendering formulations, including
vanilla NeRF and its extensions to complex dynamic scenes. We show that our
method not only enables semantic understanding in the context of scene-specific
neural fields without the use of manual labels, but also consistently improves
over the self-supervised 2D baselines. This is demonstrated by considering
various tasks, such as 2D object retrieval, 3D segmentation, and scene editing,
in diverse sequences, including long egocentric videos in the EPIC-KITCHENS
benchmark.

Comments:
- 3DV2022, Oral. Project page: https://www.robots.ox.ac.uk/~vadim/n3f/

---

## CLONeR: Camera-Lidar Fusion for Occupancy Grid-aided Neural  Representations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-02 | Alexandra Carlson, Manikandasriram Srinivasan Ramanagopal, Nathan Tseng, Matthew Johnson-Roberson, Ram Vasudevan, Katherine A. Skinner | cs.CV | [PDF](http://arxiv.org/pdf/2209.01194v4){: .btn .btn-green } |

**Abstract**: Recent advances in neural radiance fields (NeRFs) achieve state-of-the-art
novel view synthesis and facilitate dense estimation of scene properties.
However, NeRFs often fail for large, unbounded scenes that are captured under
very sparse views with the scene content concentrated far away from the camera,
as is typical for field robotics applications. In particular, NeRF-style
algorithms perform poorly: (1) when there are insufficient views with little
pose diversity, (2) when scenes contain saturation and shadows, and (3) when
finely sampling large unbounded scenes with fine structures becomes
computationally intensive.
  This paper proposes CLONeR, which significantly improves upon NeRF by
allowing it to model large outdoor driving scenes that are observed from sparse
input sensor views. This is achieved by decoupling occupancy and color learning
within the NeRF framework into separate Multi-Layer Perceptrons (MLPs) trained
using LiDAR and camera data, respectively. In addition, this paper proposes a
novel method to build differentiable 3D Occupancy Grid Maps (OGM) alongside the
NeRF model, and leverage this occupancy grid for improved sampling of points
along a ray for volumetric rendering in metric space.
  Through extensive quantitative and qualitative experiments on scenes from the
KITTI dataset, this paper demonstrates that the proposed method outperforms
state-of-the-art NeRF models on both novel view synthesis and dense depth
prediction tasks when trained on sparse input data.

Comments:
- first two authors equally contributed

---

## Cross-Spectral Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-01 | Matteo Poggi, Pierluigi Zama Ramirez, Fabio Tosi, Samuele Salti, Stefano Mattoccia, Luigi Di Stefano | cs.CV | [PDF](http://arxiv.org/pdf/2209.00648v1){: .btn .btn-green } |

**Abstract**: We propose X-NeRF, a novel method to learn a Cross-Spectral scene
representation given images captured from cameras with different light spectrum
sensitivity, based on the Neural Radiance Fields formulation. X-NeRF optimizes
camera poses across spectra during training and exploits Normalized
Cross-Device Coordinates (NXDC) to render images of different modalities from
arbitrary viewpoints, which are aligned and at the same resolution. Experiments
on 16 forward-facing scenes, featuring color, multi-spectral and infrared
images, confirm the effectiveness of X-NeRF at modeling Cross-Spectral scene
representations.

Comments:
- 3DV 2022. Project page: https://cvlab-unibo.github.io/xnerf-web/

---

## On Quantizing Implicit Neural Representations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2022-09-01 | Cameron Gordon, Shin-Fang Chng, Lachlan MacDonald, Simon Lucey | cs.CV | [PDF](http://arxiv.org/pdf/2209.01019v1){: .btn .btn-green } |

**Abstract**: The role of quantization within implicit/coordinate neural networks is still
not fully understood. We note that using a canonical fixed quantization scheme
during training produces poor performance at low-rates due to the network
weight distributions changing over the course of training. In this work, we
show that a non-uniform quantization of neural weights can lead to significant
improvements. Specifically, we demonstrate that a clustered quantization
enables improved reconstruction. Finally, by characterising a trade-off between
quantization and network capacity, we demonstrate that it is possible (while
memory inefficient) to reconstruct signals using binary neural networks. We
demonstrate our findings experimentally on 2D image reconstruction and 3D
radiance fields; and show that simple quantization methods and architecture
search can achieve compression of NeRF to less than 16kb with minimal loss in
performance (323x smaller than the original NeRF).

Comments:
- 10 pages, 10 figures