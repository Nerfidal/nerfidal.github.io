---
layout: default
title: August
parent: 2023
nav_order: 8
---
<!---metadata--->

## GHuNeRF: Generalizable Human NeRF from a Monocular Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-31 | Chen Li, Jiahao Lin, Gim Hee Lee | cs.CV | [PDF](http://arxiv.org/pdf/2308.16576v3){: .btn .btn-green } |

**Abstract**: In this paper, we tackle the challenging task of learning a generalizable
human NeRF model from a monocular video. Although existing generalizable human
NeRFs have achieved impressive results, they require muti-view images or videos
which might not be always available. On the other hand, some works on
free-viewpoint rendering of human from monocular videos cannot be generalized
to unseen identities. In view of these limitations, we propose GHuNeRF to learn
a generalizable human NeRF model from a monocular video of the human performer.
We first introduce a visibility-aware aggregation scheme to compute vertex-wise
features, which is used to construct a 3D feature volume. The feature volume
can only represent the overall geometry of the human performer with
insufficient accuracy due to the limited resolution. To solve this, we further
enhance the volume feature with temporally aligned point-wise features using an
attention mechanism. Finally, the enhanced feature is used for predicting
density and color for each sampled point. A surface-guided sampling strategy is
also adopted to improve the efficiency for both training and inference. We
validate our approach on the widely-used ZJU-MoCap dataset, where we achieve
comparable performance with existing multi-view video based approaches. We also
test on the monocular People-Snapshot dataset and achieve better performance
than existing works when only monocular video is used. Our code is available at
the project website.

Comments:
- Add in more baseline for comparison

---

## From Pixels to Portraits: A Comprehensive Survey of Talking Head  Generation Techniques and Applications

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-30 | Shreyank N Gowda, Dheeraj Pandey, Shashank Narayana Gowda | cs.CV | [PDF](http://arxiv.org/pdf/2308.16041v1){: .btn .btn-green } |

**Abstract**: Recent advancements in deep learning and computer vision have led to a surge
of interest in generating realistic talking heads. This paper presents a
comprehensive survey of state-of-the-art methods for talking head generation.
We systematically categorises them into four main approaches: image-driven,
audio-driven, video-driven and others (including neural radiance fields (NeRF),
and 3D-based methods). We provide an in-depth analysis of each method,
highlighting their unique contributions, strengths, and limitations.
Furthermore, we thoroughly compare publicly available models, evaluating them
on key aspects such as inference time and human-rated quality of the generated
outputs. Our aim is to provide a clear and concise overview of the current
landscape in talking head generation, elucidating the relationships between
different approaches and identifying promising directions for future research.
This survey will serve as a valuable reference for researchers and
practitioners interested in this rapidly evolving field.

---

## Drone-NeRF: Efficient NeRF Based 3D Scene Reconstruction for Large-Scale  Drone Survey

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-30 | Zhihao Jia, Bing Wang, Changhao Chen | cs.CV | [PDF](http://arxiv.org/pdf/2308.15733v1){: .btn .btn-green } |

**Abstract**: Neural rendering has garnered substantial attention owing to its capacity for
creating realistic 3D scenes. However, its applicability to extensive scenes
remains challenging, with limitations in effectiveness. In this work, we
propose the Drone-NeRF framework to enhance the efficient reconstruction of
unbounded large-scale scenes suited for drone oblique photography using Neural
Radiance Fields (NeRF). Our approach involves dividing the scene into uniform
sub-blocks based on camera position and depth visibility. Sub-scenes are
trained in parallel using NeRF, then merged for a complete scene. We refine the
model by optimizing camera poses and guiding NeRF with a uniform sampler.
Integrating chosen samples enhances accuracy. A hash-coded fusion MLP
accelerates density representation, yielding RGB and Depth outputs. Our
framework accounts for sub-scene constraints, reduces parallel-training noise,
handles shadow occlusion, and merges sub-regions for a polished rendering
result. This Drone-NeRF framework demonstrates promising capabilities in
addressing challenges related to scene complexity, rendering efficiency, and
accuracy in drone-obtained imagery.

Comments:
- 15 pages, 7 figures, in submission

---

## Efficient Ray Sampling for Radiance Fields Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-29 | Shilei Sun, Ming Liu, Zhongyi Fan, Yuxue Liu, Chengwei Lv, Liquan Dong, Lingqin Kong | cs.CV | [PDF](http://arxiv.org/pdf/2308.15547v1){: .btn .btn-green } |

**Abstract**: Accelerating neural radiance fields training is of substantial practical
value, as the ray sampling strategy profoundly impacts network convergence.
More efficient ray sampling can thus directly enhance existing NeRF models'
training efficiency. We therefore propose a novel ray sampling approach for
neural radiance fields that improves training efficiency while retaining
photorealistic rendering results. First, we analyze the relationship between
the pixel loss distribution of sampled rays and rendering quality. This reveals
redundancy in the original NeRF's uniform ray sampling. Guided by this finding,
we develop a sampling method leveraging pixel regions and depth boundaries. Our
main idea is to sample fewer rays in training views, yet with each ray more
informative for scene fitting. Sampling probability increases in pixel areas
exhibiting significant color and depth variation, greatly reducing wasteful
rays from other regions without sacrificing precision. Through this method, not
only can the convergence of the network be accelerated, but the spatial
geometry of a scene can also be perceived more accurately. Rendering outputs
are enhanced, especially for texture-complex regions. Experiments demonstrate
that our method significantly outperforms state-of-the-art techniques on public
benchmark datasets.

Comments:
- 15 pages

---

## Pose-Free Neural Radiance Fields via Implicit Pose Regularization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-29 | Jiahui Zhang, Fangneng Zhan, Yingchen Yu, Kunhao Liu, Rongliang Wu, Xiaoqin Zhang, Ling Shao, Shijian Lu | cs.CV | [PDF](http://arxiv.org/pdf/2308.15049v1){: .btn .btn-green } |

**Abstract**: Pose-free neural radiance fields (NeRF) aim to train NeRF with unposed
multi-view images and it has achieved very impressive success in recent years.
Most existing works share the pipeline of training a coarse pose estimator with
rendered images at first, followed by a joint optimization of estimated poses
and neural radiance field. However, as the pose estimator is trained with only
rendered images, the pose estimation is usually biased or inaccurate for real
images due to the domain gap between real images and rendered images, leading
to poor robustness for the pose estimation of real images and further local
minima in joint optimization. We design IR-NeRF, an innovative pose-free NeRF
that introduces implicit pose regularization to refine pose estimator with
unposed real images and improve the robustness of the pose estimation for real
images. With a collection of 2D images of a specific scene, IR-NeRF constructs
a scene codebook that stores scene features and captures the scene-specific
pose distribution implicitly as priors. Thus, the robustness of pose estimation
can be promoted with the scene priors according to the rationale that a 2D real
image can be well reconstructed from the scene codebook only when its estimated
pose lies within the pose distribution. Extensive experiments show that IR-NeRF
achieves superior novel view synthesis and outperforms the state-of-the-art
consistently across multiple synthetic and real datasets.

Comments:
- Accepted by ICCV2023

---

## Multi-Modal Neural Radiance Field for Monocular Dense SLAM with a  Light-Weight ToF Sensor



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-28 | Xinyang Liu, Yijin Li, Yanbin Teng, Hujun Bao, Guofeng Zhang, Yinda Zhang, Zhaopeng Cui | cs.CV | [PDF](http://arxiv.org/pdf/2308.14383v1){: .btn .btn-green } |

**Abstract**: Light-weight time-of-flight (ToF) depth sensors are compact and
cost-efficient, and thus widely used on mobile devices for tasks such as
autofocus and obstacle detection. However, due to the sparse and noisy depth
measurements, these sensors have rarely been considered for dense geometry
reconstruction. In this work, we present the first dense SLAM system with a
monocular camera and a light-weight ToF sensor. Specifically, we propose a
multi-modal implicit scene representation that supports rendering both the
signals from the RGB camera and light-weight ToF sensor which drives the
optimization by comparing with the raw sensor inputs. Moreover, in order to
guarantee successful pose tracking and reconstruction, we exploit a predicted
depth as an intermediate supervision and develop a coarse-to-fine optimization
strategy for efficient learning of the implicit representation. At last, the
temporal information is explicitly exploited to deal with the noisy signals
from light-weight ToF sensors to improve the accuracy and robustness of the
system. Experiments demonstrate that our system well exploits the signals of
light-weight ToF sensors and achieves competitive results both on camera
tracking and dense scene reconstruction. Project page:
\url{https://zju3dv.github.io/tof_slam/}.

Comments:
- Accepted to ICCV 2023 (Oral). Project Page:
  https://zju3dv.github.io/tof_slam/

---

## Flexible Techniques for Differentiable Rendering with 3D Gaussians



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-28 | Leonid Keselman, Martial Hebert | cs.CV | [PDF](http://arxiv.org/pdf/2308.14737v1){: .btn .btn-green } |

**Abstract**: Fast, reliable shape reconstruction is an essential ingredient in many
computer vision applications. Neural Radiance Fields demonstrated that
photorealistic novel view synthesis is within reach, but was gated by
performance requirements for fast reconstruction of real scenes and objects.
Several recent approaches have built on alternative shape representations, in
particular, 3D Gaussians. We develop extensions to these renderers, such as
integrating differentiable optical flow, exporting watertight meshes and
rendering per-ray normals. Additionally, we show how two of the recent methods
are interoperable with each other. These reconstructions are quick, robust, and
easily performed on GPU or CPU. For code and visual examples, see
https://leonidk.github.io/fmb-plus

---

## CLNeRF: Continual Learning Meets NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-28 | Zhipeng Cai, Matthias Mueller | cs.CV | [PDF](http://arxiv.org/pdf/2308.14816v1){: .btn .btn-green } |

**Abstract**: Novel view synthesis aims to render unseen views given a set of calibrated
images. In practical applications, the coverage, appearance or geometry of the
scene may change over time, with new images continuously being captured.
Efficiently incorporating such continuous change is an open challenge. Standard
NeRF benchmarks only involve scene coverage expansion. To study other practical
scene changes, we propose a new dataset, World Across Time (WAT), consisting of
scenes that change in appearance and geometry over time. We also propose a
simple yet effective method, CLNeRF, which introduces continual learning (CL)
to Neural Radiance Fields (NeRFs). CLNeRF combines generative replay and the
Instant Neural Graphics Primitives (NGP) architecture to effectively prevent
catastrophic forgetting and efficiently update the model when new data arrives.
We also add trainable appearance and geometry embeddings to NGP, allowing a
single compact model to handle complex scene changes. Without the need to store
historical images, CLNeRF trained sequentially over multiple scans of a
changing scene performs on-par with the upper bound model trained on all scans
at once. Compared to other CL baselines CLNeRF performs much better across
standard benchmarks and WAT. The source code, and the WAT dataset are available
at https://github.com/IntelLabs/CLNeRF. Video presentation is available at:
https://youtu.be/nLRt6OoDGq0?si=8yD6k-8MMBJInQPs

Comments:
- Accepted to ICCV 2023

---

## Sparse3D: Distilling Multiview-Consistent Diffusion for Object  Reconstruction from Sparse Views



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-27 | Zi-Xin Zou, Weihao Cheng, Yan-Pei Cao, Shi-Sheng Huang, Ying Shan, Song-Hai Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2308.14078v2){: .btn .btn-green } |

**Abstract**: Reconstructing 3D objects from extremely sparse views is a long-standing and
challenging problem. While recent techniques employ image diffusion models for
generating plausible images at novel viewpoints or for distilling pre-trained
diffusion priors into 3D representations using score distillation sampling
(SDS), these methods often struggle to simultaneously achieve high-quality,
consistent, and detailed results for both novel-view synthesis (NVS) and
geometry. In this work, we present Sparse3D, a novel 3D reconstruction method
tailored for sparse view inputs. Our approach distills robust priors from a
multiview-consistent diffusion model to refine a neural radiance field.
Specifically, we employ a controller that harnesses epipolar features from
input views, guiding a pre-trained diffusion model, such as Stable Diffusion,
to produce novel-view images that maintain 3D consistency with the input. By
tapping into 2D priors from powerful image diffusion models, our integrated
model consistently delivers high-quality results, even when faced with
open-world objects. To address the blurriness introduced by conventional SDS,
we introduce the category-score distillation sampling (C-SDS) to enhance
detail. We conduct experiments on CO3DV2 which is a multi-view dataset of
real-world objects. Both quantitative and qualitative evaluations demonstrate
that our approach outperforms previous state-of-the-art works on the metrics
regarding NVS and geometry reconstruction.

---

## Unaligned 2D to 3D Translation with Conditional Vector-Quantized Code  Diffusion using Transformers



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-27 | Abril Corona-Figueroa, Sam Bond-Taylor, Neelanjan Bhowmik, Yona Falinie A. Gaus, Toby P. Breckon, Hubert P. H. Shum, Chris G. Willcocks | cs.CV | [PDF](http://arxiv.org/pdf/2308.14152v1){: .btn .btn-green } |

**Abstract**: Generating 3D images of complex objects conditionally from a few 2D views is
a difficult synthesis problem, compounded by issues such as domain gap and
geometric misalignment. For instance, a unified framework such as Generative
Adversarial Networks cannot achieve this unless they explicitly define both a
domain-invariant and geometric-invariant joint latent distribution, whereas
Neural Radiance Fields are generally unable to handle both issues as they
optimize at the pixel level. By contrast, we propose a simple and novel 2D to
3D synthesis approach based on conditional diffusion with vector-quantized
codes. Operating in an information-rich code space enables high-resolution 3D
synthesis via full-coverage attention across the views. Specifically, we
generate the 3D codes (e.g. for CT images) conditional on previously generated
3D codes and the entire codebook of two 2D views (e.g. 2D X-rays). Qualitative
and quantitative results demonstrate state-of-the-art performance over
specialized methods across varied evaluation criteria, including fidelity
metrics such as density, coverage, and distortion metrics for two complex
volumetric imagery datasets from in real-world scenarios.

Comments:
- Camera-ready version for ICCV 2023

---

## InsertNeRF: Instilling Generalizability into NeRF with HyperNet Modules

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-26 | Yanqi Bao, Tianyu Ding, Jing Huo, Wenbin Li, Yuxin Li, Yang Gao | cs.CV | [PDF](http://arxiv.org/pdf/2308.13897v1){: .btn .btn-green } |

**Abstract**: Generalizing Neural Radiance Fields (NeRF) to new scenes is a significant
challenge that existing approaches struggle to address without extensive
modifications to vanilla NeRF framework. We introduce InsertNeRF, a method for
INStilling gEneRalizabiliTy into NeRF. By utilizing multiple plug-and-play
HyperNet modules, InsertNeRF dynamically tailors NeRF's weights to specific
reference scenes, transforming multi-scale sampling-aware features into
scene-specific representations. This novel design allows for more accurate and
efficient representations of complex appearances and geometries. Experiments
show that this method not only achieves superior generalization performance but
also provides a flexible pathway for integration with other NeRF-like systems,
even in sparse input settings. Code will be available
https://github.com/bbbbby-99/InsertNeRF.

---

## Relighting Neural Radiance Fields with Shadow and Highlight Hints



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-25 | Chong Zeng, Guojun Chen, Yue Dong, Pieter Peers, Hongzhi Wu, Xin Tong | cs.CV | [PDF](http://arxiv.org/pdf/2308.13404v1){: .btn .btn-green } |

**Abstract**: This paper presents a novel neural implicit radiance representation for free
viewpoint relighting from a small set of unstructured photographs of an object
lit by a moving point light source different from the view position. We express
the shape as a signed distance function modeled by a multi layer perceptron. In
contrast to prior relightable implicit neural representations, we do not
disentangle the different reflectance components, but model both the local and
global reflectance at each point by a second multi layer perceptron that, in
addition, to density features, the current position, the normal (from the
signed distace function), view direction, and light position, also takes shadow
and highlight hints to aid the network in modeling the corresponding high
frequency light transport effects. These hints are provided as a suggestion,
and we leave it up to the network to decide how to incorporate these in the
final relit result. We demonstrate and validate our neural implicit
representation on synthetic and real scenes exhibiting a wide variety of
shapes, material properties, and global illumination light transport.

Comments:
- Accepted to SIGGRAPH 2023. Author's version. Project page:
  https://nrhints.github.io/

---

## Improving NeRF Quality by Progressive Camera Placement for Unrestricted  Navigation in Complex Environments

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-24 | Georgios Kopanas, George Drettakis | cs.CV | [PDF](http://arxiv.org/pdf/2309.00014v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields, or NeRFs, have drastically improved novel view
synthesis and 3D reconstruction for rendering. NeRFs achieve impressive results
on object-centric reconstructions, but the quality of novel view synthesis with
free-viewpoint navigation in complex environments (rooms, houses, etc) is often
problematic. While algorithmic improvements play an important role in the
resulting quality of novel view synthesis, in this work, we show that because
optimizing a NeRF is inherently a data-driven process, good quality data play a
fundamental role in the final quality of the reconstruction. As a consequence,
it is critical to choose the data samples -- in this case the cameras -- in a
way that will eventually allow the optimization to converge to a solution that
allows free-viewpoint navigation with good quality. Our main contribution is an
algorithm that efficiently proposes new camera placements that improve visual
quality with minimal assumptions. Our solution can be used with any NeRF model
and outperforms baselines and similar work.

---

## NOVA: NOvel View Augmentation for Neural Composition of Dynamic Objects

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-24 | Dakshit Agrawal, Jiajie Xu, Siva Karthik Mustikovela, Ioannis Gkioulekas, Ashish Shrivastava, Yuning Chai | cs.CV | [PDF](http://arxiv.org/pdf/2308.12560v1){: .btn .btn-green } |

**Abstract**: We propose a novel-view augmentation (NOVA) strategy to train NeRFs for
photo-realistic 3D composition of dynamic objects in a static scene. Compared
to prior work, our framework significantly reduces blending artifacts when
inserting multiple dynamic objects into a 3D scene at novel views and times;
achieves comparable PSNR without the need for additional ground truth
modalities like optical flow; and overall provides ease, flexibility, and
scalability in neural composition. Our codebase is on GitHub.

Comments:
- Accepted for publication in ICCV Computer Vision for Metaverse
  Workshop 2023 (code is available at https://github.com/dakshitagrawal/NoVA)

---

## ARF-Plus: Controlling Perceptual Factors in Artistic Radiance Fields for  3D Scene Stylization



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-23 | Wenzhao Li, Tianhao Wu, Fangcheng Zhong, Cengiz Oztireli | cs.CV | [PDF](http://arxiv.org/pdf/2308.12452v2){: .btn .btn-green } |

**Abstract**: The radiance fields style transfer is an emerging field that has recently
gained popularity as a means of 3D scene stylization, thanks to the outstanding
performance of neural radiance fields in 3D reconstruction and view synthesis.
We highlight a research gap in radiance fields style transfer, the lack of
sufficient perceptual controllability, motivated by the existing concept in the
2D image style transfer. In this paper, we present ARF-Plus, a 3D neural style
transfer framework offering manageable control over perceptual factors, to
systematically explore the perceptual controllability in 3D scene stylization.
Four distinct types of controls - color preservation control, (style pattern)
scale control, spatial (selective stylization area) control, and depth
enhancement control - are proposed and integrated into this framework. Results
from real-world datasets, both quantitative and qualitative, show that the four
types of controls in our ARF-Plus framework successfully accomplish their
corresponding perceptual controls when stylizing 3D scenes. These techniques
work well for individual style inputs as well as for the simultaneous
application of multiple styles within a scene. This unlocks a realm of
limitless possibilities, allowing customized modifications of stylization
effects and flexible merging of the strengths of different styles, ultimately
enabling the creation of novel and eye-catching stylistic effects on 3D scenes.

---

## Pose Modulated Avatars from Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-23 | Chunjin Song, Bastian Wandt, Helge Rhodin | cs.CV | [PDF](http://arxiv.org/pdf/2308.11951v3){: .btn .btn-green } |

**Abstract**: It is now possible to reconstruct dynamic human motion and shape from a
sparse set of cameras using Neural Radiance Fields (NeRF) driven by an
underlying skeleton. However, a challenge remains to model the deformation of
cloth and skin in relation to skeleton pose. Unlike existing avatar models that
are learned implicitly or rely on a proxy surface, our approach is motivated by
the observation that different poses necessitate unique frequency assignments.
Neglecting this distinction yields noisy artifacts in smooth areas or blurs
fine-grained texture and shape details in sharp regions. We develop a
two-branch neural network that is adaptive and explicit in the frequency
domain. The first branch is a graph neural network that models correlations
among body parts locally, taking skeleton pose as input. The second branch
combines these correlation features to a set of global frequencies and then
modulates the feature encoding. Our experiments demonstrate that our network
outperforms state-of-the-art methods in terms of preserving details and
generalization capabilities.

---

## Blending-NeRF: Text-Driven Localized Editing in Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-23 | Hyeonseop Song, Seokhun Choi, Hoseok Do, Chul Lee, Taehyeong Kim | cs.CV | [PDF](http://arxiv.org/pdf/2308.11974v2){: .btn .btn-green } |

**Abstract**: Text-driven localized editing of 3D objects is particularly difficult as
locally mixing the original 3D object with the intended new object and style
effects without distorting the object's form is not a straightforward process.
To address this issue, we propose a novel NeRF-based model, Blending-NeRF,
which consists of two NeRF networks: pretrained NeRF and editable NeRF.
Additionally, we introduce new blending operations that allow Blending-NeRF to
properly edit target regions which are localized by text. By using a pretrained
vision-language aligned model, CLIP, we guide Blending-NeRF to add new objects
with varying colors and densities, modify textures, and remove parts of the
original object. Our extensive experiments demonstrate that Blending-NeRF
produces naturally and locally edited 3D objects from various text prompts. Our
project page is available at https://seokhunchoi.github.io/Blending-NeRF/

Comments:
- Accepted to ICCV 2023. The first two authors contributed equally to
  this work

---

## Enhancing NeRF akin to Enhancing LLMs: Generalizable NeRF Transformer  with Mixture-of-View-Experts

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-22 | Wenyan Cong, Hanxue Liang, Peihao Wang, Zhiwen Fan, Tianlong Chen, Mukund Varma, Yi Wang, Zhangyang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2308.11793v1){: .btn .btn-green } |

**Abstract**: Cross-scene generalizable NeRF models, which can directly synthesize novel
views of unseen scenes, have become a new spotlight of the NeRF field. Several
existing attempts rely on increasingly end-to-end "neuralized" architectures,
i.e., replacing scene representation and/or rendering modules with performant
neural networks such as transformers, and turning novel view synthesis into a
feed-forward inference pipeline. While those feedforward "neuralized"
architectures still do not fit diverse scenes well out of the box, we propose
to bridge them with the powerful Mixture-of-Experts (MoE) idea from large
language models (LLMs), which has demonstrated superior generalization ability
by balancing between larger overall model capacity and flexible per-instance
specialization. Starting from a recent generalizable NeRF architecture called
GNT, we first demonstrate that MoE can be neatly plugged in to enhance the
model. We further customize a shared permanent expert and a geometry-aware
consistency loss to enforce cross-scene consistency and spatial smoothness
respectively, which are essential for generalizable view synthesis. Our
proposed model, dubbed GNT with Mixture-of-View-Experts (GNT-MOVE), has
experimentally shown state-of-the-art results when transferring to unseen
scenes, indicating remarkably better cross-scene generalization in both
zero-shot and few-shot settings. Our codes are available at
https://github.com/VITA-Group/GNT-MOVE.

Comments:
- Accepted by ICCV2023

---

## SAMSNeRF: Segment Anything Model (SAM) Guides Dynamic Surgical Scene  Reconstruction by Neural Radiance Field (NeRF)

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-22 | Ange Lou, Yamin Li, Xing Yao, Yike Zhang, Jack Noble | cs.CV | [PDF](http://arxiv.org/pdf/2308.11774v1){: .btn .btn-green } |

**Abstract**: The accurate reconstruction of surgical scenes from surgical videos is
critical for various applications, including intraoperative navigation and
image-guided robotic surgery automation. However, previous approaches, mainly
relying on depth estimation, have limited effectiveness in reconstructing
surgical scenes with moving surgical tools. To address this limitation and
provide accurate 3D position prediction for surgical tools in all frames, we
propose a novel approach called SAMSNeRF that combines Segment Anything Model
(SAM) and Neural Radiance Field (NeRF) techniques. Our approach generates
accurate segmentation masks of surgical tools using SAM, which guides the
refinement of the dynamic surgical scene reconstruction by NeRF. Our
experimental results on public endoscopy surgical videos demonstrate that our
approach successfully reconstructs high-fidelity dynamic surgical scenes and
accurately reflects the spatial information of surgical tools. Our proposed
approach can significantly enhance surgical navigation and automation by
providing surgeons with accurate 3D position information of surgical tools
during surgery.The source code will be released soon.

---

## Novel-view Synthesis and Pose Estimation for Hand-Object Interaction  from Sparse Views

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-22 | Wentian Qu, Zhaopeng Cui, Yinda Zhang, Chenyu Meng, Cuixia Ma, Xiaoming Deng, Hongan Wang | cs.CV | [PDF](http://arxiv.org/pdf/2308.11198v1){: .btn .btn-green } |

**Abstract**: Hand-object interaction understanding and the barely addressed novel view
synthesis are highly desired in the immersive communication, whereas it is
challenging due to the high deformation of hand and heavy occlusions between
hand and object. In this paper, we propose a neural rendering and pose
estimation system for hand-object interaction from sparse views, which can also
enable 3D hand-object interaction editing. We share the inspiration from recent
scene understanding work that shows a scene specific model built beforehand can
significantly improve and unblock vision tasks especially when inputs are
sparse, and extend it to the dynamic hand-object interaction scenario and
propose to solve the problem in two stages. We first learn the shape and
appearance prior knowledge of hands and objects separately with the neural
representation at the offline stage. During the online stage, we design a
rendering-based joint model fitting framework to understand the dynamic
hand-object interaction with the pre-built hand and object models as well as
interaction priors, which thereby overcomes penetration and separation issues
between hand and object and also enables novel view synthesis. In order to get
stable contact during the hand-object interaction process in a sequence, we
propose a stable contact loss to make the contact region to be consistent.
Experiments demonstrate that our method outperforms the state-of-the-art
methods. Code and dataset are available in project webpage
https://iscas3dv.github.io/HO-NeRF.

---

## Efficient View Synthesis with Neural Radiance Distribution Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-22 | Yushuang Wu, Xiao Li, Jinglu Wang, Xiaoguang Han, Shuguang Cui, Yan Lu | cs.CV | [PDF](http://arxiv.org/pdf/2308.11130v1){: .btn .btn-green } |

**Abstract**: Recent work on Neural Radiance Fields (NeRF) has demonstrated significant
advances in high-quality view synthesis. A major limitation of NeRF is its low
rendering efficiency due to the need for multiple network forwardings to render
a single pixel. Existing methods to improve NeRF either reduce the number of
required samples or optimize the implementation to accelerate the network
forwarding. Despite these efforts, the problem of multiple sampling persists
due to the intrinsic representation of radiance fields. In contrast, Neural
Light Fields (NeLF) reduce the computation cost of NeRF by querying only one
single network forwarding per pixel. To achieve a close visual quality to NeRF,
existing NeLF methods require significantly larger network capacities which
limits their rendering efficiency in practice. In this work, we propose a new
representation called Neural Radiance Distribution Field (NeRDF) that targets
efficient view synthesis in real-time. Specifically, we use a small network
similar to NeRF while preserving the rendering speed with a single network
forwarding per pixel as in NeLF. The key is to model the radiance distribution
along each ray with frequency basis and predict frequency weights using the
network. Pixel values are then computed via volume rendering on radiance
distributions. Experiments show that our proposed method offers a better
trade-off among speed, quality, and network size than existing methods: we
achieve a ~254x speed-up over NeRF with similar network size, with only a
marginal performance decline. Our project page is at
yushuang-wu.github.io/NeRDF.

Comments:
- Accepted by ICCV2023

---

## CamP: Camera Preconditioning for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-21 | Keunhong Park, Philipp Henzler, Ben Mildenhall, Jonathan T. Barron, Ricardo Martin-Brualla | cs.CV | [PDF](http://arxiv.org/pdf/2308.10902v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) can be optimized to obtain high-fidelity 3D
scene reconstructions of objects and large-scale scenes. However, NeRFs require
accurate camera parameters as input -- inaccurate camera parameters result in
blurry renderings. Extrinsic and intrinsic camera parameters are usually
estimated using Structure-from-Motion (SfM) methods as a pre-processing step to
NeRF, but these techniques rarely yield perfect estimates. Thus, prior works
have proposed jointly optimizing camera parameters alongside a NeRF, but these
methods are prone to local minima in challenging settings. In this work, we
analyze how different camera parameterizations affect this joint optimization
problem, and observe that standard parameterizations exhibit large differences
in magnitude with respect to small perturbations, which can lead to an
ill-conditioned optimization problem. We propose using a proxy problem to
compute a whitening transform that eliminates the correlation between camera
parameters and normalizes their effects, and we propose to use this transform
as a preconditioner for the camera parameters during joint optimization. Our
preconditioned camera optimization significantly improves reconstruction
quality on scenes from the Mip-NeRF 360 dataset: we reduce error rates (RMSE)
by 67% compared to state-of-the-art NeRF approaches that do not optimize for
cameras like Zip-NeRF, and by 29% relative to state-of-the-art joint
optimization approaches using the camera parameterization of SCNeRF. Our
approach is easy to implement, does not significantly increase runtime, can be
applied to a wide variety of camera parameterizations, and can
straightforwardly be incorporated into other NeRF-like models.

Comments:
- SIGGRAPH Asia 2023, Project page: https://camp-nerf.github.io

---

## Strata-NeRF : Neural Radiance Fields for Stratified Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-20 | Ankit Dhiman, Srinath R, Harsh Rangwani, Rishubh Parihar, Lokesh R Boregowda, Srinath Sridhar, R Venkatesh Babu | cs.CV | [PDF](http://arxiv.org/pdf/2308.10337v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) approaches learn the underlying 3D
representation of a scene and generate photo-realistic novel views with high
fidelity. However, most proposed settings concentrate on modelling a single
object or a single level of a scene. However, in the real world, we may capture
a scene at multiple levels, resulting in a layered capture. For example,
tourists usually capture a monument's exterior structure before capturing the
inner structure. Modelling such scenes in 3D with seamless switching between
levels can drastically improve immersive experiences. However, most existing
techniques struggle in modelling such scenes. We propose Strata-NeRF, a single
neural radiance field that implicitly captures a scene with multiple levels.
Strata-NeRF achieves this by conditioning the NeRFs on Vector Quantized (VQ)
latent representations which allow sudden changes in scene structure. We
evaluate the effectiveness of our approach in multi-layered synthetic dataset
comprising diverse scenes and then further validate its generalization on the
real-world RealEstate10K dataset. We find that Strata-NeRF effectively captures
stratified scenes, minimizes artifacts, and synthesizes high-fidelity views
compared to existing approaches.

Comments:
- ICCV 2023, Project Page: https://ankitatiisc.github.io/Strata-NeRF/

---

## AltNeRF: Learning Robust Neural Radiance Field via Alternating  Depth-Pose Optimization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-19 | Kun Wang, Zhiqiang Yan, Huang Tian, Zhenyu Zhang, Xiang Li, Jun Li, Jian Yang | cs.CV | [PDF](http://arxiv.org/pdf/2308.10001v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have shown promise in generating realistic
novel views from sparse scene images. However, existing NeRF approaches often
encounter challenges due to the lack of explicit 3D supervision and imprecise
camera poses, resulting in suboptimal outcomes. To tackle these issues, we
propose AltNeRF -- a novel framework designed to create resilient NeRF
representations using self-supervised monocular depth estimation (SMDE) from
monocular videos, without relying on known camera poses. SMDE in AltNeRF
masterfully learns depth and pose priors to regulate NeRF training. The depth
prior enriches NeRF's capacity for precise scene geometry depiction, while the
pose prior provides a robust starting point for subsequent pose refinement.
Moreover, we introduce an alternating algorithm that harmoniously melds NeRF
outputs into SMDE through a consistence-driven mechanism, thus enhancing the
integrity of depth priors. This alternation empowers AltNeRF to progressively
refine NeRF representations, yielding the synthesis of realistic novel views.
Additionally, we curate a distinctive dataset comprising indoor videos captured
via mobile devices. Extensive experiments showcase the compelling capabilities
of AltNeRF in generating high-fidelity and robust novel views that closely
resemble reality.

---

## Semantic-Human: Neural Rendering of Humans from Monocular Video with  Human Parsing

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-19 | Jie Zhang, Pengcheng Shi, Zaiwang Gu, Yiyang Zhou, Zhi Wang | cs.CV | [PDF](http://arxiv.org/pdf/2308.09894v1){: .btn .btn-green } |

**Abstract**: The neural rendering of humans is a topic of great research significance.
However, previous works mostly focus on achieving photorealistic details,
neglecting the exploration of human parsing. Additionally, classical semantic
work are all limited in their ability to efficiently represent fine results in
complex motions. Human parsing is inherently related to radiance
reconstruction, as similar appearance and geometry often correspond to similar
semantic part. Furthermore, previous works often design a motion field that
maps from the observation space to the canonical space, while it tends to
exhibit either underfitting or overfitting, resulting in limited
generalization. In this paper, we present Semantic-Human, a novel method that
achieves both photorealistic details and viewpoint-consistent human parsing for
the neural rendering of humans. Specifically, we extend neural radiance fields
(NeRF) to jointly encode semantics, appearance and geometry to achieve accurate
2D semantic labels using noisy pseudo-label supervision. Leveraging the
inherent consistency and smoothness properties of NeRF, Semantic-Human achieves
consistent human parsing in both continuous and novel views. We also introduce
constraints derived from the SMPL surface for the motion field and
regularization for the recovered volumetric geometry. We have evaluated the
model using the ZJU-MoCap dataset, and the obtained highly competitive results
demonstrate the effectiveness of our proposed Semantic-Human. We also showcase
various compelling applications, including label denoising, label synthesis and
image editing, and empirically validate its advantageous properties.

---

## HollowNeRF: Pruning Hashgrid-Based NeRFs with Trainable Collision  Mitigation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-19 | Xiufeng Xie, Riccardo Gherardi, Zhihong Pan, Stephen Huang | cs.CV | [PDF](http://arxiv.org/pdf/2308.10122v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) have garnered significant attention, with
recent works such as Instant-NGP accelerating NeRF training and evaluation
through a combination of hashgrid-based positional encoding and neural
networks. However, effectively leveraging the spatial sparsity of 3D scenes
remains a challenge. To cull away unnecessary regions of the feature grid,
existing solutions rely on prior knowledge of object shape or periodically
estimate object shape during training by repeated model evaluations, which are
costly and wasteful.
  To address this issue, we propose HollowNeRF, a novel compression solution
for hashgrid-based NeRF which automatically sparsifies the feature grid during
the training phase. Instead of directly compressing dense features, HollowNeRF
trains a coarse 3D saliency mask that guides efficient feature pruning, and
employs an alternating direction method of multipliers (ADMM) pruner to
sparsify the 3D saliency mask during training. By exploiting the sparsity in
the 3D scene to redistribute hash collisions, HollowNeRF improves rendering
quality while using a fraction of the parameters of comparable state-of-the-art
solutions, leading to a better cost-accuracy trade-off. Our method delivers
comparable rendering quality to Instant-NGP, while utilizing just 31% of the
parameters. In addition, our solution can achieve a PSNR accuracy gain of up to
1dB using only 56% of the parameters.

Comments:
- Accepted to ICCV 2023

---

## DReg-NeRF: Deep Registration for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-18 | Yu Chen, Gim Hee Lee | cs.CV | [PDF](http://arxiv.org/pdf/2308.09386v1){: .btn .btn-green } |

**Abstract**: Although Neural Radiance Fields (NeRF) is popular in the computer vision
community recently, registering multiple NeRFs has yet to gain much attention.
Unlike the existing work, NeRF2NeRF, which is based on traditional optimization
methods and needs human annotated keypoints, we propose DReg-NeRF to solve the
NeRF registration problem on object-centric scenes without human intervention.
After training NeRF models, our DReg-NeRF first extracts features from the
occupancy grid in NeRF. Subsequently, our DReg-NeRF utilizes a transformer
architecture with self-attention and cross-attention layers to learn the
relations between pairwise NeRF blocks. In contrast to state-of-the-art (SOTA)
point cloud registration methods, the decoupled correspondences are supervised
by surface fields without any ground truth overlapping labels. We construct a
novel view synthesis dataset with 1,700+ 3D objects obtained from Objaverse to
train our network. When evaluated on the test set, our proposed method beats
the SOTA point cloud registration methods by a large margin, with a mean
$\text{RPE}=9.67^{\circ}$ and a mean $\text{RTE}=0.038$.
  Our code is available at https://github.com/AIBluefisher/DReg-NeRF.

Comments:
- Accepted at ICCV 2023

---

## MonoNeRD: NeRF-like Representations for Monocular 3D Object Detection

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-18 | Junkai Xu, Liang Peng, Haoran Cheng, Hao Li, Wei Qian, Ke Li, Wenxiao Wang, Deng Cai | cs.CV | [PDF](http://arxiv.org/pdf/2308.09421v2){: .btn .btn-green } |

**Abstract**: In the field of monocular 3D detection, it is common practice to utilize
scene geometric clues to enhance the detector's performance. However, many
existing works adopt these clues explicitly such as estimating a depth map and
back-projecting it into 3D space. This explicit methodology induces sparsity in
3D representations due to the increased dimensionality from 2D to 3D, and leads
to substantial information loss, especially for distant and occluded objects.
To alleviate this issue, we propose MonoNeRD, a novel detection framework that
can infer dense 3D geometry and occupancy. Specifically, we model scenes with
Signed Distance Functions (SDF), facilitating the production of dense 3D
representations. We treat these representations as Neural Radiance Fields
(NeRF) and then employ volume rendering to recover RGB images and depth maps.
To the best of our knowledge, this work is the first to introduce volume
rendering for M3D, and demonstrates the potential of implicit reconstruction
for image-based 3D perception. Extensive experiments conducted on the KITTI-3D
benchmark and Waymo Open Dataset demonstrate the effectiveness of MonoNeRD.
Codes are available at https://github.com/cskkxjk/MonoNeRD.

Comments:
- Accepted by ICCV 2023

---

## Watch Your Steps: Local Image and Scene Editing by Text Instructions

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-17 | Ashkan Mirzaei, Tristan Aumentado-Armstrong, Marcus A. Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G. Derpanis, Igor Gilitschenski | cs.CV | [PDF](http://arxiv.org/pdf/2308.08947v1){: .btn .btn-green } |

**Abstract**: Denoising diffusion models have enabled high-quality image generation and
editing. We present a method to localize the desired edit region implicit in a
text instruction. We leverage InstructPix2Pix (IP2P) and identify the
discrepancy between IP2P predictions with and without the instruction. This
discrepancy is referred to as the relevance map. The relevance map conveys the
importance of changing each pixel to achieve the edits, and is used to to guide
the modifications. This guidance ensures that the irrelevant pixels remain
unchanged. Relevance maps are further used to enhance the quality of
text-guided editing of 3D scenes in the form of neural radiance fields. A field
is trained on relevance maps of training views, denoted as the relevance field,
defining the 3D region within which modifications should be made. We perform
iterative updates on the training views guided by rendered relevance maps from
the relevance field. Our method achieves state-of-the-art performance on both
image and NeRF editing tasks. Project page:
https://ashmrz.github.io/WatchYourSteps/

Comments:
- Project page: https://ashmrz.github.io/WatchYourSteps/

---

## Language-enhanced RNR-Map: Querying Renderable Neural Radiance Field  maps with natural language



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-17 | Francesco Taioli, Federico Cunico, Federico Girella, Riccardo Bologna, Alessandro Farinelli, Marco Cristani | cs.CV | [PDF](http://arxiv.org/pdf/2308.08854v1){: .btn .btn-green } |

**Abstract**: We present Le-RNR-Map, a Language-enhanced Renderable Neural Radiance map for
Visual Navigation with natural language query prompts. The recently proposed
RNR-Map employs a grid structure comprising latent codes positioned at each
pixel. These latent codes, which are derived from image observation, enable: i)
image rendering given a camera pose, since they are converted to Neural
Radiance Field; ii) image navigation and localization with astonishing
accuracy. On top of this, we enhance RNR-Map with CLIP-based embedding latent
codes, allowing natural language search without additional label data. We
evaluate the effectiveness of this map in single and multi-object searches. We
also investigate its compatibility with a Large Language Model as an
"affordance query resolver". Code and videos are available at
https://intelligolabs.github.io/Le-RNR-Map/

Comments:
- Accepted at ICCVW23 VLAR

---

## Ref-DVGO: Reflection-Aware Direct Voxel Grid Optimization for an  Improved Quality-Efficiency Trade-Off in Reflective Scene Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-16 | Georgios Kouros, Minye Wu, Shubham Shrivastava, Sushruth Nagesh, Punarjay Chakravarty, Tinne Tuytelaars | cs.CV | [PDF](http://arxiv.org/pdf/2308.08530v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have revolutionized the field of novel view
synthesis, demonstrating remarkable performance. However, the modeling and
rendering of reflective objects remain challenging problems. Recent methods
have shown significant improvements over the baselines in handling reflective
scenes, albeit at the expense of efficiency. In this work, we aim to strike a
balance between efficiency and quality. To this end, we investigate an
implicit-explicit approach based on conventional volume rendering to enhance
the reconstruction quality and accelerate the training and rendering processes.
We adopt an efficient density-based grid representation and reparameterize the
reflected radiance in our pipeline. Our proposed reflection-aware approach
achieves a competitive quality efficiency trade-off compared to competing
methods. Based on our experimental results, we propose and discuss hypotheses
regarding the factors influencing the results of density-based methods for
reconstructing reflective objects. The source code is available at
https://github.com/gkouros/ref-dvgo.

Comments:
- 5 pages, 4 figures, 3 tables, ICCV TRICKY 2023 Workshop

---

## SceNeRFlow: Time-Consistent Reconstruction of General Dynamic Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-16 | Edith Tretschk, Vladislav Golyanik, Michael Zollhoefer, Aljaz Bozic, Christoph Lassner, Christian Theobalt | cs.CV | [PDF](http://arxiv.org/pdf/2308.08258v1){: .btn .btn-green } |

**Abstract**: Existing methods for the 4D reconstruction of general, non-rigidly deforming
objects focus on novel-view synthesis and neglect correspondences. However,
time consistency enables advanced downstream tasks like 3D editing, motion
analysis, or virtual-asset creation. We propose SceNeRFlow to reconstruct a
general, non-rigid scene in a time-consistent manner. Our dynamic-NeRF method
takes multi-view RGB videos and background images from static cameras with
known camera parameters as input. It then reconstructs the deformations of an
estimated canonical model of the geometry and appearance in an online fashion.
Since this canonical model is time-invariant, we obtain correspondences even
for long-term, long-range motions. We employ neural scene representations to
parametrize the components of our method. Like prior dynamic-NeRF methods, we
use a backwards deformation model. We find non-trivial adaptations of this
model necessary to handle larger motions: We decompose the deformations into a
strongly regularized coarse component and a weakly regularized fine component,
where the coarse component also extends the deformation field into the space
surrounding the object, which enables tracking over time. We show
experimentally that, unlike prior work that only handles small motion, our
method enables the reconstruction of studio-scale motions.

Comments:
- Project page: https://vcai.mpi-inf.mpg.de/projects/scenerflow/

---

## Neural radiance fields in the industrial and robotics domain:  applications, research opportunities and use cases

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-14 | Eugen Šlapak, Enric Pardo, Matúš Dopiriak, Taras Maksymyuk, Juraj Gazda | cs.RO | [PDF](http://arxiv.org/pdf/2308.07118v2){: .btn .btn-green } |

**Abstract**: The proliferation of technologies, such as extended reality (XR), has
increased the demand for high-quality three-dimensional (3D) graphical
representations. Industrial 3D applications encompass computer-aided design
(CAD), finite element analysis (FEA), scanning, and robotics. However, current
methods employed for industrial 3D representations suffer from high
implementation costs and reliance on manual human input for accurate 3D
modeling. To address these challenges, neural radiance fields (NeRFs) have
emerged as a promising approach for learning 3D scene representations based on
provided training 2D images. Despite a growing interest in NeRFs, their
potential applications in various industrial subdomains are still unexplored.
In this paper, we deliver a comprehensive examination of NeRF industrial
applications while also providing direction for future research endeavors. We
also present a series of proof-of-concept experiments that demonstrate the
potential of NeRFs in the industrial domain. These experiments include
NeRF-based video compression techniques and using NeRFs for 3D motion
estimation in the context of collision avoidance. In the video compression
experiment, our results show compression savings up to 48\% and 74\% for
resolutions of 1920x1080 and 300x168, respectively. The motion estimation
experiment used a 3D animation of a robotic arm to train Dynamic-NeRF (D-NeRF)
and achieved an average peak signal-to-noise ratio (PSNR) of disparity map with
the value of 23 dB and an structural similarity index measure (SSIM) 0.97.

---

## S3IM: Stochastic Structural SIMilarity and Its Unreasonable  Effectiveness for Neural Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-14 | Zeke Xie, Xindi Yang, Yujie Yang, Qi Sun, Yixiang Jiang, Haoran Wang, Yunfeng Cai, Mingming Sun | cs.CV | [PDF](http://arxiv.org/pdf/2308.07032v1){: .btn .btn-green } |

**Abstract**: Recently, Neural Radiance Field (NeRF) has shown great success in rendering
novel-view images of a given scene by learning an implicit representation with
only posed RGB images. NeRF and relevant neural field methods (e.g., neural
surface representation) typically optimize a point-wise loss and make
point-wise predictions, where one data point corresponds to one pixel.
Unfortunately, this line of research failed to use the collective supervision
of distant pixels, although it is known that pixels in an image or scene can
provide rich structural information. To the best of our knowledge, we are the
first to design a nonlocal multiplex training paradigm for NeRF and relevant
neural field methods via a novel Stochastic Structural SIMilarity (S3IM) loss
that processes multiple data points as a whole set instead of process multiple
inputs independently. Our extensive experiments demonstrate the unreasonable
effectiveness of S3IM in improving NeRF and neural surface representation for
nearly free. The improvements of quality metrics can be particularly
significant for those relatively difficult tasks: e.g., the test MSE loss
unexpectedly drops by more than 90% for TensoRF and DVGO over eight novel view
synthesis tasks; a 198% F-score gain and a 64% Chamfer $L_{1}$ distance
reduction for NeuS over eight surface reconstruction tasks. Moreover, S3IM is
consistently robust even with sparse inputs, corrupted images, and dynamic
scenes.

Comments:
- ICCV 2023 main conference. Code: https://github.com/Madaoer/S3IM. 14
  pages, 5 figures, 17 tables

---

## VERF: Runtime Monitoring of Pose Estimation with Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-11 | Dominic Maggio, Courtney Mario, Luca Carlone | cs.RO | [PDF](http://arxiv.org/pdf/2308.05939v1){: .btn .btn-green } |

**Abstract**: We present VERF, a collection of two methods (VERF-PnP and VERF-Light) for
providing runtime assurance on the correctness of a camera pose estimate of a
monocular camera without relying on direct depth measurements. We leverage the
ability of NeRF (Neural Radiance Fields) to render novel RGB perspectives of a
scene. We only require as input the camera image whose pose is being estimated,
an estimate of the camera pose we want to monitor, and a NeRF model containing
the scene pictured by the camera. We can then predict if the pose estimate is
within a desired distance from the ground truth and justify our prediction with
a level of confidence. VERF-Light does this by rendering a viewpoint with NeRF
at the estimated pose and estimating its relative offset to the sensor image up
to scale. Since scene scale is unknown, the approach renders another auxiliary
image and reasons over the consistency of the optical flows across the three
images. VERF-PnP takes a different approach by rendering a stereo pair of
images with NeRF and utilizing the Perspective-n-Point (PnP) algorithm. We
evaluate both methods on the LLFF dataset, on data from a Unitree A1 quadruped
robot, and on data collected from Blue Origin's sub-orbital New Shepard rocket
to demonstrate the effectiveness of the proposed pose monitoring method across
a range of scene scales. We also show monitoring can be completed in under half
a second on a 3090 GPU.

---

## Focused Specific Objects NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-11 | Yuesong Li, Feng Pan, Helong Yan, Xiuli Xin, Xiaoxue Feng | cs.CV | [PDF](http://arxiv.org/pdf/2308.05970v1){: .btn .btn-green } |

**Abstract**: Most NeRF-based models are designed for learning the entire scene, and
complex scenes can lead to longer learning times and poorer rendering effects.
This paper utilizes scene semantic priors to make improvements in fast
training, allowing the network to focus on the specific targets and not be
affected by complex backgrounds. The training speed can be increased by 7.78
times with better rendering effect, and small to medium sized targets can be
rendered faster. In addition, this improvement applies to all NeRF-based
models. Considering the inherent multi-view consistency and smoothness of NeRF,
this paper also studies weak supervision by sparsely sampling negative ray
samples. With this method, training can be further accelerated and rendering
quality can be maintained. Finally, this paper extends pixel semantic and color
rendering formulas and proposes a new scene editing technique that can achieve
unique displays of the specific semantic targets or masking them in rendering.
To address the problem of unsupervised regions incorrect inferences in the
scene, we also designed a self-supervised loop that combines morphological
operations and clustering.

Comments:
- 17 pages,32 figures

---

## WaveNeRF: Wavelet-based Generalizable Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-09 | Muyu Xu, Fangneng Zhan, Jiahui Zhang, Yingchen Yu, Xiaoqin Zhang, Christian Theobalt, Ling Shao, Shijian Lu | cs.CV | [PDF](http://arxiv.org/pdf/2308.04826v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has shown impressive performance in novel view
synthesis via implicit scene representation. However, it usually suffers from
poor scalability as requiring densely sampled images for each new scene.
Several studies have attempted to mitigate this problem by integrating
Multi-View Stereo (MVS) technique into NeRF while they still entail a
cumbersome fine-tuning process for new scenes. Notably, the rendering quality
will drop severely without this fine-tuning process and the errors mainly
appear around the high-frequency features. In the light of this observation, we
design WaveNeRF, which integrates wavelet frequency decomposition into MVS and
NeRF to achieve generalizable yet high-quality synthesis without any per-scene
optimization. To preserve high-frequency information when generating 3D feature
volumes, WaveNeRF builds Multi-View Stereo in the Wavelet domain by integrating
the discrete wavelet transform into the classical cascade MVS, which
disentangles high-frequency information explicitly. With that, disentangled
frequency features can be injected into classic NeRF via a novel hybrid neural
renderer to yield faithful high-frequency details, and an intuitive
frequency-guided sampling strategy can be designed to suppress artifacts around
high-frequency regions. Extensive experiments over three widely studied
benchmarks show that WaveNeRF achieves superior generalizable radiance field
modeling when only given three images as input.

Comments:
- Accepted to ICCV 2023. Project website:
  https://mxuai.github.io/WaveNeRF/

---

## A General Implicit Framework for Fast NeRF Composition and Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-09 | Xinyu Gao, Ziyi Yang, Yunlu Zhao, Yuxiang Sun, Xiaogang Jin, Changqing Zou | cs.CV | [PDF](http://arxiv.org/pdf/2308.04669v4){: .btn .btn-green } |

**Abstract**: A variety of Neural Radiance Fields (NeRF) methods have recently achieved
remarkable success in high render speed. However, current accelerating methods
are specialized and incompatible with various implicit methods, preventing
real-time composition over various types of NeRF works. Because NeRF relies on
sampling along rays, it is possible to provide general guidance for
acceleration. To that end, we propose a general implicit pipeline for composing
NeRF objects quickly. Our method enables the casting of dynamic shadows within
or between objects using analytical light sources while allowing multiple NeRF
objects to be seamlessly placed and rendered together with any arbitrary rigid
transformations. Mainly, our work introduces a new surface representation known
as Neural Depth Fields (NeDF) that quickly determines the spatial relationship
between objects by allowing direct intersection computation between rays and
implicit surfaces. It leverages an intersection neural network to query NeRF
for acceleration instead of depending on an explicit spatial structure.Our
proposed method is the first to enable both the progressive and interactive
composition of NeRF objects. Additionally, it also serves as a previewing
plugin for a range of existing NeRF works.

Comments:
- AAAI 2024

---

## Digging into Depth Priors for Outdoor Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-08 | Chen Wang, Jiadai Sun, Lina Liu, Chenming Wu, Zhelun Shen, Dayan Wu, Yuchao Dai, Liangjun Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2308.04413v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have demonstrated impressive performance in
vision and graphics tasks, such as novel view synthesis and immersive reality.
However, the shape-radiance ambiguity of radiance fields remains a challenge,
especially in the sparse viewpoints setting. Recent work resorts to integrating
depth priors into outdoor NeRF training to alleviate the issue. However, the
criteria for selecting depth priors and the relative merits of different priors
have not been thoroughly investigated. Moreover, the relative merits of
selecting different approaches to use the depth priors is also an unexplored
problem. In this paper, we provide a comprehensive study and evaluation of
employing depth priors to outdoor neural radiance fields, covering common depth
sensing technologies and most application ways. Specifically, we conduct
extensive experiments with two representative NeRF methods equipped with four
commonly-used depth priors and different depth usages on two widely used
outdoor datasets. Our experimental results reveal several interesting findings
that can potentially benefit practitioners and researchers in training their
NeRF models with depth priors. Project Page:
https://cwchenwang.github.io/outdoor-nerf-depth

Comments:
- Accepted to ACM MM 2023. Project Page:
  https://cwchenwang.github.io/outdoor-nerf-depth

---

## 3D Gaussian Splatting for Real-Time Radiance Field Rendering

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-08 | Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis | cs.GR | [PDF](http://arxiv.org/pdf/2308.04079v1){: .btn .btn-green } |

**Abstract**: Radiance Field methods have recently revolutionized novel-view synthesis of
scenes captured with multiple photos or videos. However, achieving high visual
quality still requires neural networks that are costly to train and render,
while recent faster methods inevitably trade off speed for quality. For
unbounded and complete scenes (rather than isolated objects) and 1080p
resolution rendering, no current method can achieve real-time display rates. We
introduce three key elements that allow us to achieve state-of-the-art visual
quality while maintaining competitive training times and importantly allow
high-quality real-time (>= 30 fps) novel-view synthesis at 1080p resolution.
First, starting from sparse points produced during camera calibration, we
represent the scene with 3D Gaussians that preserve desirable properties of
continuous volumetric radiance fields for scene optimization while avoiding
unnecessary computation in empty space; Second, we perform interleaved
optimization/density control of the 3D Gaussians, notably optimizing
anisotropic covariance to achieve an accurate representation of the scene;
Third, we develop a fast visibility-aware rendering algorithm that supports
anisotropic splatting and both accelerates training and allows realtime
rendering. We demonstrate state-of-the-art visual quality and real-time
rendering on several established datasets.

Comments:
- https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## Mirror-NeRF: Learning Neural Radiance Fields for Mirrors with  Whitted-Style Ray Tracing

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-07 | Junyi Zeng, Chong Bao, Rui Chen, Zilong Dong, Guofeng Zhang, Hujun Bao, Zhaopeng Cui | cs.CV | [PDF](http://arxiv.org/pdf/2308.03280v1){: .btn .btn-green } |

**Abstract**: Recently, Neural Radiance Fields (NeRF) has exhibited significant success in
novel view synthesis, surface reconstruction, etc. However, since no physical
reflection is considered in its rendering pipeline, NeRF mistakes the
reflection in the mirror as a separate virtual scene, leading to the inaccurate
reconstruction of the mirror and multi-view inconsistent reflections in the
mirror. In this paper, we present a novel neural rendering framework, named
Mirror-NeRF, which is able to learn accurate geometry and reflection of the
mirror and support various scene manipulation applications with mirrors, such
as adding new objects or mirrors into the scene and synthesizing the
reflections of these new objects in mirrors, controlling mirror roughness, etc.
To achieve this goal, we propose a unified radiance field by introducing the
reflection probability and tracing rays following the light transport model of
Whitted Ray Tracing, and also develop several techniques to facilitate the
learning process. Experiments and comparisons on both synthetic and real
datasets demonstrate the superiority of our method. The code and supplementary
material are available on the project webpage:
https://zju3dv.github.io/Mirror-NeRF/.

Comments:
- Accepted to ACM Multimedia 2023. Project Page:
  https://zju3dv.github.io/Mirror-NeRF/

---

## Where and How: Mitigating Confusion in Neural Radiance Fields from  Sparse Inputs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-05 | Yanqi Bao, Yuxin Li, Jing Huo, Tianyu Ding, Xinyue Liang, Wenbin Li, Yang Gao | cs.CV | [PDF](http://arxiv.org/pdf/2308.02908v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields from Sparse input} (NeRF-S) have shown great potential
in synthesizing novel views with a limited number of observed viewpoints.
However, due to the inherent limitations of sparse inputs and the gap between
non-adjacent views, rendering results often suffer from over-fitting and foggy
surfaces, a phenomenon we refer to as "CONFUSION" during volume rendering. In
this paper, we analyze the root cause of this confusion and attribute it to two
fundamental questions: "WHERE" and "HOW". To this end, we present a novel
learning framework, WaH-NeRF, which effectively mitigates confusion by tackling
the following challenges: (i)"WHERE" to Sample? in NeRF-S -- we introduce a
Deformable Sampling strategy and a Weight-based Mutual Information Loss to
address sample-position confusion arising from the limited number of
viewpoints; and (ii) "HOW" to Predict? in NeRF-S -- we propose a
Semi-Supervised NeRF learning Paradigm based on pose perturbation and a
Pixel-Patch Correspondence Loss to alleviate prediction confusion caused by the
disparity between training and testing viewpoints. By integrating our proposed
modules and loss functions, WaH-NeRF outperforms previous methods under the
NeRF-S setting. Code is available https://github.com/bbbbby-99/WaH-NeRF.

Comments:
- Accepted In Proceedings of the 31st ACM International Conference on
  Multimedia (MM' 23)

---

## Learning Unified Decompositional and Compositional NeRF for Editable  Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-05 | Yuxin Wang, Wayne Wu, Dan Xu | cs.CV | [PDF](http://arxiv.org/pdf/2308.02840v1){: .btn .btn-green } |

**Abstract**: Implicit neural representations have shown powerful capacity in modeling
real-world 3D scenes, offering superior performance in novel view synthesis. In
this paper, we target a more challenging scenario, i.e., joint scene novel view
synthesis and editing based on implicit neural scene representations.
State-of-the-art methods in this direction typically consider building separate
networks for these two tasks (i.e., view synthesis and editing). Thus, the
modeling of interactions and correlations between these two tasks is very
limited, which, however, is critical for learning high-quality scene
representations. To tackle this problem, in this paper, we propose a unified
Neural Radiance Field (NeRF) framework to effectively perform joint scene
decomposition and composition for modeling real-world scenes. The decomposition
aims at learning disentangled 3D representations of different objects and the
background, allowing for scene editing, while scene composition models an
entire scene representation for novel view synthesis. Specifically, with a
two-stage NeRF framework, we learn a coarse stage for predicting a global
radiance field as guidance for point sampling, and in the second fine-grained
stage, we perform scene decomposition by a novel one-hot object radiance field
regularization module and a pseudo supervision via inpainting to handle
ambiguous background regions occluded by objects. The decomposed object-level
radiance fields are further composed by using activations from the
decomposition module. Extensive quantitative and qualitative results show the
effectiveness of our method for scene decomposition and composition,
outperforming state-of-the-art methods for both novel-view synthesis and
editing tasks.

Comments:
- ICCV2023, Project Page: https://w-ted.github.io/publications/udc-nerf

---

## NeRFs: The Search for the Best 3D Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-05 | Ravi Ramamoorthi | cs.CV | [PDF](http://arxiv.org/pdf/2308.02751v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields or NeRFs have become the representation of choice for
problems in view synthesis or image-based rendering, as well as in many other
applications across computer graphics and vision, and beyond. At their core,
NeRFs describe a new representation of 3D scenes or 3D geometry. Instead of
meshes, disparity maps, multiplane images or even voxel grids, they represent
the scene as a continuous volume, with volumetric parameters like
view-dependent radiance and volume density obtained by querying a neural
network. The NeRF representation has now been widely used, with thousands of
papers extending or building on it every year, multiple authors and websites
providing overviews and surveys, and numerous industrial applications and
startup companies. In this article, we briefly review the NeRF representation,
and describe the three decades-long quest to find the best 3D representation
for view synthesis and related problems, culminating in the NeRF papers. We
then describe new developments in terms of NeRF representations and make some
observations and insights regarding the future of 3D representations.

Comments:
- Updated based on feedback in-person and via e-mail at SIGGRAPH 2023.
  In particular, I have added references and discussion of seminal SIGGRAPH
  image-based rendering papers, and better put the recent Kerbl et al. work in
  context, with more references

---

## ES-MVSNet: Efficient Framework for End-to-end Self-supervised Multi-View  Stereo

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-04 | Qiang Zhou, Chaohui Yu, Jingliang Li, Yuang Liu, Jing Wang, Zhibin Wang | cs.CV | [PDF](http://arxiv.org/pdf/2308.02191v1){: .btn .btn-green } |

**Abstract**: Compared to the multi-stage self-supervised multi-view stereo (MVS) method,
the end-to-end (E2E) approach has received more attention due to its concise
and efficient training pipeline. Recent E2E self-supervised MVS approaches have
integrated third-party models (such as optical flow models, semantic
segmentation models, NeRF models, etc.) to provide additional consistency
constraints, which grows GPU memory consumption and complicates the model's
structure and training pipeline. In this work, we propose an efficient
framework for end-to-end self-supervised MVS, dubbed ES-MVSNet. To alleviate
the high memory consumption of current E2E self-supervised MVS frameworks, we
present a memory-efficient architecture that reduces memory usage by 43%
without compromising model performance. Furthermore, with the novel design of
asymmetric view selection policy and region-aware depth consistency, we achieve
state-of-the-art performance among E2E self-supervised MVS methods, without
relying on third-party models for additional consistency signals. Extensive
experiments on DTU and Tanks&Temples benchmarks demonstrate that the proposed
ES-MVSNet approach achieves state-of-the-art performance among E2E
self-supervised MVS methods and competitive performance to many supervised and
multi-stage self-supervised methods.

Comments:
- arXiv admin note: text overlap with arXiv:2203.03949 by other authors

---

## Incorporating Season and Solar Specificity into Renderings made by a  NeRF Architecture using Satellite Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-02 | Michael Gableman, Avinash Kak | cs.CV | [PDF](http://arxiv.org/pdf/2308.01262v2){: .btn .btn-green } |

**Abstract**: As a result of Shadow NeRF and Sat-NeRF, it is possible to take the solar
angle into account in a NeRF-based framework for rendering a scene from a novel
viewpoint using satellite images for training. Our work extends those
contributions and shows how one can make the renderings season-specific. Our
main challenge was creating a Neural Radiance Field (NeRF) that could render
seasonal features independently of viewing angle and solar angle while still
being able to render shadows. We teach our network to render seasonal features
by introducing one more input variable -- time of the year. However, the small
training datasets typical of satellite imagery can introduce ambiguities in
cases where shadows are present in the same location for every image of a
particular season. We add additional terms to the loss function to discourage
the network from using seasonal features for accounting for shadows. We show
the performance of our network on eight Areas of Interest containing images
captured by the Maxar WorldView-3 satellite. This evaluation includes tests
measuring the ability of our framework to accurately render novel views,
generate height maps, predict shadows, and specify seasonal features
independently from shadows. Our ablation studies justify the choices made for
network design parameters.

Comments:
- 18 pages, 17 figures, 10 tables

---

## Context-Aware Talking-Head Video Editing

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-01 | Songlin Yang, Wei Wang, Jun Ling, Bo Peng, Xu Tan, Jing Dong | cs.MM | [PDF](http://arxiv.org/pdf/2308.00462v3){: .btn .btn-green } |

**Abstract**: Talking-head video editing aims to efficiently insert, delete, and substitute
the word of a pre-recorded video through a text transcript editor. The key
challenge for this task is obtaining an editing model that generates new
talking-head video clips which simultaneously have accurate lip synchronization
and motion smoothness. Previous approaches, including 3DMM-based (3D Morphable
Model) methods and NeRF-based (Neural Radiance Field) methods, are sub-optimal
in that they either require minutes of source videos and days of training time
or lack the disentangled control of verbal (e.g., lip motion) and non-verbal
(e.g., head pose and expression) representations for video clip insertion. In
this work, we fully utilize the video context to design a novel framework for
talking-head video editing, which achieves efficiency, disentangled motion
control, and sequential smoothness. Specifically, we decompose this framework
to motion prediction and motion-conditioned rendering: (1) We first design an
animation prediction module that efficiently obtains smooth and lip-sync motion
sequences conditioned on the driven speech. This module adopts a
non-autoregressive network to obtain context prior and improve the prediction
efficiency, and it learns a speech-animation mapping prior with better
generalization to novel speech from a multi-identity video dataset. (2) We then
introduce a neural rendering module to synthesize the photo-realistic and
full-head video frames given the predicted motion sequence. This module adopts
a pre-trained head topology and uses only few frames for efficient fine-tuning
to obtain a person-specific rendering model. Extensive experiments demonstrate
that our method efficiently achieves smoother editing results with higher image
quality and lip accuracy using less data than previous methods.

Comments:
- The version of this paper needs to be further improved

---

## Robust Single-view Cone-beam X-ray Pose Estimation with Neural Tuned  Tomography (NeTT) and Masked Neural Radiance Fields (mNeRF)

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-01 | Chaochao Zhou, Syed Hasib Akhter Faruqui, Abhinav Patel, Ramez N. Abdalla, Michael C. Hurley, Ali Shaibani, Matthew B. Potts, Babak S. Jahromi, Leon Cho, Sameer A. Ansari, Donald R. Cantrell | cs.CV | [PDF](http://arxiv.org/pdf/2308.00214v2){: .btn .btn-green } |

**Abstract**: Many tasks performed in image-guided, mini-invasive, medical procedures can
be cast as pose estimation problems, where an X-ray projection is utilized to
reach a target in 3D space. Expanding on recent advances in the differentiable
rendering of optically reflective materials, we introduce new methods for pose
estimation of radiolucent objects using X-ray projections, and we demonstrate
the critical role of optimal view synthesis in performing this task. We first
develop an algorithm (DiffDRR) that efficiently computes Digitally
Reconstructed Radiographs (DRRs) and leverages automatic differentiation within
TensorFlow. Pose estimation is performed by iterative gradient descent using a
loss function that quantifies the similarity of the DRR synthesized from a
randomly initialized pose and the true fluoroscopic image at the target pose.
We propose two novel methods for high-fidelity view synthesis, Neural Tuned
Tomography (NeTT) and masked Neural Radiance Fields (mNeRF). Both methods rely
on classic Cone-Beam Computerized Tomography (CBCT); NeTT directly optimizes
the CBCT densities, while the non-zero values of mNeRF are constrained by a 3D
mask of the anatomic region segmented from CBCT. We demonstrate that both NeTT
and mNeRF distinctly improve pose estimation within our framework. By defining
a successful pose estimate to be a 3D angle error of less than 3 deg, we find
that NeTT and mNeRF can achieve similar results, both with overall success
rates more than 93%. However, the computational cost of NeTT is significantly
lower than mNeRF in both training and pose estimation. Furthermore, we show
that a NeTT trained for a single subject can generalize to synthesize
high-fidelity DRRs and ensure robust pose estimations for all other subjects.
Therefore, we suggest that NeTT is an attractive option for robust pose
estimation using fluoroscopic projections.

---

## High-Fidelity Eye Animatable Neural Radiance Fields for Human Face

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-08-01 | Hengfei Wang, Zhongqun Zhang, Yihua Cheng, Hyung Jin Chang | cs.CV | [PDF](http://arxiv.org/pdf/2308.00773v3){: .btn .btn-green } |

**Abstract**: Face rendering using neural radiance fields (NeRF) is a rapidly developing
research area in computer vision. While recent methods primarily focus on
controlling facial attributes such as identity and expression, they often
overlook the crucial aspect of modeling eyeball rotation, which holds
importance for various downstream tasks. In this paper, we aim to learn a face
NeRF model that is sensitive to eye movements from multi-view images. We
address two key challenges in eye-aware face NeRF learning: how to effectively
capture eyeball rotation for training and how to construct a manifold for
representing eyeball rotation. To accomplish this, we first fit FLAME, a
well-established parametric face model, to the multi-view images considering
multi-view consistency. Subsequently, we introduce a new Dynamic Eye-aware NeRF
(DeNeRF). DeNeRF transforms 3D points from different views into a canonical
space to learn a unified face NeRF model. We design an eye deformation field
for the transformation, including rigid transformation, e.g., eyeball rotation,
and non-rigid transformation. Through experiments conducted on the ETH-XGaze
dataset, we demonstrate that our model is capable of generating high-fidelity
images with accurate eyeball rotation and non-rigid periocular deformation,
even under novel viewing angles. Furthermore, we show that utilizing the
rendered images can effectively enhance gaze estimation performance.

Comments:
- BMVC2023 Oral