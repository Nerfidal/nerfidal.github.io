---
layout: default
title: January
parent: 2024
nav_order: 1
---
<!---metadata--->

## GaussianBody: Clothed Human Reconstruction via 3d Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-18 | Mengtian Li, Shengxiang Yao, Zhifeng Xie, Keyu Chen, Yu-Gang Jiang | cs.CV | [PDF](http://arxiv.org/pdf/2401.09720v1){: .btn .btn-green } |

**Abstract**: In this work, we propose a novel clothed human reconstruction method called
GaussianBody, based on 3D Gaussian Splatting. Compared with the costly neural
radiance based models, 3D Gaussian Splatting has recently demonstrated great
performance in terms of training time and rendering quality. However, applying
the static 3D Gaussian Splatting model to the dynamic human reconstruction
problem is non-trivial due to complicated non-rigid deformations and rich cloth
details. To address these challenges, our method considers explicit pose-guided
deformation to associate dynamic Gaussians across the canonical space and the
observation space, introducing a physically-based prior with regularized
transformations helps mitigate ambiguity between the two spaces. During the
training process, we further propose a pose refinement strategy to update the
pose regression for compensating the inaccurate initial estimation and a
split-with-scale mechanism to enhance the density of regressed point clouds.
The experiments validate that our method can achieve state-of-the-art
photorealistic novel-view rendering results with high-quality details for
dynamic clothed human bodies, along with explicit geometry reconstruction.

---

## IPR-NeRF: Ownership Verification meets Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-17 | Win Kent Ong, Kam Woh Ng, Chee Seng Chan, Yi Zhe Song, Tao Xiang | cs.CV | [PDF](http://arxiv.org/pdf/2401.09495v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) models have gained significant attention in the
computer vision community in the recent past with state-of-the-art visual
quality and produced impressive demonstrations. Since then, technopreneurs have
sought to leverage NeRF models into a profitable business. Therefore, NeRF
models make it worth the risk of plagiarizers illegally copying,
re-distributing, or misusing those models. This paper proposes a comprehensive
intellectual property (IP) protection framework for the NeRF model in both
black-box and white-box settings, namely IPR-NeRF. In the black-box setting, a
diffusion-based solution is introduced to embed and extract the watermark via a
two-stage optimization process. In the white-box setting, a designated digital
signature is embedded into the weights of the NeRF model by adopting the sign
loss objective. Our extensive experiments demonstrate that not only does our
approach maintain the fidelity (\ie, the rendering quality) of IPR-NeRF models,
but it is also robust against both ambiguity and removal attacks compared to
prior arts.

Comments:
- 21 pages

---

## ICON: Incremental CONfidence for Joint Pose and Radiance Field  Optimization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-17 | Weiyao Wang, Pierre Gleize, Hao Tang, Xingyu Chen, Kevin J Liang, Matt Feiszli | cs.CV | [PDF](http://arxiv.org/pdf/2401.08937v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) exhibit remarkable performance for Novel View
Synthesis (NVS) given a set of 2D images. However, NeRF training requires
accurate camera pose for each input view, typically obtained by
Structure-from-Motion (SfM) pipelines. Recent works have attempted to relax
this constraint, but they still often rely on decent initial poses which they
can refine. Here we aim at removing the requirement for pose initialization. We
present Incremental CONfidence (ICON), an optimization procedure for training
NeRFs from 2D video frames. ICON only assumes smooth camera motion to estimate
initial guess for poses. Further, ICON introduces ``confidence": an adaptive
measure of model quality used to dynamically reweight gradients. ICON relies on
high-confidence poses to learn NeRF, and high-confidence 3D structure (as
encoded by NeRF) to learn poses. We show that ICON, without prior pose
initialization, achieves superior performance in both CO3D and HO3D versus
methods which use SfM pose.

---

## Fast Dynamic 3D Object Generation from a Single-view Video

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-16 | Zijie Pan, Zeyu Yang, Xiatian Zhu, Li Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2401.08742v1){: .btn .btn-green } |

**Abstract**: Generating dynamic three-dimensional (3D) object from a single-view video is
challenging due to the lack of 4D labeled data. Existing methods extend
text-to-3D pipelines by transferring off-the-shelf image generation models such
as score distillation sampling, but they are slow and expensive to scale (e.g.,
150 minutes per object) due to the need for back-propagating the
information-limited supervision signals through a large pretrained model. To
address this limitation, we propose an efficient video-to-4D object generation
framework called Efficient4D. It generates high-quality spacetime-consistent
images under different camera views, and then uses them as labeled data to
directly train a novel 4D Gaussian splatting model with explicit point cloud
geometry, enabling real-time rendering under continuous camera trajectories.
Extensive experiments on synthetic and real videos show that Efficient4D offers
a remarkable 10-fold increase in speed when compared to prior art alternatives
while preserving the same level of innovative view synthesis quality. For
example, Efficient4D takes only 14 minutes to model a dynamic object.

Comments:
- Technical report

---

## Forging Vision Foundation Models for Autonomous Driving: Challenges,  Methodologies, and Opportunities

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-16 | Xu Yan, Haiming Zhang, Yingjie Cai, Jingming Guo, Weichao Qiu, Bin Gao, Kaiqiang Zhou, Yue Zhao, Huan Jin, Jiantao Gao, Zhen Li, Lihui Jiang, Wei Zhang, Hongbo Zhang, Dengxin Dai, Bingbing Liu | cs.CV | [PDF](http://arxiv.org/pdf/2401.08045v1){: .btn .btn-green } |

**Abstract**: The rise of large foundation models, trained on extensive datasets, is
revolutionizing the field of AI. Models such as SAM, DALL-E2, and GPT-4
showcase their adaptability by extracting intricate patterns and performing
effectively across diverse tasks, thereby serving as potent building blocks for
a wide range of AI applications. Autonomous driving, a vibrant front in AI
applications, remains challenged by the lack of dedicated vision foundation
models (VFMs). The scarcity of comprehensive training data, the need for
multi-sensor integration, and the diverse task-specific architectures pose
significant obstacles to the development of VFMs in this field. This paper
delves into the critical challenge of forging VFMs tailored specifically for
autonomous driving, while also outlining future directions. Through a
systematic analysis of over 250 papers, we dissect essential techniques for VFM
development, including data preparation, pre-training strategies, and
downstream task adaptation. Moreover, we explore key advancements such as NeRF,
diffusion models, 3D Gaussian Splatting, and world models, presenting a
comprehensive roadmap for future research. To empower researchers, we have
built and maintained https://github.com/zhanghm1995/Forge_VFM4AD, an
open-access repository constantly updated with the latest advancements in
forging VFMs for autonomous driving.

Comments:
- Github Repo: https://github.com/zhanghm1995/Forge_VFM4AD

---

## ProvNeRF: Modeling per Point Provenance in NeRFs as a Stochastic Process

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-16 | Kiyohiro Nakayama, Mikaela Angelina Uy, Yang You, Ke Li, Leonidas Guibas | cs.CV | [PDF](http://arxiv.org/pdf/2401.08140v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) have gained popularity across various
applications. However, they face challenges in the sparse view setting, lacking
sufficient constraints from volume rendering. Reconstructing and understanding
a 3D scene from sparse and unconstrained cameras is a long-standing problem in
classical computer vision with diverse applications. While recent works have
explored NeRFs in sparse, unconstrained view scenarios, their focus has been
primarily on enhancing reconstruction and novel view synthesis. Our approach
takes a broader perspective by posing the question: "from where has each point
been seen?" -- which gates how well we can understand and reconstruct it. In
other words, we aim to determine the origin or provenance of each 3D point and
its associated information under sparse, unconstrained views. We introduce
ProvNeRF, a model that enriches a traditional NeRF representation by
incorporating per-point provenance, modeling likely source locations for each
point. We achieve this by extending implicit maximum likelihood estimation
(IMLE) for stochastic processes. Notably, our method is compatible with any
pre-trained NeRF model and the associated training camera poses. We demonstrate
that modeling per-point provenance offers several advantages, including
uncertainty estimation, criteria-based view selection, and improved novel view
synthesis, compared to state-of-the-art methods. Please visit our project page
at https://provnerf.github.io

---

## 6-DoF Grasp Pose Evaluation and Optimization via Transfer Learning from  NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-15 | Gergely Sóti, Xi Huang, Christian Wurll, Björn Hein | cs.RO | [PDF](http://arxiv.org/pdf/2401.07935v1){: .btn .btn-green } |

**Abstract**: We address the problem of robotic grasping of known and unknown objects using
implicit behavior cloning. We train a grasp evaluation model from a small
number of demonstrations that outputs higher values for grasp candidates that
are more likely to succeed in grasping. This evaluation model serves as an
objective function, that we maximize to identify successful grasps. Key to our
approach is the utilization of learned implicit representations of visual and
geometric features derived from a pre-trained NeRF. Though trained exclusively
in a simulated environment with simplified objects and 4-DoF top-down grasps,
our evaluation model and optimization procedure demonstrate generalization to
6-DoF grasps and novel objects both in simulation and in real-world settings,
without the need for additional data. Supplementary material is available at:
https://gergely-soti.github.io/grasp

---

## CoSSegGaussians: Compact and Swift Scene Segmenting 3D Gaussians

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-11 | Bin Dou, Tianyu Zhang, Yongjia Ma, Zhaohui Wang, Zejian Yuan | cs.CV | [PDF](http://arxiv.org/pdf/2401.05925v1){: .btn .btn-green } |

**Abstract**: We propose Compact and Swift Segmenting 3D Gaussians(CoSSegGaussians), a
method for compact 3D-consistent scene segmentation at fast rendering speed
with only RGB images input. Previous NeRF-based 3D segmentation methods have
relied on implicit or voxel neural scene representation and ray-marching volume
rendering which are time consuming. Recent 3D Gaussian Splatting significantly
improves the rendering speed, however, existing Gaussians-based segmentation
methods(eg: Gaussian Grouping) fail to provide compact segmentation masks
especially in zero-shot segmentation, which is mainly caused by the lack of
robustness and compactness for straightforwardly assigning learnable parameters
to each Gaussian when encountering inconsistent 2D machine-generated labels.
Our method aims to achieve compact and reliable zero-shot scene segmentation
swiftly by mapping fused spatial and semantically meaningful features for each
Gaussian point with a shallow decoding network. Specifically, our method
firstly optimizes Gaussian points' position, convariance and color attributes
under the supervision of RGB images. After Gaussian Locating, we distill
multi-scale DINO features extracted from images through unprojection to each
Gaussian, which is then incorporated with spatial features from the fast point
features processing network, i.e. RandLA-Net. Then the shallow decoding MLP is
applied to the multi-scale fused features to obtain compact segmentation.
Experimental results show that our model can perform high-quality zero-shot
scene segmentation, as our model outperforms other segmentation methods on both
semantic and panoptic segmentation task, meanwhile consumes approximately only
10% segmenting time compared to NeRF-based segmentation. Code and more results
will be available at https://David-Dou.github.io/CoSSegGaussians

---

## GO-NeRF: Generating Virtual Objects in Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-11 | Peng Dai, Feitong Tan, Xin Yu, Yinda Zhang, Xiaojuan Qi | cs.CV | [PDF](http://arxiv.org/pdf/2401.05750v1){: .btn .btn-green } |

**Abstract**: Despite advances in 3D generation, the direct creation of 3D objects within
an existing 3D scene represented as NeRF remains underexplored. This process
requires not only high-quality 3D object generation but also seamless
composition of the generated 3D content into the existing NeRF. To this end, we
propose a new method, GO-NeRF, capable of utilizing scene context for
high-quality and harmonious 3D object generation within an existing NeRF. Our
method employs a compositional rendering formulation that allows the generated
3D objects to be seamlessly composited into the scene utilizing learned
3D-aware opacity maps without introducing unintended scene modification.
Moreover, we also develop tailored optimization objectives and training
strategies to enhance the model's ability to exploit scene context and mitigate
artifacts, such as floaters, originating from 3D object generation within a
scene. Extensive experiments on both feed-forward and $360^o$ scenes show the
superior performance of our proposed GO-NeRF in generating objects harmoniously
composited with surrounding scenes and synthesizing high-quality novel view
images. Project page at {\url{https://daipengwa.github.io/GO-NeRF/}.

Comments:
- 12 pages

---

## TriNeRFLet: A Wavelet Based Multiscale Triplane NeRF Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-11 | Rajaei Khatib, Raja Giryes | cs.CV | [PDF](http://arxiv.org/pdf/2401.06191v1){: .btn .btn-green } |

**Abstract**: In recent years, the neural radiance field (NeRF) model has gained popularity
due to its ability to recover complex 3D scenes. Following its success, many
approaches proposed different NeRF representations in order to further improve
both runtime and performance. One such example is Triplane, in which NeRF is
represented using three 2D feature planes. This enables easily using existing
2D neural networks in this framework, e.g., to generate the three planes.
Despite its advantage, the triplane representation lagged behind in its 3D
recovery quality compared to NeRF solutions. In this work, we propose
TriNeRFLet, a 2D wavelet-based multiscale triplane representation for NeRF,
which closes the 3D recovery performance gap and is competitive with current
state-of-the-art methods. Building upon the triplane framework, we also propose
a novel super-resolution (SR) technique that combines a diffusion model with
TriNeRFLet for improving NeRF resolution.

Comments:
- webpage link: https://rajaeekh.github.io/trinerflet-web

---

## Gaussian Shadow Casting for Neural Characters

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-11 | Luis Bolanos, Shih-Yang Su, Helge Rhodin | cs.CV | [PDF](http://arxiv.org/pdf/2401.06116v1){: .btn .btn-green } |

**Abstract**: Neural character models can now reconstruct detailed geometry and texture
from video, but they lack explicit shadows and shading, leading to artifacts
when generating novel views and poses or during relighting. It is particularly
difficult to include shadows as they are a global effect and the required
casting of secondary rays is costly. We propose a new shadow model using a
Gaussian density proxy that replaces sampling with a simple analytic formula.
It supports dynamic motion and is tailored for shadow computation, thereby
avoiding the affine projection approximation and sorting required by the
closely related Gaussian splatting. Combined with a deferred neural rendering
model, our Gaussian shadows enable Lambertian shading and shadow casting with
minimal overhead. We demonstrate improved reconstructions, with better
separation of albedo, shading, and shadows in challenging outdoor scenes with
direct sun light and hard shadows. Our method is able to optimize the light
direction without any input from the user. As a result, novel poses have fewer
shadow artifacts and relighting in novel scenes is more realistic compared to
the state-of-the-art methods, providing new ways to pose neural characters in
novel environments, increasing their applicability.

Comments:
- 14 pages, 13 figures

---

## TRIPS: Trilinear Point Splatting for Real-Time Radiance Field Rendering

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-11 | Linus Franke, Darius Rückert, Laura Fink, Marc Stamminger | cs.CV | [PDF](http://arxiv.org/pdf/2401.06003v1){: .btn .btn-green } |

**Abstract**: Point-based radiance field rendering has demonstrated impressive results for
novel view synthesis, offering a compelling blend of rendering quality and
computational efficiency. However, also latest approaches in this domain are
not without their shortcomings. 3D Gaussian Splatting [Kerbl and Kopanas et al.
2023] struggles when tasked with rendering highly detailed scenes, due to
blurring and cloudy artifacts. On the other hand, ADOP [R\"uckert et al. 2022]
can accommodate crisper images, but the neural reconstruction network decreases
performance, it grapples with temporal instability and it is unable to
effectively address large gaps in the point cloud.
  In this paper, we present TRIPS (Trilinear Point Splatting), an approach that
combines ideas from both Gaussian Splatting and ADOP. The fundamental concept
behind our novel technique involves rasterizing points into a screen-space
image pyramid, with the selection of the pyramid layer determined by the
projected point size. This approach allows rendering arbitrarily large points
using a single trilinear write. A lightweight neural network is then used to
reconstruct a hole-free image including detail beyond splat resolution.
Importantly, our render pipeline is entirely differentiable, allowing for
automatic optimization of both point sizes and positions.
  Our evaluation demonstrate that TRIPS surpasses existing state-of-the-art
methods in terms of rendering quality while maintaining a real-time frame rate
of 60 frames per second on readily available hardware. This performance extends
to challenging scenarios, such as scenes featuring intricate geometry,
expansive landscapes, and auto-exposed footage.

---

## Fast High Dynamic Range Radiance Fields for Dynamic Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-11 | Guanjun Wu, Taoran Yi, Jiemin Fang, Wenyu Liu, Xinggang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2401.06052v1){: .btn .btn-green } |

**Abstract**: Neural Radiances Fields (NeRF) and their extensions have shown great success
in representing 3D scenes and synthesizing novel-view images. However, most
NeRF methods take in low-dynamic-range (LDR) images, which may lose details,
especially with nonuniform illumination. Some previous NeRF methods attempt to
introduce high-dynamic-range (HDR) techniques but mainly target static scenes.
To extend HDR NeRF methods to wider applications, we propose a dynamic HDR NeRF
framework, named HDR-HexPlane, which can learn 3D scenes from dynamic 2D images
captured with various exposures. A learnable exposure mapping function is
constructed to obtain adaptive exposure values for each image. Based on the
monotonically increasing prior, a camera response function is designed for
stable learning. With the proposed model, high-quality novel-view images at any
time point can be rendered with any desired exposure. We further construct a
dataset containing multiple dynamic scenes captured with diverse exposures for
evaluation. All the datasets and code are available at
\url{https://guanjunwu.github.io/HDR-HexPlane/}.

Comments:
- 3DV 2024. Project page: https://guanjunwu.github.io/HDR-HexPlane

---

## Diffusion Priors for Dynamic View Synthesis from Monocular Videos

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-10 | Chaoyang Wang, Peiye Zhuang, Aliaksandr Siarohin, Junli Cao, Guocheng Qian, Hsin-Ying Lee, Sergey Tulyakov | cs.CV | [PDF](http://arxiv.org/pdf/2401.05583v1){: .btn .btn-green } |

**Abstract**: Dynamic novel view synthesis aims to capture the temporal evolution of visual
content within videos. Existing methods struggle to distinguishing between
motion and structure, particularly in scenarios where camera poses are either
unknown or constrained compared to object motion. Furthermore, with information
solely from reference images, it is extremely challenging to hallucinate unseen
regions that are occluded or partially observed in the given videos. To address
these issues, we first finetune a pretrained RGB-D diffusion model on the video
frames using a customization technique. Subsequently, we distill the knowledge
from the finetuned model to a 4D representations encompassing both dynamic and
static Neural Radiance Fields (NeRF) components. The proposed pipeline achieves
geometric consistency while preserving the scene identity. We perform thorough
experiments to evaluate the efficacy of the proposed method qualitatively and
quantitatively. Our results demonstrate the robustness and utility of our
approach in challenging cases, further advancing dynamic novel view synthesis.

---

## FPRF: Feed-Forward Photorealistic Style Transfer of Large-Scale 3D  Neural Radiance Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-10 | GeonU Kim, Kim Youwang, Tae-Hyun Oh | cs.CV | [PDF](http://arxiv.org/pdf/2401.05516v1){: .btn .btn-green } |

**Abstract**: We present FPRF, a feed-forward photorealistic style transfer method for
large-scale 3D neural radiance fields. FPRF stylizes large-scale 3D scenes with
arbitrary, multiple style reference images without additional optimization
while preserving multi-view appearance consistency. Prior arts required tedious
per-style/-scene optimization and were limited to small-scale 3D scenes. FPRF
efficiently stylizes large-scale 3D scenes by introducing a style-decomposed 3D
neural radiance field, which inherits AdaIN's feed-forward stylization
machinery, supporting arbitrary style reference images. Furthermore, FPRF
supports multi-reference stylization with the semantic correspondence matching
and local AdaIN, which adds diverse user control for 3D scene styles. FPRF also
preserves multi-view consistency by applying semantic matching and style
transfer processes directly onto queried features in 3D space. In experiments,
we demonstrate that FPRF achieves favorable photorealistic quality 3D scene
stylization for large-scale scenes with diverse reference images. Project page:
https://kim-geonu.github.io/FPRF/

Comments:
- Project page: https://kim-geonu.github.io/FPRF/

---

## InseRF: Text-Driven Generative Object Insertion in Neural 3D Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-10 | Mohamad Shahbazi, Liesbeth Claessens, Michael Niemeyer, Edo Collins, Alessio Tonioni, Luc Van Gool, Federico Tombari | cs.CV | [PDF](http://arxiv.org/pdf/2401.05335v1){: .btn .btn-green } |

**Abstract**: We introduce InseRF, a novel method for generative object insertion in the
NeRF reconstructions of 3D scenes. Based on a user-provided textual description
and a 2D bounding box in a reference viewpoint, InseRF generates new objects in
3D scenes. Recently, methods for 3D scene editing have been profoundly
transformed, owing to the use of strong priors of text-to-image diffusion
models in 3D generative modeling. Existing methods are mostly effective in
editing 3D scenes via style and appearance changes or removing existing
objects. Generating new objects, however, remains a challenge for such methods,
which we address in this study. Specifically, we propose grounding the 3D
object insertion to a 2D object insertion in a reference view of the scene. The
2D edit is then lifted to 3D using a single-view object reconstruction method.
The reconstructed object is then inserted into the scene, guided by the priors
of monocular depth estimation methods. We evaluate our method on various 3D
scenes and provide an in-depth analysis of the proposed components. Our
experiments with generative insertion of objects in several 3D scenes indicate
the effectiveness of our method compared to the existing methods. InseRF is
capable of controllable and 3D-consistent object insertion without requiring
explicit 3D information as input. Please visit our project page at
https://mohamad-shahbazi.github.io/inserf.

---

## CTNeRF: Cross-Time Transformer for Dynamic Neural Radiance Field from  Monocular Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-10 | Xingyu Miao, Yang Bai, Haoran Duan, Yawen Huang, Fan Wan, Yang Long, Yefeng Zheng | cs.CV | [PDF](http://arxiv.org/pdf/2401.04861v1){: .btn .btn-green } |

**Abstract**: The goal of our work is to generate high-quality novel views from monocular
videos of complex and dynamic scenes. Prior methods, such as DynamicNeRF, have
shown impressive performance by leveraging time-varying dynamic radiation
fields. However, these methods have limitations when it comes to accurately
modeling the motion of complex objects, which can lead to inaccurate and blurry
renderings of details. To address this limitation, we propose a novel approach
that builds upon a recent generalization NeRF, which aggregates nearby views
onto new viewpoints. However, such methods are typically only effective for
static scenes. To overcome this challenge, we introduce a module that operates
in both the time and frequency domains to aggregate the features of object
motion. This allows us to learn the relationship between frames and generate
higher-quality images. Our experiments demonstrate significant improvements
over state-of-the-art methods on dynamic scene datasets. Specifically, our
approach outperforms existing methods in terms of both the accuracy and visual
quality of the synthesized views.

---

## AGG: Amortized Generative 3D Gaussians for Single Image to 3D

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-08 | Dejia Xu, Ye Yuan, Morteza Mardani, Sifei Liu, Jiaming Song, Zhangyang Wang, Arash Vahdat | cs.CV | [PDF](http://arxiv.org/pdf/2401.04099v1){: .btn .btn-green } |

**Abstract**: Given the growing need for automatic 3D content creation pipelines, various
3D representations have been studied to generate 3D objects from a single
image. Due to its superior rendering efficiency, 3D Gaussian splatting-based
models have recently excelled in both 3D reconstruction and generation. 3D
Gaussian splatting approaches for image to 3D generation are often
optimization-based, requiring many computationally expensive score-distillation
steps. To overcome these challenges, we introduce an Amortized Generative 3D
Gaussian framework (AGG) that instantly produces 3D Gaussians from a single
image, eliminating the need for per-instance optimization. Utilizing an
intermediate hybrid representation, AGG decomposes the generation of 3D
Gaussian locations and other appearance attributes for joint optimization.
Moreover, we propose a cascaded pipeline that first generates a coarse
representation of the 3D data and later upsamples it with a 3D Gaussian
super-resolution module. Our method is evaluated against existing
optimization-based 3D Gaussian frameworks and sampling-based pipelines
utilizing other 3D representations, where AGG showcases competitive generation
abilities both qualitatively and quantitatively while being several orders of
magnitude faster. Project page: https://ir1d.github.io/AGG/

Comments:
- Project page: https://ir1d.github.io/AGG/

---

## A Survey on 3D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-08 | Guikun Chen, Wenguan Wang | cs.CV | [PDF](http://arxiv.org/pdf/2401.03890v1){: .btn .btn-green } |

**Abstract**: 3D Gaussian splatting (3D GS) has recently emerged as a transformative
technique in the explicit radiance field and computer graphics landscape. This
innovative approach, characterized by the utilization of millions of 3D
Gaussians, represents a significant departure from the neural radiance field
(NeRF) methodologies, which predominantly use implicit, coordinate-based models
to map spatial coordinates to pixel values. 3D GS, with its explicit scene
representations and differentiable rendering algorithms, not only promises
real-time rendering capabilities but also introduces unprecedented levels of
control and editability. This positions 3D GS as a potential game-changer for
the next generation of 3D reconstruction and representation. In the present
paper, we provide the first systematic overview of the recent developments and
critical contributions in the domain of 3D GS. We begin with a detailed
exploration of the underlying principles and the driving forces behind the
advent of 3D GS, setting the stage for understanding its significance. A focal
point of our discussion is the practical applicability of 3D GS. By
facilitating real-time performance, 3D GS opens up a plethora of applications,
ranging from virtual reality to interactive media and beyond. This is
complemented by a comparative analysis of leading 3D GS models, evaluated
across various benchmark tasks to highlight their performance and practical
utility. The survey concludes by identifying current challenges and suggesting
potential avenues for future research in this domain. Through this survey, we
aim to provide a valuable resource for both newcomers and seasoned researchers,
fostering further exploration and advancement in applicable and explicit
radiance field representation.

Comments:
- Ongoing project

---

## NeRFmentation: NeRF-based Augmentation for Monocular Depth Estimation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-08 | Casimir Feldmann, Niall Siegenheim, Nikolas Hars, Lovro Rabuzin, Mert Ertugrul, Luca Wolfart, Marc Pollefeys, Zuria Bauer, Martin R. Oswald | cs.CV | [PDF](http://arxiv.org/pdf/2401.03771v1){: .btn .btn-green } |

**Abstract**: The capabilities of monocular depth estimation (MDE) models are limited by
the availability of sufficient and diverse datasets. In the case of MDE models
for autonomous driving, this issue is exacerbated by the linearity of the
captured data trajectories. We propose a NeRF-based data augmentation pipeline
to introduce synthetic data with more diverse viewing directions into training
datasets and demonstrate the benefits of our approach to model performance and
robustness. Our data augmentation pipeline, which we call "NeRFmentation",
trains NeRFs on each scene in the dataset, filters out subpar NeRFs based on
relevant metrics, and uses them to generate synthetic RGB-D images captured
from new viewing directions. In this work, we apply our technique in
conjunction with three state-of-the-art MDE architectures on the popular
autonomous driving dataset KITTI, augmenting its training set of the Eigen
split. We evaluate the resulting performance gain on the original test set, a
separate popular driving set, and our own synthetic test set.

---

## RustNeRF: Robust Neural Radiance Field with Low-Quality Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-06 | Mengfei Li, Ming Lu, Xiaofang Li, Shanghang Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2401.03257v1){: .btn .btn-green } |

**Abstract**: Recent work on Neural Radiance Fields (NeRF) exploits multi-view 3D
consistency, achieving impressive results in 3D scene modeling and
high-fidelity novel-view synthesis. However, there are limitations. First,
existing methods assume enough high-quality images are available for training
the NeRF model, ignoring real-world image degradation. Second, previous methods
struggle with ambiguity in the training set due to unmodeled inconsistencies
among different views. In this work, we present RustNeRF for real-world
high-quality NeRF. To improve NeRF's robustness under real-world inputs, we
train a 3D-aware preprocessing network that incorporates real-world degradation
modeling. We propose a novel implicit multi-view guidance to address
information loss during image degradation and restoration. Extensive
experiments demonstrate RustNeRF's advantages over existing approaches under
real-world degradation. The code will be released.

---

## Hi-Map: Hierarchical Factorized Radiance Field for High-Fidelity  Monocular Dense Mapping

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-06 | Tongyan Hua, Haotian Bai, Zidong Cao, Ming Liu, Dacheng Tao, Lin Wang | cs.CV | [PDF](http://arxiv.org/pdf/2401.03203v1){: .btn .btn-green } |

**Abstract**: In this paper, we introduce Hi-Map, a novel monocular dense mapping approach
based on Neural Radiance Field (NeRF). Hi-Map is exceptional in its capacity to
achieve efficient and high-fidelity mapping using only posed RGB inputs. Our
method eliminates the need for external depth priors derived from e.g., a depth
estimation model. Our key idea is to represent the scene as a hierarchical
feature grid that encodes the radiance and then factorizes it into feature
planes and vectors. As such, the scene representation becomes simpler and more
generalizable for fast and smooth convergence on new observations. This allows
for efficient computation while alleviating noise patterns by reducing the
complexity of the scene representation. Buttressed by the hierarchical
factorized representation, we leverage the Sign Distance Field (SDF) as a proxy
of rendering for inferring the volume density, demonstrating high mapping
fidelity. Moreover, we introduce a dual-path encoding strategy to strengthen
the photometric cues and further boost the mapping quality, especially for the
distant and textureless regions. Extensive experiments demonstrate our method's
superiority in geometric and textural accuracy over the state-of-the-art
NeRF-based monocular mapping methods.

---

## Progress and Prospects in 3D Generative AI: A Technical Overview  including 3D human

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-05 | Song Bai, Jie Li | cs.AI | [PDF](http://arxiv.org/pdf/2401.02620v1){: .btn .btn-green } |

**Abstract**: While AI-generated text and 2D images continue to expand its territory, 3D
generation has gradually emerged as a trend that cannot be ignored. Since the
year 2023 an abundant amount of research papers has emerged in the domain of 3D
generation. This growth encompasses not just the creation of 3D objects, but
also the rapid development of 3D character and motion generation. Several key
factors contribute to this progress. The enhanced fidelity in stable diffusion,
coupled with control methods that ensure multi-view consistency, and realistic
human models like SMPL-X, contribute synergistically to the production of 3D
models with remarkable consistency and near-realistic appearances. The
advancements in neural network-based 3D storing and rendering models, such as
Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have
accelerated the efficiency and realism of neural rendered models. Furthermore,
the multimodality capabilities of large language models have enabled language
inputs to transcend into human motion outputs. This paper aims to provide a
comprehensive overview and summary of the relevant papers published mostly
during the latter half year of 2023. It will begin by discussing the AI
generated object models in 3D, followed by the generated 3D human models, and
finally, the generated 3D human motions, culminating in a conclusive summary
and a vision for the future.

---

## FED-NeRF: Achieve High 3D Consistency and Temporal Coherence for Face  Video Editing on Dynamic NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-05 | Hao Zhang, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2401.02616v1){: .btn .btn-green } |

**Abstract**: The success of the GAN-NeRF structure has enabled face editing on NeRF to
maintain 3D view consistency. However, achieving simultaneously multi-view
consistency and temporal coherence while editing video sequences remains a
formidable challenge. This paper proposes a novel face video editing
architecture built upon the dynamic face GAN-NeRF structure, which effectively
utilizes video sequences to restore the latent code and 3D face geometry. By
editing the latent code, multi-view consistent editing on the face can be
ensured, as validated by multiview stereo reconstruction on the resulting
edited images in our dynamic NeRF. As the estimation of face geometries occurs
on a frame-by-frame basis, this may introduce a jittering issue. We propose a
stabilizer that maintains temporal coherence by preserving smooth changes of
face expressions in consecutive frames. Quantitative and qualitative analyses
reveal that our method, as the pioneering 4D face video editor, achieves
state-of-the-art performance in comparison to existing 2D or 3D-based
approaches independently addressing identity and motion. Codes will be
released.

Comments:
- Our code will be available at: https://github.com/ZHANG1023/FED-NeRF

---

## Characterizing Satellite Geometry via Accelerated 3D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-05 | Van Minh Nguyen, Emma Sandidge, Trupti Mahendrakar, Ryan T. White | cs.CV | [PDF](http://arxiv.org/pdf/2401.02588v1){: .btn .btn-green } |

**Abstract**: The accelerating deployment of spacecraft in orbit have generated interest in
on-orbit servicing (OOS), inspection of spacecraft, and active debris removal
(ADR). Such missions require precise rendezvous and proximity operations in the
vicinity of non-cooperative, possible unknown, resident space objects. Safety
concerns with manned missions and lag times with ground-based control
necessitate complete autonomy. This requires robust characterization of the
target's geometry. In this article, we present an approach for mapping
geometries of satellites on orbit based on 3D Gaussian Splatting that can run
on computing resources available on current spaceflight hardware. We
demonstrate model training and 3D rendering performance on a
hardware-in-the-loop satellite mock-up under several realistic lighting and
motion conditions. Our model is shown to be capable of training on-board and
rendering higher quality novel views of an unknown satellite nearly 2 orders of
magnitude faster than previous NeRF-based algorithms. Such on-board
capabilities are critical to enable downstream machine intelligence tasks
necessary for autonomous guidance, navigation, and control tasks.

Comments:
- 11 pages, 5 figures

---

## PEGASUS: Physically Enhanced Gaussian Splatting Simulation System for  6DOF Object Pose Dataset Generation

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-04 | Lukas Meyer, Floris Erich, Yusuke Yoshiyasu, Marc Stamminger, Noriaki Ando, Yukiyasu Domae | cs.CV | [PDF](http://arxiv.org/pdf/2401.02281v1){: .btn .btn-green } |

**Abstract**: We introduce Physically Enhanced Gaussian Splatting Simulation System
(PEGASUS) for 6DOF object pose dataset generation, a versatile dataset
generator based on 3D Gaussian Splatting. Environment and object
representations can be easily obtained using commodity cameras to reconstruct
with Gaussian Splatting. PEGASUS allows the composition of new scenes by
merging the respective underlying Gaussian Splatting point cloud of an
environment with one or multiple objects. Leveraging a physics engine enables
the simulation of natural object placement within a scene through interaction
between meshes extracted for the objects and the environment. Consequently, an
extensive amount of new scenes - static or dynamic - can be created by
combining different environments and objects. By rendering scenes from various
perspectives, diverse data points such as RGB images, depth maps, semantic
masks, and 6DoF object poses can be extracted. Our study demonstrates that
training on data generated by PEGASUS enables pose estimation networks to
successfully transfer from synthetic data to real-world data. Moreover, we
introduce the Ramen dataset, comprising 30 Japanese cup noodle items. This
dataset includes spherical scans that captures images from both object
hemisphere and the Gaussian Splatting reconstruction, making them compatible
with PEGASUS.

Comments:
- Project Page: https://meyerls.github.io/pegasus_web

---

## SIGNeRF: Scene Integrated Generation for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-03 | Jan-Niklas Dihlmann, Andreas Engelhardt, Hendrik Lensch | cs.CV | [PDF](http://arxiv.org/pdf/2401.01647v1){: .btn .btn-green } |

**Abstract**: Advances in image diffusion models have recently led to notable improvements
in the generation of high-quality images. In combination with Neural Radiance
Fields (NeRFs), they enabled new opportunities in 3D generation. However, most
generative 3D approaches are object-centric and applying them to editing
existing photorealistic scenes is not trivial. We propose SIGNeRF, a novel
approach for fast and controllable NeRF scene editing and scene-integrated
object generation. A new generative update strategy ensures 3D consistency
across the edited images, without requiring iterative optimization. We find
that depth-conditioned diffusion models inherently possess the capability to
generate 3D consistent views by requesting a grid of images instead of single
views. Based on these insights, we introduce a multi-view reference sheet of
modified images. Our method updates an image collection consistently based on
the reference sheet and refines the original NeRF with the newly generated
image set in one go. By exploiting the depth conditioning mechanism of the
image diffusion model, we gain fine control over the spatial location of the
edit and enforce shape guidance by a selected region or an external mesh.

Comments:
- Project Page: https://signerf.jdihlmann.com

---

## FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D  Scene Understanding

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-03 | Xingxing Zuo, Pouya Samangouei, Yunwen Zhou, Yan Di, Mingyang Li | cs.CV | [PDF](http://arxiv.org/pdf/2401.01970v1){: .btn .btn-green } |

**Abstract**: Precisely perceiving the geometric and semantic properties of real-world 3D
objects is crucial for the continued evolution of augmented reality and robotic
applications. To this end, we present \algfull{} (\algname{}), which
incorporates vision-language embeddings of foundation models into 3D Gaussian
Splatting (GS). The key contribution of this work is an efficient method to
reconstruct and represent 3D vision-language models. This is achieved by
distilling feature maps generated from image-based foundation models into those
rendered from our 3D model. To ensure high-quality rendering and fast training,
we introduce a novel scene representation by integrating strengths from both GS
and multi-resolution hash encodings (MHE). Our effective training procedure
also introduces a pixel alignment loss that makes the rendered feature distance
of same semantic entities close, following the pixel-level semantic boundaries.
Our results demonstrate remarkable multi-view semantic consistency,
facilitating diverse downstream tasks, beating state-of-the-art methods by
$\mathbf{10.2}$ percent on open-vocabulary language-based object detection,
despite that we are $\mathbf{851\times}$ faster for inference. This research
explores the intersection of vision, language, and 3D scene representation,
paving the way for enhanced scene understanding in uncontrolled real-world
environments. We plan to release the code upon paper acceptance.

Comments:
- 19 pages, Project page coming soon

---

## Street Gaussians for Modeling Dynamic Urban Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-02 | Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, Sida Peng | cs.CV | [PDF](http://arxiv.org/pdf/2401.01339v1){: .btn .btn-green } |

**Abstract**: This paper aims to tackle the problem of modeling dynamic urban street scenes
from monocular videos. Recent methods extend NeRF by incorporating tracked
vehicle poses to animate vehicles, enabling photo-realistic view synthesis of
dynamic urban street scenes. However, significant limitations are their slow
training and rendering speed, coupled with the critical need for high precision
in tracked vehicle poses. We introduce Street Gaussians, a new explicit scene
representation that tackles all these limitations. Specifically, the dynamic
urban street is represented as a set of point clouds equipped with semantic
logits and 3D Gaussians, each associated with either a foreground vehicle or
the background. To model the dynamics of foreground object vehicles, each
object point cloud is optimized with optimizable tracked poses, along with a
dynamic spherical harmonics model for the dynamic appearance. The explicit
representation allows easy composition of object vehicles and background, which
in turn allows for scene editing operations and rendering at 133 FPS
(1066$\times$1600 resolution) within half an hour of training. The proposed
method is evaluated on multiple challenging benchmarks, including KITTI and
Waymo Open datasets. Experiments show that the proposed method consistently
outperforms state-of-the-art methods across all datasets. Furthermore, the
proposed representation delivers performance on par with that achieved using
precise ground-truth poses, despite relying only on poses from an off-the-shelf
tracker. The code is available at https://zju3dv.github.io/street_gaussians/.

Comments:
- Project page: https://zju3dv.github.io/street_gaussians/

---

## Noise-NeRF: Hide Information in Neural Radiance Fields using Trainable  Noise

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-02 | Qinglong Huang, Yong Liao, Yanbin Hao, Pengyuan Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2401.01216v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) have been proposed as an innovative 3D
representation method. While attracting lots of attention, NeRF faces critical
issues such as information confidentiality and security. Steganography is a
technique used to embed information in another object as a means of protecting
information security. Currently, there are few related studies on NeRF
steganography, facing challenges in low steganography quality, model weight
damage, and a limited amount of steganographic information. This paper proposes
a novel NeRF steganography method based on trainable noise: Noise-NeRF.
Furthermore, we propose the Adaptive Pixel Selection strategy and Pixel
Perturbation strategy to improve the steganography quality and efficiency. The
extensive experiments on open-source datasets show that Noise-NeRF provides
state-of-the-art performances in both steganography quality and rendering
quality, as well as effectiveness in super-resolution image steganography.

---

## 3D Visibility-aware Generalizable Neural Radiance Fields for Interacting  Hands

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-02 | Xuan Huang, Hanhui Li, Zejun Yang, Zhisheng Wang, Xiaodan Liang | cs.CV | [PDF](http://arxiv.org/pdf/2401.00979v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) are promising 3D representations for scenes,
objects, and humans. However, most existing methods require multi-view inputs
and per-scene training, which limits their real-life applications. Moreover,
current methods focus on single-subject cases, leaving scenes of interacting
hands that involve severe inter-hand occlusions and challenging view variations
remain unsolved. To tackle these issues, this paper proposes a generalizable
visibility-aware NeRF (VA-NeRF) framework for interacting hands. Specifically,
given an image of interacting hands as input, our VA-NeRF first obtains a
mesh-based representation of hands and extracts their corresponding geometric
and textural features. Subsequently, a feature fusion module that exploits the
visibility of query points and mesh vertices is introduced to adaptively merge
features of both hands, enabling the recovery of features in unseen areas.
Additionally, our VA-NeRF is optimized together with a novel discriminator
within an adversarial learning paradigm. In contrast to conventional
discriminators that predict a single real/fake label for the synthesized image,
the proposed discriminator generates a pixel-wise visibility map, providing
fine-grained supervision for unseen areas and encouraging the VA-NeRF to
improve the visual quality of synthesized images. Experiments on the
Interhand2.6M dataset demonstrate that our proposed VA-NeRF outperforms
conventional NeRFs significantly. Project Page:
\url{https://github.com/XuanHuang0/VANeRF}.

Comments:
- Accepted by AAAI-24

---

## Deblurring 3D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-01 | Byeonghyeon Lee, Howoong Lee, Xiangyu Sun, Usman Ali, Eunbyung Park | cs.CV | [PDF](http://arxiv.org/pdf/2401.00834v1){: .btn .btn-green } |

**Abstract**: Recent studies in Radiance Fields have paved the robust way for novel view
synthesis with their photorealistic rendering quality. Nevertheless, they
usually employ neural networks and volumetric rendering, which are costly to
train and impede their broad use in various real-time applications due to the
lengthy rendering time. Lately 3D Gaussians splatting-based approach has been
proposed to model the 3D scene, and it achieves remarkable visual quality while
rendering the images in real-time. However, it suffers from severe degradation
in the rendering quality if the training images are blurry. Blurriness commonly
occurs due to the lens defocusing, object motion, and camera shake, and it
inevitably intervenes in clean image acquisition. Several previous studies have
attempted to render clean and sharp images from blurry input images using
neural fields. The majority of those works, however, are designed only for
volumetric rendering-based neural radiance fields and are not straightforwardly
applicable to rasterization-based 3D Gaussian splatting methods. Thus, we
propose a novel real-time deblurring framework, deblurring 3D Gaussian
Splatting, using a small Multi-Layer Perceptron (MLP) that manipulates the
covariance of each 3D Gaussian to model the scene blurriness. While deblurring
3D Gaussian Splatting can still enjoy real-time rendering, it can reconstruct
fine and sharp details from blurry images. A variety of experiments have been
conducted on the benchmark, and the results have revealed the effectiveness of
our approach for deblurring. Qualitative results are available at
https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/

Comments:
- 19 pages, 8 figures

---

## Sharp-NeRF: Grid-based Fast Deblurring Neural Radiance Fields Using  Sharpness Prior

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-01 | Byeonghyeon Lee, Howoong Lee, Usman Ali, Eunbyung Park | cs.CV | [PDF](http://arxiv.org/pdf/2401.00825v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have shown remarkable performance in neural
rendering-based novel view synthesis. However, NeRF suffers from severe visual
quality degradation when the input images have been captured under imperfect
conditions, such as poor illumination, defocus blurring, and lens aberrations.
Especially, defocus blur is quite common in the images when they are normally
captured using cameras. Although few recent studies have proposed to render
sharp images of considerably high-quality, yet they still face many key
challenges. In particular, those methods have employed a Multi-Layer Perceptron
(MLP) based NeRF, which requires tremendous computational time. To overcome
these shortcomings, this paper proposes a novel technique Sharp-NeRF -- a
grid-based NeRF that renders clean and sharp images from the input blurry
images within half an hour of training. To do so, we used several grid-based
kernels to accurately model the sharpness/blurriness of the scene. The
sharpness level of the pixels is computed to learn the spatially varying blur
kernels. We have conducted experiments on the benchmarks consisting of blurry
images and have evaluated full-reference and non-reference metrics. The
qualitative and quantitative results have revealed that our approach renders
the sharp novel views with vivid colors and fine details, and it has
considerably faster training time than the previous works. Our project page is
available at https://benhenryl.github.io/SharpNeRF/

Comments:
- Accepted to WACV 2024

---

## GD^2-NeRF: Generative Detail Compensation via GAN and Diffusion for  One-shot Generalizable Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2024-01-01 | Xiao Pan, Zongxin Yang, Shuai Bai, Yi Yang | cs.CV | [PDF](http://arxiv.org/pdf/2401.00616v2){: .btn .btn-green } |

**Abstract**: In this paper, we focus on the One-shot Novel View Synthesis (O-NVS) task
which targets synthesizing photo-realistic novel views given only one reference
image per scene. Previous One-shot Generalizable Neural Radiance Fields
(OG-NeRF) methods solve this task in an inference-time finetuning-free manner,
yet suffer the blurry issue due to the encoder-only architecture that highly
relies on the limited reference image. On the other hand, recent
diffusion-based image-to-3d methods show vivid plausible results via distilling
pre-trained 2D diffusion models into a 3D representation, yet require tedious
per-scene optimization. Targeting these issues, we propose the GD$^2$-NeRF, a
Generative Detail compensation framework via GAN and Diffusion that is both
inference-time finetuning-free and with vivid plausible details. In detail,
following a coarse-to-fine strategy, GD$^2$-NeRF is mainly composed of a
One-stage Parallel Pipeline (OPP) and a 3D-consistent Detail Enhancer
(Diff3DE). At the coarse stage, OPP first efficiently inserts the GAN model
into the existing OG-NeRF pipeline for primarily relieving the blurry issue
with in-distribution priors captured from the training dataset, achieving a
good balance between sharpness (LPIPS, FID) and fidelity (PSNR, SSIM). Then, at
the fine stage, Diff3DE further leverages the pre-trained image diffusion
models to complement rich out-distribution details while maintaining decent 3D
consistency. Extensive experiments on both the synthetic and real-world
datasets show that GD$^2$-NeRF noticeably improves the details while without
per-scene finetuning.

Comments:
- Reading with Macbook Preview is recommended for best quality;
  Submitted to Journal