---
layout: default
title: December
parent: 2023
nav_order: 12
---
<!---metadata--->

## Inpaint4DNeRF: Promptable Spatio-Temporal NeRF Inpainting with  Generative Diffusion Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-30 | Han Jiang, Haosen Sun, Ruoxuan Li, Chi-Keung Tang, Yu-Wing Tai | cs.CV | [PDF](http://arxiv.org/pdf/2401.00208v1){: .btn .btn-green } |

**Abstract**: Current Neural Radiance Fields (NeRF) can generate photorealistic novel
views. For editing 3D scenes represented by NeRF, with the advent of generative
models, this paper proposes Inpaint4DNeRF to capitalize on state-of-the-art
stable diffusion models (e.g., ControlNet) for direct generation of the
underlying completed background content, regardless of static or dynamic. The
key advantages of this generative approach for NeRF inpainting are twofold.
First, after rough mask propagation, to complete or fill in previously occluded
content, we can individually generate a small subset of completed images with
plausible content, called seed images, from which simple 3D geometry proxies
can be derived. Second and the remaining problem is thus 3D multiview
consistency among all completed images, now guided by the seed images and their
3D proxies. Without other bells and whistles, our generative Inpaint4DNeRF
baseline framework is general which can be readily extended to 4D dynamic
NeRFs, where temporal consistency can be naturally handled in a similar way as
our multiview consistency.

---

## PlanarNeRF: Online Learning of Planar Primitives with Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-30 | Zheng Chen, Qingan Yan, Huangying Zhan, Changjiang Cai, Xiangyu Xu, Yuzhong Huang, Weihan Wang, Ziyue Feng, Lantao Liu, Yi Xu | cs.CV | [PDF](http://arxiv.org/pdf/2401.00871v1){: .btn .btn-green } |

**Abstract**: Identifying spatially complete planar primitives from visual data is a
crucial task in computer vision. Prior methods are largely restricted to either
2D segment recovery or simplifying 3D structures, even with extensive plane
annotations. We present PlanarNeRF, a novel framework capable of detecting
dense 3D planes through online learning. Drawing upon the neural field
representation, PlanarNeRF brings three major contributions. First, it enhances
3D plane detection with concurrent appearance and geometry knowledge. Second, a
lightweight plane fitting module is proposed to estimate plane parameters.
Third, a novel global memory bank structure with an update mechanism is
introduced, ensuring consistent cross-frame correspondence. The flexible
architecture of PlanarNeRF allows it to function in both 2D-supervised and
self-supervised solutions, in each of which it can effectively learn from
sparse training signals, significantly improving training efficiency. Through
extensive experiments, we demonstrate the effectiveness of PlanarNeRF in
various scenarios and remarkable improvement over existing works.

---

## Informative Rays Selection for Few-Shot Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-29 | Marco Orsingher, Anthony Dell'Eva, Paolo Zani, Paolo Medici, Massimo Bertozzi | cs.CV | [PDF](http://arxiv.org/pdf/2312.17561v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have recently emerged as a powerful method for
image-based 3D reconstruction, but the lengthy per-scene optimization limits
their practical usage, especially in resource-constrained settings. Existing
approaches solve this issue by reducing the number of input views and
regularizing the learned volumetric representation with either complex losses
or additional inputs from other modalities. In this paper, we present KeyNeRF,
a simple yet effective method for training NeRF in few-shot scenarios by
focusing on key informative rays. Such rays are first selected at camera level
by a view selection algorithm that promotes baseline diversity while
guaranteeing scene coverage, then at pixel level by sampling from a probability
distribution based on local image entropy. Our approach performs favorably
against state-of-the-art methods, while requiring minimal changes to existing
NeRF codebases.

Comments:
- To appear at VISAPP 2024

---

## DreamGaussian4D: Generative 4D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-28 | Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, Ziwei Liu | cs.CV | [PDF](http://arxiv.org/pdf/2312.17142v2){: .btn .btn-green } |

**Abstract**: Remarkable progress has been made in 4D content generation recently. However,
existing methods suffer from long optimization time, lack of motion
controllability, and a low level of detail. In this paper, we introduce
DreamGaussian4D, an efficient 4D generation framework that builds on 4D
Gaussian Splatting representation. Our key insight is that the explicit
modeling of spatial transformations in Gaussian Splatting makes it more
suitable for the 4D generation setting compared with implicit representations.
DreamGaussian4D reduces the optimization time from several hours to just a few
minutes, allows flexible control of the generated 3D motion, and produces
animated meshes that can be efficiently rendered in 3D engines.

Comments:
- Technical report. Project page is at
  https://jiawei-ren.github.io/projects/dreamgaussian4d Code is at
  https://github.com/jiawei-ren/dreamgaussian4d

---

## City-on-Web: Real-time Neural Rendering of Large-scale Scenes on the Web

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-27 | Kaiwen Song, Juyong Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2312.16457v1){: .btn .btn-green } |

**Abstract**: NeRF has significantly advanced 3D scene reconstruction, capturing intricate
details across various environments. Existing methods have successfully
leveraged radiance field baking to facilitate real-time rendering of small
scenes. However, when applied to large-scale scenes, these techniques encounter
significant challenges, struggling to provide a seamless real-time experience
due to limited resources in computation, memory, and bandwidth. In this paper,
we propose City-on-Web, which represents the whole scene by partitioning it
into manageable blocks, each with its own Level-of-Detail, ensuring high
fidelity, efficient memory management and fast rendering. Meanwhile, we
carefully design the training and inference process such that the final
rendering result on web is consistent with training. Thanks to our novel
representation and carefully designed training/inference process, we are the
first to achieve real-time rendering of large-scale scenes in
resource-constrained environments. Extensive experimental results demonstrate
that our method facilitates real-time rendering of large-scale scenes on a web
platform, achieving 32FPS at 1080P resolution with an RTX 3060 GPU, while
simultaneously achieving a quality that closely rivals that of state-of-the-art
methods. Project page: https://ustc3dv.github.io/City-on-Web/

Comments:
- Project page: https://ustc3dv.github.io/City-on-Web/

---

## DL3DV-10K: A Large-Scale Scene Dataset for Deep Learning-based 3D Vision

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-26 | Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, Xuanmao Li, Xingpeng Sun, Rohan Ashok, Aniruddha Mukherjee, Hao Kang, Xiangrui Kong, Gang Hua, Tianyi Zhang, Bedrich Benes, Aniket Bera | cs.CV | [PDF](http://arxiv.org/pdf/2312.16256v2){: .btn .btn-green } |

**Abstract**: We have witnessed significant progress in deep learning-based 3D vision,
ranging from neural radiance field (NeRF) based 3D representation learning to
applications in novel view synthesis (NVS). However, existing scene-level
datasets for deep learning-based 3D vision, limited to either synthetic
environments or a narrow selection of real-world scenes, are quite
insufficient. This insufficiency not only hinders a comprehensive benchmark of
existing methods but also caps what could be explored in deep learning-based 3D
analysis. To address this critical gap, we present DL3DV-10K, a large-scale
scene dataset, featuring 51.2 million frames from 10,510 videos captured from
65 types of point-of-interest (POI) locations, covering both bounded and
unbounded scenes, with different levels of reflection, transparency, and
lighting. We conducted a comprehensive benchmark of recent NVS methods on
DL3DV-10K, which revealed valuable insights for future research in NVS. In
addition, we have obtained encouraging results in a pilot study to learn
generalizable NeRF from DL3DV-10K, which manifests the necessity of a
large-scale scene-level dataset to forge a path toward a foundation model for
learning 3D representation. Our DL3DV-10K dataset, benchmark results, and
models will be publicly accessible at https://dl3dv-10k.github.io/DL3DV-10K/.

---

## Pano-NeRF: Synthesizing High Dynamic Range Novel Views with Geometry  from Sparse Low Dynamic Range Panoramic Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-26 | Zhan Lu, Qian Zheng, Boxin Shi, Xudong Jiang | cs.CV | [PDF](http://arxiv.org/pdf/2312.15942v1){: .btn .btn-green } |

**Abstract**: Panoramic imaging research on geometry recovery and High Dynamic Range (HDR)
reconstruction becomes a trend with the development of Extended Reality (XR).
Neural Radiance Fields (NeRF) provide a promising scene representation for both
tasks without requiring extensive prior data. However, in the case of inputting
sparse Low Dynamic Range (LDR) panoramic images, NeRF often degrades with
under-constrained geometry and is unable to reconstruct HDR radiance from LDR
inputs. We observe that the radiance from each pixel in panoramic images can be
modeled as both a signal to convey scene lighting information and a light
source to illuminate other pixels. Hence, we propose the irradiance fields from
sparse LDR panoramic images, which increases the observation counts for
faithful geometry recovery and leverages the irradiance-radiance attenuation
for HDR reconstruction. Extensive experiments demonstrate that the irradiance
fields outperform state-of-the-art methods on both geometry recovery and HDR
reconstruction and validate their effectiveness. Furthermore, we show a
promising byproduct of spatially-varying lighting estimation. The code is
available at https://github.com/Lu-Zhan/Pano-NeRF.

---

## LangSplat: 3D Language Gaussian Splatting

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-26 | Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, Hanspeter Pfister | cs.CV | [PDF](http://arxiv.org/pdf/2312.16084v1){: .btn .btn-green } |

**Abstract**: Human lives in a 3D world and commonly uses natural language to interact with
a 3D scene. Modeling a 3D language field to support open-ended language queries
in 3D has gained increasing attention recently. This paper introduces
LangSplat, which constructs a 3D language field that enables precise and
efficient open-vocabulary querying within 3D spaces. Unlike existing methods
that ground CLIP language embeddings in a NeRF model, LangSplat advances the
field by utilizing a collection of 3D Gaussians, each encoding language
features distilled from CLIP, to represent the language field. By employing a
tile-based splatting technique for rendering language features, we circumvent
the costly rendering process inherent in NeRF. Instead of directly learning
CLIP embeddings, LangSplat first trains a scene-wise language autoencoder and
then learns language features on the scene-specific latent space, thereby
alleviating substantial memory demands imposed by explicit modeling. Existing
methods struggle with imprecise and vague 3D language fields, which fail to
discern clear boundaries between objects. We delve into this issue and propose
to learn hierarchical semantics using SAM, thereby eliminating the need for
extensively querying the language field across various scales and the
regularization of DINO features. Extensive experiments on open-vocabulary 3D
object localization and semantic segmentation demonstrate that LangSplat
significantly outperforms the previous state-of-the-art method LERF by a large
margin. Notably, LangSplat is extremely efficient, achieving a {\speed}
$\times$ speedup compared to LERF at the resolution of 1440 $\times$ 1080. We
strongly recommend readers to check out our video results at
https://langsplat.github.io

Comments:
- Project Page: https://langsplat.github.io

---

## 2D-Guided 3D Gaussian Segmentation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-26 | Kun Lan, Haoran Li, Haolin Shi, Wenjun Wu, Yong Liao, Lin Wang, Pengyuan Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2312.16047v1){: .btn .btn-green } |

**Abstract**: Recently, 3D Gaussian, as an explicit 3D representation method, has
demonstrated strong competitiveness over NeRF (Neural Radiance Fields) in terms
of expressing complex scenes and training duration. These advantages signal a
wide range of applications for 3D Gaussians in 3D understanding and editing.
Meanwhile, the segmentation of 3D Gaussians is still in its infancy. The
existing segmentation methods are not only cumbersome but also incapable of
segmenting multiple objects simultaneously in a short amount of time. In
response, this paper introduces a 3D Gaussian segmentation method implemented
with 2D segmentation as supervision. This approach uses input 2D segmentation
maps to guide the learning of the added 3D Gaussian semantic information, while
nearest neighbor clustering and statistical filtering refine the segmentation
results. Experiments show that our concise method can achieve comparable
performances on mIOU and mAcc for multi-object segmentation as previous
single-object segmentation methods.

---

## Neural BSSRDF: Object Appearance Representation Including Heterogeneous  Subsurface Scattering



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-25 | Thomson TG, Jeppe Revall Frisvad, Ravi Ramamoorthi, Henrik Wann Jensen | cs.GR | [PDF](http://arxiv.org/pdf/2312.15711v1){: .btn .btn-green } |

**Abstract**: Monte Carlo rendering of translucent objects with heterogeneous scattering
properties is often expensive both in terms of memory and computation. If we do
path tracing and use a high dynamic range lighting environment, the rendering
becomes computationally heavy. We propose a compact and efficient neural method
for representing and rendering the appearance of heterogeneous translucent
objects. The neural representation function resembles a bidirectional
scattering-surface reflectance distribution function (BSSRDF). However,
conventional BSSRDF models assume a planar half-space medium and only surface
variation of the material, which is often not a good representation of the
appearance of real-world objects. Our method represents the BSSRDF of a full
object taking its geometry and heterogeneities into account. This is similar to
a neural radiance field, but our representation works for an arbitrary distant
lighting environment. In a sense, we present a version of neural precomputed
radiance transfer that captures all-frequency relighting of heterogeneous
translucent objects. We use a multi-layer perceptron (MLP) with skip
connections to represent the appearance of an object as a function of spatial
position, direction of observation, and direction of incidence. The latter is
considered a directional light incident across the entire non-self-shadowed
part of the object. We demonstrate the ability of our method to store highly
complex materials while having high accuracy when comparing to reference images
of the represented object in unseen lighting environments. As compared with
path tracing of a heterogeneous light scattering volume behind a refractive
interface, our method more easily enables importance sampling of the directions
of incidence and can be integrated into existing rendering frameworks while
achieving interactive frame rates.

---

## SUNDIAL: 3D Satellite Understanding through Direct, Ambient, and Complex  Lighting Decomposition

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-24 | Nikhil Behari, Akshat Dave, Kushagra Tiwary, William Yang, Ramesh Raskar | cs.CV | [PDF](http://arxiv.org/pdf/2312.16215v1){: .btn .btn-green } |

**Abstract**: 3D modeling from satellite imagery is essential in areas of environmental
science, urban planning, agriculture, and disaster response. However,
traditional 3D modeling techniques face unique challenges in the remote sensing
context, including limited multi-view baselines over extensive regions, varying
direct, ambient, and complex illumination conditions, and time-varying scene
changes across captures. In this work, we introduce SUNDIAL, a comprehensive
approach to 3D reconstruction of satellite imagery using neural radiance
fields. We jointly learn satellite scene geometry, illumination components, and
sun direction in this single-model approach, and propose a secondary shadow ray
casting technique to 1) improve scene geometry using oblique sun angles to
render shadows, 2) enable physically-based disentanglement of scene albedo and
illumination, and 3) determine the components of illumination from direct,
ambient (sky), and complex sources. To achieve this, we incorporate lighting
cues and geometric priors from remote sensing literature in a neural rendering
approach, modeling physical properties of satellite scenes such as shadows,
scattered sky illumination, and complex illumination and shading of vegetation
and water. We evaluate the performance of SUNDIAL against existing NeRF-based
techniques for satellite scene modeling and demonstrate improved scene and
lighting disentanglement, novel view and lighting rendering, and geometry and
sun direction estimation on challenging scenes with small baselines, sparse
inputs, and variable illumination.

Comments:
- 8 pages, 6 figures

---

## Human101: Training 100+FPS Human Gaussians in 100s from 1 View

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-23 | Mingwei Li, Jiachen Tao, Zongxin Yang, Yi Yang | cs.CV | [PDF](http://arxiv.org/pdf/2312.15258v1){: .btn .btn-green } |

**Abstract**: Reconstructing the human body from single-view videos plays a pivotal role in
the virtual reality domain. One prevalent application scenario necessitates the
rapid reconstruction of high-fidelity 3D digital humans while simultaneously
ensuring real-time rendering and interaction. Existing methods often struggle
to fulfill both requirements. In this paper, we introduce Human101, a novel
framework adept at producing high-fidelity dynamic 3D human reconstructions
from 1-view videos by training 3D Gaussians in 100 seconds and rendering in
100+ FPS. Our method leverages the strengths of 3D Gaussian Splatting, which
provides an explicit and efficient representation of 3D humans. Standing apart
from prior NeRF-based pipelines, Human101 ingeniously applies a Human-centric
Forward Gaussian Animation method to deform the parameters of 3D Gaussians,
thereby enhancing rendering speed (i.e., rendering 1024-resolution images at an
impressive 60+ FPS and rendering 512-resolution images at 100+ FPS).
Experimental results indicate that our approach substantially eclipses current
methods, clocking up to a 10 times surge in frames per second and delivering
comparable or superior rendering quality. Code and demos will be released at
https://github.com/longxiang-ai/Human101.

Comments:
- Website: https://github.com/longxiang-ai/Human101

---

## Efficient Deformable Tissue Reconstruction via Orthogonal Neural Plane

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-23 | Chen Yang, Kailing Wang, Yuehao Wang, Qi Dou, Xiaokang Yang, Wei Shen | cs.CV | [PDF](http://arxiv.org/pdf/2312.15253v1){: .btn .btn-green } |

**Abstract**: Intraoperative imaging techniques for reconstructing deformable tissues in
vivo are pivotal for advanced surgical systems. Existing methods either
compromise on rendering quality or are excessively computationally intensive,
often demanding dozens of hours to perform, which significantly hinders their
practical application. In this paper, we introduce Fast Orthogonal Plane
(Forplane), a novel, efficient framework based on neural radiance fields (NeRF)
for the reconstruction of deformable tissues. We conceptualize surgical
procedures as 4D volumes, and break them down into static and dynamic fields
comprised of orthogonal neural planes. This factorization iscretizes the
four-dimensional space, leading to a decreased memory usage and faster
optimization. A spatiotemporal importance sampling scheme is introduced to
improve performance in regions with tool occlusion as well as large motions and
accelerate training. An efficient ray marching method is applied to skip
sampling among empty regions, significantly improving inference speed. Forplane
accommodates both binocular and monocular endoscopy videos, demonstrating its
extensive applicability and flexibility. Our experiments, carried out on two in
vivo datasets, the EndoNeRF and Hamlyn datasets, demonstrate the effectiveness
of our framework. In all cases, Forplane substantially accelerates both the
optimization process (by over 100 times) and the inference process (by over 15
times) while maintaining or even improving the quality across a variety of
non-rigid deformations. This significant performance improvement promises to be
a valuable asset for future intraoperative surgical applications. The code of
our project is now available at https://github.com/Loping151/ForPlane.

---

## CaLDiff: Camera Localization in NeRF via Pose Diffusion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-23 | Rashik Shrestha, Bishad Koju, Abhigyan Bhusal, Danda Pani Paudel, François Rameau | cs.CV | [PDF](http://arxiv.org/pdf/2312.15242v1){: .btn .btn-green } |

**Abstract**: With the widespread use of NeRF-based implicit 3D representation, the need
for camera localization in the same representation becomes manifestly apparent.
Doing so not only simplifies the localization process -- by avoiding an
outside-the-NeRF-based localization -- but also has the potential to offer the
benefit of enhanced localization. This paper studies the problem of localizing
cameras in NeRF using a diffusion model for camera pose adjustment. More
specifically, given a pre-trained NeRF model, we train a diffusion model that
iteratively updates randomly initialized camera poses, conditioned upon the
image to be localized. At test time, a new camera is localized in two steps:
first, coarse localization using the proposed pose diffusion process, followed
by local refinement steps of a pose inversion process in NeRF. In fact, the
proposed camera localization by pose diffusion (CaLDiff) method also integrates
the pose inversion steps within the diffusion process. Such integration offers
significantly better localization, thanks to our downstream refinement-aware
diffusion process. Our exhaustive experiments on challenging real-world data
validate our method by providing significantly better results than the compared
methods and the established baselines. Our source code will be made publicly
available.

---

## INFAMOUS-NeRF: ImproviNg FAce MOdeling Using Semantically-Aligned  Hypernetworks with Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-23 | Andrew Hou, Feng Liu, Zhiyuan Ren, Michel Sarkis, Ning Bi, Yiying Tong, Xiaoming Liu | cs.CV | [PDF](http://arxiv.org/pdf/2312.16197v1){: .btn .btn-green } |

**Abstract**: We propose INFAMOUS-NeRF, an implicit morphable face model that introduces
hypernetworks to NeRF to improve the representation power in the presence of
many training subjects. At the same time, INFAMOUS-NeRF resolves the classic
hypernetwork tradeoff of representation power and editability by learning
semantically-aligned latent spaces despite the subject-specific models, all
without requiring a large pretrained model. INFAMOUS-NeRF further introduces a
novel constraint to improve NeRF rendering along the face boundary. Our
constraint can leverage photometric surface rendering and multi-view
supervision to guide surface color prediction and improve rendering near the
surface. Finally, we introduce a novel, loss-guided adaptive sampling method
for more effective NeRF training by reducing the sampling redundancy. We show
quantitatively and qualitatively that our method achieves higher representation
power than prior face modeling methods in both controlled and in-the-wild
settings. Code and models will be released upon publication.

---

## Deformable 3D Gaussian Splatting for Animatable Human Avatars

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-22 | HyunJun Jung, Nikolas Brasch, Jifei Song, Eduardo Perez-Pellitero, Yiren Zhou, Zhihao Li, Nassir Navab, Benjamin Busam | cs.CV | [PDF](http://arxiv.org/pdf/2312.15059v1){: .btn .btn-green } |

**Abstract**: Recent advances in neural radiance fields enable novel view synthesis of
photo-realistic images in dynamic settings, which can be applied to scenarios
with human animation. Commonly used implicit backbones to establish accurate
models, however, require many input views and additional annotations such as
human masks, UV maps and depth maps. In this work, we propose ParDy-Human
(Parameterized Dynamic Human Avatar), a fully explicit approach to construct a
digital avatar from as little as a single monocular sequence. ParDy-Human
introduces parameter-driven dynamics into 3D Gaussian Splatting where 3D
Gaussians are deformed by a human pose model to animate the avatar. Our method
is composed of two parts: A first module that deforms canonical 3D Gaussians
according to SMPL vertices and a consecutive module that further takes their
designed joint encodings and predicts per Gaussian deformations to deal with
dynamics beyond SMPL vertex deformations. Images are then synthesized by a
rasterizer. ParDy-Human constitutes an explicit model for realistic dynamic
human avatars which requires significantly fewer training views and images. Our
avatars learning is free of additional annotations such as masks and can be
trained with variable backgrounds while inferring full-resolution images
efficiently even on consumer hardware. We provide experimental evidence to show
that ParDy-Human outperforms state-of-the-art methods on ZJU-MoCap and
THUman4.0 datasets both quantitatively and visually.

---

## PoseGen: Learning to Generate 3D Human Pose Dataset with NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-22 | Mohsen Gholami, Rabab Ward, Z. Jane Wang | cs.CV | [PDF](http://arxiv.org/pdf/2312.14915v1){: .btn .btn-green } |

**Abstract**: This paper proposes an end-to-end framework for generating 3D human pose
datasets using Neural Radiance Fields (NeRF). Public datasets generally have
limited diversity in terms of human poses and camera viewpoints, largely due to
the resource-intensive nature of collecting 3D human pose data. As a result,
pose estimators trained on public datasets significantly underperform when
applied to unseen out-of-distribution samples. Previous works proposed
augmenting public datasets by generating 2D-3D pose pairs or rendering a large
amount of random data. Such approaches either overlook image rendering or
result in suboptimal datasets for pre-trained models. Here we propose PoseGen,
which learns to generate a dataset (human 3D poses and images) with a feedback
loss from a given pre-trained pose estimator. In contrast to prior art, our
generated data is optimized to improve the robustness of the pre-trained model.
The objective of PoseGen is to learn a distribution of data that maximizes the
prediction error of a given pre-trained model. As the learned data distribution
contains OOD samples of the pre-trained model, sampling data from such a
distribution for further fine-tuning a pre-trained model improves the
generalizability of the model. This is the first work that proposes NeRFs for
3D human data generation. NeRFs are data-driven and do not require 3D scans of
humans. Therefore, using NeRF for data generation is a new direction for
convenient user-specific data generation. Our extensive experiments show that
the proposed PoseGen improves two baseline models (SPIN and HybrIK) on four
datasets with an average 6% relative improvement.

---

## Density Uncertainty Quantification with NeRF-Ensembles: Impact of Data  and Scene Constraints

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-22 | Miriam Jäger, Steven Landgraf, Boris Jutzi | cs.CV | [PDF](http://arxiv.org/pdf/2312.14664v1){: .btn .btn-green } |

**Abstract**: In the fields of computer graphics, computer vision and photogrammetry,
Neural Radiance Fields (NeRFs) are a major topic driving current research and
development. However, the quality of NeRF-generated 3D scene reconstructions
and subsequent surface reconstructions, heavily relies on the network output,
particularly the density. Regarding this critical aspect, we propose to utilize
NeRF-Ensembles that provide a density uncertainty estimate alongside the mean
density. We demonstrate that data constraints such as low-quality images and
poses lead to a degradation of the training process, increased density
uncertainty and decreased predicted density. Even with high-quality input data,
the density uncertainty varies based on scene constraints such as acquisition
constellations, occlusions and material properties. NeRF-Ensembles not only
provide a tool for quantifying the uncertainty but exhibit two promising
advantages: Enhanced robustness and artifact removal. Through the utilization
of NeRF-Ensembles instead of single NeRFs, small outliers are removed, yielding
a smoother output with improved completeness of structures. Furthermore,
applying percentile-based thresholds on density uncertainty outliers proves to
be effective for the removal of large (foggy) artifacts in post-processing. We
conduct our methodology on 3 different datasets: (i) synthetic benchmark
dataset, (ii) real benchmark dataset, (iii) real data under realistic recording
conditions and sensors.

Comments:
- 21 pages, 12 figures, 5 tables

---

## SyncDreamer for 3D Reconstruction of Endangered Animal Species with NeRF  and NeuS

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-21 | Ahmet Haydar Ornek, Deniz Sen, Esmanur Civil | cs.CV | [PDF](http://arxiv.org/pdf/2312.13832v1){: .btn .btn-green } |

**Abstract**: The main aim of this study is to demonstrate how innovative view synthesis
and 3D reconstruction techniques can be used to create models of endangered
species using monocular RGB images. To achieve this, we employed SyncDreamer to
produce unique perspectives and NeuS and NeRF to reconstruct 3D
representations. We chose four different animals, including the oriental stork,
frog, dragonfly, and tiger, as our subjects for this study. Our results show
that the combination of SyncDreamer, NeRF, and NeuS techniques can successfully
create 3D models of endangered animals. However, we also observed that NeuS
produced blurry images, while NeRF generated sharper but noisier images. This
study highlights the potential of modeling endangered animals and offers a new
direction for future research in this field. By showcasing the effectiveness of
these advanced techniques, we hope to encourage further exploration and
development of techniques for preserving and studying endangered species.

Comments:
- 8 figures

---

## Visual Tomography: Physically Faithful Volumetric Models of Partially  Translucent Objects

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-21 | David Nakath, Xiangyu Weng, Mengkun She, Kevin Köser | cs.CV | [PDF](http://arxiv.org/pdf/2312.13494v1){: .btn .btn-green } |

**Abstract**: When created faithfully from real-world data, Digital 3D representations of
objects can be useful for human or computer-assisted analysis. Such models can
also serve for generating training data for machine learning approaches in
settings where data is difficult to obtain or where too few training data
exists, e.g. by providing novel views or images in varying conditions. While
the vast amount of visual 3D reconstruction approaches focus on non-physical
models, textured object surfaces or shapes, in this contribution we propose a
volumetric reconstruction approach that obtains a physical model including the
interior of partially translucent objects such as plankton or insects. Our
technique photographs the object under different poses in front of a bright
white light source and computes absorption and scattering per voxel. It can be
interpreted as visual tomography that we solve by inverse raytracing. We
additionally suggest a method to convert non-physical NeRF media into a
physically-based volumetric grid for initialization and illustrate the
usefulness of the approach using two real-world plankton validation sets, the
lab-scanned models being finally also relighted and virtually submerged in a
scenario with augmented medium and illumination conditions. Please visit the
project homepage at www.marine.informatik.uni-kiel.de/go/vito

Comments:
- Accepted for publication at 3DV '24

---

## Gaussian Splatting with NeRF-based Color and Opacity

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-21 | Dawid Malarz, Weronika Smolak, Jacek Tabor, Sławomir Tadeja, Przemysław Spurek | cs.CV | [PDF](http://arxiv.org/pdf/2312.13729v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have demonstrated the remarkable potential of
neural networks to capture the intricacies of 3D objects. By encoding the shape
and color information within neural network weights, NeRFs excel at producing
strikingly sharp novel views of 3D objects. Recently, numerous generalizations
of NeRFs utilizing generative models have emerged, expanding its versatility.
In contrast, Gaussian Splatting (GS) offers a similar renders quality with
faster training and inference as it does not need neural networks to work. We
encode information about the 3D objects in the set of Gaussian distributions
that can be rendered in 3D similarly to classical meshes. Unfortunately, GS are
difficult to condition since they usually require circa hundred thousand
Gaussian components. To mitigate the caveats of both models, we propose a
hybrid model that uses GS representation of the 3D object's shape and
NeRF-based encoding of color and opacity. Our model uses Gaussian distributions
with trainable positions (i.e. means of Gaussian), shape (i.e. covariance of
Gaussian), color and opacity, and neural network, which takes parameters of
Gaussian and viewing direction to produce changes in color and opacity.
Consequently, our model better describes shadows, light reflections, and
transparency of 3D objects.

---

## Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed  Diffusion Models

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-21 | Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, Karsten Kreis | cs.CV | [PDF](http://arxiv.org/pdf/2312.13763v2){: .btn .btn-green } |

**Abstract**: Text-guided diffusion models have revolutionized image and video generation
and have also been successfully used for optimization-based 3D object
synthesis. Here, we instead focus on the underexplored text-to-4D setting and
synthesize dynamic, animated 3D objects using score distillation methods with
an additional temporal dimension. Compared to previous work, we pursue a novel
compositional generation-based approach, and combine text-to-image,
text-to-video, and 3D-aware multiview diffusion models to provide feedback
during 4D object optimization, thereby simultaneously enforcing temporal
consistency, high-quality visual appearance and realistic geometry. Our method,
called Align Your Gaussians (AYG), leverages dynamic 3D Gaussian Splatting with
deformation fields as 4D representation. Crucial to AYG is a novel method to
regularize the distribution of the moving 3D Gaussians and thereby stabilize
the optimization and induce motion. We also propose a motion amplification
mechanism as well as a new autoregressive synthesis scheme to generate and
combine multiple 4D sequences for longer generation. These techniques allow us
to synthesize vivid dynamic scenes, outperform previous work qualitatively and
quantitatively and achieve state-of-the-art text-to-4D performance. Due to the
Gaussian 4D representation, different 4D animations can be seamlessly combined,
as we demonstrate. AYG opens up promising avenues for animation, simulation and
digital content creation as well as synthetic data generation.

Comments:
- Project page:
  https://research.nvidia.com/labs/toronto-ai/AlignYourGaussians/

---

## DyBluRF: Dynamic Deblurring Neural Radiance Fields for Blurry Monocular  Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-21 | Minh-Quan Viet Bui, Jongmin Park, Jihyong Oh, Munchurl Kim | cs.CV | [PDF](http://arxiv.org/pdf/2312.13528v1){: .btn .btn-green } |

**Abstract**: Video view synthesis, allowing for the creation of visually appealing frames
from arbitrary viewpoints and times, offers immersive viewing experiences.
Neural radiance fields, particularly NeRF, initially developed for static
scenes, have spurred the creation of various methods for video view synthesis.
However, the challenge for video view synthesis arises from motion blur, a
consequence of object or camera movement during exposure, which hinders the
precise synthesis of sharp spatio-temporal views. In response, we propose a
novel dynamic deblurring NeRF framework for blurry monocular video, called
DyBluRF, consisting of an Interleave Ray Refinement (IRR) stage and a Motion
Decomposition-based Deblurring (MDD) stage. Our DyBluRF is the first that
addresses and handles the novel view synthesis for blurry monocular video. The
IRR stage jointly reconstructs dynamic 3D scenes and refines the inaccurate
camera pose information to combat imprecise pose information extracted from the
given blurry frames. The MDD stage is a novel incremental latent sharp-rays
prediction (ILSP) approach for the blurry monocular video frames by decomposing
the latent sharp rays into global camera motion and local object motion
components. Extensive experimental results demonstrate that our DyBluRF
outperforms qualitatively and quantitatively the very recent state-of-the-art
methods. Our project page including source codes and pretrained model are
publicly available at https://kaist-viclab.github.io/dyblurf-site/.

Comments:
- The first three authors contributed equally to this work. Please
  visit our project page at https://kaist-viclab.github.io/dyblurf-site/

---

## Carve3D: Improving Multi-view Reconstruction Consistency for Diffusion  Models with RL Finetuning

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-21 | Desai Xie, Jiahao Li, Hao Tan, Xin Sun, Zhixin Shu, Yi Zhou, Sai Bi, Sören Pirk, Arie E. Kaufman | cs.CV | [PDF](http://arxiv.org/pdf/2312.13980v1){: .btn .btn-green } |

**Abstract**: Recent advancements in the text-to-3D task leverage finetuned text-to-image
diffusion models to generate multi-view images, followed by NeRF
reconstruction. Yet, existing supervised finetuned (SFT) diffusion models still
suffer from multi-view inconsistency and the resulting NeRF artifacts. Although
training longer with SFT improves consistency, it also causes distribution
shift, which reduces diversity and realistic details. We argue that the SFT of
multi-view diffusion models resembles the instruction finetuning stage of the
LLM alignment pipeline and can benefit from RL finetuning (RLFT) methods.
Essentially, RLFT methods optimize models beyond their SFT data distribution by
using their own outputs, effectively mitigating distribution shift. To this
end, we introduce Carve3D, a RLFT method coupled with the Multi-view
Reconstruction Consistency (MRC) metric, to improve the consistency of
multi-view diffusion models. To compute MRC on a set of multi-view images, we
compare them with their corresponding renderings of the reconstructed NeRF at
the same viewpoints. We validate the robustness of MRC with extensive
experiments conducted under controlled inconsistency levels. We enhance the
base RLFT algorithm to stabilize the training process, reduce distribution
shift, and identify scaling laws. Through qualitative and quantitative
experiments, along with a user study, we demonstrate Carve3D's improved
multi-view consistency, the resulting superior NeRF reconstruction quality, and
minimal distribution shift compared to longer SFT. Project webpage:
https://desaixie.github.io/carve-3d.

Comments:
- Project webpage: https://desaixie.github.io/carve-3d

---

## Virtual Pets: Animatable Animal Generation in 3D Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-21 | Yen-Chi Cheng, Chieh Hubert Lin, Chaoyang Wang, Yash Kant, Sergey Tulyakov, Alexander Schwing, Liangyan Gui, Hsin-Ying Lee | cs.CV | [PDF](http://arxiv.org/pdf/2312.14154v1){: .btn .btn-green } |

**Abstract**: Toward unlocking the potential of generative models in immersive 4D
experiences, we introduce Virtual Pet, a novel pipeline to model realistic and
diverse motions for target animal species within a 3D environment. To
circumvent the limited availability of 3D motion data aligned with
environmental geometry, we leverage monocular internet videos and extract
deformable NeRF representations for the foreground and static NeRF
representations for the background. For this, we develop a reconstruction
strategy, encompassing species-level shared template learning and per-video
fine-tuning. Utilizing the reconstructed data, we then train a conditional 3D
motion model to learn the trajectory and articulation of foreground animals in
the context of 3D backgrounds. We showcase the efficacy of our pipeline with
comprehensive qualitative and quantitative evaluations using cat videos. We
also demonstrate versatility across unseen cats and indoor environments,
producing temporally coherent 4D outputs for enriched virtual experiences.

Comments:
- Preprint. Project page: https://yccyenchicheng.github.io/VirtualPets/

---

## PlatoNeRF: 3D Reconstruction in Plato's Cave via Single-View Two-Bounce  Lidar

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-21 | Tzofi Klinghoffer, Xiaoyu Xiang, Siddharth Somasundaram, Yuchen Fan, Christian Richardt, Ramesh Raskar, Rakesh Ranjan | cs.CV | [PDF](http://arxiv.org/pdf/2312.14239v1){: .btn .btn-green } |

**Abstract**: 3D reconstruction from a single-view is challenging because of the ambiguity
from monocular cues and lack of information about occluded regions. Neural
radiance fields (NeRF), while popular for view synthesis and 3D reconstruction,
are typically reliant on multi-view images. Existing methods for single-view 3D
reconstruction with NeRF rely on either data priors to hallucinate views of
occluded regions, which may not be physically accurate, or shadows observed by
RGB cameras, which are difficult to detect in ambient light and low albedo
backgrounds. We propose using time-of-flight data captured by a single-photon
avalanche diode to overcome these limitations. Our method models two-bounce
optical paths with NeRF, using lidar transient data for supervision. By
leveraging the advantages of both NeRF and two-bounce light measured by lidar,
we demonstrate that we can reconstruct visible and occluded geometry without
data priors or reliance on controlled ambient lighting or scene albedo. In
addition, we demonstrate improved generalization under practical constraints on
sensor spatial- and temporal-resolution. We believe our method is a promising
direction as single-photon lidars become ubiquitous on consumer devices, such
as phones, tablets, and headsets.

Comments:
- Project Page: https://platonerf.github.io/

---

## Neural Point Cloud Diffusion for Disentangled 3D Shape and Appearance  Generation



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-21 | Philipp Schröppel, Christopher Wewer, Jan Eric Lenssen, Eddy Ilg, Thomas Brox | cs.CV | [PDF](http://arxiv.org/pdf/2312.14124v1){: .btn .btn-green } |

**Abstract**: Controllable generation of 3D assets is important for many practical
applications like content creation in movies, games and engineering, as well as
in AR/VR. Recently, diffusion models have shown remarkable results in
generation quality of 3D objects. However, none of the existing models enable
disentangled generation to control the shape and appearance separately. For the
first time, we present a suitable representation for 3D diffusion models to
enable such disentanglement by introducing a hybrid point cloud and neural
radiance field approach. We model a diffusion process over point positions
jointly with a high-dimensional feature space for a local density and radiance
decoder. While the point positions represent the coarse shape of the object,
the point features allow modeling the geometry and appearance details. This
disentanglement enables us to sample both independently and therefore to
control both separately. Our approach sets a new state of the art in generation
compared to previous disentanglement-capable methods by reduced FID scores of
30-90% and is on-par with other non disentanglement-capable state-of-the art
methods.

---

## Splatter Image: Ultra-Fast Single-View 3D Reconstruction

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-20 | Stanislaw Szymanowicz, Christian Rupprecht, Andrea Vedaldi | cs.CV | [PDF](http://arxiv.org/pdf/2312.13150v1){: .btn .btn-green } |

**Abstract**: We introduce the Splatter Image, an ultra-fast approach for monocular 3D
object reconstruction which operates at 38 FPS. Splatter Image is based on
Gaussian Splatting, which has recently brought real-time rendering, fast
training, and excellent scaling to multi-view reconstruction. For the first
time, we apply Gaussian Splatting in a monocular reconstruction setting. Our
approach is learning-based, and, at test time, reconstruction only requires the
feed-forward evaluation of a neural network. The main innovation of Splatter
Image is the surprisingly straightforward design: it uses a 2D image-to-image
network to map the input image to one 3D Gaussian per pixel. The resulting
Gaussians thus have the form of an image, the Splatter Image. We further extend
the method to incorporate more than one image as input, which we do by adding
cross-view attention. Owning to the speed of the renderer (588 FPS), we can use
a single GPU for training while generating entire images at each iteration in
order to optimize perceptual metrics like LPIPS. On standard benchmarks, we
demonstrate not only fast reconstruction but also better results than recent
and much more expensive baselines in terms of PSNR, LPIPS, and other metrics.

Comments:
- Project page: https://szymanowiczs.github.io/splatter-image.html .
  Code: https://github.com/szymanowiczs/splatter-image

---

## Reducing Shape-Radiance Ambiguity in Radiance Fields with a Closed-Form  Color Estimation Method

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-20 | Qihang Fang, Yafei Song, Keqiang Li, Liefeng Bo | cs.CV | [PDF](http://arxiv.org/pdf/2312.12726v1){: .btn .btn-green } |

**Abstract**: Neural radiance field (NeRF) enables the synthesis of cutting-edge realistic
novel view images of a 3D scene. It includes density and color fields to model
the shape and radiance of a scene, respectively. Supervised by the photometric
loss in an end-to-end training manner, NeRF inherently suffers from the
shape-radiance ambiguity problem, i.e., it can perfectly fit training views but
does not guarantee decoupling the two fields correctly. To deal with this
issue, existing works have incorporated prior knowledge to provide an
independent supervision signal for the density field, including total variation
loss, sparsity loss, distortion loss, etc. These losses are based on general
assumptions about the density field, e.g., it should be smooth, sparse, or
compact, which are not adaptive to a specific scene. In this paper, we propose
a more adaptive method to reduce the shape-radiance ambiguity. The key is a
rendering method that is only based on the density field. Specifically, we
first estimate the color field based on the density field and posed images in a
closed form. Then NeRF's rendering process can proceed. We address the problems
in estimating the color field, including occlusion and non-uniformly
distributed views. Afterward, it is applied to regularize NeRF's density field.
As our regularization is guided by photometric loss, it is more adaptive
compared to existing ones. Experimental results show that our method improves
the density field of NeRF both qualitatively and quantitatively. Our code is
available at https://github.com/qihangGH/Closed-form-color-field.

Comments:
- This work has been published in NeurIPS 2023

---

## SWAGS: Sampling Windows Adaptively for Dynamic 3D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-20 | Richard Shaw, Jifei Song, Arthur Moreau, Michal Nazarczuk, Sibi Catley-Chandar, Helisa Dhamo, Eduardo Perez-Pellitero | cs.CV | [PDF](http://arxiv.org/pdf/2312.13308v1){: .btn .btn-green } |

**Abstract**: Novel view synthesis has shown rapid progress recently, with methods capable
of producing evermore photo-realistic results. 3D Gaussian Splatting has
emerged as a particularly promising method, producing high-quality renderings
of static scenes and enabling interactive viewing at real-time frame rates.
However, it is currently limited to static scenes only. In this work, we extend
3D Gaussian Splatting to reconstruct dynamic scenes. We model the dynamics of a
scene using a tunable MLP, which learns the deformation field from a canonical
space to a set of 3D Gaussians per frame. To disentangle the static and dynamic
parts of the scene, we learn a tuneable parameter for each Gaussian, which
weighs the respective MLP parameters to focus attention on the dynamic parts.
This improves the model's ability to capture dynamics in scenes with an
imbalance of static to dynamic regions. To handle scenes of arbitrary length
whilst maintaining high rendering quality, we introduce an adaptive window
sampling strategy to partition the sequence into windows based on the amount of
movement in the sequence. We train a separate dynamic Gaussian Splatting model
for each window, allowing the canonical representation to change, thus enabling
the reconstruction of scenes with significant geometric or topological changes.
Temporal consistency is enforced using a fine-tuning step with self-supervising
consistency loss on randomly sampled novel views. As a result, our method
produces high-quality renderings of general dynamic scenes with competitive
quantitative performance, which can be viewed in real-time with our dynamic
interactive viewer.

---

## ShowRoom3D: Text to High-Quality 3D Room Generation Using 3D Priors

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-20 | Weijia Mao, Yan-Pei Cao, Jia-Wei Liu, Zhongcong Xu, Mike Zheng Shou | cs.CV | [PDF](http://arxiv.org/pdf/2312.13324v1){: .btn .btn-green } |

**Abstract**: We introduce ShowRoom3D, a three-stage approach for generating high-quality
3D room-scale scenes from texts. Previous methods using 2D diffusion priors to
optimize neural radiance fields for generating room-scale scenes have shown
unsatisfactory quality. This is primarily attributed to the limitations of 2D
priors lacking 3D awareness and constraints in the training methodology. In
this paper, we utilize a 3D diffusion prior, MVDiffusion, to optimize the 3D
room-scale scene. Our contributions are in two aspects. Firstly, we propose a
progressive view selection process to optimize NeRF. This involves dividing the
training process into three stages, gradually expanding the camera sampling
scope. Secondly, we propose the pose transformation method in the second stage.
It will ensure MVDiffusion provide the accurate view guidance. As a result,
ShowRoom3D enables the generation of rooms with improved structural integrity,
enhanced clarity from any view, reduced content repetition, and higher
consistency across different perspectives. Extensive experiments demonstrate
that our method, significantly outperforms state-of-the-art approaches by a
large margin in terms of user study.

---

## SpecNeRF: Gaussian Directional Encoding for Specular Reflections

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-20 | Li Ma, Vasu Agrawal, Haithem Turki, Changil Kim, Chen Gao, Pedro Sander, Michael Zollhöfer, Christian Richardt | cs.CV | [PDF](http://arxiv.org/pdf/2312.13102v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields have achieved remarkable performance in modeling the
appearance of 3D scenes. However, existing approaches still struggle with the
view-dependent appearance of glossy surfaces, especially under complex lighting
of indoor environments. Unlike existing methods, which typically assume distant
lighting like an environment map, we propose a learnable Gaussian directional
encoding to better model the view-dependent effects under near-field lighting
conditions. Importantly, our new directional encoding captures the
spatially-varying nature of near-field lighting and emulates the behavior of
prefiltered environment maps. As a result, it enables the efficient evaluation
of preconvolved specular color at any 3D location with varying roughness
coefficients. We further introduce a data-driven geometry prior that helps
alleviate the shape radiance ambiguity in reflection modeling. We show that our
Gaussian directional encoding and geometry prior significantly improve the
modeling of challenging specular reflections in neural radiance fields, which
helps decompose appearance into more physically meaningful components.

Comments:
- Project page: https://limacv.github.io/SpecNeRF_web/

---

## Ternary-type Opacity and Hybrid Odometry for RGB-only NeRF-SLAM

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-20 | Junru Lin, Asen Nachkov, Songyou Peng, Luc Van Gool, Danda Pani Paudel | cs.CV | [PDF](http://arxiv.org/pdf/2312.13332v2){: .btn .btn-green } |

**Abstract**: The opacity of rigid 3D scenes with opaque surfaces is considered to be of a
binary type. However, we observed that this property is not followed by the
existing RGB-only NeRF-SLAM. Therefore, we are motivated to introduce this
prior into the RGB-only NeRF-SLAM pipeline. Unfortunately, the optimization
through the volumetric rendering function does not facilitate easy integration
of the desired prior. Instead, we observed that the opacity of ternary-type
(TT) is well supported. In this work, we study why ternary-type opacity is
well-suited and desired for the task at hand. In particular, we provide
theoretical insights into the process of jointly optimizing radiance and
opacity through the volumetric rendering process. Through exhaustive
experiments on benchmark datasets, we validate our claim and provide insights
into the optimization process, which we believe will unleash the potential of
RGB-only NeRF-SLAM. To foster this line of research, we also propose a simple
yet novel visual odometry scheme that uses a hybrid combination of volumetric
and warping-based image renderings. More specifically, the proposed hybrid
odometry (HO) additionally uses image warping-based coarse odometry, leading up
to an order of magnitude final speed-up. Furthermore, we show that the proposed
TT and HO well complement each other, offering state-of-the-art results on
benchmark datasets in terms of both speed and accuracy.

---

## Deep Learning on 3D Neural Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-20 | Pierluigi Zama Ramirez, Luca De Luigi, Daniele Sirocchi, Adriano Cardace, Riccardo Spezialetti, Francesco Ballerini, Samuele Salti, Luigi Di Stefano | cs.CV | [PDF](http://arxiv.org/pdf/2312.13277v1){: .btn .btn-green } |

**Abstract**: In recent years, Neural Fields (NFs) have emerged as an effective tool for
encoding diverse continuous signals such as images, videos, audio, and 3D
shapes. When applied to 3D data, NFs offer a solution to the fragmentation and
limitations associated with prevalent discrete representations. However, given
that NFs are essentially neural networks, it remains unclear whether and how
they can be seamlessly integrated into deep learning pipelines for solving
downstream tasks. This paper addresses this research problem and introduces
nf2vec, a framework capable of generating a compact latent representation for
an input NF in a single inference pass. We demonstrate that nf2vec effectively
embeds 3D objects represented by the input NFs and showcase how the resulting
embeddings can be employed in deep learning pipelines to successfully address
various tasks, all while processing exclusively NFs. We test this framework on
several NFs used to represent 3D surfaces, such as unsigned/signed distance and
occupancy fields. Moreover, we demonstrate the effectiveness of our approach
with more complex NFs that encompass both geometry and appearance of 3D objects
such as neural radiance fields.

Comments:
- Extended version of the paper "Deep Learning on Implicit Neural
  Representations of Shapes" that was presented at ICLR 2023. arXiv admin note:
  text overlap with arXiv:2302.05438

---

## UniSDF: Unifying Neural Representations for High-Fidelity 3D  Reconstruction of Complex Scenes with Reflections

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-20 | Fangjinhua Wang, Marie-Julie Rakotosaona, Michael Niemeyer, Richard Szeliski, Marc Pollefeys, Federico Tombari | cs.CV | [PDF](http://arxiv.org/pdf/2312.13285v1){: .btn .btn-green } |

**Abstract**: Neural 3D scene representations have shown great potential for 3D
reconstruction from 2D images. However, reconstructing real-world captures of
complex scenes still remains a challenge. Existing generic 3D reconstruction
methods often struggle to represent fine geometric details and do not
adequately model reflective surfaces of large-scale scenes. Techniques that
explicitly focus on reflective surfaces can model complex and detailed
reflections by exploiting better reflection parameterizations. However, we
observe that these methods are often not robust in real unbounded scenarios
where non-reflective as well as reflective components are present. In this
work, we propose UniSDF, a general purpose 3D reconstruction method that can
reconstruct large complex scenes with reflections. We investigate both
view-based as well as reflection-based color prediction parameterization
techniques and find that explicitly blending these representations in 3D space
enables reconstruction of surfaces that are more geometrically accurate,
especially for reflective surfaces. We further combine this representation with
a multi-resolution grid backbone that is trained in a coarse-to-fine manner,
enabling faster reconstructions than prior methods. Extensive experiments on
object-level datasets DTU, Shiny Blender as well as unbounded datasets Mip-NeRF
360 and Ref-NeRF real demonstrate that our method is able to robustly
reconstruct complex large-scale scenes with fine details and reflective
surfaces. Please see our project page at
https://fangjinhuawang.github.io/UniSDF.

Comments:
- Project page: https://fangjinhuawang.github.io/UniSDF

---

## NeRF-VO: Real-Time Sparse Visual Odometry with Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-20 | Jens Naumann, Binbin Xu, Stefan Leutenegger, Xingxing Zuo | cs.CV | [PDF](http://arxiv.org/pdf/2312.13471v1){: .btn .btn-green } |

**Abstract**: We introduce a novel monocular visual odometry (VO) system, NeRF-VO, that
integrates learning-based sparse visual odometry for low-latency camera
tracking and a neural radiance scene representation for sophisticated dense
reconstruction and novel view synthesis. Our system initializes camera poses
using sparse visual odometry and obtains view-dependent dense geometry priors
from a monocular depth prediction network. We harmonize the scale of poses and
dense geometry, treating them as supervisory cues to train a neural implicit
scene representation. NeRF-VO demonstrates exceptional performance in both
photometric and geometric fidelity of the scene representation by jointly
optimizing a sliding window of keyframed poses and the underlying dense
geometry, which is accomplished through training the radiance field with volume
rendering. We surpass state-of-the-art methods in pose estimation accuracy,
novel view synthesis fidelity, and dense reconstruction quality across a
variety of synthetic and real-world datasets, while achieving a higher camera
tracking frequency and consuming less GPU memory.

Comments:
- 10 tables, 4 figures

---

## Compact 3D Scene Representation via Self-Organizing Gaussian Grids

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-19 | Wieland Morgenstern, Florian Barthel, Anna Hilsmann, Peter Eisert | cs.CV | [PDF](http://arxiv.org/pdf/2312.13299v1){: .btn .btn-green } |

**Abstract**: 3D Gaussian Splatting has recently emerged as a highly promising technique
for modeling of static 3D scenes. In contrast to Neural Radiance Fields, it
utilizes efficient rasterization allowing for very fast rendering at
high-quality. However, the storage size is significantly higher, which hinders
practical deployment, e.g.~on resource constrained devices. In this paper, we
introduce a compact scene representation organizing the parameters of 3D
Gaussian Splatting (3DGS) into a 2D grid with local homogeneity, ensuring a
drastic reduction in storage requirements without compromising visual quality
during rendering. Central to our idea is the explicit exploitation of
perceptual redundancies present in natural scenes. In essence, the inherent
nature of a scene allows for numerous permutations of Gaussian parameters to
equivalently represent it. To this end, we propose a novel highly parallel
algorithm that regularly arranges the high-dimensional Gaussian parameters into
a 2D grid while preserving their neighborhood structure. During training, we
further enforce local smoothness between the sorted parameters in the grid. The
uncompressed Gaussians use the same structure as 3DGS, ensuring a seamless
integration with established renderers. Our method achieves a reduction factor
of 8x to 26x in size for complex scenes with no increase in training time,
marking a substantial leap forward in the domain of 3D scene distribution and
consumption. Additional information can be found on our project page:
https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/

---

## pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable  Generalizable 3D Reconstruction

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-19 | David Charatan, Sizhe Li, Andrea Tagliasacchi, Vincent Sitzmann | cs.CV | [PDF](http://arxiv.org/pdf/2312.12337v2){: .btn .btn-green } |

**Abstract**: We introduce pixelSplat, a feed-forward model that learns to reconstruct 3D
radiance fields parameterized by 3D Gaussian primitives from pairs of images.
Our model features real-time and memory-efficient rendering for scalable
training as well as fast 3D reconstruction at inference time. To overcome local
minima inherent to sparse and locally supported representations, we predict a
dense probability distribution over 3D and sample Gaussian means from that
probability distribution. We make this sampling operation differentiable via a
reparameterization trick, allowing us to back-propagate gradients through the
Gaussian splatting representation. We benchmark our method on wide-baseline
novel view synthesis on the real-world RealEstate10k and ACID datasets, where
we outperform state-of-the-art light field transformers and accelerate
rendering by 2.5 orders of magnitude while reconstructing an interpretable and
editable 3D radiance field.

Comments:
- Project page: https://dcharatan.github.io/pixelsplat

---

## ZS-SRT: An Efficient Zero-Shot Super-Resolution Training Method for  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-19 | Xiang Feng, Yongbo He, Yubo Wang, Chengkai Wang, Zhenzhong Kuang, Jiajun Ding, Feiwei Qin, Jun Yu, Jianping Fan | cs.CV | [PDF](http://arxiv.org/pdf/2312.12122v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have achieved great success in the task of
synthesizing novel views that preserve the same resolution as the training
views. However, it is challenging for NeRF to synthesize high-quality
high-resolution novel views with low-resolution training data. To solve this
problem, we propose a zero-shot super-resolution training framework for NeRF.
This framework aims to guide the NeRF model to synthesize high-resolution novel
views via single-scene internal learning rather than requiring any external
high-resolution training data. Our approach consists of two stages. First, we
learn a scene-specific degradation mapping by performing internal learning on a
pretrained low-resolution coarse NeRF. Second, we optimize a super-resolution
fine NeRF by conducting inverse rendering with our mapping function so as to
backpropagate the gradients from low-resolution 2D space into the
super-resolution 3D sampling space. Then, we further introduce a temporal
ensemble strategy in the inference phase to compensate for the scene estimation
errors. Our method is featured on two points: (1) it does not consume
high-resolution views or additional scene data to train super-resolution NeRF;
(2) it can speed up the training process by adopting a coarse-to-fine strategy.
By conducting extensive experiments on public datasets, we have qualitatively
and quantitatively demonstrated the effectiveness of our method.

---

## LHManip: A Dataset for Long-Horizon Language-Grounded Manipulation Tasks  in Cluttered Tabletop Environments

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-19 | Federico Ceola, Lorenzo Natale, Niko Sünderhauf, Krishan Rana | cs.RO | [PDF](http://arxiv.org/pdf/2312.12036v2){: .btn .btn-green } |

**Abstract**: Instructing a robot to complete an everyday task within our homes has been a
long-standing challenge for robotics. While recent progress in
language-conditioned imitation learning and offline reinforcement learning has
demonstrated impressive performance across a wide range of tasks, they are
typically limited to short-horizon tasks -- not reflective of those a home
robot would be expected to complete. While existing architectures have the
potential to learn these desired behaviours, the lack of the necessary
long-horizon, multi-step datasets for real robotic systems poses a significant
challenge. To this end, we present the Long-Horizon Manipulation (LHManip)
dataset comprising 200 episodes, demonstrating 20 different manipulation tasks
via real robot teleoperation. The tasks entail multiple sub-tasks, including
grasping, pushing, stacking and throwing objects in highly cluttered
environments. Each task is paired with a natural language instruction and
multi-camera viewpoints for point-cloud or NeRF reconstruction. In total, the
dataset comprises 176,278 observation-action pairs which form part of the Open
X-Embodiment dataset. The full LHManip dataset is made publicly available at
https://github.com/fedeceola/LHManip.

Comments:
- Submitted to IJRR

---

## MixRT: Mixed Neural Representations For Real-Time NeRF Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-19 | Chaojian Li, Bichen Wu, Peter Vajda,  Yingyan,  Lin | cs.CV | [PDF](http://arxiv.org/pdf/2312.11841v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has emerged as a leading technique for novel
view synthesis, owing to its impressive photorealistic reconstruction and
rendering capability. Nevertheless, achieving real-time NeRF rendering in
large-scale scenes has presented challenges, often leading to the adoption of
either intricate baked mesh representations with a substantial number of
triangles or resource-intensive ray marching in baked representations. We
challenge these conventions, observing that high-quality geometry, represented
by meshes with substantial triangles, is not necessary for achieving
photorealistic rendering quality. Consequently, we propose MixRT, a novel NeRF
representation that includes a low-quality mesh, a view-dependent displacement
map, and a compressed NeRF model. This design effectively harnesses the
capabilities of existing graphics hardware, thus enabling real-time NeRF
rendering on edge devices. Leveraging a highly-optimized WebGL-based rendering
framework, our proposed MixRT attains real-time rendering speeds on edge
devices (over 30 FPS at a resolution of 1280 x 720 on a MacBook M1 Pro laptop),
better rendering quality (0.2 PSNR higher in indoor scenes of the Unbounded-360
datasets), and a smaller storage size (less than 80% compared to
state-of-the-art methods).

Comments:
- Accepted by 3DV'24. Project Page: https://licj15.github.io/MixRT/

---

## Text-Image Conditioned Diffusion for Consistent Text-to-3D Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-19 | Yuze He, Yushi Bai, Matthieu Lin, Jenny Sheng, Yubin Hu, Qi Wang, Yu-Hui Wen, Yong-Jin Liu | cs.CV | [PDF](http://arxiv.org/pdf/2312.11774v1){: .btn .btn-green } |

**Abstract**: By lifting the pre-trained 2D diffusion models into Neural Radiance Fields
(NeRFs), text-to-3D generation methods have made great progress. Many
state-of-the-art approaches usually apply score distillation sampling (SDS) to
optimize the NeRF representations, which supervises the NeRF optimization with
pre-trained text-conditioned 2D diffusion models such as Imagen. However, the
supervision signal provided by such pre-trained diffusion models only depends
on text prompts and does not constrain the multi-view consistency. To inject
the cross-view consistency into diffusion priors, some recent works finetune
the 2D diffusion model with multi-view data, but still lack fine-grained view
coherence. To tackle this challenge, we incorporate multi-view image conditions
into the supervision signal of NeRF optimization, which explicitly enforces
fine-grained view consistency. With such stronger supervision, our proposed
text-to-3D method effectively mitigates the generation of floaters (due to
excessive densities) and completely empty spaces (due to insufficient
densities). Our quantitative evaluations on the T$^3$Bench dataset demonstrate
that our method achieves state-of-the-art performance over existing text-to-3D
methods. We will make the code publicly available.

---

## GauFRe: Gaussian Deformation Fields for Real-time Dynamic Novel View  Synthesis

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-18 | Yiqing Liang, Numair Khan, Zhengqin Li, Thu Nguyen-Phuoc, Douglas Lanman, James Tompkin, Lei Xiao | cs.CV | [PDF](http://arxiv.org/pdf/2312.11458v1){: .btn .btn-green } |

**Abstract**: We propose a method for dynamic scene reconstruction using deformable 3D
Gaussians that is tailored for monocular video. Building upon the efficiency of
Gaussian splatting, our approach extends the representation to accommodate
dynamic elements via a deformable set of Gaussians residing in a canonical
space, and a time-dependent deformation field defined by a multi-layer
perceptron (MLP). Moreover, under the assumption that most natural scenes have
large regions that remain static, we allow the MLP to focus its
representational power by additionally including a static Gaussian point cloud.
The concatenated dynamic and static point clouds form the input for the
Gaussian Splatting rasterizer, enabling real-time rendering. The differentiable
pipeline is optimized end-to-end with a self-supervised rendering loss. Our
method achieves results that are comparable to state-of-the-art dynamic neural
radiance field methods while allowing much faster optimization and rendering.
Project website: https://lynl7130.github.io/gaufre/index.html

Comments:
- 10 pages, 8 figures, 4 tables

---

## AE-NeRF: Audio Enhanced Neural Radiance Field for Few Shot Talking Head  Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-18 | Dongze Li, Kang Zhao, Wei Wang, Bo Peng, Yingya Zhang, Jing Dong, Tieniu Tan | cs.CV | [PDF](http://arxiv.org/pdf/2312.10921v1){: .btn .btn-green } |

**Abstract**: Audio-driven talking head synthesis is a promising topic with wide
applications in digital human, film making and virtual reality. Recent
NeRF-based approaches have shown superiority in quality and fidelity compared
to previous studies. However, when it comes to few-shot talking head
generation, a practical scenario where only few seconds of talking video is
available for one identity, two limitations emerge: 1) they either have no base
model, which serves as a facial prior for fast convergence, or ignore the
importance of audio when building the prior; 2) most of them overlook the
degree of correlation between different face regions and audio, e.g., mouth is
audio related, while ear is audio independent. In this paper, we present Audio
Enhanced Neural Radiance Field (AE-NeRF) to tackle the above issues, which can
generate realistic portraits of a new speaker with fewshot dataset.
Specifically, we introduce an Audio Aware Aggregation module into the feature
fusion stage of the reference scheme, where the weight is determined by the
similarity of audio between reference and target image. Then, an Audio-Aligned
Face Generation strategy is proposed to model the audio related and audio
independent regions respectively, with a dual-NeRF framework. Extensive
experiments have shown AE-NeRF surpasses the state-of-the-art on image
fidelity, audio-lip synchronization, and generalization ability, even in
limited training set or training iterations.

Comments:
- Accepted by AAAI 2024

---

## GAvatar: Animatable 3D Gaussian Avatars with Implicit Mesh Learning

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-18 | Ye Yuan, Xueting Li, Yangyi Huang, Shalini De Mello, Koki Nagano, Jan Kautz, Umar Iqbal | cs.CV | [PDF](http://arxiv.org/pdf/2312.11461v1){: .btn .btn-green } |

**Abstract**: Gaussian splatting has emerged as a powerful 3D representation that harnesses
the advantages of both explicit (mesh) and implicit (NeRF) 3D representations.
In this paper, we seek to leverage Gaussian splatting to generate realistic
animatable avatars from textual descriptions, addressing the limitations (e.g.,
flexibility and efficiency) imposed by mesh or NeRF-based representations.
However, a naive application of Gaussian splatting cannot generate high-quality
animatable avatars and suffers from learning instability; it also cannot
capture fine avatar geometries and often leads to degenerate body parts. To
tackle these problems, we first propose a primitive-based 3D Gaussian
representation where Gaussians are defined inside pose-driven primitives to
facilitate animation. Second, to stabilize and amortize the learning of
millions of Gaussians, we propose to use neural implicit fields to predict the
Gaussian attributes (e.g., colors). Finally, to capture fine avatar geometries
and extract detailed meshes, we propose a novel SDF-based implicit mesh
learning approach for 3D Gaussians that regularizes the underlying geometries
and extracts highly detailed textured meshes. Our proposed method, GAvatar,
enables the large-scale generation of diverse animatable avatars using only
text prompts. GAvatar significantly surpasses existing methods in terms of both
appearance and geometry quality, and achieves extremely fast rendering (100
fps) at 1K resolution.

Comments:
- Project website: https://nvlabs.github.io/GAvatar

---

## PNeRFLoc: Visual Localization with Point-based Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-17 | Boming Zhao, Luwei Yang, Mao Mao, Hujun Bao, Zhaopeng Cui | cs.CV | [PDF](http://arxiv.org/pdf/2312.10649v1){: .btn .btn-green } |

**Abstract**: Due to the ability to synthesize high-quality novel views, Neural Radiance
Fields (NeRF) have been recently exploited to improve visual localization in a
known environment. However, the existing methods mostly utilize NeRFs for data
augmentation to improve the regression model training, and the performance on
novel viewpoints and appearances is still limited due to the lack of geometric
constraints. In this paper, we propose a novel visual localization framework,
\ie, PNeRFLoc, based on a unified point-based representation. On the one hand,
PNeRFLoc supports the initial pose estimation by matching 2D and 3D feature
points as traditional structure-based methods; on the other hand, it also
enables pose refinement with novel view synthesis using rendering-based
optimization. Specifically, we propose a novel feature adaption module to close
the gaps between the features for visual localization and neural rendering. To
improve the efficacy and efficiency of neural rendering-based optimization, we
also develop an efficient rendering-based framework with a warping loss
function. Furthermore, several robustness techniques are developed to handle
illumination changes and dynamic objects for outdoor scenarios. Experiments
demonstrate that PNeRFLoc performs the best on synthetic data when the NeRF
model can be well learned and performs on par with the SOTA method on the
visual localization benchmark datasets.

Comments:
- Accepted to AAAI 2024

---

## Learning Dense Correspondence for NeRF-Based Face Reenactment

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-16 | Songlin Yang, Wei Wang, Yushi Lan, Xiangyu Fan, Bo Peng, Lei Yang, Jing Dong | cs.CV | [PDF](http://arxiv.org/pdf/2312.10422v2){: .btn .btn-green } |

**Abstract**: Face reenactment is challenging due to the need to establish dense
correspondence between various face representations for motion transfer. Recent
studies have utilized Neural Radiance Field (NeRF) as fundamental
representation, which further enhanced the performance of multi-view face
reenactment in photo-realism and 3D consistency. However, establishing dense
correspondence between different face NeRFs is non-trivial, because implicit
representations lack ground-truth correspondence annotations like mesh-based 3D
parametric models (e.g., 3DMM) with index-aligned vertexes. Although aligning
3DMM space with NeRF-based face representations can realize motion control, it
is sub-optimal for their limited face-only modeling and low identity fidelity.
Therefore, we are inspired to ask: Can we learn the dense correspondence
between different NeRF-based face representations without a 3D parametric model
prior? To address this challenge, we propose a novel framework, which adopts
tri-planes as fundamental NeRF representation and decomposes face tri-planes
into three components: canonical tri-planes, identity deformations, and motion.
In terms of motion control, our key contribution is proposing a Plane
Dictionary (PlaneDict) module, which efficiently maps the motion conditions to
a linear weighted addition of learnable orthogonal plane bases. To the best of
our knowledge, our framework is the first method that achieves one-shot
multi-view face reenactment without a 3D parametric model prior. Extensive
experiments demonstrate that we produce better results in fine-grained motion
control and identity preservation than previous methods.

Comments:
- Accepted by Proceedings of the AAAI Conference on Artificial
  Intelligence, 2024

---

## RANRAC: Robust Neural Scene Representations via Random Ray Consensus



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-15 | Benno Buschmann, Andreea Dogaru, Elmar Eisemann, Michael Weinmann, Bernhard Egger | cs.CV | [PDF](http://arxiv.org/pdf/2312.09780v1){: .btn .btn-green } |

**Abstract**: We introduce RANRAC, a robust reconstruction algorithm for 3D objects
handling occluded and distracted images, which is a particularly challenging
scenario that prior robust reconstruction methods cannot deal with. Our
solution supports single-shot reconstruction by involving light-field networks,
and is also applicable to photo-realistic, robust, multi-view reconstruction
from real-world images based on neural radiance fields. While the algorithm
imposes certain limitations on the scene representation and, thereby, the
supported scene types, it reliably detects and excludes inconsistent
perspectives, resulting in clean images without floating artifacts. Our
solution is based on a fuzzy adaption of the random sample consensus paradigm,
enabling its application to large scale models. We interpret the minimal number
of samples to determine the model parameters as a tunable hyperparameter. This
is applicable, as a cleaner set of samples improves reconstruction quality.
Further, this procedure also handles outliers. Especially for conditioned
models, it can result in the same local minimum in the latent space as would be
obtained with a completely clean set. We report significant improvements for
novel-view synthesis in occluded scenarios, of up to 8dB PSNR compared to the
baseline.

---

## Towards Transferable Targeted 3D Adversarial Attack in the Physical  World

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-15 | Yao Huang, Yinpeng Dong, Shouwei Ruan, Xiao Yang, Hang Su, Xingxing Wei | cs.CV | [PDF](http://arxiv.org/pdf/2312.09558v1){: .btn .btn-green } |

**Abstract**: Compared with transferable untargeted attacks, transferable targeted
adversarial attacks could specify the misclassification categories of
adversarial samples, posing a greater threat to security-critical tasks. In the
meanwhile, 3D adversarial samples, due to their potential of multi-view
robustness, can more comprehensively identify weaknesses in existing deep
learning systems, possessing great application value. However, the field of
transferable targeted 3D adversarial attacks remains vacant. The goal of this
work is to develop a more effective technique that could generate transferable
targeted 3D adversarial examples, filling the gap in this field. To achieve
this goal, we design a novel framework named TT3D that could rapidly
reconstruct from few multi-view images into Transferable Targeted 3D textured
meshes. While existing mesh-based texture optimization methods compute
gradients in the high-dimensional mesh space and easily fall into local optima,
leading to unsatisfactory transferability and distinct distortions, TT3D
innovatively performs dual optimization towards both feature grid and
Multi-layer Perceptron (MLP) parameters in the grid-based NeRF space, which
significantly enhances black-box transferability while enjoying naturalness.
Experimental results show that TT3D not only exhibits superior cross-model
transferability but also maintains considerable adaptability across different
renders and vision tasks. More importantly, we produce 3D adversarial examples
with 3D printing techniques in the real world and verify their robust
performance under various scenarios.

Comments:
- 11 pages, 7 figures

---

## Exploring the Feasibility of Generating Realistic 3D Models of  Endangered Species Using DreamGaussian: An Analysis of Elevation Angle's  Impact on Model Generation

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-15 | Selcuk Anil Karatopak, Deniz Sen | cs.CV | [PDF](http://arxiv.org/pdf/2312.09682v1){: .btn .btn-green } |

**Abstract**: Many species face the threat of extinction. It's important to study these
species and gather information about them as much as possible to preserve
biodiversity. Due to the rarity of endangered species, there is a limited
amount of data available, making it difficult to apply data requiring
generative AI methods to this domain. We aim to study the feasibility of
generating consistent and real-like 3D models of endangered animals using
limited data. Such a phenomenon leads us to utilize zero-shot stable diffusion
models that can generate a 3D model out of a single image of the target
species. This paper investigates the intricate relationship between elevation
angle and the output quality of 3D model generation, focusing on the innovative
approach presented in DreamGaussian. DreamGaussian, a novel framework utilizing
Generative Gaussian Splatting along with novel mesh extraction and refinement
algorithms, serves as the focal point of our study. We conduct a comprehensive
analysis, analyzing the effect of varying elevation angles on DreamGaussian's
ability to reconstruct 3D scenes accurately. Through an empirical evaluation,
we demonstrate how changes in elevation angle impact the generated images'
spatial coherence, structural integrity, and perceptual realism. We observed
that giving a correct elevation angle with the input image significantly
affects the result of the generated 3D model. We hope this study to be
influential for the usability of AI to preserve endangered animals; while the
penultimate aim is to obtain a model that can output biologically consistent 3D
models via small samples, the qualitative interpretation of an existing
state-of-the-art model such as DreamGaussian will be a step forward in our
goal.

---

## SLS4D: Sparse Latent Space for 4D Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-15 | Qi-Yuan Feng, Hao-Xiang Chen, Qun-Ce Xu, Tai-Jiang Mu | cs.CV | [PDF](http://arxiv.org/pdf/2312.09743v1){: .btn .btn-green } |

**Abstract**: Neural radiance field (NeRF) has achieved great success in novel view
synthesis and 3D representation for static scenarios. Existing dynamic NeRFs
usually exploit a locally dense grid to fit the deformation field; however,
they fail to capture the global dynamics and concomitantly yield models of
heavy parameters. We observe that the 4D space is inherently sparse. Firstly,
the deformation field is sparse in spatial but dense in temporal due to the
continuity of of motion. Secondly, the radiance field is only valid on the
surface of the underlying scene, usually occupying a small fraction of the
whole space. We thus propose to represent the 4D scene using a learnable sparse
latent space, a.k.a. SLS4D. Specifically, SLS4D first uses dense learnable time
slot features to depict the temporal space, from which the deformation field is
fitted with linear multi-layer perceptions (MLP) to predict the displacement of
a 3D position at any time. It then learns the spatial features of a 3D position
using another sparse latent space. This is achieved by learning the adaptive
weights of each latent code with the attention mechanism. Extensive experiments
demonstrate the effectiveness of our SLS4D: it achieves the best 4D novel view
synthesis using only about $6\%$ parameters of the most recent work.

Comments:
- 10 pages, 6 figures

---

## LAENeRF: Local Appearance Editing for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-15 | Lukas Radl, Michael Steiner, Andreas Kurz, Markus Steinberger | cs.CV | [PDF](http://arxiv.org/pdf/2312.09913v1){: .btn .btn-green } |

**Abstract**: Due to the omnipresence of Neural Radiance Fields (NeRFs), the interest
towards editable implicit 3D representations has surged over the last years.
However, editing implicit or hybrid representations as used for NeRFs is
difficult due to the entanglement of appearance and geometry encoded in the
model parameters. Despite these challenges, recent research has shown first
promising steps towards photorealistic and non-photorealistic appearance edits.
The main open issues of related work include limited interactivity, a lack of
support for local edits and large memory requirements, rendering them less
useful in practice. We address these limitations with LAENeRF, a unified
framework for photorealistic and non-photorealistic appearance editing of
NeRFs. To tackle local editing, we leverage a voxel grid as starting point for
region selection. We learn a mapping from expected ray terminations to final
output color, which can optionally be supervised by a style loss, resulting in
a framework which can perform photorealistic and non-photorealistic appearance
editing of selected regions. Relying on a single point per ray for our mapping,
we limit memory requirements and enable fast optimization. To guarantee
interactivity, we compose the output color using a set of learned, modifiable
base colors, composed with additive layer mixing. Compared to concurrent work,
LAENeRF enables recoloring and stylization while keeping processing time low.
Furthermore, we demonstrate that our approach surpasses baseline methods both
quantitatively and qualitatively.

Comments:
- Project website: https://r4dl.github.io/LAENeRF/

---

## SlimmeRF: Slimmable Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-15 | Shiran Yuan, Hao Zhao | cs.CV | [PDF](http://arxiv.org/pdf/2312.10034v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) and its variants have recently emerged as
successful methods for novel view synthesis and 3D scene reconstruction.
However, most current NeRF models either achieve high accuracy using large
model sizes, or achieve high memory-efficiency by trading off accuracy. This
limits the applicable scope of any single model, since high-accuracy models
might not fit in low-memory devices, and memory-efficient models might not
satisfy high-quality requirements. To this end, we present SlimmeRF, a model
that allows for instant test-time trade-offs between model size and accuracy
through slimming, thus making the model simultaneously suitable for scenarios
with different computing budgets. We achieve this through a newly proposed
algorithm named Tensorial Rank Incrementation (TRaIn) which increases the rank
of the model's tensorial representation gradually during training. We also
observe that our model allows for more effective trade-offs in sparse-view
scenarios, at times even achieving higher accuracy after being slimmed. We
credit this to the fact that erroneous information such as floaters tend to be
stored in components corresponding to higher ranks. Our implementation is
available at https://github.com/Shiran-Yuan/SlimmeRF.

Comments:
- 3DV 2024 Oral, Project Page: https://shiran-yuan.github.io/SlimmeRF/,
  Code: https://github.com/Shiran-Yuan/SlimmeRF/

---

## Customize-It-3D: High-Quality 3D Creation from A Single Image Using  Subject-Specific Knowledge Prior

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-15 | Nan Huang, Ting Zhang, Yuhui Yuan, Dong Chen, Shanghang Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2312.11535v2){: .btn .btn-green } |

**Abstract**: In this paper, we present a novel two-stage approach that fully utilizes the
information provided by the reference image to establish a customized knowledge
prior for image-to-3D generation. While previous approaches primarily rely on a
general diffusion prior, which struggles to yield consistent results with the
reference image, we propose a subject-specific and multi-modal diffusion model.
This model not only aids NeRF optimization by considering the shading mode for
improved geometry but also enhances texture from the coarse results to achieve
superior refinement. Both aspects contribute to faithfully aligning the 3D
content with the subject. Extensive experiments showcase the superiority of our
method, Customize-It-3D, outperforming previous works by a substantial margin.
It produces faithful 360-degree reconstructions with impressive visual quality,
making it well-suited for various applications, including text-to-3D creation.

Comments:
- Project Page: https://nnanhuang.github.io/projects/customize-it-3d/

---

## FastSR-NeRF: Improving NeRF Efficiency on Consumer Devices with A Simple  Super-Resolution Pipeline

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-15 | Chien-Yu Lin, Qichen Fu, Thomas Merth, Karren Yang, Anurag Ranjan | cs.CV | [PDF](http://arxiv.org/pdf/2312.11537v2){: .btn .btn-green } |

**Abstract**: Super-resolution (SR) techniques have recently been proposed to upscale the
outputs of neural radiance fields (NeRF) and generate high-quality images with
enhanced inference speeds. However, existing NeRF+SR methods increase training
overhead by using extra input features, loss functions, and/or expensive
training procedures such as knowledge distillation. In this paper, we aim to
leverage SR for efficiency gains without costly training or architectural
changes. Specifically, we build a simple NeRF+SR pipeline that directly
combines existing modules, and we propose a lightweight augmentation technique,
random patch sampling, for training. Compared to existing NeRF+SR methods, our
pipeline mitigates the SR computing overhead and can be trained up to 23x
faster, making it feasible to run on consumer devices such as the Apple
MacBook. Experiments show our pipeline can upscale NeRF outputs by 2-4x while
maintaining high quality, increasing inference speeds by up to 18x on an NVIDIA
V100 GPU and 12.8x on an M1 Pro chip. We conclude that SR can be a simple but
effective technique for improving the efficiency of NeRF models for consumer
devices.

Comments:
- WACV 2024 (Oral)

---

## SpectralNeRF: Physically Based Spectral Rendering with Neural Radiance  Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Ru Li, Jia Liu, Guanghui Liu, Shengping Zhang, Bing Zeng, Shuaicheng Liu | cs.CV | [PDF](http://arxiv.org/pdf/2312.08692v1){: .btn .btn-green } |

**Abstract**: In this paper, we propose SpectralNeRF, an end-to-end Neural Radiance Field
(NeRF)-based architecture for high-quality physically based rendering from a
novel spectral perspective. We modify the classical spectral rendering into two
main steps, 1) the generation of a series of spectrum maps spanning different
wavelengths, 2) the combination of these spectrum maps for the RGB output. Our
SpectralNeRF follows these two steps through the proposed multi-layer
perceptron (MLP)-based architecture (SpectralMLP) and Spectrum Attention UNet
(SAUNet). Given the ray origin and the ray direction, the SpectralMLP
constructs the spectral radiance field to obtain spectrum maps of novel views,
which are then sent to the SAUNet to produce RGB images of white-light
illumination. Applying NeRF to build up the spectral rendering is a more
physically-based way from the perspective of ray-tracing. Further, the spectral
radiance fields decompose difficult scenes and improve the performance of
NeRF-based methods. Comprehensive experimental results demonstrate the proposed
SpectralNeRF is superior to recent NeRF-based methods when synthesizing new
views on synthetic and real datasets. The codes and datasets are available at
https://github.com/liru0126/SpectralNeRF.

Comments:
- Accepted by AAAI 2024

---

## CF-NeRF: Camera Parameter Free Neural Radiance Fields with Incremental  Learning

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Qingsong Yan, Qiang Wang, Kaiyong Zhao, Jie Chen, Bo Li, Xiaowen Chu, Fei Deng | cs.CV | [PDF](http://arxiv.org/pdf/2312.08760v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have demonstrated impressive performance in
novel view synthesis. However, NeRF and most of its variants still rely on
traditional complex pipelines to provide extrinsic and intrinsic camera
parameters, such as COLMAP. Recent works, like NeRFmm, BARF, and L2G-NeRF,
directly treat camera parameters as learnable and estimate them through
differential volume rendering. However, these methods work for forward-looking
scenes with slight motions and fail to tackle the rotation scenario in
practice. To overcome this limitation, we propose a novel \underline{c}amera
parameter \underline{f}ree neural radiance field (CF-NeRF), which incrementally
reconstructs 3D representations and recovers the camera parameters inspired by
incremental structure from motion (SfM). Given a sequence of images, CF-NeRF
estimates the camera parameters of images one by one and reconstructs the scene
through initialization, implicit localization, and implicit optimization. To
evaluate our method, we use a challenging real-world dataset NeRFBuster which
provides 12 scenes under complex trajectories. Results demonstrate that CF-NeRF
is robust to camera rotation and achieves state-of-the-art results without
providing prior information and constraints.

Comments:
- Accepted at the Thirty-Eighth AAAI Conference on Artificial
  Intelligence (AAAI24)

---

## VaLID: Variable-Length Input Diffusion for Novel View Synthesis



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Shijie Li, Farhad G. Zanjani, Haitam Ben Yahia, Yuki M. Asano, Juergen Gall, Amirhossein Habibian | cs.CV | [PDF](http://arxiv.org/pdf/2312.08892v1){: .btn .btn-green } |

**Abstract**: Novel View Synthesis (NVS), which tries to produce a realistic image at the
target view given source view images and their corresponding poses, is a
fundamental problem in 3D Vision. As this task is heavily under-constrained,
some recent work, like Zero123, tries to solve this problem with generative
modeling, specifically using pre-trained diffusion models. Although this
strategy generalizes well to new scenes, compared to neural radiance
field-based methods, it offers low levels of flexibility. For example, it can
only accept a single-view image as input, despite realistic applications often
offering multiple input images. This is because the source-view images and
corresponding poses are processed separately and injected into the model at
different stages. Thus it is not trivial to generalize the model into
multi-view source images, once they are available. To solve this issue, we try
to process each pose image pair separately and then fuse them as a unified
visual representation which will be injected into the model to guide image
synthesis at the target-views. However, inconsistency and computation costs
increase as the number of input source-view images increases. To solve these
issues, the Multi-view Cross Former module is proposed which maps
variable-length input data to fix-size output data. A two-stage training
strategy is introduced to further improve the efficiency during training time.
Qualitative and quantitative evaluation over multiple datasets demonstrates the
effectiveness of the proposed method against previous approaches. The code will
be released according to the acceptance.

Comments:
- paper and supplementary material

---

## Scene 3-D Reconstruction System in Scattering Medium

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Zhuoyifan Zhang, Lu Zhang, Liang Wang, Haoming Wu | cs.CV | [PDF](http://arxiv.org/pdf/2312.09005v1){: .btn .btn-green } |

**Abstract**: The research on neural radiance fields for new view synthesis has experienced
explosive growth with the development of new models and extensions. The NERF
algorithm, suitable for underwater scenes or scattering media, is also
evolving. Existing underwater 3D reconstruction systems still face challenges
such as extensive training time and low rendering efficiency. This paper
proposes an improved underwater 3D reconstruction system to address these
issues and achieve rapid, high-quality 3D reconstruction.To begin with, we
enhance underwater videos captured by a monocular camera to correct the poor
image quality caused by the physical properties of the water medium while
ensuring consistency in enhancement across adjacent frames. Subsequently, we
perform keyframe selection on the video frames to optimize resource utilization
and eliminate the impact of dynamic objects on the reconstruction results. The
selected keyframes, after pose estimation using COLMAP, undergo a
three-dimensional reconstruction improvement process using neural radiance
fields based on multi-resolution hash coding for model construction and
rendering.

---

## iComMa: Inverting 3D Gaussians Splatting for Camera Pose Estimation via  Comparing and Matching

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Yuan Sun, Xuan Wang, Yunfan Zhang, Jie Zhang, Caigui Jiang, Yu Guo, Fei Wang | cs.CV | [PDF](http://arxiv.org/pdf/2312.09031v1){: .btn .btn-green } |

**Abstract**: We present a method named iComMa to address the 6D pose estimation problem in
computer vision. The conventional pose estimation methods typically rely on the
target's CAD model or necessitate specific network training tailored to
particular object classes. Some existing methods address mesh-free 6D pose
estimation by employing the inversion of a Neural Radiance Field (NeRF), aiming
to overcome the aforementioned constraints. However, it still suffers from
adverse initializations. By contrast, we model the pose estimation as the
problem of inverting the 3D Gaussian Splatting (3DGS) with both the comparing
and matching loss. In detail, a render-and-compare strategy is adopted for the
precise estimation of poses. Additionally, a matching module is designed to
enhance the model's robustness against adverse initializations by minimizing
the distances between 2D keypoints. This framework systematically incorporates
the distinctive characteristics and inherent rationale of render-and-compare
and matching-based approaches. This comprehensive consideration equips the
framework to effectively address a broader range of intricate and challenging
scenarios, including instances with substantial angular deviations, all while
maintaining a high level of prediction accuracy. Experimental results
demonstrate the superior precision and robustness of our proposed jointly
optimized framework when evaluated on synthetic and complex real-world data in
challenging scenarios.

Comments:
- 10 pages, 5 figures

---

## Aleth-NeRF: Illumination Adaptive NeRF with Concealing Field Assumption

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Ziteng Cui, Lin Gu, Xiao Sun, Xianzheng Ma, Yu Qiao, Tatsuya Harada | cs.CV | [PDF](http://arxiv.org/pdf/2312.09093v2){: .btn .btn-green } |

**Abstract**: The standard Neural Radiance Fields (NeRF) paradigm employs a viewer-centered
methodology, entangling the aspects of illumination and material reflectance
into emission solely from 3D points. This simplified rendering approach
presents challenges in accurately modeling images captured under adverse
lighting conditions, such as low light or over-exposure. Motivated by the
ancient Greek emission theory that posits visual perception as a result of rays
emanating from the eyes, we slightly refine the conventional NeRF framework to
train NeRF under challenging light conditions and generate normal-light
condition novel views unsupervised. We introduce the concept of a "Concealing
Field," which assigns transmittance values to the surrounding air to account
for illumination effects. In dark scenarios, we assume that object emissions
maintain a standard lighting level but are attenuated as they traverse the air
during the rendering process. Concealing Field thus compel NeRF to learn
reasonable density and colour estimations for objects even in dimly lit
situations. Similarly, the Concealing Field can mitigate over-exposed emissions
during the rendering stage. Furthermore, we present a comprehensive multi-view
dataset captured under challenging illumination conditions for evaluation. Our
code and dataset available at https://github.com/cuiziteng/Aleth-NeRF

Comments:
- AAAI 2024, code available at https://github.com/cuiziteng/Aleth-NeRF
  Modified version of previous paper arXiv:2303.05807

---

## ColNeRF: Collaboration for Generalizable Sparse Input Neural Radiance  Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Zhangkai Ni, Peiqi Yang, Wenhan Yang, Hanli Wang, Lin Ma, Sam Kwong | cs.CV | [PDF](http://arxiv.org/pdf/2312.09095v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have demonstrated impressive potential in
synthesizing novel views from dense input, however, their effectiveness is
challenged when dealing with sparse input. Existing approaches that incorporate
additional depth or semantic supervision can alleviate this issue to an extent.
However, the process of supervision collection is not only costly but also
potentially inaccurate, leading to poor performance and generalization ability
in diverse scenarios. In our work, we introduce a novel model: the
Collaborative Neural Radiance Fields (ColNeRF) designed to work with sparse
input. The collaboration in ColNeRF includes both the cooperation between
sparse input images and the cooperation between the output of the neural
radiation field. Through this, we construct a novel collaborative module that
aligns information from various views and meanwhile imposes self-supervised
constraints to ensure multi-view consistency in both geometry and appearance. A
Collaborative Cross-View Volume Integration module (CCVI) is proposed to
capture complex occlusions and implicitly infer the spatial location of
objects. Moreover, we introduce self-supervision of target rays projected in
multiple directions to ensure geometric and color consistency in adjacent
regions. Benefiting from the collaboration at the input and output ends,
ColNeRF is capable of capturing richer and more generalized scene
representation, thereby facilitating higher-quality results of the novel view
synthesis. Extensive experiments demonstrate that ColNeRF outperforms
state-of-the-art sparse input generalizable NeRF methods. Furthermore, our
approach exhibits superiority in fine-tuning towards adapting to new scenes,
achieving competitive performance compared to per-scene optimized NeRF-based
methods while significantly reducing computational costs. Our code is available
at: https://github.com/eezkni/ColNeRF.

---

## Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D  Reconstruction with Transformers

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Yan-Pei Cao, Song-Hai Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2312.09147v2){: .btn .btn-green } |

**Abstract**: Recent advancements in 3D reconstruction from single images have been driven
by the evolution of generative models. Prominent among these are methods based
on Score Distillation Sampling (SDS) and the adaptation of diffusion models in
the 3D domain. Despite their progress, these techniques often face limitations
due to slow optimization or rendering processes, leading to extensive training
and optimization times. In this paper, we introduce a novel approach for
single-view reconstruction that efficiently generates a 3D model from a single
image via feed-forward inference. Our method utilizes two transformer-based
networks, namely a point decoder and a triplane decoder, to reconstruct 3D
objects using a hybrid Triplane-Gaussian intermediate representation. This
hybrid representation strikes a balance, achieving a faster rendering speed
compared to implicit representations while simultaneously delivering superior
rendering quality than explicit representations. The point decoder is designed
for generating point clouds from single images, offering an explicit
representation which is then utilized by the triplane decoder to query Gaussian
features for each point. This design choice addresses the challenges associated
with directly regressing explicit 3D Gaussian attributes characterized by their
non-structural nature. Subsequently, the 3D Gaussians are decoded by an MLP to
enable rapid rendering through splatting. Both decoders are built upon a
scalable, transformer-based architecture and have been efficiently trained on
large-scale 3D datasets. The evaluations conducted on both synthetic datasets
and real-world images demonstrate that our method not only achieves higher
quality but also ensures a faster runtime in comparison to previous
state-of-the-art techniques. Please see our project page at
https://zouzx.github.io/TriplaneGaussian/.

Comments:
- Project Page: https://zouzx.github.io/TriplaneGaussian/

---

## 3DGS-Avatar: Animatable Avatars via Deformable 3D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas Geiger, Siyu Tang | cs.CV | [PDF](http://arxiv.org/pdf/2312.09228v2){: .btn .btn-green } |

**Abstract**: We introduce an approach that creates animatable human avatars from monocular
videos using 3D Gaussian Splatting (3DGS). Existing methods based on neural
radiance fields (NeRFs) achieve high-quality novel-view/novel-pose image
synthesis but often require days of training, and are extremely slow at
inference time. Recently, the community has explored fast grid structures for
efficient training of clothed avatars. Albeit being extremely fast at training,
these methods can barely achieve an interactive rendering frame rate with
around 15 FPS. In this paper, we use 3D Gaussian Splatting and learn a
non-rigid deformation network to reconstruct animatable clothed human avatars
that can be trained within 30 minutes and rendered at real-time frame rates
(50+ FPS). Given the explicit nature of our representation, we further
introduce as-isometric-as-possible regularizations on both the Gaussian mean
vectors and the covariance matrices, enhancing the generalization of our model
on highly articulated unseen poses. Experimental results show that our method
achieves comparable and even better performance compared to state-of-the-art
approaches on animatable avatar creation from a monocular input, while being
400x and 250x faster in training and inference, respectively.

Comments:
- Project page: https://neuralbodies.github.io/3DGS-Avatar

---

## OccNeRF: Self-Supervised Multi-Camera Occupancy Prediction with Neural  Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Chubin Zhang, Juncheng Yan, Yi Wei, Jiaxin Li, Li Liu, Yansong Tang, Yueqi Duan, Jiwen Lu | cs.CV | [PDF](http://arxiv.org/pdf/2312.09243v1){: .btn .btn-green } |

**Abstract**: As a fundamental task of vision-based perception, 3D occupancy prediction
reconstructs 3D structures of surrounding environments. It provides detailed
information for autonomous driving planning and navigation. However, most
existing methods heavily rely on the LiDAR point clouds to generate occupancy
ground truth, which is not available in the vision-based system. In this paper,
we propose an OccNeRF method for self-supervised multi-camera occupancy
prediction. Different from bounded 3D occupancy labels, we need to consider
unbounded scenes with raw image supervision. To solve the issue, we
parameterize the reconstructed occupancy fields and reorganize the sampling
strategy. The neural rendering is adopted to convert occupancy fields to
multi-camera depth maps, supervised by multi-frame photometric consistency.
Moreover, for semantic occupancy prediction, we design several strategies to
polish the prompts and filter the outputs of a pretrained open-vocabulary 2D
segmentation model. Extensive experiments for both self-supervised depth
estimation and semantic occupancy prediction tasks on nuScenes dataset
demonstrate the effectiveness of our method.

Comments:
- Code: https://github.com/LinShan-Bin/OccNeRF

---

## ZeroRF: Fast Sparse View 360° Reconstruction with Zero Pretraining

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Ruoxi Shi, Xinyue Wei, Cheng Wang, Hao Su | cs.CV | [PDF](http://arxiv.org/pdf/2312.09249v1){: .btn .btn-green } |

**Abstract**: We present ZeroRF, a novel per-scene optimization method addressing the
challenge of sparse view 360{\deg} reconstruction in neural field
representations. Current breakthroughs like Neural Radiance Fields (NeRF) have
demonstrated high-fidelity image synthesis but struggle with sparse input
views. Existing methods, such as Generalizable NeRFs and per-scene optimization
approaches, face limitations in data dependency, computational cost, and
generalization across diverse scenarios. To overcome these challenges, we
propose ZeroRF, whose key idea is to integrate a tailored Deep Image Prior into
a factorized NeRF representation. Unlike traditional methods, ZeroRF
parametrizes feature grids with a neural network generator, enabling efficient
sparse view 360{\deg} reconstruction without any pretraining or additional
regularization. Extensive experiments showcase ZeroRF's versatility and
superiority in terms of both quality and speed, achieving state-of-the-art
results on benchmark datasets. ZeroRF's significance extends to applications in
3D content generation and editing. Project page:
https://sarahweiii.github.io/zerorf/

Comments:
- Project page: https://sarahweiii.github.io/zerorf/

---

## Stable Score Distillation for High-Quality 3D Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Boshi Tang, Jianan Wang, Zhiyong Wu, Lei Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2312.09305v1){: .btn .btn-green } |

**Abstract**: Score Distillation Sampling (SDS) has exhibited remarkable performance in
conditional 3D content generation. However, a comprehensive understanding of
the SDS formulation is still lacking, hindering the development of 3D
generation. In this work, we present an interpretation of SDS as a combination
of three functional components: mode-disengaging, mode-seeking and
variance-reducing terms, and analyze the properties of each. We show that
problems such as over-smoothness and color-saturation result from the intrinsic
deficiency of the supervision terms and reveal that the variance-reducing term
introduced by SDS is sub-optimal. Additionally, we shed light on the adoption
of large Classifier-Free Guidance (CFG) scale for 3D generation. Based on the
analysis, we propose a simple yet effective approach named Stable Score
Distillation (SSD) which strategically orchestrates each term for high-quality
3D generation. Extensive experiments validate the efficacy of our approach,
demonstrating its ability to generate high-fidelity 3D content without
succumbing to issues such as over-smoothness and over-saturation, even under
low CFG conditions with the most challenging NeRF representation.

---

## LatentEditor: Text Driven Local Editing of 3D Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-14 | Umar Khalid, Hasan Iqbal, Nazmul Karim, Jing Hua, Chen Chen | cs.CV | [PDF](http://arxiv.org/pdf/2312.09313v2){: .btn .btn-green } |

**Abstract**: While neural fields have made significant strides in view synthesis and scene
reconstruction, editing them poses a formidable challenge due to their implicit
encoding of geometry and texture information from multi-view inputs. In this
paper, we introduce \textsc{LatentEditor}, an innovative framework designed to
empower users with the ability to perform precise and locally controlled
editing of neural fields using text prompts. Leveraging denoising diffusion
models, we successfully embed real-world scenes into the latent space,
resulting in a faster and more adaptable NeRF backbone for editing compared to
traditional methods. To enhance editing precision, we introduce a delta score
to calculate the 2D mask in the latent space that serves as a guide for local
modifications while preserving irrelevant regions. Our novel pixel-level
scoring approach harnesses the power of InstructPix2Pix (IP2P) to discern the
disparity between IP2P conditional and unconditional noise predictions in the
latent space. The edited latents conditioned on the 2D masks are then
iteratively updated in the training set to achieve 3D local editing. Our
approach achieves faster editing speeds and superior output quality compared to
existing 3D editing models, bridging the gap between textual instructions and
high-quality 3D scene editing in latent space. We show the superiority of our
approach on four benchmark 3D datasets, LLFF, IN2N, NeRFStudio and NeRF-Art.

Comments:
- Project Page: https://latenteditor.github.io/

---

## ProNeRF: Learning Efficient Projection-Aware Ray Sampling for  Fine-Grained Implicit Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-13 | Juan Luis Gonzalez Bello, Minh-Quan Viet Bui, Munchurl Kim | cs.CV | [PDF](http://arxiv.org/pdf/2312.08136v1){: .btn .btn-green } |

**Abstract**: Recent advances in neural rendering have shown that, albeit slow, implicit
compact models can learn a scene's geometries and view-dependent appearances
from multiple views. To maintain such a small memory footprint but achieve
faster inference times, recent works have adopted `sampler' networks that
adaptively sample a small subset of points along each ray in the implicit
neural radiance fields. Although these methods achieve up to a 10$\times$
reduction in rendering time, they still suffer from considerable quality
degradation compared to the vanilla NeRF. In contrast, we propose ProNeRF,
which provides an optimal trade-off between memory footprint (similar to NeRF),
speed (faster than HyperReel), and quality (better than K-Planes). ProNeRF is
equipped with a novel projection-aware sampling (PAS) network together with a
new training strategy for ray exploration and exploitation, allowing for
efficient fine-grained particle sampling. Our ProNeRF yields state-of-the-art
metrics, being 15-23x faster with 0.65dB higher PSNR than NeRF and yielding
0.95dB higher PSNR than the best published sampler-based method, HyperReel. Our
exploration and exploitation training strategy allows ProNeRF to learn the full
scenes' color and density distributions while also learning efficient ray
sampling focused on the highest-density regions. We provide extensive
experimental results that support the effectiveness of our method on the widely
adopted forward-facing and 360 datasets, LLFF and Blender, respectively.

Comments:
- Visit our project website at
  https://kaist-viclab.github.io/pronerf-site/

---

## Neural Radiance Fields for Transparent Object Using Visual Hull

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-13 | Heechan Yoon, Seungkyu Lee | cs.CV | [PDF](http://arxiv.org/pdf/2312.08118v1){: .btn .btn-green } |

**Abstract**: Unlike opaque object, novel view synthesis of transparent object is a
challenging task, because transparent object refracts light of background
causing visual distortions on the transparent object surface along the
viewpoint change. Recently introduced Neural Radiance Fields (NeRF) is a view
synthesis method. Thanks to its remarkable performance improvement, lots of
following applications based on NeRF in various topics have been developed.
However, if an object with a different refractive index is included in a scene
such as transparent object, NeRF shows limited performance because refracted
light ray at the surface of the transparent object is not appropriately
considered. To resolve the problem, we propose a NeRF-based method consisting
of the following three steps: First, we reconstruct a three-dimensional shape
of a transparent object using visual hull. Second, we simulate the refraction
of the rays inside of the transparent object according to Snell's law. Last, we
sample points through refracted rays and put them into NeRF. Experimental
evaluation results demonstrate that our method addresses the limitation of
conventional NeRF with transparent objects.

---

## 3DGEN: A GAN-based approach for generating novel 3D models from image  data



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-13 | Antoine Schnepf, Flavian Vasile, Ugo Tanielian | cs.CV | [PDF](http://arxiv.org/pdf/2312.08094v1){: .btn .btn-green } |

**Abstract**: The recent advances in text and image synthesis show a great promise for the
future of generative models in creative fields. However, a less explored area
is the one of 3D model generation, with a lot of potential applications to game
design, video production, and physical product design. In our paper, we present
3DGEN, a model that leverages the recent work on both Neural Radiance Fields
for object reconstruction and GAN-based image generation. We show that the
proposed architecture can generate plausible meshes for objects of the same
category as the training images and compare the resulting meshes with the
state-of-the-art baselines, leading to visible uplifts in generation quality.

Comments:
- Submitted to NeurIPS 2022 Machine Learning for Creativity and Design
  Workshop

---

## uSF: Learning Neural Semantic Field with Uncertainty

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-13 | Vsevolod Skorokhodov, Darya Drozdova, Dmitry Yudin | cs.CV | [PDF](http://arxiv.org/pdf/2312.08012v1){: .btn .btn-green } |

**Abstract**: Recently, there has been an increased interest in NeRF methods which
reconstruct differentiable representation of three-dimensional scenes. One of
the main limitations of such methods is their inability to assess the
confidence of the model in its predictions. In this paper, we propose a new
neural network model for the formation of extended vector representations,
called uSF, which allows the model to predict not only color and semantic label
of each point, but also estimate the corresponding values of uncertainty. We
show that with a small number of images available for training, a model
quantifying uncertainty performs better than a model without such
functionality. Code of the uSF approach is publicly available at
https://github.com/sevashasla/usf/.

Comments:
- 12 pages, 4 figures

---

## DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic  Autonomous Driving Scenes

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-13 | Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, Ming-Hsuan Yang | cs.CV | [PDF](http://arxiv.org/pdf/2312.07920v1){: .btn .btn-green } |

**Abstract**: We present DrivingGaussian, an efficient and effective framework for
surrounding dynamic autonomous driving scenes. For complex scenes with moving
objects, we first sequentially and progressively model the static background of
the entire scene with incremental static 3D Gaussians. We then leverage a
composite dynamic Gaussian graph to handle multiple moving objects,
individually reconstructing each object and restoring their accurate positions
and occlusion relationships within the scene. We further use a LiDAR prior for
Gaussian Splatting to reconstruct scenes with greater details and maintain
panoramic consistency. DrivingGaussian outperforms existing methods in driving
scene reconstruction and enables photorealistic surround-view synthesis with
high-fidelity and multi-camera consistency. The source code and trained models
will be released.

---

## COLMAP-Free 3D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-12 | Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A. Efros, Xiaolong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2312.07504v1){: .btn .btn-green } |

**Abstract**: While neural rendering has led to impressive advances in scene reconstruction
and novel view synthesis, it relies heavily on accurately pre-computed camera
poses. To relax this constraint, multiple efforts have been made to train
Neural Radiance Fields (NeRFs) without pre-processed camera poses. However, the
implicit representations of NeRFs provide extra challenges to optimize the 3D
structure and camera poses at the same time. On the other hand, the recently
proposed 3D Gaussian Splatting provides new opportunities given its explicit
point cloud representations. This paper leverages both the explicit geometric
representation and the continuity of the input video stream to perform novel
view synthesis without any SfM preprocessing. We process the input frames in a
sequential manner and progressively grow the 3D Gaussians set by taking one
input frame at a time, without the need to pre-compute the camera poses. Our
method significantly improves over previous approaches in view synthesis and
camera pose estimation under large motion changes. Our project page is
https://oasisyang.github.io/colmap-free-3dgs

Comments:
- Project Page: https://oasisyang.github.io/colmap-free-3dgs

---

## Unifying Correspondence, Pose and NeRF for Pose-Free Novel View  Synthesis from Stereo Pairs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-12 | Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jiaolong Yang, Seungryong Kim, Chong Luo | cs.CV | [PDF](http://arxiv.org/pdf/2312.07246v1){: .btn .btn-green } |

**Abstract**: This work delves into the task of pose-free novel view synthesis from stereo
pairs, a challenging and pioneering task in 3D vision. Our innovative
framework, unlike any before, seamlessly integrates 2D correspondence matching,
camera pose estimation, and NeRF rendering, fostering a synergistic enhancement
of these tasks. We achieve this through designing an architecture that utilizes
a shared representation, which serves as a foundation for enhanced 3D geometry
understanding. Capitalizing on the inherent interplay between the tasks, our
unified framework is trained end-to-end with the proposed training strategy to
improve overall model accuracy. Through extensive evaluations across diverse
indoor and outdoor scenes from two real-world datasets, we demonstrate that our
approach achieves substantial improvement over previous methodologies,
especially in scenarios characterized by extreme viewpoint changes and the
absence of accurate camera poses.

Comments:
- Project page: https://ku-cvlab.github.io/CoPoNeRF/

---

## WaterHE-NeRF: Water-ray Tracing Neural Radiance Fields for Underwater  Scene Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-12 | Jingchun Zhou, Tianyu Liang, Zongxin He, Dehuan Zhang, Weishi Zhang, Xianping Fu, Chongyi Li | cs.CV | [PDF](http://arxiv.org/pdf/2312.06946v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) technology demonstrates immense potential in
novel viewpoint synthesis tasks, due to its physics-based volumetric rendering
process, which is particularly promising in underwater scenes. Addressing the
limitations of existing underwater NeRF methods in handling light attenuation
caused by the water medium and the lack of real Ground Truth (GT) supervision,
this study proposes WaterHE-NeRF. We develop a new water-ray tracing field by
Retinex theory that precisely encodes color, density, and illuminance
attenuation in three-dimensional space. WaterHE-NeRF, through its illuminance
attenuation mechanism, generates both degraded and clear multi-view images and
optimizes image restoration by combining reconstruction loss with Wasserstein
distance. Additionally, the use of histogram equalization (HE) as pseudo-GT
enhances the network's accuracy in preserving original details and color
distribution. Extensive experiments on real underwater datasets and synthetic
datasets validate the effectiveness of WaterHE-NeRF. Our code will be made
publicly available.

---

## Creating Visual Effects with Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-11 | Cyrus Vachha | cs.CV | [PDF](http://arxiv.org/pdf/2401.08633v1){: .btn .btn-green } |

**Abstract**: We present a pipeline for integrating NeRFs into traditional compositing VFX
pipelines using Nerfstudio, an open-source framework for training and rendering
NeRFs. Our approach involves using Blender, a widely used open-source 3D
creation software, to align camera paths and composite NeRF renders with meshes
and other NeRFs, allowing for seamless integration of NeRFs into traditional
VFX pipelines. Our NeRF Blender add-on allows for more controlled camera
trajectories of photorealistic scenes, compositing meshes and other
environmental effects with NeRFs, and compositing multiple NeRFs in a single
scene.This approach of generating NeRF aligned camera paths can be adapted to
other 3D tool sets and workflows, enabling a more seamless integration of NeRFs
into visual effects and film production. Documentation can be found here:
https://docs.nerf.studio/extensions/blender_addon.html

Comments:
- 2 pages, 4 figures

---

## DreamControl: Control-Based Text-to-3D Generation with 3D Self-Prior

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-11 | Tianyu Huang, Yihan Zeng, Zhilu Zhang, Wan Xu, Hang Xu, Songcen Xu, Rynson W. H. Lau, Wangmeng Zuo | cs.CV | [PDF](http://arxiv.org/pdf/2312.06439v1){: .btn .btn-green } |

**Abstract**: 3D generation has raised great attention in recent years. With the success of
text-to-image diffusion models, the 2D-lifting technique becomes a promising
route to controllable 3D generation. However, these methods tend to present
inconsistent geometry, which is also known as the Janus problem. We observe
that the problem is caused mainly by two aspects, i.e., viewpoint bias in 2D
diffusion models and overfitting of the optimization objective. To address it,
we propose a two-stage 2D-lifting framework, namely DreamControl, which
optimizes coarse NeRF scenes as 3D self-prior and then generates fine-grained
objects with control-based score distillation. Specifically, adaptive viewpoint
sampling and boundary integrity metric are proposed to ensure the consistency
of generated priors. The priors are then regarded as input conditions to
maintain reasonable geometries, in which conditional LoRA and weighted score
are further proposed to optimize detailed textures. DreamControl can generate
high-quality 3D content in terms of both geometry consistency and texture
fidelity. Moreover, our control-based optimization guidance is applicable to
more downstream tasks, including user-guided generation and 3D animation. The
project page is available at https://github.com/tyhuang0428/DreamControl.

---

## Gaussian Splatting SLAM

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-11 | Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, Andrew J. Davison | cs.CV | [PDF](http://arxiv.org/pdf/2312.06741v1){: .btn .btn-green } |

**Abstract**: We present the first application of 3D Gaussian Splatting to incremental 3D
reconstruction using a single moving monocular or RGB-D camera. Our
Simultaneous Localisation and Mapping (SLAM) method, which runs live at 3fps,
utilises Gaussians as the only 3D representation, unifying the required
representation for accurate, efficient tracking, mapping, and high-quality
rendering. Several innovations are required to continuously reconstruct 3D
scenes with high fidelity from a live camera. First, to move beyond the
original 3DGS algorithm, which requires accurate poses from an offline
Structure from Motion (SfM) system, we formulate camera tracking for 3DGS using
direct optimisation against the 3D Gaussians, and show that this enables fast
and robust tracking with a wide basin of convergence. Second, by utilising the
explicit nature of the Gaussians, we introduce geometric verification and
regularisation to handle the ambiguities occurring in incremental 3D dense
reconstruction. Finally, we introduce a full SLAM system which not only
achieves state-of-the-art results in novel view synthesis and trajectory
estimation, but also reconstruction of tiny and even transparent objects.

Comments:
- First two authors contributed equally to this work. Project Page:
  https://rmurai.co.uk/projects/GaussianSplattingSLAM/ Video:
  https://www.youtube.com/watch?v=x604ghp9R_Q&ab_channel=DysonRoboticsLaboratoryatImperialCollege

---

## CorresNeRF: Image Correspondence Priors for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-11 | Yixing Lao, Xiaogang Xu, Zhipeng Cai, Xihui Liu, Hengshuang Zhao | cs.CV | [PDF](http://arxiv.org/pdf/2312.06642v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have achieved impressive results in novel view
synthesis and surface reconstruction tasks. However, their performance suffers
under challenging scenarios with sparse input views. We present CorresNeRF, a
novel method that leverages image correspondence priors computed by
off-the-shelf methods to supervise NeRF training. We design adaptive processes
for augmentation and filtering to generate dense and high-quality
correspondences. The correspondences are then used to regularize NeRF training
via the correspondence pixel reprojection and depth loss terms. We evaluate our
methods on novel view synthesis and surface reconstruction tasks with
density-based and SDF-based NeRF models on different datasets. Our method
outperforms previous methods in both photometric and geometric metrics. We show
that this simple yet effective technique of using correspondence priors can be
applied as a plug-and-play module across different NeRF variants. The project
page is at https://yxlao.github.io/corres-nerf.

---

## Learning Naturally Aggregated Appearance for Efficient 3D Editing



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-11 | Ka Leong Cheng, Qiuyu Wang, Zifan Shi, Kecheng Zheng, Yinghao Xu, Hao Ouyang, Qifeng Chen, Yujun Shen | cs.CV | [PDF](http://arxiv.org/pdf/2312.06657v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields, which represent a 3D scene as a color field and a
density field, have demonstrated great progress in novel view synthesis yet are
unfavorable for editing due to the implicitness. In view of such a deficiency,
we propose to replace the color field with an explicit 2D appearance
aggregation, also called canonical image, with which users can easily customize
their 3D editing via 2D image processing. To avoid the distortion effect and
facilitate convenient editing, we complement the canonical image with a
projection field that maps 3D points onto 2D pixels for texture lookup. This
field is carefully initialized with a pseudo canonical camera model and
optimized with offset regularity to ensure naturalness of the aggregated
appearance. Extensive experimental results on three datasets suggest that our
representation, dubbed AGAP, well supports various ways of 3D editing (e.g.,
stylization, interactive drawing, and content extraction) with no need of
re-optimization for each case, demonstrating its generalizability and
efficiency. Project page is available at https://felixcheng97.github.io/AGAP/.

Comments:
- Project Webpage: https://felixcheng97.github.io/AGAP/, Code:
  https://github.com/felixcheng97/AGAP

---

## Nuvo: Neural UV Mapping for Unruly 3D Representations



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-11 | Pratul P. Srinivasan, Stephan J. Garbin, Dor Verbin, Jonathan T. Barron, Ben Mildenhall | cs.CV | [PDF](http://arxiv.org/pdf/2312.05283v1){: .btn .btn-green } |

**Abstract**: Existing UV mapping algorithms are designed to operate on well-behaved
meshes, instead of the geometry representations produced by state-of-the-art 3D
reconstruction and generation techniques. As such, applying these methods to
the volume densities recovered by neural radiance fields and related techniques
(or meshes triangulated from such fields) results in texture atlases that are
too fragmented to be useful for tasks such as view synthesis or appearance
editing. We present a UV mapping method designed to operate on geometry
produced by 3D reconstruction and generation techniques. Instead of computing a
mapping defined on a mesh's vertices, our method Nuvo uses a neural field to
represent a continuous UV mapping, and optimizes it to be a valid and
well-behaved mapping for just the set of visible points, i.e. only points that
affect the scene's appearance. We show that our model is robust to the
challenges posed by ill-behaved geometry, and that it produces editable UV
mappings that can represent detailed appearance.

Comments:
- Project page at https://pratulsrinivasan.github.io/nuvo

---

## TeTriRF: Temporal Tri-Plane Radiance Fields for Efficient Free-Viewpoint  Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-10 | Minye Wu, Zehao Wang, Georgios Kouros, Tinne Tuytelaars | cs.CV | [PDF](http://arxiv.org/pdf/2312.06713v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) revolutionize the realm of visual media by
providing photorealistic Free-Viewpoint Video (FVV) experiences, offering
viewers unparalleled immersion and interactivity. However, the technology's
significant storage requirements and the computational complexity involved in
generation and rendering currently limit its broader application. To close this
gap, this paper presents Temporal Tri-Plane Radiance Fields (TeTriRF), a novel
technology that significantly reduces the storage size for Free-Viewpoint Video
(FVV) while maintaining low-cost generation and rendering. TeTriRF introduces a
hybrid representation with tri-planes and voxel grids to support scaling up to
long-duration sequences and scenes with complex motions or rapid changes. We
propose a group training scheme tailored to achieving high training efficiency
and yielding temporally consistent, low-entropy scene representations.
Leveraging these properties of the representations, we introduce a compression
pipeline with off-the-shelf video codecs, achieving an order of magnitude less
storage size compared to the state-of-the-art. Our experiments demonstrate that
TeTriRF can achieve competitive quality with a higher compression rate.

Comments:
- 13 pages, 11 figures

---

## ASH: Animatable Gaussian Splats for Efficient and Photoreal Human  Rendering

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-10 | Haokai Pang, Heming Zhu, Adam Kortylewski, Christian Theobalt, Marc Habermann | cs.CV | [PDF](http://arxiv.org/pdf/2312.05941v1){: .btn .btn-green } |

**Abstract**: Real-time rendering of photorealistic and controllable human avatars stands
as a cornerstone in Computer Vision and Graphics. While recent advances in
neural implicit rendering have unlocked unprecedented photorealism for digital
avatars, real-time performance has mostly been demonstrated for static scenes
only. To address this, we propose ASH, an animatable Gaussian splatting
approach for photorealistic rendering of dynamic humans in real-time. We
parameterize the clothed human as animatable 3D Gaussians, which can be
efficiently splatted into image space to generate the final rendering. However,
naively learning the Gaussian parameters in 3D space poses a severe challenge
in terms of compute. Instead, we attach the Gaussians onto a deformable
character model, and learn their parameters in 2D texture space, which allows
leveraging efficient 2D convolutional architectures that easily scale with the
required number of Gaussians. We benchmark ASH with competing methods on
pose-controllable avatars, demonstrating that our method outperforms existing
real-time methods by a large margin and shows comparable or even better results
than offline methods.

Comments:
- 13 pages, 7 figures. For project page, see
  https://vcai.mpi-inf.mpg.de/projects/ash/

---

## Learning for CasADi: Data-driven Models in Numerical Optimization



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-10 | Tim Salzmann, Jon Arrizabalaga, Joel Andersson, Marco Pavone, Markus Ryll | eess.SY | [PDF](http://arxiv.org/pdf/2312.05873v1){: .btn .btn-green } |

**Abstract**: While real-world problems are often challenging to analyze analytically, deep
learning excels in modeling complex processes from data. Existing optimization
frameworks like CasADi facilitate seamless usage of solvers but face challenges
when integrating learned process models into numerical optimizations. To
address this gap, we present the Learning for CasADi (L4CasADi) framework,
enabling the seamless integration of PyTorch-learned models with CasADi for
efficient and potentially hardware-accelerated numerical optimization. The
applicability of L4CasADi is demonstrated with two tutorial examples: First, we
optimize a fish's trajectory in a turbulent river for energy efficiency where
the turbulent flow is represented by a PyTorch model. Second, we demonstrate
how an implicit Neural Radiance Field environment representation can be easily
leveraged for optimal control with L4CasADi. L4CasADi, along with examples and
documentation, is available under MIT license at
https://github.com/Tim-Salzmann/l4casadi

---

## NeVRF: Neural Video-based Radiance Fields for Long-duration Sequences

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-10 | Minye Wu, Tinne Tuytelaars | cs.CV | [PDF](http://arxiv.org/pdf/2312.05855v1){: .btn .btn-green } |

**Abstract**: Adopting Neural Radiance Fields (NeRF) to long-duration dynamic sequences has
been challenging. Existing methods struggle to balance between quality and
storage size and encounter difficulties with complex scene changes such as
topological changes and large motions. To tackle these issues, we propose a
novel neural video-based radiance fields (NeVRF) representation. NeVRF marries
neural radiance field with image-based rendering to support photo-realistic
novel view synthesis on long-duration dynamic inward-looking scenes. We
introduce a novel multi-view radiance blending approach to predict radiance
directly from multi-view videos. By incorporating continual learning
techniques, NeVRF can efficiently reconstruct frames from sequential data
without revisiting previous frames, enabling long-duration free-viewpoint
video. Furthermore, with a tailored compression approach, NeVRF can compactly
represent dynamic scenes, making dynamic radiance fields more practical in
real-world scenarios. Our extensive experiments demonstrate the effectiveness
of NeVRF in enabling long-duration sequence rendering, sequential data
reconstruction, and compact data storage.

Comments:
- 11 pages, 12 figures

---

## IL-NeRF: Incremental Learning for Neural Radiance Fields with Camera  Pose Alignment

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-10 | Letian Zhang, Ming Li, Chen Chen, Jie Xu | cs.CV | [PDF](http://arxiv.org/pdf/2312.05748v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) is a promising approach for generating
photorealistic images and representing complex scenes. However, when processing
data sequentially, it can suffer from catastrophic forgetting, where previous
data is easily forgotten after training with new data. Existing incremental
learning methods using knowledge distillation assume that continuous data
chunks contain both 2D images and corresponding camera pose parameters,
pre-estimated from the complete dataset. This poses a paradox as the necessary
camera pose must be estimated from the entire dataset, even though the data
arrives sequentially and future chunks are inaccessible. In contrast, we focus
on a practical scenario where camera poses are unknown. We propose IL-NeRF, a
novel framework for incremental NeRF training, to address this challenge.
IL-NeRF's key idea lies in selecting a set of past camera poses as references
to initialize and align the camera poses of incoming image data. This is
followed by a joint optimization of camera poses and replay-based NeRF
distillation. Our experiments on real-world indoor and outdoor scenes show that
IL-NeRF handles incremental NeRF training and outperforms the baselines by up
to $54.04\%$ in rendering quality.

---

## R2-Talker: Realistic Real-Time Talking Head Synthesis with Hash Grid  Landmarks Encoding and Progressive Multilayer Conditioning

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-09 | Zhiling Ye, LiangGuo Zhang, Dingheng Zeng, Quan Lu, Ning Jiang | cs.CV | [PDF](http://arxiv.org/pdf/2312.05572v1){: .btn .btn-green } |

**Abstract**: Dynamic NeRFs have recently garnered growing attention for 3D talking
portrait synthesis. Despite advances in rendering speed and visual quality,
challenges persist in enhancing efficiency and effectiveness. We present
R2-Talker, an efficient and effective framework enabling realistic real-time
talking head synthesis. Specifically, using multi-resolution hash grids, we
introduce a novel approach for encoding facial landmarks as conditional
features. This approach losslessly encodes landmark structures as conditional
features, decoupling input diversity, and conditional spaces by mapping
arbitrary landmarks to a unified feature space. We further propose a scheme of
progressive multilayer conditioning in the NeRF rendering pipeline for
effective conditional feature fusion. Our new approach has the following
advantages as demonstrated by extensive experiments compared with the
state-of-the-art works: 1) The lossless input encoding enables acquiring more
precise features, yielding superior visual quality. The decoupling of inputs
and conditional spaces improves generalizability. 2) The fusing of conditional
features and MLP outputs at each MLP layer enhances conditional impact,
resulting in more accurate lip synthesis and better visual quality. 3) It
compactly structures the fusion of conditional features, significantly
enhancing computational efficiency.

---

## Robo360: A 3D Omnispective Multi-Material Robotic Manipulation Dataset

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-09 | Litian Liang, Liuyu Bian, Caiwei Xiao, Jialin Zhang, Linghao Chen, Isabella Liu, Fanbo Xiang, Zhiao Huang, Hao Su | cs.CV | [PDF](http://arxiv.org/pdf/2312.06686v1){: .btn .btn-green } |

**Abstract**: Building robots that can automate labor-intensive tasks has long been the
core motivation behind the advancements in computer vision and the robotics
community. Recent interest in leveraging 3D algorithms, particularly neural
fields, has led to advancements in robot perception and physical understanding
in manipulation scenarios. However, the real world's complexity poses
significant challenges. To tackle these challenges, we present Robo360, a
dataset that features robotic manipulation with a dense view coverage, which
enables high-quality 3D neural representation learning, and a diverse set of
objects with various physical and optical properties and facilitates research
in various object manipulation and physical world modeling tasks. We confirm
the effectiveness of our dataset using existing dynamic NeRF and evaluate its
potential in learning multi-view policies. We hope that Robo360 can open new
research directions yet to be explored at the intersection of understanding the
physical world in 3D and robot control.

---

## CoGS: Controllable Gaussian Splatting

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-09 | Heng Yu, Joel Julin, Zoltán Á. Milacski, Koichiro Niinuma, László A. Jeni | cs.CV | [PDF](http://arxiv.org/pdf/2312.05664v1){: .btn .btn-green } |

**Abstract**: Capturing and re-animating the 3D structure of articulated objects present
significant barriers. On one hand, methods requiring extensively calibrated
multi-view setups are prohibitively complex and resource-intensive, limiting
their practical applicability. On the other hand, while single-camera Neural
Radiance Fields (NeRFs) offer a more streamlined approach, they have excessive
training and rendering costs. 3D Gaussian Splatting would be a suitable
alternative but for two reasons. Firstly, existing methods for 3D dynamic
Gaussians require synchronized multi-view cameras, and secondly, the lack of
controllability in dynamic scenarios. We present CoGS, a method for
Controllable Gaussian Splatting, that enables the direct manipulation of scene
elements, offering real-time control of dynamic scenes without the prerequisite
of pre-computing control signals. We evaluated CoGS using both synthetic and
real-world datasets that include dynamic objects that differ in degree of
difficulty. In our evaluations, CoGS consistently outperformed existing dynamic
and controllable neural representations in terms of visual fidelity.

Comments:
- 10 pages, in submission

---

## Multi-view Inversion for 3D-aware Generative Adversarial Networks

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-08 | Florian Barthel, Anna Hilsmann, Peter Eisert | cs.CV | [PDF](http://arxiv.org/pdf/2312.05330v1){: .btn .btn-green } |

**Abstract**: Current 3D GAN inversion methods for human heads typically use only one
single frontal image to reconstruct the whole 3D head model. This leaves out
meaningful information when multi-view data or dynamic videos are available.
Our method builds on existing state-of-the-art 3D GAN inversion techniques to
allow for consistent and simultaneous inversion of multiple views of the same
subject. We employ a multi-latent extension to handle inconsistencies present
in dynamic face videos to re-synthesize consistent 3D representations from the
sequence. As our method uses additional information about the target subject,
we observe significant enhancements in both geometric accuracy and image
quality, particularly when rendering from wide viewing angles. Moreover, we
demonstrate the editability of our inverted 3D renderings, which distinguishes
them from NeRF-based scene reconstructions.

---

## 360° Volumetric Portrait Avatar



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-08 | Jalees Nehvi, Berna Kabadayi, Julien Valentin, Justus Thies | cs.CV | [PDF](http://arxiv.org/pdf/2312.05311v1){: .btn .btn-green } |

**Abstract**: We propose 360{\deg} Volumetric Portrait (3VP) Avatar, a novel method for
reconstructing 360{\deg} photo-realistic portrait avatars of human subjects
solely based on monocular video inputs. State-of-the-art monocular avatar
reconstruction methods rely on stable facial performance capturing. However,
the common usage of 3DMM-based facial tracking has its limits; side-views can
hardly be captured and it fails, especially, for back-views, as required inputs
like facial landmarks or human parsing masks are missing. This results in
incomplete avatar reconstructions that only cover the frontal hemisphere. In
contrast to this, we propose a template-based tracking of the torso, head and
facial expressions which allows us to cover the appearance of a human subject
from all sides. Thus, given a sequence of a subject that is rotating in front
of a single camera, we train a neural volumetric representation based on neural
radiance fields. A key challenge to construct this representation is the
modeling of appearance changes, especially, in the mouth region (i.e., lips and
teeth). We, therefore, propose a deformation-field-based blend basis which
allows us to interpolate between different appearance states. We evaluate our
approach on captured real-world data and compare against state-of-the-art
monocular reconstruction methods. In contrast to those, our method is the first
monocular technique that reconstructs an entire 360{\deg} avatar.

Comments:
- Project page: https://jalees018.github.io/3VP-Avatar/

---

## SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational  Score Distillation



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-08 | Thuan Hoang Nguyen, Anh Tran | cs.CV | [PDF](http://arxiv.org/pdf/2312.05239v1){: .btn .btn-green } |

**Abstract**: Despite their ability to generate high-resolution and diverse images from
text prompts, text-to-image diffusion models often suffer from slow iterative
sampling processes. Model distillation is one of the most effective directions
to accelerate these models. However, previous distillation methods fail to
retain the generation quality while requiring a significant amount of images
for training, either from real data or synthetically generated by the teacher
model. In response to this limitation, we present a novel image-free
distillation scheme named $\textbf{SwiftBrush}$. Drawing inspiration from
text-to-3D synthesis, in which a 3D neural radiance field that aligns with the
input prompt can be obtained from a 2D text-to-image diffusion prior via a
specialized loss without the use of any 3D data ground-truth, our approach
re-purposes that same loss for distilling a pretrained multi-step text-to-image
model to a student network that can generate high-fidelity images with just a
single inference step. In spite of its simplicity, our model stands as one of
the first one-step text-to-image generators that can produce images of
comparable quality to Stable Diffusion without reliance on any training image
data. Remarkably, SwiftBrush achieves an FID score of $\textbf{16.67}$ and a
CLIP score of $\textbf{0.29}$ on the COCO-30K benchmark, achieving competitive
results or even substantially surpassing existing state-of-the-art distillation
techniques.

Comments:
- Project Page: https://thuanz123.github.io/swiftbrush/

---

## TriHuman : A Real-time and Controllable Tri-plane Representation for  Detailed Human Geometry and Appearance Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-08 | Heming Zhu, Fangneng Zhan, Christian Theobalt, Marc Habermann | cs.CV | [PDF](http://arxiv.org/pdf/2312.05161v1){: .btn .btn-green } |

**Abstract**: Creating controllable, photorealistic, and geometrically detailed digital
doubles of real humans solely from video data is a key challenge in Computer
Graphics and Vision, especially when real-time performance is required. Recent
methods attach a neural radiance field (NeRF) to an articulated structure,
e.g., a body model or a skeleton, to map points into a pose canonical space
while conditioning the NeRF on the skeletal pose. These approaches typically
parameterize the neural field with a multi-layer perceptron (MLP) leading to a
slow runtime. To address this drawback, we propose TriHuman a novel
human-tailored, deformable, and efficient tri-plane representation, which
achieves real-time performance, state-of-the-art pose-controllable geometry
synthesis as well as photorealistic rendering quality. At the core, we
non-rigidly warp global ray samples into our undeformed tri-plane texture
space, which effectively addresses the problem of global points being mapped to
the same tri-plane locations. We then show how such a tri-plane feature
representation can be conditioned on the skeletal motion to account for dynamic
appearance and geometry changes. Our results demonstrate a clear step towards
higher quality in terms of geometry and appearance modeling of humans as well
as runtime performance.

---

## Learn to Optimize Denoising Scores for 3D Generation: A Unified and  Improved Diffusion Prior on NeRF and 3D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-08 | Xiaofeng Yang, Yiwen Chen, Cheng Chen, Chi Zhang, Yi Xu, Xulei Yang, Fayao Liu, Guosheng Lin | cs.CV | [PDF](http://arxiv.org/pdf/2312.04820v1){: .btn .btn-green } |

**Abstract**: We propose a unified framework aimed at enhancing the diffusion priors for 3D
generation tasks. Despite the critical importance of these tasks, existing
methodologies often struggle to generate high-caliber results. We begin by
examining the inherent limitations in previous diffusion priors. We identify a
divergence between the diffusion priors and the training procedures of
diffusion models that substantially impairs the quality of 3D generation. To
address this issue, we propose a novel, unified framework that iteratively
optimizes both the 3D model and the diffusion prior. Leveraging the different
learnable parameters of the diffusion prior, our approach offers multiple
configurations, affording various trade-offs between performance and
implementation complexity. Notably, our experimental results demonstrate that
our method markedly surpasses existing techniques, establishing new
state-of-the-art in the realm of text-to-3D generation. Furthermore, our
approach exhibits impressive performance on both NeRF and the newly introduced
3D Gaussian Splatting backbones. Additionally, our framework yields insightful
contributions to the understanding of recent score distillation methods, such
as the VSD and DDS loss.

---

## Reality's Canvas, Language's Brush: Crafting 3D Avatars from Monocular  Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-08 | Yuchen Rao, Eduardo Perez Pellitero, Benjamin Busam, Yiren Zhou, Jifei Song | cs.CV | [PDF](http://arxiv.org/pdf/2312.04784v1){: .btn .btn-green } |

**Abstract**: Recent advancements in 3D avatar generation excel with multi-view supervision
for photorealistic models. However, monocular counterparts lag in quality
despite broader applicability. We propose ReCaLab to close this gap. ReCaLab is
a fully-differentiable pipeline that learns high-fidelity 3D human avatars from
just a single RGB video. A pose-conditioned deformable NeRF is optimized to
volumetrically represent a human subject in canonical T-pose. The canonical
representation is then leveraged to efficiently associate viewpoint-agnostic
textures using 2D-3D correspondences. This enables to separately generate
albedo and shading which jointly compose an RGB prediction. The design allows
to control intermediate results for human pose, body shape, texture, and
lighting with text prompts. An image-conditioned diffusion model thereby helps
to animate appearance and pose of the 3D avatar to create video sequences with
previously unseen human motion. Extensive experiments show that ReCaLab
outperforms previous monocular approaches in terms of image quality for image
synthesis tasks. ReCaLab even outperforms multi-view methods that leverage up
to 19x more synchronized videos for the task of novel pose rendering. Moreover,
natural language offers an intuitive user interface for creative manipulation
of 3D human avatars.

Comments:
- Video link: https://youtu.be/Oz83z1es2J4

---

## Correspondences of the Third Kind: Camera Pose Estimation from Object  Reflection

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-07 | Kohei Yamashita, Vincent Lepetit, Ko Nishino | cs.CV | [PDF](http://arxiv.org/pdf/2312.04527v1){: .btn .btn-green } |

**Abstract**: Computer vision has long relied on two kinds of correspondences: pixel
correspondences in images and 3D correspondences on object surfaces. Is there
another kind, and if there is, what can they do for us? In this paper, we
introduce correspondences of the third kind we call reflection correspondences
and show that they can help estimate camera pose by just looking at objects
without relying on the background. Reflection correspondences are point
correspondences in the reflected world, i.e., the scene reflected by the object
surface. The object geometry and reflectance alters the scene geometrically and
radiometrically, respectively, causing incorrect pixel correspondences.
Geometry recovered from each image is also hampered by distortions, namely
generalized bas-relief ambiguity, leading to erroneous 3D correspondences. We
show that reflection correspondences can resolve the ambiguities arising from
these distortions. We introduce a neural correspondence estimator and a RANSAC
algorithm that fully leverages all three kinds of correspondences for robust
and accurate joint camera pose and object shape estimation just from the object
appearance. The method expands the horizon of numerous downstream tasks,
including camera pose estimation for appearance modeling (e.g., NeRF) and
motion estimation of reflective objects (e.g., cars on the road), to name a
few, as it relieves the requirement of overlapping background.

---

## Identity-Obscured Neural Radiance Fields: Privacy-Preserving 3D Facial  Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-07 | Jiayi Kong, Baixin Xu, Xurui Song, Chen Qian, Jun Luo, Ying He | cs.CV | [PDF](http://arxiv.org/pdf/2312.04106v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) typically require a complete set of images
taken from multiple camera perspectives to accurately reconstruct geometric
details. However, this approach raise significant privacy concerns in the
context of facial reconstruction. The critical need for privacy protection
often leads invidividuals to be reluctant in sharing their facial images, due
to fears of potential misuse or security risks. Addressing these concerns, we
propose a method that leverages privacy-preserving images for reconstructing 3D
head geometry within the NeRF framework. Our method stands apart from
traditional facial reconstruction techniques as it does not depend on RGB
information from images containing sensitive facial data. Instead, it
effectively generates plausible facial geometry using a series of
identity-obscured inputs, thereby protecting facial privacy.

---

## Towards 4D Human Video Stylization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-07 | Tiantian Wang, Xinxin Zuo, Fangzhou Mu, Jian Wang, Ming-Hsuan Yang | cs.CV | [PDF](http://arxiv.org/pdf/2312.04143v1){: .btn .btn-green } |

**Abstract**: We present a first step towards 4D (3D and time) human video stylization,
which addresses style transfer, novel view synthesis and human animation within
a unified framework. While numerous video stylization methods have been
developed, they are often restricted to rendering images in specific viewpoints
of the input video, lacking the capability to generalize to novel views and
novel poses in dynamic scenes. To overcome these limitations, we leverage
Neural Radiance Fields (NeRFs) to represent videos, conducting stylization in
the rendered feature space. Our innovative approach involves the simultaneous
representation of both the human subject and the surrounding scene using two
NeRFs. This dual representation facilitates the animation of human subjects
across various poses and novel viewpoints. Specifically, we introduce a novel
geometry-guided tri-plane representation, significantly enhancing feature
representation robustness compared to direct tri-plane optimization. Following
the video reconstruction, stylization is performed within the NeRFs' rendered
feature space. Extensive experiments demonstrate that the proposed method
strikes a superior balance between stylized textures and temporal coherence,
surpassing existing approaches. Furthermore, our framework uniquely extends its
capabilities to accommodate novel poses and viewpoints, making it a versatile
tool for creative human video stylization.

Comments:
- Under Review

---

## Multi-View Unsupervised Image Generation with Cross Attention Guidance

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-07 | Llukman Cerkezi, Aram Davtyan, Sepehr Sameni, Paolo Favaro | cs.CV | [PDF](http://arxiv.org/pdf/2312.04337v1){: .btn .btn-green } |

**Abstract**: The growing interest in novel view synthesis, driven by Neural Radiance Field
(NeRF) models, is hindered by scalability issues due to their reliance on
precisely annotated multi-view images. Recent models address this by
fine-tuning large text2image diffusion models on synthetic multi-view data.
Despite robust zero-shot generalization, they may need post-processing and can
face quality issues due to the synthetic-real domain gap. This paper introduces
a novel pipeline for unsupervised training of a pose-conditioned diffusion
model on single-category datasets. With the help of pretrained self-supervised
Vision Transformers (DINOv2), we identify object poses by clustering the
dataset through comparing visibility and locations of specific object parts.
The pose-conditioned diffusion model, trained on pose labels, and equipped with
cross-frame attention at inference time ensures cross-view consistency, that is
further aided by our novel hard-attention guidance. Our model, MIRAGE,
surpasses prior work in novel view synthesis on real images. Furthermore,
MIRAGE is robust to diverse textures and geometries, as demonstrated with our
experiments on synthetic images generated with pretrained Stable Diffusion.

---

## NeuSD: Surface Completion with Multi-View Text-to-Image Diffusion



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-07 | Savva Ignatyev, Daniil Selikhanovych, Oleg Voynov, Yiqun Wang, Peter Wonka, Stamatios Lefkimmiatis, Evgeny Burnaev | cs.CV | [PDF](http://arxiv.org/pdf/2312.04654v1){: .btn .btn-green } |

**Abstract**: We present a novel method for 3D surface reconstruction from multiple images
where only a part of the object of interest is captured. Our approach builds on
two recent developments: surface reconstruction using neural radiance fields
for the reconstruction of the visible parts of the surface, and guidance of
pre-trained 2D diffusion models in the form of Score Distillation Sampling
(SDS) to complete the shape in unobserved regions in a plausible manner. We
introduce three components. First, we suggest employing normal maps as a pure
geometric representation for SDS instead of color renderings which are
entangled with the appearance information. Second, we introduce the freezing of
the SDS noise during training which results in more coherent gradients and
better convergence. Third, we propose Multi-View SDS as a way to condition the
generation of the non-observable part of the surface without fine-tuning or
making changes to the underlying 2D Stable Diffusion model. We evaluate our
approach on the BlendedMVS dataset demonstrating significant qualitative and
quantitative improvements over competing methods.

---

## MonoGaussianAvatar: Monocular Gaussian Point-based Head Avatar

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-07 | Yufan Chen, Lizhen Wang, Qijing Li, Hongjiang Xiao, Shengping Zhang, Hongxun Yao, Yebin Liu | cs.CV | [PDF](http://arxiv.org/pdf/2312.04558v1){: .btn .btn-green } |

**Abstract**: The ability to animate photo-realistic head avatars reconstructed from
monocular portrait video sequences represents a crucial step in bridging the
gap between the virtual and real worlds. Recent advancements in head avatar
techniques, including explicit 3D morphable meshes (3DMM), point clouds, and
neural implicit representation have been exploited for this ongoing research.
However, 3DMM-based methods are constrained by their fixed topologies,
point-based approaches suffer from a heavy training burden due to the extensive
quantity of points involved, and the last ones suffer from limitations in
deformation flexibility and rendering efficiency. In response to these
challenges, we propose MonoGaussianAvatar (Monocular Gaussian Point-based Head
Avatar), a novel approach that harnesses 3D Gaussian point representation
coupled with a Gaussian deformation field to learn explicit head avatars from
monocular portrait videos. We define our head avatars with Gaussian points
characterized by adaptable shapes, enabling flexible topology. These points
exhibit movement with a Gaussian deformation field in alignment with the target
pose and expression of a person, facilitating efficient deformation.
Additionally, the Gaussian points have controllable shape, size, color, and
opacity combined with Gaussian splatting, allowing for efficient training and
rendering. Experiments demonstrate the superior performance of our method,
which achieves state-of-the-art results among previous methods.

Comments:
- The link to our projectpage is
  https://yufan1012.github.io/MonoGaussianAvatar

---

## EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-07 | Sharath Girish, Kamal Gupta, Abhinav Shrivastava | cs.CV | [PDF](http://arxiv.org/pdf/2312.04564v1){: .btn .btn-green } |

**Abstract**: Recently, 3D Gaussian splatting (3D-GS) has gained popularity in novel-view
scene synthesis. It addresses the challenges of lengthy training times and slow
rendering speeds associated with Neural Radiance Fields (NeRFs). Through rapid,
differentiable rasterization of 3D Gaussians, 3D-GS achieves real-time
rendering and accelerated training. They, however, demand substantial memory
resources for both training and storage, as they require millions of Gaussians
in their point cloud representation for each scene. We present a technique
utilizing quantized embeddings to significantly reduce memory storage
requirements and a coarse-to-fine training strategy for a faster and more
stable optimization of the Gaussian point clouds. Our approach results in scene
representations with fewer Gaussians and quantized representations, leading to
faster training times and rendering speeds for real-time rendering of high
resolution scenes. We reduce memory by more than an order of magnitude all
while maintaining the reconstruction quality. We validate the effectiveness of
our approach on a variety of datasets and scenes preserving the visual quality
while consuming 10-20x less memory and faster training/inference speed. Project
page and code is available https://efficientgaussian.github.io

Comments:
- Website: https://efficientgaussian.github.io Code:
  https://github.com/Sharath-girish/efficientgaussian

---

## MuRF: Multi-Baseline Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-07 | Haofei Xu, Anpei Chen, Yuedong Chen, Christos Sakaridis, Yulun Zhang, Marc Pollefeys, Andreas Geiger, Fisher Yu | cs.CV | [PDF](http://arxiv.org/pdf/2312.04565v1){: .btn .btn-green } |

**Abstract**: We present Multi-Baseline Radiance Fields (MuRF), a general feed-forward
approach to solving sparse view synthesis under multiple different baseline
settings (small and large baselines, and different number of input views). To
render a target novel view, we discretize the 3D space into planes parallel to
the target image plane, and accordingly construct a target view frustum volume.
Such a target volume representation is spatially aligned with the target view,
which effectively aggregates relevant information from the input views for
high-quality rendering. It also facilitates subsequent radiance field
regression with a convolutional network thanks to its axis-aligned nature. The
3D context modeled by the convolutional network enables our method to synthesis
sharper scene structures than prior works. Our MuRF achieves state-of-the-art
performance across multiple different baseline settings and diverse scenarios
ranging from simple objects (DTU) to complex indoor and outdoor scenes
(RealEstate10K and LLFF). We also show promising zero-shot generalization
abilities on the Mip-NeRF 360 dataset, demonstrating the general applicability
of MuRF.

Comments:
- Project page: https://haofeixu.github.io/murf/

---

## VOODOO 3D: Volumetric Portrait Disentanglement for One-Shot 3D Head  Reenactment



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-07 | Phong Tran, Egor Zakharov, Long-Nhat Ho, Anh Tuan Tran, Liwen Hu, Hao Li | cs.CV | [PDF](http://arxiv.org/pdf/2312.04651v1){: .btn .btn-green } |

**Abstract**: We present a 3D-aware one-shot head reenactment method based on a fully
volumetric neural disentanglement framework for source appearance and driver
expressions. Our method is real-time and produces high-fidelity and
view-consistent output, suitable for 3D teleconferencing systems based on
holographic displays. Existing cutting-edge 3D-aware reenactment methods often
use neural radiance fields or 3D meshes to produce view-consistent appearance
encoding, but, at the same time, they rely on linear face models, such as 3DMM,
to achieve its disentanglement with facial expressions. As a result, their
reenactment results often exhibit identity leakage from the driver or have
unnatural expressions. To address these problems, we propose a neural
self-supervised disentanglement approach that lifts both the source image and
driver video frame into a shared 3D volumetric representation based on
tri-planes. This representation can then be freely manipulated with expression
tri-planes extracted from the driving images and rendered from an arbitrary
view using neural radiance fields. We achieve this disentanglement via
self-supervised learning on a large in-the-wild video dataset. We further
introduce a highly effective fine-tuning approach to improve the
generalizability of the 3D lifting using the same real-world data. We
demonstrate state-of-the-art performance on a wide range of datasets, and also
showcase high-quality 3D-aware head reenactment on highly challenging and
diverse subjects, including non-frontal head poses and complex expressions for
both source and driver.

---

## Inpaint3D: 3D Scene Content Generation using 2D Inpainting Diffusion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-06 | Kira Prabhu, Jane Wu, Lynn Tsai, Peter Hedman, Dan B Goldman, Ben Poole, Michael Broxton | cs.CV | [PDF](http://arxiv.org/pdf/2312.03869v1){: .btn .btn-green } |

**Abstract**: This paper presents a novel approach to inpainting 3D regions of a scene,
given masked multi-view images, by distilling a 2D diffusion model into a
learned 3D scene representation (e.g. a NeRF). Unlike 3D generative methods
that explicitly condition the diffusion model on camera pose or multi-view
information, our diffusion model is conditioned only on a single masked 2D
image. Nevertheless, we show that this 2D diffusion model can still serve as a
generative prior in a 3D multi-view reconstruction problem where we optimize a
NeRF using a combination of score distillation sampling and NeRF reconstruction
losses. Predicted depth is used as additional supervision to encourage accurate
geometry. We compare our approach to 3D inpainting methods that focus on object
removal. Because our method can generate content to fill any 3D masked region,
we additionally demonstrate 3D object completion, 3D object replacement, and 3D
scene completion.

---

## HiFi4G: High-Fidelity Human Performance Rendering via Compact Gaussian  Splatting



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-06 | Yuheng Jiang, Zhehao Shen, Penghao Wang, Zhuo Su, Yu Hong, Yingliang Zhang, Jingyi Yu, Lan Xu | cs.CV | [PDF](http://arxiv.org/pdf/2312.03461v2){: .btn .btn-green } |

**Abstract**: We have recently seen tremendous progress in photo-real human modeling and
rendering. Yet, efficiently rendering realistic human performance and
integrating it into the rasterization pipeline remains challenging. In this
paper, we present HiFi4G, an explicit and compact Gaussian-based approach for
high-fidelity human performance rendering from dense footage. Our core
intuition is to marry the 3D Gaussian representation with non-rigid tracking,
achieving a compact and compression-friendly representation. We first propose a
dual-graph mechanism to obtain motion priors, with a coarse deformation graph
for effective initialization and a fine-grained Gaussian graph to enforce
subsequent constraints. Then, we utilize a 4D Gaussian optimization scheme with
adaptive spatial-temporal regularizers to effectively balance the non-rigid
prior and Gaussian updating. We also present a companion compression scheme
with residual compensation for immersive experiences on various platforms. It
achieves a substantial compression rate of approximately 25 times, with less
than 2MB of storage per frame. Extensive experiments demonstrate the
effectiveness of our approach, which significantly outperforms existing
approaches in terms of optimization speed, rendering quality, and storage
overhead.

---

## Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-06 | Youtian Lin, Zuozhuo Dai, Siyu Zhu, Yao Yao | cs.CV | [PDF](http://arxiv.org/pdf/2312.03431v1){: .btn .btn-green } |

**Abstract**: We introduce Gaussian-Flow, a novel point-based approach for fast dynamic
scene reconstruction and real-time rendering from both multi-view and monocular
videos. In contrast to the prevalent NeRF-based approaches hampered by slow
training and rendering speeds, our approach harnesses recent advancements in
point-based 3D Gaussian Splatting (3DGS). Specifically, a novel Dual-Domain
Deformation Model (DDDM) is proposed to explicitly model attribute deformations
of each Gaussian point, where the time-dependent residual of each attribute is
captured by a polynomial fitting in the time domain, and a Fourier series
fitting in the frequency domain. The proposed DDDM is capable of modeling
complex scene deformations across long video footage, eliminating the need for
training separate 3DGS for each frame or introducing an additional implicit
neural field to model 3D dynamics. Moreover, the explicit deformation modeling
for discretized Gaussian points ensures ultra-fast training and rendering of a
4D scene, which is comparable to the original 3DGS designed for static 3D
reconstruction. Our proposed approach showcases a substantial efficiency
improvement, achieving a $5\times$ faster training speed compared to the
per-frame 3DGS modeling. In addition, quantitative results demonstrate that the
proposed Gaussian-Flow significantly outperforms previous leading methods in
novel view rendering quality. Project page:
https://nju-3dv.github.io/projects/Gaussian-Flow

---

## Artist-Friendly Relightable and Animatable Neural Heads

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-06 | Yingyan Xu, Prashanth Chandran, Sebastian Weiss, Markus Gross, Gaspard Zoss, Derek Bradley | cs.CV | [PDF](http://arxiv.org/pdf/2312.03420v1){: .btn .btn-green } |

**Abstract**: An increasingly common approach for creating photo-realistic digital avatars
is through the use of volumetric neural fields. The original neural radiance
field (NeRF) allowed for impressive novel view synthesis of static heads when
trained on a set of multi-view images, and follow up methods showed that these
neural representations can be extended to dynamic avatars. Recently, new
variants also surpassed the usual drawback of baked-in illumination in neural
representations, showing that static neural avatars can be relit in any
environment. In this work we simultaneously tackle both the motion and
illumination problem, proposing a new method for relightable and animatable
neural heads. Our method builds on a proven dynamic avatar approach based on a
mixture of volumetric primitives, combined with a recently-proposed lightweight
hardware setup for relightable neural fields, and includes a novel architecture
that allows relighting dynamic neural avatars performing unseen expressions in
any environment, even with nearfield illumination and viewpoints.

---

## Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-06 | Vladimir Yugay, Yue Li, Theo Gevers, Martin R. Oswald | cs.CV | [PDF](http://arxiv.org/pdf/2312.10070v1){: .btn .btn-green } |

**Abstract**: We present a new dense simultaneous localization and mapping (SLAM) method
that uses Gaussian splats as a scene representation. The new representation
enables interactive-time reconstruction and photo-realistic rendering of
real-world and synthetic scenes. We propose novel strategies for seeding and
optimizing Gaussian splats to extend their use from multiview offline scenarios
to sequential monocular RGBD input data setups. In addition, we extend Gaussian
splats to encode geometry and experiment with tracking against this scene
representation. Our method achieves state-of-the-art rendering quality on both
real-world and synthetic datasets while being competitive in reconstruction
performance and runtime.

---

## Evaluating the point cloud of individual trees generated from images  based on Neural Radiance fields (NeRF) method

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-06 | Hongyu Huang, Guoji Tian, Chongcheng Chen | cs.CV | [PDF](http://arxiv.org/pdf/2312.03372v1){: .btn .btn-green } |

**Abstract**: Three-dimensional (3D) reconstruction of trees has always been a key task in
precision forestry management and research. Due to the complex branch
morphological structure of trees themselves and the occlusions from tree stems,
branches and foliage, it is difficult to recreate a complete three-dimensional
tree model from a two-dimensional image by conventional photogrammetric
methods. In this study, based on tree images collected by various cameras in
different ways, the Neural Radiance Fields (NeRF) method was used for
individual tree reconstruction and the exported point cloud models are compared
with point cloud derived from photogrammetric reconstruction and laser scanning
methods. The results show that the NeRF method performs well in individual tree
3D reconstruction, as it has higher successful reconstruction rate, better
reconstruction in the canopy area, it requires less amount of images as input.
Compared with photogrammetric reconstruction method, NeRF has significant
advantages in reconstruction efficiency and is adaptable to complex scenes, but
the generated point cloud tends to be noisy and low resolution. The accuracy of
tree structural parameters (tree height and diameter at breast height)
extracted from the photogrammetric point cloud is still higher than those of
derived from the NeRF point cloud. The results of this study illustrate the
great potential of NeRF method for individual tree reconstruction, and it
provides new ideas and research directions for 3D reconstruction and
visualization of complex forest scenes.

Comments:
- 25 pages; 6 figures

---

## RING-NeRF: A Versatile Architecture based on Residual Implicit Neural  Grids

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-06 | Doriand Petit, Steve Bourgeois, Dumitru Pavel, Vincent Gay-Bellile, Florian Chabot, Loic Barthe | cs.CV | [PDF](http://arxiv.org/pdf/2312.03357v1){: .btn .btn-green } |

**Abstract**: Since their introduction, Neural Fields have become very popular for 3D
reconstruction and new view synthesis. Recent researches focused on
accelerating the process, as well as improving the robustness to variation of
the observation distance and limited number of supervised viewpoints. However,
those approaches often led to dedicated solutions that cannot be easily
combined. To tackle this issue, we introduce a new simple but efficient
architecture named RING-NeRF, based on Residual Implicit Neural Grids, that
provides a control on the level of detail of the mapping function between the
scene and the latent spaces. Associated with a distance-aware forward mapping
mechanism and a continuous coarse-to-fine reconstruction process, our versatile
architecture demonstrates both fast training and state-of-the-art performances
in terms of: (1) anti-aliased rendering, (2) reconstruction quality from few
supervised viewpoints, and (3) robustness in the absence of appropriate
scene-specific initialization for SDF-based NeRFs. We also demonstrate that our
architecture can dynamically add grids to increase the details of the
reconstruction, opening the way to adaptive reconstruction.

---

## SO-NeRF: Active View Planning for NeRF using Surrogate Objectives

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-06 | Keifer Lee, Shubham Gupta, Sunglyoung Kim, Bhargav Makwana, Chao Chen, Chen Feng | cs.CV | [PDF](http://arxiv.org/pdf/2312.03266v1){: .btn .btn-green } |

**Abstract**: Despite the great success of Neural Radiance Fields (NeRF), its
data-gathering process remains vague with only a general rule of thumb of
sampling as densely as possible. The lack of understanding of what actually
constitutes good views for NeRF makes it difficult to actively plan a sequence
of views that yield the maximal reconstruction quality. We propose Surrogate
Objectives for Active Radiance Fields (SOAR), which is a set of interpretable
functions that evaluates the goodness of views using geometric and photometric
visual cues - surface coverage, geometric complexity, textural complexity, and
ray diversity. Moreover, by learning to infer the SOAR scores from a deep
network, SOARNet, we are able to effectively select views in mere seconds
instead of hours, without the need for prior visits to all the candidate views
or training any radiance field during such planning. Our experiments show
SOARNet outperforms the baselines with $\sim$80x speed-up while achieving
better or comparable reconstruction qualities. We finally show that SOAR is
model-agnostic, thus it generalizes across fully neural-implicit to fully
explicit approaches.

Comments:
- 13 pages

---

## Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled  Feature Fields

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-06 | Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, Achuta Kadambi | cs.CV | [PDF](http://arxiv.org/pdf/2312.03203v1){: .btn .btn-green } |

**Abstract**: 3D scene representations have gained immense popularity in recent years.
Methods that use Neural Radiance fields are versatile for traditional tasks
such as novel view synthesis. In recent times, some work has emerged that aims
to extend the functionality of NeRF beyond view synthesis, for semantically
aware tasks such as editing and segmentation using 3D feature field
distillation from 2D foundation models. However, these methods have two major
limitations: (a) they are limited by the rendering speed of NeRF pipelines, and
(b) implicitly represented feature fields suffer from continuity artifacts
reducing feature quality. Recently, 3D Gaussian Splatting has shown
state-of-the-art performance on real-time radiance field rendering. In this
work, we go one step further: in addition to radiance field rendering, we
enable 3D Gaussian splatting on arbitrary-dimension semantic features via 2D
foundation model distillation. This translation is not straightforward: naively
incorporating feature fields in the 3DGS framework leads to warp-level
divergence. We propose architectural and training changes to efficiently avert
this problem. Our proposed method is general, and our experiments showcase
novel view semantic segmentation, language-guided editing and segment anything
through learning feature fields from state-of-the-art 2D foundation models such
as SAM and CLIP-LSeg. Across experiments, our distillation method is able to
provide comparable or better results, while being significantly faster to both
train and render. Additionally, to the best of our knowledge, we are the first
method to enable point and bounding-box prompting for radiance field
manipulation, by leveraging the SAM model. Project website at:
https://feature-3dgs.github.io/

---

## MVHumanNet: A Large-scale Dataset of Multi-view Daily Dressing Human  Captures

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-05 | Zhangyang Xiong, Chenghong Li, Kenkun Liu, Hongjie Liao, Jianqiao Hu, Junyi Zhu, Shuliang Ning, Lingteng Qiu, Chongjie Wang, Shijie Wang, Shuguang Cui, Xiaoguang Han | cs.CV | [PDF](http://arxiv.org/pdf/2312.02963v1){: .btn .btn-green } |

**Abstract**: In this era, the success of large language models and text-to-image models
can be attributed to the driving force of large-scale datasets. However, in the
realm of 3D vision, while remarkable progress has been made with models trained
on large-scale synthetic and real-captured object data like Objaverse and
MVImgNet, a similar level of progress has not been observed in the domain of
human-centric tasks partially due to the lack of a large-scale human dataset.
Existing datasets of high-fidelity 3D human capture continue to be mid-sized
due to the significant challenges in acquiring large-scale high-quality 3D
human data. To bridge this gap, we present MVHumanNet, a dataset that comprises
multi-view human action sequences of 4,500 human identities. The primary focus
of our work is on collecting human data that features a large number of diverse
identities and everyday clothing using a multi-view human capture system, which
facilitates easily scalable data collection. Our dataset contains 9,000 daily
outfits, 60,000 motion sequences and 645 million frames with extensive
annotations, including human masks, camera parameters, 2D and 3D keypoints,
SMPL/SMPLX parameters, and corresponding textual descriptions. To explore the
potential of MVHumanNet in various 2D and 3D visual tasks, we conducted pilot
studies on view-consistent action recognition, human NeRF reconstruction,
text-driven view-unconstrained human image generation, as well as 2D
view-unconstrained human image and 3D avatar generation. Extensive experiments
demonstrate the performance improvements and effective applications enabled by
the scale provided by MVHumanNet. As the current largest-scale 3D human
dataset, we hope that the release of MVHumanNet data with annotations will
foster further innovations in the domain of 3D human-centric tasks at scale.

Comments:
- Project page: https://x-zhangyang.github.io/MVHumanNet/

---

## FINER: Flexible spectral-bias tuning in Implicit NEural Representation  by Variable-periodic Activation Functions



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-05 | Zhen Liu, Hao Zhu, Qi Zhang, Jingde Fu, Weibing Deng, Zhan Ma, Yanwen Guo, Xun Cao | cs.CV | [PDF](http://arxiv.org/pdf/2312.02434v1){: .btn .btn-green } |

**Abstract**: Implicit Neural Representation (INR), which utilizes a neural network to map
coordinate inputs to corresponding attributes, is causing a revolution in the
field of signal processing. However, current INR techniques suffer from a
restricted capability to tune their supported frequency set, resulting in
imperfect performance when representing complex signals with multiple
frequencies. We have identified that this frequency-related problem can be
greatly alleviated by introducing variable-periodic activation functions, for
which we propose FINER. By initializing the bias of the neural network within
different ranges, sub-functions with various frequencies in the
variable-periodic function are selected for activation. Consequently, the
supported frequency set of FINER can be flexibly tuned, leading to improved
performance in signal representation. We demonstrate the capabilities of FINER
in the contexts of 2D image fitting, 3D signed distance field representation,
and 5D neural radiance fields optimization, and we show that it outperforms
existing INRs.

Comments:
- 10 pages, 9 figures

---

## HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-05 | Helisa Dhamo, Yinyu Nie, Arthur Moreau, Jifei Song, Richard Shaw, Yiren Zhou, Eduardo Pérez-Pellitero | cs.CV | [PDF](http://arxiv.org/pdf/2312.02902v1){: .btn .btn-green } |

**Abstract**: 3D head animation has seen major quality and runtime improvements over the
last few years, particularly empowered by the advances in differentiable
rendering and neural radiance fields. Real-time rendering is a highly desirable
goal for real-world applications. We propose HeadGaS, the first model to use 3D
Gaussian Splats (3DGS) for 3D head reconstruction and animation. In this paper
we introduce a hybrid model that extends the explicit representation from 3DGS
with a base of learnable latent features, which can be linearly blended with
low-dimensional parameters from parametric head models to obtain
expression-dependent final color and opacity values. We demonstrate that
HeadGaS delivers state-of-the-art results in real-time inference frame rates,
which surpasses baselines by up to ~2dB, while accelerating rendering speed by
over x10.

---

## Prompt2NeRF-PIL: Fast NeRF Generation via Pretrained Implicit Latent

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-05 | Jianmeng Liu, Yuyao Zhang, Zeyuan Meng, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2312.02568v1){: .btn .btn-green } |

**Abstract**: This paper explores promptable NeRF generation (e.g., text prompt or single
image prompt) for direct conditioning and fast generation of NeRF parameters
for the underlying 3D scenes, thus undoing complex intermediate steps while
providing full 3D generation with conditional control. Unlike previous
diffusion-CLIP-based pipelines that involve tedious per-prompt optimizations,
Prompt2NeRF-PIL is capable of generating a variety of 3D objects with a single
forward pass, leveraging a pre-trained implicit latent space of NeRF
parameters. Furthermore, in zero-shot tasks, our experiments demonstrate that
the NeRFs produced by our method serve as semantically informative
initializations, significantly accelerating the inference process of existing
prompt-to-NeRF methods. Specifically, we will show that our approach speeds up
the text-to-NeRF model DreamFusion and the 3D reconstruction speed of the
image-to-NeRF method Zero-1-to-3 by 3 to 5 times.

---

## Alchemist: Parametric Control of Material Properties with Diffusion  Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-05 | Prafull Sharma, Varun Jampani, Yuanzhen Li, Xuhui Jia, Dmitry Lagun, Fredo Durand, William T. Freeman, Mark Matthews | cs.CV | [PDF](http://arxiv.org/pdf/2312.02970v1){: .btn .btn-green } |

**Abstract**: We propose a method to control material attributes of objects like roughness,
metallic, albedo, and transparency in real images. Our method capitalizes on
the generative prior of text-to-image models known for photorealism, employing
a scalar value and instructions to alter low-level material properties.
Addressing the lack of datasets with controlled material attributes, we
generated an object-centric synthetic dataset with physically-based materials.
Fine-tuning a modified pre-trained text-to-image model on this synthetic
dataset enables us to edit material properties in real-world images while
preserving all other attributes. We show the potential application of our model
to material edited NeRFs.

---

## HybridNeRF: Efficient Neural Rendering via Adaptive Volumetric Surfaces

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-05 | Haithem Turki, Vasu Agrawal, Samuel Rota Bulò, Lorenzo Porzi, Peter Kontschieder, Deva Ramanan, Michael Zollhöfer, Christian Richardt | cs.CV | [PDF](http://arxiv.org/pdf/2312.03160v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields provide state-of-the-art view synthesis quality but
tend to be slow to render. One reason is that they make use of volume
rendering, thus requiring many samples (and model queries) per ray at render
time. Although this representation is flexible and easy to optimize, most
real-world objects can be modeled more efficiently with surfaces instead of
volumes, requiring far fewer samples per ray. This observation has spurred
considerable progress in surface representations such as signed distance
functions, but these may struggle to model semi-opaque and thin structures. We
propose a method, HybridNeRF, that leverages the strengths of both
representations by rendering most objects as surfaces while modeling the
(typically) small fraction of challenging regions volumetrically. We evaluate
HybridNeRF against the challenging Eyeful Tower dataset along with other
commonly used view synthesis datasets. When comparing to state-of-the-art
baselines, including recent rasterization-based approaches, we improve error
rates by 15-30% while achieving real-time framerates (at least 36 FPS) for
virtual-reality resolutions (2Kx2K).

Comments:
- Project page: https://haithemturki.com/hybrid-nerf/

---

## GauHuman: Articulated Gaussian Splatting from Monocular Human Videos

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-05 | Shoukang Hu, Ziwei Liu | cs.CV | [PDF](http://arxiv.org/pdf/2312.02973v1){: .btn .btn-green } |

**Abstract**: We present, GauHuman, a 3D human model with Gaussian Splatting for both fast
training (1 ~ 2 minutes) and real-time rendering (up to 189 FPS), compared with
existing NeRF-based implicit representation modelling frameworks demanding
hours of training and seconds of rendering per frame. Specifically, GauHuman
encodes Gaussian Splatting in the canonical space and transforms 3D Gaussians
from canonical space to posed space with linear blend skinning (LBS), in which
effective pose and LBS refinement modules are designed to learn fine details of
3D humans under negligible computational cost. Moreover, to enable fast
optimization of GauHuman, we initialize and prune 3D Gaussians with 3D human
prior, while splitting/cloning via KL divergence guidance, along with a novel
merge operation for further speeding up. Extensive experiments on ZJU_Mocap and
MonoCap datasets demonstrate that GauHuman achieves state-of-the-art
performance quantitatively and qualitatively with fast training and real-time
rendering speed. Notably, without sacrificing rendering quality, GauHuman can
fast model the 3D human performer with ~13k 3D Gaussians.

Comments:
- project page: https://skhu101.github.io/GauHuman/; code:
  https://github.com/skhu101/GauHuman

---

## C-NERF: Representing Scene Changes as Directional Consistency  Difference-based NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-05 | Rui Huang, Binbin Jiang, Qingyi Zhao, William Wang, Yuxiang Zhang, Qing Guo | cs.CV | [PDF](http://arxiv.org/pdf/2312.02751v2){: .btn .btn-green } |

**Abstract**: In this work, we aim to detect the changes caused by object variations in a
scene represented by the neural radiance fields (NeRFs). Given an arbitrary
view and two sets of scene images captured at different timestamps, we can
predict the scene changes in that view, which has significant potential
applications in scene monitoring and measuring. We conducted preliminary
studies and found that such an exciting task cannot be easily achieved by
utilizing existing NeRFs and 2D change detection methods with many false or
missing detections. The main reason is that the 2D change detection is based on
the pixel appearance difference between spatial-aligned image pairs and
neglects the stereo information in the NeRF. To address the limitations, we
propose the C-NERF to represent scene changes as directional consistency
difference-based NeRF, which mainly contains three modules. We first perform
the spatial alignment of two NeRFs captured before and after changes. Then, we
identify the change points based on the direction-consistent constraint; that
is, real change points have similar change representations across view
directions, but fake change points do not. Finally, we design the change map
rendering process based on the built NeRFs and can generate the change map of
an arbitrarily specified view direction. To validate the effectiveness, we
build a new dataset containing ten scenes covering diverse scenarios with
different changing objects. Our approach surpasses state-of-the-art 2D change
detection and NeRF-based methods by a significant margin.

---

## ReconFusion: 3D Reconstruction with Diffusion Priors

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-05 | Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P. Srinivasan, Dor Verbin, Jonathan T. Barron, Ben Poole, Aleksander Holynski | cs.CV | [PDF](http://arxiv.org/pdf/2312.02981v1){: .btn .btn-green } |

**Abstract**: 3D reconstruction methods such as Neural Radiance Fields (NeRFs) excel at
rendering photorealistic novel views of complex scenes. However, recovering a
high-quality NeRF typically requires tens to hundreds of input images,
resulting in a time-consuming capture process. We present ReconFusion to
reconstruct real-world scenes using only a few photos. Our approach leverages a
diffusion prior for novel view synthesis, trained on synthetic and multiview
datasets, which regularizes a NeRF-based 3D reconstruction pipeline at novel
camera poses beyond those captured by the set of input images. Our method
synthesizes realistic geometry and texture in underconstrained regions while
preserving the appearance of observed regions. We perform an extensive
evaluation across various real-world datasets, including forward-facing and
360-degree scenes, demonstrating significant performance improvements over
previous few-view NeRF reconstruction approaches.

Comments:
- Project page: https://reconfusion.github.io/

---

## Mathematical Supplement for the $\texttt{gsplat}$ Library

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Vickie Ye, Angjoo Kanazawa | cs.MS | [PDF](http://arxiv.org/pdf/2312.02121v1){: .btn .btn-green } |

**Abstract**: This report provides the mathematical details of the gsplat library, a
modular toolbox for efficient differentiable Gaussian splatting, as proposed by
Kerbl et al. It provides a self-contained reference for the computations
involved in the forward and backward passes of differentiable Gaussian
splatting. To facilitate practical usage and development, we provide a user
friendly Python API that exposes each component of the forward and backward
passes in rasterization at github.com/nerfstudio-project/gsplat .

Comments:
- Find the library at: https://docs.gsplat.studio/

---

## Customize your NeRF: Adaptive Source Driven 3D Scene Editing via  Local-Global Iterative Training

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Runze He, Shaofei Huang, Xuecheng Nie, Tianrui Hui, Luoqi Liu, Jiao Dai, Jizhong Han, Guanbin Li, Si Liu | cs.CV | [PDF](http://arxiv.org/pdf/2312.01663v1){: .btn .btn-green } |

**Abstract**: In this paper, we target the adaptive source driven 3D scene editing task by
proposing a CustomNeRF model that unifies a text description or a reference
image as the editing prompt. However, obtaining desired editing results
conformed with the editing prompt is nontrivial since there exist two
significant challenges, including accurate editing of only foreground regions
and multi-view consistency given a single-view reference image. To tackle the
first challenge, we propose a Local-Global Iterative Editing (LGIE) training
scheme that alternates between foreground region editing and full-image
editing, aimed at foreground-only manipulation while preserving the background.
For the second challenge, we also design a class-guided regularization that
exploits class priors within the generation model to alleviate the
inconsistency problem among different views in image-driven editing. Extensive
experiments show that our CustomNeRF produces precise editing results under
various real scenes for both text- and image-driven settings.

Comments:
- 14 pages, 13 figures, project website: https://customnerf.github.io/

---

## Fast and accurate sparse-view CBCT reconstruction using meta-learned  neural attenuation field and hash-encoding regularization



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Heejun Shin, Taehee Kim, Jongho Lee, Se Young Chun, Seungryung Cho, Dongmyung Shin | eess.IV | [PDF](http://arxiv.org/pdf/2312.01689v2){: .btn .btn-green } |

**Abstract**: Cone beam computed tomography (CBCT) is an emerging medical imaging technique
to visualize the internal anatomical structures of patients. During a CBCT
scan, several projection images of different angles or views are collectively
utilized to reconstruct a tomographic image. However, reducing the number of
projections in a CBCT scan while preserving the quality of a reconstructed
image is challenging due to the nature of an ill-posed inverse problem.
Recently, a neural attenuation field (NAF) method was proposed by adopting a
neural radiance field algorithm as a new way for CBCT reconstruction,
demonstrating fast and promising results using only 50 views. However,
decreasing the number of projections is still preferable to reduce potential
radiation exposure, and a faster reconstruction time is required considering a
typical scan time. In this work, we propose a fast and accurate sparse-view
CBCT reconstruction (FACT) method to provide better reconstruction quality and
faster optimization speed in the minimal number of view acquisitions ($<$ 50
views). In the FACT method, we meta-trained a neural network and a hash-encoder
using a few scans (= 15), and a new regularization technique is utilized to
reconstruct the details of an anatomical structure. In conclusion, we have
shown that the FACT method produced better, and faster reconstruction results
over the other conventional algorithms based on CBCT scans of different body
parts (chest, head, and abdomen) and CT vendors (Siemens, Phillips, and GE).

---

## SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, Xiaojuan Qi | cs.CV | [PDF](http://arxiv.org/pdf/2312.14937v2){: .btn .btn-green } |

**Abstract**: Novel view synthesis for dynamic scenes is still a challenging problem in
computer vision and graphics. Recently, Gaussian splatting has emerged as a
robust technique to represent static scenes and enable high-quality and
real-time novel view synthesis. Building upon this technique, we propose a new
representation that explicitly decomposes the motion and appearance of dynamic
scenes into sparse control points and dense Gaussians, respectively. Our key
idea is to use sparse control points, significantly fewer in number than the
Gaussians, to learn compact 6 DoF transformation bases, which can be locally
interpolated through learned interpolation weights to yield the motion field of
3D Gaussians. We employ a deformation MLP to predict time-varying 6 DoF
transformations for each control point, which reduces learning complexities,
enhances learning abilities, and facilitates obtaining temporal and spatial
coherent motion patterns. Then, we jointly learn the 3D Gaussians, the
canonical space locations of control points, and the deformation MLP to
reconstruct the appearance, geometry, and dynamics of 3D scenes. During
learning, the location and number of control points are adaptively adjusted to
accommodate varying motion complexities in different regions, and an ARAP loss
following the principle of as rigid as possible is developed to enforce spatial
continuity and local rigidity of learned motions. Finally, thanks to the
explicit sparse motion representation and its decomposition from appearance,
our method can enable user-controlled motion editing while retaining
high-fidelity appearances. Extensive experiments demonstrate that our approach
outperforms existing approaches on novel view synthesis with a high rendering
speed and enables novel appearance-preserved motion editing applications.
Project page: https://yihua7.github.io/SC-GS-web/

Comments:
- Code link: https://github.com/yihua7/SC-GS

---

## ColonNeRF: Neural Radiance Fields for High-Fidelity Long-Sequence  Colonoscopy Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Yufei Shi, Beijia Lu, Jia-Wei Liu, Ming Li, Mike Zheng Shou | cs.CV | [PDF](http://arxiv.org/pdf/2312.02015v1){: .btn .btn-green } |

**Abstract**: Colonoscopy reconstruction is pivotal for diagnosing colorectal cancer.
However, accurate long-sequence colonoscopy reconstruction faces three major
challenges: (1) dissimilarity among segments of the colon due to its meandering
and convoluted shape; (2) co-existence of simple and intricately folded
geometry structures; (3) sparse viewpoints due to constrained camera
trajectories. To tackle these challenges, we introduce a new reconstruction
framework based on neural radiance field (NeRF), named ColonNeRF, which
leverages neural rendering for novel view synthesis of long-sequence
colonoscopy. Specifically, to reconstruct the entire colon in a piecewise
manner, our ColonNeRF introduces a region division and integration module,
effectively reducing shape dissimilarity and ensuring geometric consistency in
each segment. To learn both the simple and complex geometry in a unified
framework, our ColonNeRF incorporates a multi-level fusion module that
progressively models the colon regions from easy to hard. Additionally, to
overcome the challenges from sparse views, we devise a DensiNet module for
densifying camera poses under the guidance of semantic consistency. We conduct
extensive experiments on both synthetic and real-world datasets to evaluate our
ColonNeRF. Quantitatively, our ColonNeRF outperforms existing methods on two
benchmarks over four evaluation metrics. Notably, our LPIPS-ALEX scores exhibit
a substantial increase of about 67%-85% on the SimCol-to-3D dataset.
Qualitatively, our reconstruction visualizations show much clearer textures and
more accurate geometric details. These sufficiently demonstrate our superior
performance over the state-of-the-art methods.

Comments:
- for Project Page, see https://showlab.github.io/ColonNeRF/

---

## GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain, Matthias Nießner | cs.CV | [PDF](http://arxiv.org/pdf/2312.02069v1){: .btn .btn-green } |

**Abstract**: We introduce GaussianAvatars, a new method to create photorealistic head
avatars that are fully controllable in terms of expression, pose, and
viewpoint. The core idea is a dynamic 3D representation based on 3D Gaussian
splats that are rigged to a parametric morphable face model. This combination
facilitates photorealistic rendering while allowing for precise animation
control via the underlying parametric model, e.g., through expression transfer
from a driving sequence or by manually changing the morphable model parameters.
We parameterize each splat by a local coordinate frame of a triangle and
optimize for explicit displacement offset to obtain a more accurate geometric
representation. During avatar reconstruction, we jointly optimize for the
morphable model parameters and Gaussian splat parameters in an end-to-end
fashion. We demonstrate the animation capabilities of our photorealistic avatar
in several challenging scenarios. For instance, we show reenactments from a
driving video, where our method outperforms existing works by a significant
margin.

Comments:
- Project page: https://shenhanqian.github.io/gaussian-avatars

---

## PointNeRF++: A multi-scale, point-based Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Weiwei Sun, Eduard Trulls, Yang-Che Tseng, Sneha Sambandam, Gopal Sharma, Andrea Tagliasacchi, Kwang Moo Yi | cs.CV | [PDF](http://arxiv.org/pdf/2312.02362v1){: .btn .btn-green } |

**Abstract**: Point clouds offer an attractive source of information to complement images
in neural scene representations, especially when few images are available.
Neural rendering methods based on point clouds do exist, but they do not
perform well when the point cloud quality is low -- e.g., sparse or incomplete,
which is often the case with real-world data. We overcome these problems with a
simple representation that aggregates point clouds at multiple scale levels
with sparse voxel grids at different resolutions. To deal with point cloud
sparsity, we average across multiple scale levels -- but only among those that
are valid, i.e., that have enough neighboring points in proximity to the ray of
a pixel. To help model areas without points, we add a global voxel at the
coarsest scale, thus unifying "classical" and point-based NeRF formulations. We
validate our method on the NeRF Synthetic, ScanNet, and KITTI-360 datasets,
outperforming the state of the art by a significant margin.

---

## Fast View Synthesis of Casual Videos



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Yao-Chih Lee, Zhoutong Zhang, Kevin Blackburn-Matzen, Simon Niklaus, Jianming Zhang, Jia-Bin Huang, Feng Liu | cs.CV | [PDF](http://arxiv.org/pdf/2312.02135v1){: .btn .btn-green } |

**Abstract**: Novel view synthesis from an in-the-wild video is difficult due to challenges
like scene dynamics and lack of parallax. While existing methods have shown
promising results with implicit neural radiance fields, they are slow to train
and render. This paper revisits explicit video representations to synthesize
high-quality novel views from a monocular video efficiently. We treat static
and dynamic video content separately. Specifically, we build a global static
scene model using an extended plane-based scene representation to synthesize
temporally coherent novel video. Our plane-based scene representation is
augmented with spherical harmonics and displacement maps to capture
view-dependent effects and model non-planar complex surface geometry. We opt to
represent the dynamic content as per-frame point clouds for efficiency. While
such representations are inconsistency-prone, minor temporal inconsistencies
are perceptually masked due to motion. We develop a method to quickly estimate
such a hybrid video representation and render novel views in real time. Our
experiments show that our method can render high-quality novel views from an
in-the-wild video with comparable quality to state-of-the-art methods while
being 100x faster in training and enabling real-time rendering.

Comments:
- Project page: https://casual-fvs.github.io/

---

## MANUS: Markerless Hand-Object Grasp Capture using Articulated 3D  Gaussians

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Chandradeep Pokhariya, Ishaan N Shah, Angela Xing, Zekun Li, Kefan Chen, Avinash Sharma, Srinath Sridhar | cs.CV | [PDF](http://arxiv.org/pdf/2312.02137v1){: .btn .btn-green } |

**Abstract**: Understanding how we grasp objects with our hands has important applications
in areas like robotics and mixed reality. However, this challenging problem
requires accurate modeling of the contact between hands and objects. To capture
grasps, existing methods use skeletons, meshes, or parametric models that can
cause misalignments resulting in inaccurate contacts. We present MANUS, a
method for Markerless Hand-Object Grasp Capture using Articulated 3D Gaussians.
We build a novel articulated 3D Gaussians representation that extends 3D
Gaussian splatting for high-fidelity representation of articulating hands.
Since our representation uses Gaussian primitives, it enables us to efficiently
and accurately estimate contacts between the hand and the object. For the most
accurate results, our method requires tens of camera views that current
datasets do not provide. We therefore build MANUS-Grasps, a new dataset that
contains hand-object grasps viewed from 53 cameras across 30+ scenes, 3
subjects, and comprising over 7M frames. In addition to extensive qualitative
results, we also show that our method outperforms others on a quantitative
contact evaluation method that uses paint transfer from the object to the hand.

---

## GPS-Gaussian: Generalizable Pixel-wise 3D Gaussian Splatting for  Real-time Human Novel View Synthesis

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu, Shengping Zhang, Liqiang Nie, Yebin Liu | cs.CV | [PDF](http://arxiv.org/pdf/2312.02155v1){: .btn .btn-green } |

**Abstract**: We present a new approach, termed GPS-Gaussian, for synthesizing novel views
of a character in a real-time manner. The proposed method enables 2K-resolution
rendering under a sparse-view camera setting. Unlike the original Gaussian
Splatting or neural implicit rendering methods that necessitate per-subject
optimizations, we introduce Gaussian parameter maps defined on the source views
and regress directly Gaussian Splatting properties for instant novel view
synthesis without any fine-tuning or optimization. To this end, we train our
Gaussian parameter regression module on a large amount of human scan data,
jointly with a depth estimation module to lift 2D parameter maps to 3D space.
The proposed framework is fully differentiable and experiments on several
datasets demonstrate that our method outperforms state-of-the-art methods while
achieving an exceeding rendering speed.

Comments:
- The link to our projectpage is https://shunyuanzheng.github.io

---

## Mesh-Guided Neural Implicit Field Editing



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Can Wang, Mingming He, Menglei Chai, Dongdong Chen, Jing Liao | cs.CV | [PDF](http://arxiv.org/pdf/2312.02157v1){: .btn .btn-green } |

**Abstract**: Neural implicit fields have emerged as a powerful 3D representation for
reconstructing and rendering photo-realistic views, yet they possess limited
editability. Conversely, explicit 3D representations, such as polygonal meshes,
offer ease of editing but may not be as suitable for rendering high-quality
novel views. To harness the strengths of both representations, we propose a new
approach that employs a mesh as a guiding mechanism in editing the neural
radiance field. We first introduce a differentiable method using marching
tetrahedra for polygonal mesh extraction from the neural implicit field and
then design a differentiable color extractor to assign colors obtained from the
volume renderings to this extracted mesh. This differentiable colored mesh
allows gradient back-propagation from the explicit mesh to the implicit fields,
empowering users to easily manipulate the geometry and color of neural implicit
fields. To enhance user control from coarse-grained to fine-grained levels, we
introduce an octree-based structure into its optimization. This structure
prioritizes the edited regions and the surface part, making our method achieve
fine-grained edits to the neural implicit field and accommodate various user
modifications, including object additions, component removals, specific area
deformations, and adjustments to local and global colors. Through extensive
experiments involving diverse scenes and editing operations, we have
demonstrated the capabilities and effectiveness of our method. Our project page
is: \url{https://cassiepython.github.io/MNeuEdit/}

Comments:
- Project page: https://cassiepython.github.io/MNeuEdit/

---

## Calibrated Uncertainties for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Niki Amini-Naieni, Tomas Jakab, Andrea Vedaldi, Ronald Clark | cs.CV | [PDF](http://arxiv.org/pdf/2312.02350v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields have achieved remarkable results for novel view
synthesis but still lack a crucial component: precise measurement of
uncertainty in their predictions. Probabilistic NeRF methods have tried to
address this, but their output probabilities are not typically accurately
calibrated, and therefore do not capture the true confidence levels of the
model. Calibration is a particularly challenging problem in the sparse-view
setting, where additional held-out data is unavailable for fitting a calibrator
that generalizes to the test distribution. In this paper, we introduce the
first method for obtaining calibrated uncertainties from NeRF models. Our
method is based on a robust and efficient metric to calculate per-pixel
uncertainties from the predictive posterior distribution. We propose two
techniques that eliminate the need for held-out data. The first, based on patch
sampling, involves training two NeRF models for each scene. The second is a
novel meta-calibrator that only requires the training of one NeRF model. Our
proposed approach for obtaining calibrated uncertainties achieves
state-of-the-art uncertainty in the sparse-view setting while maintaining image
quality. We further demonstrate our method's effectiveness in applications such
as view enhancement and next-best view selection.

---

## Re-Nerfing: Enforcing Geometric Constraints on Neural Radiance Fields  through Novel Views Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-04 | Felix Tristram, Stefano Gasperini, Federico Tombari, Nassir Navab, Benjamin Busam | cs.CV | [PDF](http://arxiv.org/pdf/2312.02255v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have shown remarkable novel view synthesis
capabilities even in large-scale, unbounded scenes, albeit requiring hundreds
of views or introducing artifacts in sparser settings. Their optimization
suffers from shape-radiance ambiguities wherever only a small visual overlap is
available. This leads to erroneous scene geometry and artifacts. In this paper,
we propose Re-Nerfing, a simple and general multi-stage approach that leverages
NeRF's own view synthesis to address these limitations. With Re-Nerfing, we
increase the scene's coverage and enhance the geometric consistency of novel
views as follows: First, we train a NeRF with the available views. Then, we use
the optimized NeRF to synthesize pseudo-views next to the original ones to
simulate a stereo or trifocal setup. Finally, we train a second NeRF with both
original and pseudo views while enforcing structural, epipolar constraints via
the newly synthesized images. Extensive experiments on the mip-NeRF 360 dataset
show the effectiveness of Re-Nerfing across denser and sparser input scenarios,
bringing improvements to the state-of-the-art Zip-NeRF, even when trained with
all views.

Comments:
- Code will be released upon acceptance

---

## SANeRF-HQ: Segment Anything for NeRF in High Quality

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-03 | Yichen Liu, Benran Hu, Chi-Keung Tang, Yu-Wing Tai | cs.CV | [PDF](http://arxiv.org/pdf/2312.01531v1){: .btn .btn-green } |

**Abstract**: Recently, the Segment Anything Model (SAM) has showcased remarkable
capabilities of zero-shot segmentation, while NeRF (Neural Radiance Fields) has
gained popularity as a method for various 3D problems beyond novel view
synthesis. Though there exist initial attempts to incorporate these two methods
into 3D segmentation, they face the challenge of accurately and consistently
segmenting objects in complex scenarios. In this paper, we introduce the
Segment Anything for NeRF in High Quality (SANeRF-HQ) to achieve high quality
3D segmentation of any object in a given scene. SANeRF-HQ utilizes SAM for
open-world object segmentation guided by user-supplied prompts, while
leveraging NeRF to aggregate information from different viewpoints. To overcome
the aforementioned challenges, we employ density field and RGB similarity to
enhance the accuracy of segmentation boundary during the aggregation.
Emphasizing on segmentation accuracy, we evaluate our method quantitatively on
multiple NeRF datasets where high-quality ground-truths are available or
manually annotated. SANeRF-HQ shows a significant quality improvement over
previous state-of-the-art methods in NeRF object segmentation, provides higher
flexibility for object localization, and enables more consistent object
segmentation across multiple views. Additional information can be found at
https://lyclyc52.github.io/SANeRF-HQ/.

---

## WavePlanes: A compact Wavelet representation for Dynamic Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-03 | Adrian Azzarelli, Nantheera Anantrasirichai, David R Bull | cs.CV | [PDF](http://arxiv.org/pdf/2312.02218v1){: .btn .btn-green } |

**Abstract**: Dynamic Neural Radiance Fields (Dynamic NeRF) enhance NeRF technology to
model moving scenes. However, they are resource intensive and challenging to
compress. To address this issue, this paper presents WavePlanes, a fast and
more compact explicit model. We propose a multi-scale space and space-time
feature plane representation using N-level 2-D wavelet coefficients. The
inverse discrete wavelet transform reconstructs N feature signals at varying
detail, which are linearly decoded to approximate the color and density of
volumes in a 4-D grid. Exploiting the sparsity of wavelet coefficients, we
compress a Hash Map containing only non-zero coefficients and their locations
on each plane. This results in a compressed model size of ~12 MB. Compared with
state-of-the-art plane-based models, WavePlanes is up to 15x smaller, less
computationally demanding and achieves comparable results in as little as one
hour of training - without requiring custom CUDA code or high performance
computing resources. Additionally, we propose new feature fusion schemes that
work as well as previously proposed schemes while providing greater
interpretability. Our code is available at:
https://github.com/azzarelli/waveplanes/

---

## VideoRF: Rendering Dynamic Radiance Fields as 2D Feature Video Streams

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-03 | Liao Wang, Kaixin Yao, Chengcheng Guo, Zhirui Zhang, Qiang Hu, Jingyi Yu, Lan Xu, Minye Wu | cs.CV | [PDF](http://arxiv.org/pdf/2312.01407v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) excel in photorealistically rendering static
scenes. However, rendering dynamic, long-duration radiance fields on ubiquitous
devices remains challenging, due to data storage and computational constraints.
In this paper, we introduce VideoRF, the first approach to enable real-time
streaming and rendering of dynamic radiance fields on mobile platforms. At the
core is a serialized 2D feature image stream representing the 4D radiance field
all in one. We introduce a tailored training scheme directly applied to this 2D
domain to impose the temporal and spatial redundancy of the feature image
stream. By leveraging the redundancy, we show that the feature image stream can
be efficiently compressed by 2D video codecs, which allows us to exploit video
hardware accelerators to achieve real-time decoding. On the other hand, based
on the feature image stream, we propose a novel rendering pipeline for VideoRF,
which has specialized space mappings to query radiance properties efficiently.
Paired with a deferred shading model, VideoRF has the capability of real-time
rendering on mobile devices thanks to its efficiency. We have developed a
real-time interactive player that enables online streaming and rendering of
dynamic scenes, offering a seamless and immersive free-viewpoint experience
across a range of devices, from desktops to mobile phones.

Comments:
- Project page, see https://aoliao12138.github.io/VideoRF

---

## Self-Evolving Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-02 | Jaewoo Jung, Jisang Han, Jiwon Kang, Seongchan Kim, Min-Seop Kwak, Seungryong Kim | cs.CV | [PDF](http://arxiv.org/pdf/2312.01003v2){: .btn .btn-green } |

**Abstract**: Recently, neural radiance field (NeRF) has shown remarkable performance in
novel view synthesis and 3D reconstruction. However, it still requires abundant
high-quality images, limiting its applicability in real-world scenarios. To
overcome this limitation, recent works have focused on training NeRF only with
sparse viewpoints by giving additional regularizations, often called few-shot
NeRF. We observe that due to the under-constrained nature of the task, solely
using additional regularization is not enough to prevent the model from
overfitting to sparse viewpoints. In this paper, we propose a novel framework,
dubbed Self-Evolving Neural Radiance Fields (SE-NeRF), that applies a
self-training framework to NeRF to address these problems. We formulate
few-shot NeRF into a teacher-student framework to guide the network to learn a
more robust representation of the scene by training the student with additional
pseudo labels generated from the teacher. By distilling ray-level pseudo labels
using distinct distillation schemes for reliable and unreliable rays obtained
with our novel reliability estimation method, we enable NeRF to learn a more
accurate and robust geometry of the 3D scene. We show and evaluate that
applying our self-training framework to existing models improves the quality of
the rendered images and achieves state-of-the-art performance in multiple
settings.

Comments:
- 34 pages, 21 figures Our project page can be found at :
  https://ku-cvlab.github.io/SE-NeRF/

---

## StableDreamer: Taming Noisy Score Distillation Sampling for Text-to-3D

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-02 | Pengsheng Guo, Hans Hao, Adam Caccavale, Zhongzheng Ren, Edward Zhang, Qi Shan, Aditya Sankar, Alexander G. Schwing, Alex Colburn, Fangchang Ma | cs.CV | [PDF](http://arxiv.org/pdf/2312.02189v1){: .btn .btn-green } |

**Abstract**: In the realm of text-to-3D generation, utilizing 2D diffusion models through
score distillation sampling (SDS) frequently leads to issues such as blurred
appearances and multi-faced geometry, primarily due to the intrinsically noisy
nature of the SDS loss. Our analysis identifies the core of these challenges as
the interaction among noise levels in the 2D diffusion process, the
architecture of the diffusion network, and the 3D model representation. To
overcome these limitations, we present StableDreamer, a methodology
incorporating three advances. First, inspired by InstructNeRF2NeRF, we
formalize the equivalence of the SDS generative prior and a simple supervised
L2 reconstruction loss. This finding provides a novel tool to debug SDS, which
we use to show the impact of time-annealing noise levels on reducing
multi-faced geometries. Second, our analysis shows that while image-space
diffusion contributes to geometric precision, latent-space diffusion is crucial
for vivid color rendition. Based on this observation, StableDreamer introduces
a two-stage training strategy that effectively combines these aspects,
resulting in high-fidelity 3D models. Third, we adopt an anisotropic 3D
Gaussians representation, replacing Neural Radiance Fields (NeRFs), to enhance
the overall quality, reduce memory usage during training, and accelerate
rendering speeds, and better capture semi-transparent objects. StableDreamer
reduces multi-face geometries, generates fine details, and converges stably.

---

## Volumetric Rendering with Baked Quadrature Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-02 | Gopal Sharma, Daniel Rebain, Kwang Moo Yi, Andrea Tagliasacchi | cs.GR | [PDF](http://arxiv.org/pdf/2312.02202v1){: .btn .btn-green } |

**Abstract**: We propose a novel Neural Radiance Field (NeRF) representation for non-opaque
scenes that allows fast inference by utilizing textured polygons. Despite the
high-quality novel view rendering that NeRF provides, a critical limitation is
that it relies on volume rendering that can be computationally expensive and
does not utilize the advancements in modern graphics hardware. Existing methods
for this problem fall short when it comes to modelling volumetric effects as
they rely purely on surface rendering. We thus propose to model the scene with
polygons, which can then be used to obtain the quadrature points required to
model volumetric effects, and also their opacity and colour from the texture.
To obtain such polygonal mesh, we train a specialized field whose
zero-crossings would correspond to the quadrature points when volume rendering,
and perform marching cubes on this field. We then rasterize the polygons and
utilize the fragment shaders to obtain the final colour image. Our method
allows rendering on various devices and easy integration with existing graphics
frameworks while keeping the benefits of volume rendering alive.

---

## DISTWAR: Fast Differentiable Rendering on Raster-based Rendering  Pipelines

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-01 | Sankeerth Durvasula, Adrian Zhao, Fan Chen, Ruofan Liang, Pawan Kumar Sanjaya, Nandita Vijaykumar | cs.CV | [PDF](http://arxiv.org/pdf/2401.05345v1){: .btn .btn-green } |

**Abstract**: Differentiable rendering is a technique used in an important emerging class
of visual computing applications that involves representing a 3D scene as a
model that is trained from 2D images using gradient descent. Recent works (e.g.
3D Gaussian Splatting) use a rasterization pipeline to enable rendering high
quality photo-realistic imagery at high speeds from these learned 3D models.
These methods have been demonstrated to be very promising, providing
state-of-art quality for many important tasks. However, training a model to
represent a scene is still a time-consuming task even when using powerful GPUs.
In this work, we observe that the gradient computation phase during training is
a significant bottleneck on GPUs due to the large number of atomic operations
that need to be processed. These atomic operations overwhelm atomic units in
the L2 partitions causing stalls. To address this challenge, we leverage the
observations that during the gradient computation: (1) for most warps, all
threads atomically update the same memory locations; and (2) warps generate
varying amounts of atomic traffic (since some threads may be inactive). We
propose DISTWAR, a software-approach to accelerate atomic operations based on
two key ideas: First, we enable warp-level reduction of threads at the SM
sub-cores using registers to leverage the locality in intra-warp atomic
updates. Second, we distribute the atomic computation between the warp-level
reduction at the SM and the L2 atomic units to increase the throughput of
atomic computation. Warps with many threads performing atomic updates to the
same memory locations are scheduled at the SM, and the rest using L2 atomic
units. We implement DISTWAR using existing warp-level primitives. We evaluate
DISTWAR on widely used raster-based differentiable rendering workloads. We
demonstrate significant speedups of 2.44x on average (up to 5.7x).

---

## Segment Any 3D Gaussians

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-01 | Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, Qi Tian | cs.CV | [PDF](http://arxiv.org/pdf/2312.00860v1){: .btn .btn-green } |

**Abstract**: Interactive 3D segmentation in radiance fields is an appealing task since its
importance in 3D scene understanding and manipulation. However, existing
methods face challenges in either achieving fine-grained, multi-granularity
segmentation or contending with substantial computational overhead, inhibiting
real-time interaction. In this paper, we introduce Segment Any 3D GAussians
(SAGA), a novel 3D interactive segmentation approach that seamlessly blends a
2D segmentation foundation model with 3D Gaussian Splatting (3DGS), a recent
breakthrough of radiance fields. SAGA efficiently embeds multi-granularity 2D
segmentation results generated by the segmentation foundation model into 3D
Gaussian point features through well-designed contrastive training. Evaluation
on existing benchmarks demonstrates that SAGA can achieve competitive
performance with state-of-the-art methods. Moreover, SAGA achieves
multi-granularity segmentation and accommodates various prompts, including
points, scribbles, and 2D masks. Notably, SAGA can finish the 3D segmentation
within milliseconds, achieving nearly 1000x acceleration compared to previous
SOTA. The project page is at https://jumpat.github.io/SAGA.

Comments:
- Work in progress. Project page: https://jumpat.github.io/SAGA

---

## Gaussian Grouping: Segment and Edit Anything in 3D Scenes

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-01 | Mingqiao Ye, Martin Danelljan, Fisher Yu, Lei Ke | cs.CV | [PDF](http://arxiv.org/pdf/2312.00732v1){: .btn .btn-green } |

**Abstract**: The recent Gaussian Splatting achieves high-quality and real-time novel-view
synthesis of the 3D scenes. However, it is solely concentrated on the
appearance and geometry modeling, while lacking in fine-grained object-level
scene understanding. To address this issue, we propose Gaussian Grouping, which
extends Gaussian Splatting to jointly reconstruct and segment anything in
open-world 3D scenes. We augment each Gaussian with a compact Identity
Encoding, allowing the Gaussians to be grouped according to their object
instance or stuff membership in the 3D scene. Instead of resorting to expensive
3D labels, we supervise the Identity Encodings during the differentiable
rendering by leveraging the 2D mask predictions by SAM, along with introduced
3D spatial consistency regularization. Comparing to the implicit NeRF
representation, we show that the discrete and grouped 3D Gaussians can
reconstruct, segment and edit anything in 3D with high visual quality, fine
granularity and efficiency. Based on Gaussian Grouping, we further propose a
local Gaussian Editing scheme, which shows efficacy in versatile scene editing
applications, including 3D object removal, inpainting, colorization and scene
recomposition. Our code and models will be at
https://github.com/lkeab/gaussian-grouping.

Comments:
- We propose Gaussian Grouping, which extends Gaussian Splatting to
  fine-grained open-world 3D scene understanding. Github:
  https://github.com/lkeab/gaussian-grouping

---

## FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-01 | Zehao Zhu, Zhiwen Fan, Yifan Jiang, Zhangyang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2312.00451v1){: .btn .btn-green } |

**Abstract**: Novel view synthesis from limited observations remains an important and
persistent task. However, high efficiency in existing NeRF-based few-shot view
synthesis is often compromised to obtain an accurate 3D representation. To
address this challenge, we propose a few-shot view synthesis framework based on
3D Gaussian Splatting that enables real-time and photo-realistic view synthesis
with as few as three training views. The proposed method, dubbed FSGS, handles
the extremely sparse initialized SfM points with a thoughtfully designed
Gaussian Unpooling process. Our method iteratively distributes new Gaussians
around the most representative locations, subsequently infilling local details
in vacant areas. We also integrate a large-scale pre-trained monocular depth
estimator within the Gaussians optimization process, leveraging online
augmented views to guide the geometric optimization towards an optimal
solution. Starting from sparse points observed from limited input viewpoints,
our FSGS can accurately grow into unseen regions, comprehensively covering the
scene and boosting the rendering quality of novel views. Overall, FSGS achieves
state-of-the-art performance in both accuracy and rendering efficiency across
diverse datasets, including LLFF, Mip-NeRF360, and Blender. Project website:
https://zehaozhu.github.io/FSGS/.

Comments:
- Project page: https://zehaozhu.github.io/FSGS/

---

## NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting  Guidance

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-12-01 | Hanlin Chen, Chen Li, Gim Hee Lee | cs.CV | [PDF](http://arxiv.org/pdf/2312.00846v1){: .btn .btn-green } |

**Abstract**: Existing neural implicit surface reconstruction methods have achieved
impressive performance in multi-view 3D reconstruction by leveraging explicit
geometry priors such as depth maps or point clouds as regularization. However,
the reconstruction results still lack fine details because of the over-smoothed
depth map or sparse point cloud. In this work, we propose a neural implicit
surface reconstruction pipeline with guidance from 3D Gaussian Splatting to
recover highly detailed surfaces. The advantage of 3D Gaussian Splatting is
that it can generate dense point clouds with detailed structure. Nonetheless, a
naive adoption of 3D Gaussian Splatting can fail since the generated points are
the centers of 3D Gaussians that do not necessarily lie on the surface. We thus
introduce a scale regularizer to pull the centers close to the surface by
enforcing the 3D Gaussians to be extremely thin. Moreover, we propose to refine
the point cloud from 3D Gaussians Splatting with the normal priors from the
surface predicted by neural implicit models instead of using a fixed set of
points as guidance. Consequently, the quality of surface reconstruction
improves from the guidance of the more accurate 3D Gaussian splatting. By
jointly optimizing the 3D Gaussian Splatting and the neural implicit model, our
approach benefits from both representations and generates complete surfaces
with intricate details. Experiments on Tanks and Temples verify the
effectiveness of our proposed method.