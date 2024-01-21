---
layout: default
title: May
parent: 2023
nav_order: 5
---
<!---metadata--->

## Control4D: Efficient 4D Portrait Editing with Text

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-31 | Ruizhi Shao, Jingxiang Sun, Cheng Peng, Zerong Zheng, Boyao Zhou, Hongwen Zhang, Yebin Liu | cs.CV | [PDF](http://arxiv.org/pdf/2305.20082v2){: .btn .btn-green } |

**Abstract**: We introduce Control4D, an innovative framework for editing dynamic 4D
portraits using text instructions. Our method addresses the prevalent
challenges in 4D editing, notably the inefficiencies of existing 4D
representations and the inconsistent editing effect caused by diffusion-based
editors. We first propose GaussianPlanes, a novel 4D representation that makes
Gaussian Splatting more structured by applying plane-based decomposition in 3D
space and time. This enhances both efficiency and robustness in 4D editing.
Furthermore, we propose to leverage a 4D generator to learn a more continuous
generation space from inconsistent edited images produced by the
diffusion-based editor, which effectively improves the consistency and quality
of 4D editing. Comprehensive evaluation demonstrates the superiority of
Control4D, including significantly reduced training time, high-quality
rendering, and spatial-temporal consistency in 4D portrait editing. The link to
our project website is https://control4darxiv.github.io.

Comments:
- The link to our project website is https://control4darxiv.github.io

---

## DaRF: Boosting Radiance Fields from Sparse Inputs with Monocular Depth  Adaptation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-30 | Jiuhn Song, Seonghoon Park, Honggyu An, Seokju Cho, Min-Seop Kwak, Sungjin Cho, Seungryong Kim | cs.CV | [PDF](http://arxiv.org/pdf/2305.19201v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) shows powerful performance in novel view
synthesis and 3D geometry reconstruction, but it suffers from critical
performance degradation when the number of known viewpoints is drastically
reduced. Existing works attempt to overcome this problem by employing external
priors, but their success is limited to certain types of scenes or datasets.
Employing monocular depth estimation (MDE) networks, pretrained on large-scale
RGB-D datasets, with powerful generalization capability would be a key to
solving this problem: however, using MDE in conjunction with NeRF comes with a
new set of challenges due to various ambiguity problems exhibited by monocular
depths. In this light, we propose a novel framework, dubbed D\"aRF, that
achieves robust NeRF reconstruction with a handful of real-world images by
combining the strengths of NeRF and monocular depth estimation through online
complementary training. Our framework imposes the MDE network's powerful
geometry prior to NeRF representation at both seen and unseen viewpoints to
enhance its robustness and coherence. In addition, we overcome the ambiguity
problems of monocular depths through patch-wise scale-shift fitting and
geometry distillation, which adapts the MDE network to produce depths aligned
accurately with NeRF geometry. Experiments show our framework achieves
state-of-the-art results both quantitatively and qualitatively, demonstrating
consistent and reliable performance in both indoor and outdoor real-world
datasets. Project page is available at https://ku-cvlab.github.io/DaRF/.

Comments:
- To appear at NeurIPS 2023. Project Page:
  https://ku-cvlab.github.io/DaRF/

---

## Template-free Articulated Neural Point Clouds for Reposable View  Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-30 | Lukas Uzolas, Elmar Eisemann, Petr Kellnhofer | cs.CV | [PDF](http://arxiv.org/pdf/2305.19065v2){: .btn .btn-green } |

**Abstract**: Dynamic Neural Radiance Fields (NeRFs) achieve remarkable visual quality when
synthesizing novel views of time-evolving 3D scenes. However, the common
reliance on backward deformation fields makes reanimation of the captured
object poses challenging. Moreover, the state of the art dynamic models are
often limited by low visual fidelity, long reconstruction time or specificity
to narrow application domains. In this paper, we present a novel method
utilizing a point-based representation and Linear Blend Skinning (LBS) to
jointly learn a Dynamic NeRF and an associated skeletal model from even sparse
multi-view video. Our forward-warping approach achieves state-of-the-art visual
fidelity when synthesizing novel views and poses while significantly reducing
the necessary learning time when compared to existing work. We demonstrate the
versatility of our representation on a variety of articulated objects from
common datasets and obtain reposable 3D reconstructions without the need of
object-specific skeletal templates. Code will be made available at
https://github.com/lukasuz/Articulated-Point-NeRF.

---

## HiFA: High-fidelity Text-to-3D Generation with Advanced Diffusion  Guidance

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-30 | Junzhe Zhu, Peiye Zhuang | cs.CV | [PDF](http://arxiv.org/pdf/2305.18766v3){: .btn .btn-green } |

**Abstract**: The advancements in automatic text-to-3D generation have been remarkable.
Most existing methods use pre-trained text-to-image diffusion models to
optimize 3D representations like Neural Radiance Fields (NeRFs) via
latent-space denoising score matching. Yet, these methods often result in
artifacts and inconsistencies across different views due to their suboptimal
optimization approaches and limited understanding of 3D geometry. Moreover, the
inherent constraints of NeRFs in rendering crisp geometry and stable textures
usually lead to a two-stage optimization to attain high-resolution details.
This work proposes holistic sampling and smoothing approaches to achieve
high-quality text-to-3D generation, all in a single-stage optimization. We
compute denoising scores in the text-to-image diffusion model's latent and
image spaces. Instead of randomly sampling timesteps (also referred to as noise
levels in denoising score matching), we introduce a novel timestep annealing
approach that progressively reduces the sampled timestep throughout
optimization. To generate high-quality renderings in a single-stage
optimization, we propose regularization for the variance of z-coordinates along
NeRF rays. To address texture flickering issues in NeRFs, we introduce a kernel
smoothing technique that refines importance sampling weights coarse-to-fine,
ensuring accurate and thorough sampling in high-density regions. Extensive
experiments demonstrate the superiority of our method over previous approaches,
enabling the generation of highly detailed and view-consistent 3D assets
through a single-stage training process.

Comments:
- Project page: https://hifa-team.github.io/HiFA-site/

---

## Volume Feature Rendering for Fast Neural Radiance Field Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-29 | Kang Han, Wei Xiang, Lu Yu | cs.CV | [PDF](http://arxiv.org/pdf/2305.17916v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) are able to synthesize realistic novel views
from multi-view images captured from distinct positions and perspectives. In
NeRF's rendering pipeline, neural networks are used to represent a scene
independently or transform queried learnable feature vector of a point to the
expected color or density. With the aid of geometry guides either in occupancy
grids or proposal networks, the number of neural network evaluations can be
reduced from hundreds to dozens in the standard volume rendering framework.
Instead of rendering yielded color after neural network evaluation, we propose
to render the queried feature vectors of a ray first and then transform the
rendered feature vector to the final pixel color by a neural network. This
fundamental change to the standard volume rendering framework requires only one
single neural network evaluation to render a pixel, which substantially lowers
the high computational complexity of the rendering framework attributed to a
large number of neural network evaluations. Consequently, we can use a
comparably larger neural network to achieve a better rendering quality while
maintaining the same training and rendering time costs. Our model achieves the
state-of-the-art rendering quality on both synthetic and real-world datasets
while requiring a training time of several minutes.

---

## Towards a Robust Framework for NeRF Evaluation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-29 | Adrian Azzarelli, Nantheera Anantrasirichai, David R Bull | cs.CV | [PDF](http://arxiv.org/pdf/2305.18079v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) research has attracted significant attention
recently, with 3D modelling, virtual/augmented reality, and visual effects
driving its application. While current NeRF implementations can produce high
quality visual results, there is a conspicuous lack of reliable methods for
evaluating them. Conventional image quality assessment methods and analytical
metrics (e.g. PSNR, SSIM, LPIPS etc.) only provide approximate indicators of
performance since they generalise the ability of the entire NeRF pipeline.
Hence, in this paper, we propose a new test framework which isolates the neural
rendering network from the NeRF pipeline and then performs a parametric
evaluation by training and evaluating the NeRF on an explicit radiance field
representation. We also introduce a configurable approach for generating
representations specifically for evaluation purposes. This employs ray-casting
to transform mesh models into explicit NeRF samples, as well as to "shade"
these representations. Combining these two approaches, we demonstrate how
different "tasks" (scenes with different visual effects or learning strategies)
and types of networks (NeRFs and depth-wise implicit neural representations
(INRs)) can be evaluated within this framework. Additionally, we propose a
novel metric to measure task complexity of the framework which accounts for the
visual parameters and the distribution of the spatial data. Our approach offers
the potential to create a comparative objective evaluation framework for NeRF
methods.

Comments:
- 9 pages, 2 main experiments, 2 additional experiments

---

## Compact Real-time Radiance Fields with Neural Codebook



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-29 | Lingzhi Li, Zhongshu Wang, Zhen Shen, Li Shen, Ping Tan | cs.CV | [PDF](http://arxiv.org/pdf/2305.18163v1){: .btn .btn-green } |

**Abstract**: Reconstructing neural radiance fields with explicit volumetric
representations, demonstrated by Plenoxels, has shown remarkable advantages on
training and rendering efficiency, while grid-based representations typically
induce considerable overhead for storage and transmission. In this work, we
present a simple and effective framework for pursuing compact radiance fields
from the perspective of compression methodology. By exploiting intrinsic
properties exhibiting in grid models, a non-uniform compression stem is
developed to significantly reduce model complexity and a novel parameterized
module, named Neural Codebook, is introduced for better encoding high-frequency
details specific to per-scene models via a fast optimization. Our approach can
achieve over 40 $\times$ reduction on grid model storage with competitive
rendering quality. In addition, the method can achieve real-time rendering
speed with 180 fps, realizing significant advantage on storage cost compared to
real-time rendering methods.

Comments:
- Accepted by ICME 2023

---

## PlaNeRF: SVD Unsupervised 3D Plane Regularization for NeRF Large-Scale  Scene Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-26 | Fusang Wang, Arnaud Louys, Nathan Piasco, Moussab Bennehar, Luis Roldão, Dzmitry Tsishkou | cs.CV | [PDF](http://arxiv.org/pdf/2305.16914v4){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) enable 3D scene reconstruction from 2D images
and camera poses for Novel View Synthesis (NVS). Although NeRF can produce
photorealistic results, it often suffers from overfitting to training views,
leading to poor geometry reconstruction, especially in low-texture areas. This
limitation restricts many important applications which require accurate
geometry, such as extrapolated NVS, HD mapping and scene editing. To address
this limitation, we propose a new method to improve NeRF's 3D structure using
only RGB images and semantic maps. Our approach introduces a novel plane
regularization based on Singular Value Decomposition (SVD), that does not rely
on any geometric prior. In addition, we leverage the Structural Similarity
Index Measure (SSIM) in our loss design to properly initialize the volumetric
representation of NeRF. Quantitative and qualitative results show that our
method outperforms popular regularization approaches in accurate geometry
reconstruction for large-scale outdoor scenes and achieves SoTA rendering
quality on the KITTI-360 NVS benchmark.

Comments:
- Accepted to 3DV 2023

---

## ZeroAvatar: Zero-shot 3D Avatar Generation from a Single Image

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-25 | Zhenzhen Weng, Zeyu Wang, Serena Yeung | cs.CV | [PDF](http://arxiv.org/pdf/2305.16411v1){: .btn .btn-green } |

**Abstract**: Recent advancements in text-to-image generation have enabled significant
progress in zero-shot 3D shape generation. This is achieved by score
distillation, a methodology that uses pre-trained text-to-image diffusion
models to optimize the parameters of a 3D neural presentation, e.g. Neural
Radiance Field (NeRF). While showing promising results, existing methods are
often not able to preserve the geometry of complex shapes, such as human
bodies. To address this challenge, we present ZeroAvatar, a method that
introduces the explicit 3D human body prior to the optimization process.
Specifically, we first estimate and refine the parameters of a parametric human
body from a single image. Then during optimization, we use the posed parametric
body as additional geometry constraint to regularize the diffusion model as
well as the underlying density field. Lastly, we propose a UV-guided texture
regularization term to further guide the completion of texture on invisible
body parts. We show that ZeroAvatar significantly enhances the robustness and
3D consistency of optimization-based image-to-3D avatar generation,
outperforming existing zero-shot image-to-3D methods.

---

## Interactive Segment Anything NeRF with Feature Imitation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-25 | Xiaokang Chen, Jiaxiang Tang, Diwen Wan, Jingbo Wang, Gang Zeng | cs.CV | [PDF](http://arxiv.org/pdf/2305.16233v1){: .btn .btn-green } |

**Abstract**: This paper investigates the potential of enhancing Neural Radiance Fields
(NeRF) with semantics to expand their applications. Although NeRF has been
proven useful in real-world applications like VR and digital creation, the lack
of semantics hinders interaction with objects in complex scenes. We propose to
imitate the backbone feature of off-the-shelf perception models to achieve
zero-shot semantic segmentation with NeRF. Our framework reformulates the
segmentation process by directly rendering semantic features and only applying
the decoder from perception models. This eliminates the need for expensive
backbones and benefits 3D consistency. Furthermore, we can project the learned
semantics onto extracted mesh surfaces for real-time interaction. With the
state-of-the-art Segment Anything Model (SAM), our framework accelerates
segmentation by 16 times with comparable mask quality. The experimental results
demonstrate the efficacy and computational advantages of our approach. Project
page: \url{https://me.kiui.moe/san/}.

Comments:
- Technical Report

---

## ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with  Variational Score Distillation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-25 | Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, Jun Zhu | cs.LG | [PDF](http://arxiv.org/pdf/2305.16213v2){: .btn .btn-green } |

**Abstract**: Score distillation sampling (SDS) has shown great promise in text-to-3D
generation by distilling pretrained large-scale text-to-image diffusion models,
but suffers from over-saturation, over-smoothing, and low-diversity problems.
In this work, we propose to model the 3D parameter as a random variable instead
of a constant as in SDS and present variational score distillation (VSD), a
principled particle-based variational framework to explain and address the
aforementioned issues in text-to-3D generation. We show that SDS is a special
case of VSD and leads to poor samples with both small and large CFG weights. In
comparison, VSD works well with various CFG weights as ancestral sampling from
diffusion models and simultaneously improves the diversity and sample quality
with a common CFG weight (i.e., $7.5$). We further present various improvements
in the design space for text-to-3D such as distillation time schedule and
density initialization, which are orthogonal to the distillation algorithm yet
not well explored. Our overall approach, dubbed ProlificDreamer, can generate
high rendering resolution (i.e., $512\times512$) and high-fidelity NeRF with
rich structure and complex effects (e.g., smoke and drops). Further,
initialized from NeRF, meshes fine-tuned by VSD are meticulously detailed and
photo-realistic. Project page and codes:
https://ml.cs.tsinghua.edu.cn/prolificdreamer/

Comments:
- NeurIPS 2023 (Spotlight)

---

## Deceptive-NeRF: Enhancing NeRF Reconstruction using Pseudo-Observations  from Diffusion Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-24 | Xinhang Liu, Jiaben Chen, Shiu-hong Kao, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2305.15171v3){: .btn .btn-green } |

**Abstract**: We introduce Deceptive-NeRF, a novel methodology for few-shot NeRF
reconstruction, which leverages diffusion models to synthesize plausible
pseudo-observations to improve the reconstruction. This approach unfolds
through three key steps: 1) reconstructing a coarse NeRF from sparse input
data; 2) utilizing the coarse NeRF to render images and subsequently generating
pseudo-observations based on them; 3) training a refined NeRF model utilizing
input images augmented with pseudo-observations. We develop a deceptive
diffusion model that adeptly transitions RGB images and depth maps from coarse
NeRFs into photo-realistic pseudo-observations, all while preserving scene
semantics for reconstruction. Furthermore, we propose a progressive strategy
for training the Deceptive-NeRF, using the current NeRF renderings to create
pseudo-observations that enhance the next iteration's NeRF. Extensive
experiments demonstrate that our approach is capable of synthesizing
photo-realistic novel views, even for highly complex scenes with very sparse
inputs. Codes will be released.

---

## InpaintNeRF360: Text-Guided 3D Inpainting on Unbounded Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-24 | Dongqing Wang, Tong Zhang, Alaa Abboud, Sabine Süsstrunk | cs.CV | [PDF](http://arxiv.org/pdf/2305.15094v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) can generate highly realistic novel views.
However, editing 3D scenes represented by NeRF across 360-degree views,
particularly removing objects while preserving geometric and photometric
consistency, remains a challenging problem due to NeRF's implicit scene
representation. In this paper, we propose InpaintNeRF360, a unified framework
that utilizes natural language instructions as guidance for inpainting
NeRF-based 3D scenes.Our approach employs a promptable segmentation model by
generating multi-modal prompts from the encoded text for multiview
segmentation. We apply depth-space warping to enforce viewing consistency in
the segmentations, and further refine the inpainted NeRF model using perceptual
priors to ensure visual plausibility. InpaintNeRF360 is capable of
simultaneously removing multiple objects or modifying object appearance based
on text instructions while synthesizing 3D viewing-consistent and
photo-realistic inpainting. Through extensive experiments on both unbounded and
frontal-facing scenes trained through NeRF, we demonstrate the effectiveness of
our approach and showcase its potential to enhance the editability of implicit
radiance fields.

---

## OD-NeRF: Efficient Training of On-the-Fly Dynamic Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-24 | Zhiwen Yan, Chen Li, Gim Hee Lee | cs.CV | [PDF](http://arxiv.org/pdf/2305.14831v1){: .btn .btn-green } |

**Abstract**: Dynamic neural radiance fields (dynamic NeRFs) have demonstrated impressive
results in novel view synthesis on 3D dynamic scenes. However, they often
require complete video sequences for training followed by novel view synthesis,
which is similar to playing back the recording of a dynamic 3D scene. In
contrast, we propose OD-NeRF to efficiently train and render dynamic NeRFs
on-the-fly which instead is capable of streaming the dynamic scene. When
training on-the-fly, the training frames become available sequentially and the
model is trained and rendered frame-by-frame. The key challenge of efficient
on-the-fly training is how to utilize the radiance field estimated from the
previous frames effectively. To tackle this challenge, we propose: 1) a NeRF
model conditioned on the multi-view projected colors to implicitly track
correspondence between the current and previous frames, and 2) a transition and
update algorithm that leverages the occupancy grid from the last frame to
sample efficiently at the current frame. Our algorithm can achieve an
interactive speed of 6FPS training and rendering on synthetic dynamic scenes
on-the-fly, and a significant speed-up compared to the state-of-the-art on
real-world dynamic scenes.

---

## Weakly Supervised 3D Open-vocabulary Segmentation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-23 | Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, Shijian Lu | cs.CV | [PDF](http://arxiv.org/pdf/2305.14093v4){: .btn .btn-green } |

**Abstract**: Open-vocabulary segmentation of 3D scenes is a fundamental function of human
perception and thus a crucial objective in computer vision research. However,
this task is heavily impeded by the lack of large-scale and diverse 3D
open-vocabulary segmentation datasets for training robust and generalizable
models. Distilling knowledge from pre-trained 2D open-vocabulary segmentation
models helps but it compromises the open-vocabulary feature as the 2D models
are mostly finetuned with close-vocabulary datasets. We tackle the challenges
in 3D open-vocabulary segmentation by exploiting pre-trained foundation models
CLIP and DINO in a weakly supervised manner. Specifically, given only the
open-vocabulary text descriptions of the objects in a scene, we distill the
open-vocabulary multimodal knowledge and object reasoning capability of CLIP
and DINO into a neural radiance field (NeRF), which effectively lifts 2D
features into view-consistent 3D segmentation. A notable aspect of our approach
is that it does not require any manual segmentation annotations for either the
foundation models or the distillation process. Extensive experiments show that
our method even outperforms fully supervised models trained with segmentation
annotations in certain scenes, suggesting that 3D open-vocabulary segmentation
can be effectively learned from 2D images and text-image pairs. Code is
available at \url{https://github.com/Kunhao-Liu/3D-OVS}.

Comments:
- Accepted to NeurIPS 2023

---

## NeRFuser: Large-Scale Scene Representation by NeRF Fusion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-22 | Jiading Fang, Shengjie Lin, Igor Vasiljevic, Vitor Guizilini, Rares Ambrus, Adrien Gaidon, Gregory Shakhnarovich, Matthew R. Walter | cs.CV | [PDF](http://arxiv.org/pdf/2305.13307v1){: .btn .btn-green } |

**Abstract**: A practical benefit of implicit visual representations like Neural Radiance
Fields (NeRFs) is their memory efficiency: large scenes can be efficiently
stored and shared as small neural nets instead of collections of images.
However, operating on these implicit visual data structures requires extending
classical image-based vision techniques (e.g., registration, blending) from
image sets to neural fields. Towards this goal, we propose NeRFuser, a novel
architecture for NeRF registration and blending that assumes only access to
pre-generated NeRFs, and not the potentially large sets of images used to
generate them. We propose registration from re-rendering, a technique to infer
the transformation between NeRFs based on images synthesized from individual
NeRFs. For blending, we propose sample-based inverse distance weighting to
blend visual information at the ray-sample level. We evaluate NeRFuser on
public benchmarks and a self-collected object-centric indoor dataset, showing
the robustness of our method, including to views that are challenging to render
from the individual source NeRFs.

Comments:
- Code available at https://github.com/ripl/nerfuser

---

## Registering Neural Radiance Fields as 3D Density Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-22 | Han Jiang, Ruoxuan Li, Haosen Sun, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2305.12843v1){: .btn .btn-green } |

**Abstract**: No significant work has been done to directly merge two partially overlapping
scenes using NeRF representations. Given pre-trained NeRF models of a 3D scene
with partial overlapping, this paper aligns them with a rigid transform, by
generalizing the traditional registration pipeline, that is, key point
detection and point set registration, to operate on 3D density fields. To
describe corner points as key points in 3D, we propose to use universal
pre-trained descriptor-generating neural networks that can be trained and
tested on different scenes. We perform experiments to demonstrate that the
descriptor networks can be conveniently trained using a contrastive learning
strategy. We demonstrate that our method, as a global approach, can effectively
register NeRF models, thus making possible future large-scale NeRF construction
by registering its smaller and overlapping NeRFs captured individually.

---

## Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-19 | Jingbo Zhang, Xiaoyu Li, Ziyu Wan, Can Wang, Jing Liao | cs.CV | [PDF](http://arxiv.org/pdf/2305.11588v1){: .btn .btn-green } |

**Abstract**: Text-driven 3D scene generation is widely applicable to video gaming, film
industry, and metaverse applications that have a large demand for 3D scenes.
However, existing text-to-3D generation methods are limited to producing 3D
objects with simple geometries and dreamlike styles that lack realism. In this
work, we present Text2NeRF, which is able to generate a wide range of 3D scenes
with complicated geometric structures and high-fidelity textures purely from a
text prompt. To this end, we adopt NeRF as the 3D representation and leverage a
pre-trained text-to-image diffusion model to constrain the 3D reconstruction of
the NeRF to reflect the scene description. Specifically, we employ the
diffusion model to infer the text-related image as the content prior and use a
monocular depth estimation method to offer the geometric prior. Both content
and geometric priors are utilized to update the NeRF model. To guarantee
textured and geometric consistency between different views, we introduce a
progressive scene inpainting and updating strategy for novel view synthesis of
the scene. Our method requires no additional training data but only a natural
language description of the scene as the input. Extensive experiments
demonstrate that our Text2NeRF outperforms existing methods in producing
photo-realistic, multi-view consistent, and diverse 3D scenes from a variety of
natural language prompts.

Comments:
- Homepage: https://eckertzhang.github.io/Text2NeRF.github.io/

---

## MVPSNet: Fast Generalizable Multi-view Photometric Stereo

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-18 | Dongxu Zhao, Daniel Lichy, Pierre-Nicolas Perrin, Jan-Michael Frahm, Soumyadip Sengupta | cs.CV | [PDF](http://arxiv.org/pdf/2305.11167v1){: .btn .btn-green } |

**Abstract**: We propose a fast and generalizable solution to Multi-view Photometric Stereo
(MVPS), called MVPSNet. The key to our approach is a feature extraction network
that effectively combines images from the same view captured under multiple
lighting conditions to extract geometric features from shading cues for stereo
matching. We demonstrate these features, termed `Light Aggregated Feature Maps'
(LAFM), are effective for feature matching even in textureless regions, where
traditional multi-view stereo methods fail. Our method produces similar
reconstruction results to PS-NeRF, a state-of-the-art MVPS method that
optimizes a neural network per-scene, while being 411$\times$ faster (105
seconds vs. 12 hours) in inference. Additionally, we introduce a new synthetic
dataset for MVPS, sMVPS, which is shown to be effective to train a
generalizable MVPS method.

---

## ConsistentNeRF: Enhancing Neural Radiance Fields with 3D Consistency for  Sparse View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-18 | Shoukang Hu, Kaichen Zhou, Kaiyu Li, Longhui Yu, Lanqing Hong, Tianyang Hu, Zhenguo Li, Gim Hee Lee, Ziwei Liu | cs.CV | [PDF](http://arxiv.org/pdf/2305.11031v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) has demonstrated remarkable 3D reconstruction
capabilities with dense view images. However, its performance significantly
deteriorates under sparse view settings. We observe that learning the 3D
consistency of pixels among different views is crucial for improving
reconstruction quality in such cases. In this paper, we propose ConsistentNeRF,
a method that leverages depth information to regularize both multi-view and
single-view 3D consistency among pixels. Specifically, ConsistentNeRF employs
depth-derived geometry information and a depth-invariant loss to concentrate on
pixels that exhibit 3D correspondence and maintain consistent depth
relationships. Extensive experiments on recent representative works reveal that
our approach can considerably enhance model performance in sparse view
conditions, achieving improvements of up to 94% in PSNR, 76% in SSIM, and 31%
in LPIPS compared to the vanilla baselines across various benchmarks, including
DTU, NeRF Synthetic, and LLFF.

Comments:
- https://github.com/skhu101/ConsistentNeRF

---

## MultiPlaneNeRF: Neural Radiance Field with Non-Trainable Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-17 | Dominik Zimny, Artur Kasymov, Adam Kania, Jacek Tabor, Maciej Zięba, Przemysław Spurek | cs.CV | [PDF](http://arxiv.org/pdf/2305.10579v2){: .btn .btn-green } |

**Abstract**: NeRF is a popular model that efficiently represents 3D objects from 2D
images. However, vanilla NeRF has some important limitations. NeRF must be
trained on each object separately. The training time is long since we encode
the object's shape and color in neural network weights. Moreover, NeRF does not
generalize well to unseen data. In this paper, we present MultiPlaneNeRF -- a
model that simultaneously solves the above problems. Our model works directly
on 2D images. We project 3D points on 2D images to produce non-trainable
representations. The projection step is not parametrized and a very shallow
decoder can efficiently process the representation. Furthermore, we can train
MultiPlaneNeRF on a large data set and force our implicit decoder to generalize
across many objects. Consequently, we can only replace the 2D images (without
additional training) to produce a NeRF representation of the new object. In the
experimental section, we demonstrate that MultiPlaneNeRF achieves results
comparable to state-of-the-art models for synthesizing new views and has
generalization properties. Additionally, MultiPlane decoder can be used as a
component in large generative models like GANs.

---

## OR-NeRF: Object Removing from 3D Scenes Guided by Multiview Segmentation  with Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-17 | Youtan Yin, Zhoujie Fu, Fan Yang, Guosheng Lin | cs.CV | [PDF](http://arxiv.org/pdf/2305.10503v3){: .btn .btn-green } |

**Abstract**: The emergence of Neural Radiance Fields (NeRF) for novel view synthesis has
increased interest in 3D scene editing. An essential task in editing is
removing objects from a scene while ensuring visual reasonability and multiview
consistency. However, current methods face challenges such as time-consuming
object labeling, limited capability to remove specific targets, and compromised
rendering quality after removal. This paper proposes a novel object-removing
pipeline, named OR-NeRF, that can remove objects from 3D scenes with user-given
points or text prompts on a single view, achieving better performance in less
time than previous works. Our method spreads user annotations to all views
through 3D geometry and sparse correspondence, ensuring 3D consistency with
less processing burden. Then recent 2D segmentation model Segment-Anything
(SAM) is applied to predict masks, and a 2D inpainting model is used to
generate color supervision. Finally, our algorithm applies depth supervision
and perceptual loss to maintain consistency in geometry and appearance after
object removal. Experimental results demonstrate that our method achieves
better editing quality with less time than previous works, considering both
quality and quantity.

Comments:
- project site: https://ornerf.github.io/ (codes available)

---

## NerfBridge: Bringing Real-time, Online Neural Radiance Field Training to  Robotics

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-16 | Javier Yu, Jun En Low, Keiko Nagami, Mac Schwager | cs.RO | [PDF](http://arxiv.org/pdf/2305.09761v1){: .btn .btn-green } |

**Abstract**: This work was presented at the IEEE International Conference on Robotics and
Automation 2023 Workshop on Unconventional Spatial Representations.
  Neural radiance fields (NeRFs) are a class of implicit scene representations
that model 3D environments from color images. NeRFs are expressive, and can
model the complex and multi-scale geometry of real world environments, which
potentially makes them a powerful tool for robotics applications. Modern NeRF
training libraries can generate a photo-realistic NeRF from a static data set
in just a few seconds, but are designed for offline use and require a slow pose
optimization pre-computation step.
  In this work we propose NerfBridge, an open-source bridge between the Robot
Operating System (ROS) and the popular Nerfstudio library for real-time, online
training of NeRFs from a stream of images. NerfBridge enables rapid development
of research on applications of NeRFs in robotics by providing an extensible
interface to the efficient training pipelines and model libraries provided by
Nerfstudio. As an example use case we outline a hardware setup that can be used
NerfBridge to train a NeRF from images captured by a camera mounted to a
quadrotor in both indoor and outdoor environments.
  For accompanying video https://youtu.be/EH0SLn-RcDg and code
https://github.com/javieryu/nerf_bridge.

---

## Curvature-Aware Training for Coordinate Networks



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-15 | Hemanth Saratchandran, Shin-Fang Chng, Sameera Ramasinghe, Lachlan MacDonald, Simon Lucey | cs.CV | [PDF](http://arxiv.org/pdf/2305.08552v1){: .btn .btn-green } |

**Abstract**: Coordinate networks are widely used in computer vision due to their ability
to represent signals as compressed, continuous entities. However, training
these networks with first-order optimizers can be slow, hindering their use in
real-time applications. Recent works have opted for shallow voxel-based
representations to achieve faster training, but this sacrifices memory
efficiency. This work proposes a solution that leverages second-order
optimization methods to significantly reduce training times for coordinate
networks while maintaining their compressibility. Experiments demonstrate the
effectiveness of this approach on various signal modalities, such as audio,
images, videos, shape reconstruction, and neural radiance fields.

---

## MV-Map: Offboard HD-Map Generation with Multi-view Consistency

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-15 | Ziyang Xie, Ziqi Pang, Yu-Xiong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2305.08851v3){: .btn .btn-green } |

**Abstract**: While bird's-eye-view (BEV) perception models can be useful for building
high-definition maps (HD-Maps) with less human labor, their results are often
unreliable and demonstrate noticeable inconsistencies in the predicted HD-Maps
from different viewpoints. This is because BEV perception is typically set up
in an 'onboard' manner, which restricts the computation and consequently
prevents algorithms from reasoning multiple views simultaneously. This paper
overcomes these limitations and advocates a more practical 'offboard' HD-Map
generation setup that removes the computation constraints, based on the fact
that HD-Maps are commonly reusable infrastructures built offline in data
centers. To this end, we propose a novel offboard pipeline called MV-Map that
capitalizes multi-view consistency and can handle an arbitrary number of frames
with the key design of a 'region-centric' framework. In MV-Map, the target
HD-Maps are created by aggregating all the frames of onboard predictions,
weighted by the confidence scores assigned by an 'uncertainty network'. To
further enhance multi-view consistency, we augment the uncertainty network with
the global 3D structure optimized by a voxelized neural radiance field
(Voxel-NeRF). Extensive experiments on nuScenes show that our MV-Map
significantly improves the quality of HD-Maps, further highlighting the
importance of offboard methods for HD-Map generation.

Comments:
- ICCV 2023

---

## BundleRecon: Ray Bundle-Based 3D Neural Reconstruction



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-12 | Weikun Zhang, Jianke Zhu | cs.CV | [PDF](http://arxiv.org/pdf/2305.07342v1){: .btn .btn-green } |

**Abstract**: With the growing popularity of neural rendering, there has been an increasing
number of neural implicit multi-view reconstruction methods. While many models
have been enhanced in terms of positional encoding, sampling, rendering, and
other aspects to improve the reconstruction quality, current methods do not
fully leverage the information among neighboring pixels during the
reconstruction process. To address this issue, we propose an enhanced model
called BundleRecon. In the existing approaches, sampling is performed by a
single ray that corresponds to a single pixel. In contrast, our model samples a
patch of pixels using a bundle of rays, which incorporates information from
neighboring pixels. Furthermore, we design bundle-based constraints to further
improve the reconstruction quality. Experimental results demonstrate that
BundleRecon is compatible with the existing neural implicit multi-view
reconstruction methods and can improve their reconstruction quality.

Comments:
- CVPR 2023 workshop XRNeRF: Advances in NeRF for the Metaverse

---

## SparseGNV: Generating Novel Views of Indoor Scenes with Sparse Input  Views



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-11 | Weihao Cheng, Yan-Pei Cao, Ying Shan | cs.CV | [PDF](http://arxiv.org/pdf/2305.07024v1){: .btn .btn-green } |

**Abstract**: We study to generate novel views of indoor scenes given sparse input views.
The challenge is to achieve both photorealism and view consistency. We present
SparseGNV: a learning framework that incorporates 3D structures and image
generative models to generate novel views with three modules. The first module
builds a neural point cloud as underlying geometry, providing contextual
information and guidance for the target novel view. The second module utilizes
a transformer-based network to map the scene context and the guidance into a
shared latent space and autoregressively decodes the target view in the form of
discrete image tokens. The third module reconstructs the tokens into the image
of the target view. SparseGNV is trained across a large indoor scene dataset to
learn generalizable priors. Once trained, it can efficiently generate novel
views of an unseen indoor scene in a feed-forward manner. We evaluate SparseGNV
on both real-world and synthetic indoor scenes and demonstrate that it
outperforms state-of-the-art methods based on either neural radiance fields or
conditional image generation.

Comments:
- 10 pages, 6 figures

---

## HumanRF: High-Fidelity Neural Radiance Fields for Humans in Motion



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-10 | Mustafa Işık, Martin Rünz, Markos Georgopoulos, Taras Khakhulin, Jonathan Starck, Lourdes Agapito, Matthias Nießner | cs.CV | [PDF](http://arxiv.org/pdf/2305.06356v2){: .btn .btn-green } |

**Abstract**: Representing human performance at high-fidelity is an essential building
block in diverse applications, such as film production, computer games or
videoconferencing. To close the gap to production-level quality, we introduce
HumanRF, a 4D dynamic neural scene representation that captures full-body
appearance in motion from multi-view video input, and enables playback from
novel, unseen viewpoints. Our novel representation acts as a dynamic video
encoding that captures fine details at high compression rates by factorizing
space-time into a temporal matrix-vector decomposition. This allows us to
obtain temporally coherent reconstructions of human actors for long sequences,
while representing high-resolution details even in the context of challenging
motion. While most research focuses on synthesizing at resolutions of 4MP or
lower, we address the challenge of operating at 12MP. To this end, we introduce
ActorsHQ, a novel multi-view dataset that provides 12MP footage from 160
cameras for 16 sequences with high-fidelity, per-frame mesh reconstructions. We
demonstrate challenges that emerge from using such high-resolution data and
show that our newly introduced HumanRF effectively leverages this data, making
a significant step towards production-level quality novel view synthesis.

Comments:
- Project webpage: https://synthesiaresearch.github.io/humanrf Dataset
  webpage: https://www.actors-hq.com/ Video:
  https://www.youtube.com/watch?v=OTnhiLLE7io Code:
  https://github.com/synthesiaresearch/humanrf

---

## Generative AI meets 3D: A Survey on Text-to-3D in AIGC Era

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-10 | Chenghao Li, Chaoning Zhang, Atish Waghwase, Lik-Hang Lee, Francois Rameau, Yang Yang, Sung-Ho Bae, Choong Seon Hong | cs.CV | [PDF](http://arxiv.org/pdf/2305.06131v2){: .btn .btn-green } |

**Abstract**: Generative AI (AIGC, a.k.a. AI generated content) has made remarkable
progress in the past few years, among which text-guided content generation is
the most practical one since it enables the interaction between human
instruction and AIGC. Due to the development in text-to-image as well 3D
modeling technologies (like NeRF), text-to-3D has become a newly emerging yet
highly active research field. Our work conducts the first yet comprehensive
survey on text-to-3D to help readers interested in this direction quickly catch
up with its fast development. First, we introduce 3D data representations,
including both Euclidean data and non-Euclidean data. On top of that, we
introduce various foundation technologies as well as summarize how recent works
combine those foundation technologies to realize satisfactory text-to-3D.
Moreover, we summarize how text-to-3D technology is used in various
applications, including avatar generation, texture generation, shape
transformation, and scene generation.

---

## NeRF2: Neural Radio-Frequency Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-10 | Xiaopeng Zhao, Zhenlin An, Qingrui Pan, Lei Yang | cs.NI | [PDF](http://arxiv.org/pdf/2305.06118v2){: .btn .btn-green } |

**Abstract**: Although Maxwell discovered the physical laws of electromagnetic waves 160
years ago, how to precisely model the propagation of an RF signal in an
electrically large and complex environment remains a long-standing problem. The
difficulty is in the complex interactions between the RF signal and the
obstacles (e.g., reflection, diffraction, etc.). Inspired by the great success
of using a neural network to describe the optical field in computer vision, we
propose a neural radio-frequency radiance field, NeRF$^\textbf{2}$, which
represents a continuous volumetric scene function that makes sense of an RF
signal's propagation. Particularly, after training with a few signal
measurements, NeRF$^\textbf{2}$ can tell how/what signal is received at any
position when it knows the position of a transmitter. As a physical-layer
neural network, NeRF$^\textbf{2}$ can take advantage of the learned statistic
model plus the physical model of ray tracing to generate a synthetic dataset
that meets the training demands of application-layer artificial neural networks
(ANNs). Thus, we can boost the performance of ANNs by the proposed
turbo-learning, which mixes the true and synthetic datasets to intensify the
training. Our experiment results show that turbo-learning can enhance
performance with an approximate 50% increase. We also demonstrate the power of
NeRF$^\textbf{2}$ in the field of indoor localization and 5G MIMO.

---

## Instant-NeRF: Instant On-Device Neural Radiance Field Training via  Algorithm-Accelerator Co-Designed Near-Memory Processing

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-09 | Yang Zhao, Shang Wu, Jingqun Zhang, Sixu Li, Chaojian Li, Yingyan Lin | cs.CV | [PDF](http://arxiv.org/pdf/2305.05766v1){: .btn .btn-green } |

**Abstract**: Instant on-device Neural Radiance Fields (NeRFs) are in growing demand for
unleashing the promise of immersive AR/VR experiences, but are still limited by
their prohibitive training time. Our profiling analysis reveals a memory-bound
inefficiency in NeRF training. To tackle this inefficiency, near-memory
processing (NMP) promises to be an effective solution, but also faces
challenges due to the unique workloads of NeRFs, including the random hash
table lookup, random point processing sequence, and heterogeneous bottleneck
steps. Therefore, we propose the first NMP framework, Instant-NeRF, dedicated
to enabling instant on-device NeRF training. Experiments on eight datasets
consistently validate the effectiveness of Instant-NeRF.

Comments:
- Accepted by DAC 2023

---

## PET-NeuS: Positional Encoding Tri-Planes for Neural Surfaces

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-09 | Yiqun Wang, Ivan Skorokhodov, Peter Wonka | cs.CV | [PDF](http://arxiv.org/pdf/2305.05594v1){: .btn .btn-green } |

**Abstract**: A signed distance function (SDF) parametrized by an MLP is a common
ingredient of neural surface reconstruction. We build on the successful recent
method NeuS to extend it by three new components. The first component is to
borrow the tri-plane representation from EG3D and represent signed distance
fields as a mixture of tri-planes and MLPs instead of representing it with MLPs
only. Using tri-planes leads to a more expressive data structure but will also
introduce noise in the reconstructed surface. The second component is to use a
new type of positional encoding with learnable weights to combat noise in the
reconstruction process. We divide the features in the tri-plane into multiple
frequency scales and modulate them with sin and cos functions of different
frequencies. The third component is to use learnable convolution operations on
the tri-plane features using self-attention convolution to produce features
with different frequency bands. The experiments show that PET-NeuS achieves
high-fidelity surface reconstruction on standard datasets. Following previous
work and using the Chamfer metric as the most important way to measure surface
reconstruction quality, we are able to improve upon the NeuS baseline by 57% on
Nerf-synthetic (0.84 compared to 1.97) and by 15.5% on DTU (0.71 compared to
0.84). The qualitative evaluation reveals how our method can better control the
interference of high-frequency noise. Code available at
\url{https://github.com/yiqun-wang/PET-NeuS}.

Comments:
- CVPR 2023; 20 Pages; Project page:
  \url{https://github.com/yiqun-wang/PET-NeuS}

---

## AvatarReX: Real-time Expressive Full-body Avatars

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-08 | Zerong Zheng, Xiaochen Zhao, Hongwen Zhang, Boning Liu, Yebin Liu | cs.CV | [PDF](http://arxiv.org/pdf/2305.04789v1){: .btn .btn-green } |

**Abstract**: We present AvatarReX, a new method for learning NeRF-based full-body avatars
from video data. The learnt avatar not only provides expressive control of the
body, hands and the face together, but also supports real-time animation and
rendering. To this end, we propose a compositional avatar representation, where
the body, hands and the face are separately modeled in a way that the
structural prior from parametric mesh templates is properly utilized without
compromising representation flexibility. Furthermore, we disentangle the
geometry and appearance for each part. With these technical designs, we propose
a dedicated deferred rendering pipeline, which can be executed in real-time
framerate to synthesize high-quality free-view images. The disentanglement of
geometry and appearance also allows us to design a two-pass training strategy
that combines volume rendering and surface rendering for network training. In
this way, patch-level supervision can be applied to force the network to learn
sharp appearance details on the basis of geometry estimation. Overall, our
method enables automatic construction of expressive full-body avatars with
real-time rendering capability, and can generate photo-realistic images with
dynamic details for novel body motions and facial expressions.

Comments:
- To appear in SIGGRAPH 2023 Journal Track. Project page at
  https://liuyebin.com/AvatarRex/

---

## NerfAcc: Efficient Sampling Accelerates NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-08 | Ruilong Li, Hang Gao, Matthew Tancik, Angjoo Kanazawa | cs.CV | [PDF](http://arxiv.org/pdf/2305.04966v2){: .btn .btn-green } |

**Abstract**: Optimizing and rendering Neural Radiance Fields is computationally expensive
due to the vast number of samples required by volume rendering. Recent works
have included alternative sampling approaches to help accelerate their methods,
however, they are often not the focus of the work. In this paper, we
investigate and compare multiple sampling approaches and demonstrate that
improved sampling is generally applicable across NeRF variants under an unified
concept of transmittance estimator. To facilitate future experiments, we
develop NerfAcc, a Python toolbox that provides flexible APIs for incorporating
advanced sampling methods into NeRF related methods. We demonstrate its
flexibility by showing that it can reduce the training time of several recent
NeRF methods by 1.5x to 20x with minimal modifications to the existing
codebase. Additionally, highly customized NeRFs, such as Instant-NGP, can be
implemented in native PyTorch using NerfAcc.

Comments:
- Website: https://www.nerfacc.com

---

## HashCC: Lightweight Method to Improve the Quality of the Camera-less  NeRF Scene Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-07 | Jan Olszewski | cs.CV | [PDF](http://arxiv.org/pdf/2305.04296v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields has become a prominent method of scene generation via
view synthesis. A critical requirement for the original algorithm to learn
meaningful scene representation is camera pose information for each image in a
data set. Current approaches try to circumnavigate this assumption with
moderate success, by learning approximate camera positions alongside learning
neural representations of a scene. This requires complicated camera models,
causing a long and complicated training process, or results in a lack of
texture and sharp details in rendered scenes. In this work we introduce Hash
Color Correction (HashCC) -- a lightweight method for improving Neural Radiance
Fields rendered image quality, applicable also in situations where camera
positions for a given set of images are unknown.

---

## Multi-Space Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-07 | Ze-Xin Yin, Jiaxiong Qiu, Ming-Ming Cheng, Bo Ren | cs.CV | [PDF](http://arxiv.org/pdf/2305.04268v1){: .btn .btn-green } |

**Abstract**: Existing Neural Radiance Fields (NeRF) methods suffer from the existence of
reflective objects, often resulting in blurry or distorted rendering. Instead
of calculating a single radiance field, we propose a multi-space neural
radiance field (MS-NeRF) that represents the scene using a group of feature
fields in parallel sub-spaces, which leads to a better understanding of the
neural network toward the existence of reflective and refractive objects. Our
multi-space scheme works as an enhancement to existing NeRF methods, with only
small computational overheads needed for training and inferring the extra-space
outputs. We demonstrate the superiority and compatibility of our approach using
three representative NeRF-based models, i.e., NeRF, Mip-NeRF, and Mip-NeRF 360.
Comparisons are performed on a novelly constructed dataset consisting of 25
synthetic scenes and 7 real captured scenes with complex reflection and
refraction, all having 360-degree viewpoints. Extensive experiments show that
our approach significantly outperforms the existing single-space NeRF methods
for rendering high-quality scenes concerned with complex light paths through
mirror-like objects. Our code and dataset will be publicly available at
https://zx-yin.github.io/msnerf.

Comments:
- CVPR 2023, 10 pages, 12 figures

---

## General Neural Gauge Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-05 | Fangneng Zhan, Lingjie Liu, Adam Kortylewski, Christian Theobalt | cs.CV | [PDF](http://arxiv.org/pdf/2305.03462v2){: .btn .btn-green } |

**Abstract**: The recent advance of neural fields, such as neural radiance fields, has
significantly pushed the boundary of scene representation learning. Aiming to
boost the computation efficiency and rendering quality of 3D scenes, a popular
line of research maps the 3D coordinate system to another measuring system,
e.g., 2D manifolds and hash tables, for modeling neural fields. The conversion
of coordinate systems can be typically dubbed as \emph{gauge transformation},
which is usually a pre-defined mapping function, e.g., orthogonal projection or
spatial hash function. This begs a question: can we directly learn a desired
gauge transformation along with the neural field in an end-to-end manner? In
this work, we extend this problem to a general paradigm with a taxonomy of
discrete \& continuous cases, and develop a learning framework to jointly
optimize gauge transformations and neural fields. To counter the problem that
the learning of gauge transformations can collapse easily, we derive a general
regularization mechanism from the principle of information conservation during
the gauge transformation. To circumvent the high computation cost in gauge
learning with regularization, we directly derive an information-invariant gauge
transformation which allows to preserve scene information inherently and yield
superior performance. Project: https://fnzhan.com/Neural-Gauge-Fields

Comments:
- ICLR 2023

---

## NeRF-QA: Neural Radiance Fields Quality Assessment Database

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-04 | Pedro Martin, António Rodrigues, João Ascenso, Maria Paula Queluz | cs.MM | [PDF](http://arxiv.org/pdf/2305.03176v1){: .btn .btn-green } |

**Abstract**: This short paper proposes a new database - NeRF-QA - containing 48 videos
synthesized with seven NeRF based methods, along with their perceived quality
scores, resulting from subjective assessment tests; for the videos selection,
both real and synthetic, 360 degrees scenes were considered. This database will
allow to evaluate the suitability, to NeRF based synthesized views, of existing
objective quality metrics and also the development of new quality metrics,
specific for this case.

---

## NeuralEditor: Editing Neural Radiance Fields via Manipulating Point  Clouds

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-04 | Jun-Kun Chen, Jipeng Lyu, Yu-Xiong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2305.03049v1){: .btn .btn-green } |

**Abstract**: This paper proposes NeuralEditor that enables neural radiance fields (NeRFs)
natively editable for general shape editing tasks. Despite their impressive
results on novel-view synthesis, it remains a fundamental challenge for NeRFs
to edit the shape of the scene. Our key insight is to exploit the explicit
point cloud representation as the underlying structure to construct NeRFs,
inspired by the intuitive interpretation of NeRF rendering as a process that
projects or "plots" the associated 3D point cloud to a 2D image plane. To this
end, NeuralEditor introduces a novel rendering scheme based on deterministic
integration within K-D tree-guided density-adaptive voxels, which produces both
high-quality rendering results and precise point clouds through optimization.
NeuralEditor then performs shape editing via mapping associated points between
point clouds. Extensive evaluation shows that NeuralEditor achieves
state-of-the-art performance in both shape deformation and scene morphing
tasks. Notably, NeuralEditor supports both zero-shot inference and further
fine-tuning over the edited scene. Our code, benchmark, and demo video are
available at https://immortalco.github.io/NeuralEditor.

Comments:
- CVPR 2023

---

## Single-Shot Implicit Morphable Faces with Consistent Texture  Parameterization



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-04 | Connor Z. Lin, Koki Nagano, Jan Kautz, Eric R. Chan, Umar Iqbal, Leonidas Guibas, Gordon Wetzstein, Sameh Khamis | cs.CV | [PDF](http://arxiv.org/pdf/2305.03043v1){: .btn .btn-green } |

**Abstract**: There is a growing demand for the accessible creation of high-quality 3D
avatars that are animatable and customizable. Although 3D morphable models
provide intuitive control for editing and animation, and robustness for
single-view face reconstruction, they cannot easily capture geometric and
appearance details. Methods based on neural implicit representations, such as
signed distance functions (SDF) or neural radiance fields, approach
photo-realism, but are difficult to animate and do not generalize well to
unseen data. To tackle this problem, we propose a novel method for constructing
implicit 3D morphable face models that are both generalizable and intuitive for
editing. Trained from a collection of high-quality 3D scans, our face model is
parameterized by geometry, expression, and texture latent codes with a learned
SDF and explicit UV texture parameterization. Once trained, we can reconstruct
an avatar from a single in-the-wild image by leveraging the learned prior to
project the image into the latent space of our model. Our implicit morphable
face models can be used to render an avatar from novel views, animate facial
expressions by modifying expression codes, and edit textures by directly
painting on the learned UV-texture maps. We demonstrate quantitatively and
qualitatively that our method improves upon photo-realism, geometry, and
expression accuracy compared to state-of-the-art methods.

Comments:
- SIGGRAPH 2023, Project Page:
  https://research.nvidia.com/labs/toronto-ai/ssif

---

## NeRSemble: Multi-view Radiance Field Reconstruction of Human Heads



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-04 | Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim Walter, Matthias Nießner | cs.CV | [PDF](http://arxiv.org/pdf/2305.03027v1){: .btn .btn-green } |

**Abstract**: We focus on reconstructing high-fidelity radiance fields of human heads,
capturing their animations over time, and synthesizing re-renderings from novel
viewpoints at arbitrary time steps. To this end, we propose a new multi-view
capture setup composed of 16 calibrated machine vision cameras that record
time-synchronized images at 7.1 MP resolution and 73 frames per second. With
our setup, we collect a new dataset of over 4700 high-resolution,
high-framerate sequences of more than 220 human heads, from which we introduce
a new human head reconstruction benchmark. The recorded sequences cover a wide
range of facial dynamics, including head motions, natural expressions,
emotions, and spoken language. In order to reconstruct high-fidelity human
heads, we propose Dynamic Neural Radiance Fields using Hash Ensembles
(NeRSemble). We represent scene dynamics by combining a deformation field and
an ensemble of 3D multi-resolution hash encodings. The deformation field allows
for precise modeling of simple scene movements, while the ensemble of hash
encodings helps to represent complex dynamics. As a result, we obtain radiance
field representations of human heads that capture motion over time and
facilitate re-rendering of arbitrary novel viewpoints. In a series of
experiments, we explore the design choices of our method and demonstrate that
our approach outperforms state-of-the-art dynamic radiance field approaches by
a significant margin.

Comments:
- Siggraph 2023, Project Page:
  https://tobias-kirschstein.github.io/nersemble/ , Video:
  https://youtu.be/a-OAWqBzldU

---

## Floaters No More: Radiance Field Gradient Scaling for Improved  Near-Camera Training

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-04 | Julien Philip, Valentin Deschaintre | cs.CV | [PDF](http://arxiv.org/pdf/2305.02756v2){: .btn .btn-green } |

**Abstract**: NeRF acquisition typically requires careful choice of near planes for the
different cameras or suffers from background collapse, creating floating
artifacts on the edges of the captured scene. The key insight of this work is
that background collapse is caused by a higher density of samples in regions
near cameras. As a result of this sampling imbalance, near-camera volumes
receive significantly more gradients, leading to incorrect density buildup. We
propose a gradient scaling approach to counter-balance this sampling imbalance,
removing the need for near planes, while preventing background collapse. Our
method can be implemented in a few lines, does not induce any significant
overhead, and is compatible with most NeRF implementations.

Comments:
- EGSR 2023

---

## Semantic-aware Generation of Multi-view Portrait Drawings

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-04 | Biao Ma, Fei Gao, Chang Jiang, Nannan Wang, Gang Xu | cs.CV | [PDF](http://arxiv.org/pdf/2305.02618v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) based methods have shown amazing performance in
synthesizing 3D-consistent photographic images, but fail to generate multi-view
portrait drawings. The key is that the basic assumption of these methods -- a
surface point is consistent when rendered from different views -- doesn't hold
for drawings. In a portrait drawing, the appearance of a facial point may
changes when viewed from different angles. Besides, portrait drawings usually
present little 3D information and suffer from insufficient training data. To
combat this challenge, in this paper, we propose a Semantic-Aware GEnerator
(SAGE) for synthesizing multi-view portrait drawings. Our motivation is that
facial semantic labels are view-consistent and correlate with drawing
techniques. We therefore propose to collaboratively synthesize multi-view
semantic maps and the corresponding portrait drawings. To facilitate training,
we design a semantic-aware domain translator, which generates portrait drawings
based on features of photographic faces. In addition, use data augmentation via
synthesis to mitigate collapsed results. We apply SAGE to synthesize multi-view
portrait drawings in diverse artistic styles. Experimental results show that
SAGE achieves significantly superior or highly competitive performance,
compared to existing 3D-aware image synthesis methods. The codes are available
at https://github.com/AiArt-HDU/SAGE.

Comments:
- Accepted by IJCAI 2023

---

## Real-Time Radiance Fields for Single-Image Portrait View Synthesis



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-03 | Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, Koki Nagano | cs.CV | [PDF](http://arxiv.org/pdf/2305.02310v1){: .btn .btn-green } |

**Abstract**: We present a one-shot method to infer and render a photorealistic 3D
representation from a single unposed image (e.g., face portrait) in real-time.
Given a single RGB input, our image encoder directly predicts a canonical
triplane representation of a neural radiance field for 3D-aware novel view
synthesis via volume rendering. Our method is fast (24 fps) on consumer
hardware, and produces higher quality results than strong GAN-inversion
baselines that require test-time optimization. To train our triplane encoder
pipeline, we use only synthetic data, showing how to distill the knowledge from
a pretrained 3D GAN into a feedforward encoder. Technical contributions include
a Vision Transformer-based triplane encoder, a camera data augmentation
strategy, and a well-designed loss function for synthetic data training. We
benchmark against the state-of-the-art methods, demonstrating significant
improvements in robustness and image quality in challenging real-world
settings. We showcase our results on portraits of faces (FFHQ) and cats (AFHQ),
but our algorithm can also be applied in the future to other categories with a
3D-aware image generator.

Comments:
- Project page: https://research.nvidia.com/labs/nxp/lp3d/

---

## Shap-E: Generating Conditional 3D Implicit Functions



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-03 | Heewoo Jun, Alex Nichol | cs.CV | [PDF](http://arxiv.org/pdf/2305.02463v1){: .btn .btn-green } |

**Abstract**: We present Shap-E, a conditional generative model for 3D assets. Unlike
recent work on 3D generative models which produce a single output
representation, Shap-E directly generates the parameters of implicit functions
that can be rendered as both textured meshes and neural radiance fields. We
train Shap-E in two stages: first, we train an encoder that deterministically
maps 3D assets into the parameters of an implicit function; second, we train a
conditional diffusion model on outputs of the encoder. When trained on a large
dataset of paired 3D and text data, our resulting models are capable of
generating complex and diverse 3D assets in a matter of seconds. When compared
to Point-E, an explicit generative model over point clouds, Shap-E converges
faster and reaches comparable or better sample quality despite modeling a
higher-dimensional, multi-representation output space. We release model
weights, inference code, and samples at https://github.com/openai/shap-e.

Comments:
- 23 pages, 13 figures

---

## LatentAvatar: Learning Latent Expression Code for Expressive Neural Head  Avatar

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-02 | Yuelang Xu, Hongwen Zhang, Lizhen Wang, Xiaochen Zhao, Han Huang, Guojun Qi, Yebin Liu | cs.CV | [PDF](http://arxiv.org/pdf/2305.01190v2){: .btn .btn-green } |

**Abstract**: Existing approaches to animatable NeRF-based head avatars are either built
upon face templates or use the expression coefficients of templates as the
driving signal. Despite the promising progress, their performances are heavily
bound by the expression power and the tracking accuracy of the templates. In
this work, we present LatentAvatar, an expressive neural head avatar driven by
latent expression codes. Such latent expression codes are learned in an
end-to-end and self-supervised manner without templates, enabling our method to
get rid of expression and tracking issues. To achieve this, we leverage a
latent head NeRF to learn the person-specific latent expression codes from a
monocular portrait video, and further design a Y-shaped network to learn the
shared latent expression codes of different subjects for cross-identity
reenactment. By optimizing the photometric reconstruction objectives in NeRF,
the latent expression codes are learned to be 3D-aware while faithfully
capturing the high-frequency detailed expressions. Moreover, by learning a
mapping between the latent expression code learned in shared and
person-specific settings, LatentAvatar is able to perform expressive
reenactment between different subjects. Experimental results show that our
LatentAvatar is able to capture challenging expressions and the subtle movement
of teeth and even eyeballs, which outperforms previous state-of-the-art
solutions in both quantitative and qualitative comparisons. Project page:
https://www.liuyebin.com/latentavatar.

Comments:
- Accepted by SIGGRAPH 2023

---

## Federated Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-02 | Lachlan Holden, Feras Dayoub, David Harvey, Tat-Jun Chin | cs.CV | [PDF](http://arxiv.org/pdf/2305.01163v1){: .btn .btn-green } |

**Abstract**: The ability of neural radiance fields or NeRFs to conduct accurate 3D
modelling has motivated application of the technique to scene representation.
Previous approaches have mainly followed a centralised learning paradigm, which
assumes that all training images are available on one compute node for
training. In this paper, we consider training NeRFs in a federated manner,
whereby multiple compute nodes, each having acquired a distinct set of
observations of the overall scene, learn a common NeRF in parallel. This
supports the scenario of cooperatively modelling a scene using multiple agents.
Our contribution is the first federated learning algorithm for NeRF, which
splits the training effort across multiple compute nodes and obviates the need
to pool the images at a central node. A technique based on low-rank
decomposition of NeRF layers is introduced to reduce bandwidth consumption to
transmit the model parameters for aggregation. Transferring compressed models
instead of the raw data also contributes to the privacy of the data collecting
agents.

Comments:
- 10 pages, 7 figures

---

## Neural LiDAR Fields for Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-02 | Shengyu Huang, Zan Gojcic, Zian Wang, Francis Williams, Yoni Kasten, Sanja Fidler, Konrad Schindler, Or Litany | cs.CV | [PDF](http://arxiv.org/pdf/2305.01643v2){: .btn .btn-green } |

**Abstract**: We present Neural Fields for LiDAR (NFL), a method to optimise a neural field
scene representation from LiDAR measurements, with the goal of synthesizing
realistic LiDAR scans from novel viewpoints. NFL combines the rendering power
of neural fields with a detailed, physically motivated model of the LiDAR
sensing process, thus enabling it to accurately reproduce key sensor behaviors
like beam divergence, secondary returns, and ray dropping. We evaluate NFL on
synthetic and real LiDAR scans and show that it outperforms explicit
reconstruct-then-simulate methods as well as other NeRF-style methods on LiDAR
novel view synthesis task. Moreover, we show that the improved realism of the
synthesized views narrows the domain gap to real scans and translates to better
registration and semantic segmentation performance.

Comments:
- ICCV 2023 - camera ready. Project page:
  https://research.nvidia.com/labs/toronto-ai/nfl/

---

## GeneFace++: Generalized and Stable Real-Time Audio-Driven 3D Talking  Face Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-05-01 | Zhenhui Ye, Jinzheng He, Ziyue Jiang, Rongjie Huang, Jiawei Huang, Jinglin Liu, Yi Ren, Xiang Yin, Zejun Ma, Zhou Zhao | cs.CV | [PDF](http://arxiv.org/pdf/2305.00787v1){: .btn .btn-green } |

**Abstract**: Generating talking person portraits with arbitrary speech audio is a crucial
problem in the field of digital human and metaverse. A modern talking face
generation method is expected to achieve the goals of generalized audio-lip
synchronization, good video quality, and high system efficiency. Recently,
neural radiance field (NeRF) has become a popular rendering technique in this
field since it could achieve high-fidelity and 3D-consistent talking face
generation with a few-minute-long training video. However, there still exist
several challenges for NeRF-based methods: 1) as for the lip synchronization,
it is hard to generate a long facial motion sequence of high temporal
consistency and audio-lip accuracy; 2) as for the video quality, due to the
limited data used to train the renderer, it is vulnerable to out-of-domain
input condition and produce bad rendering results occasionally; 3) as for the
system efficiency, the slow training and inference speed of the vanilla NeRF
severely obstruct its usage in real-world applications. In this paper, we
propose GeneFace++ to handle these challenges by 1) utilizing the pitch contour
as an auxiliary feature and introducing a temporal loss in the facial motion
prediction process; 2) proposing a landmark locally linear embedding method to
regulate the outliers in the predicted motion sequence to avoid robustness
issues; 3) designing a computationally efficient NeRF-based motion-to-video
renderer to achieves fast training and real-time inference. With these
settings, GeneFace++ becomes the first NeRF-based method that achieves stable
and real-time talking face generation with generalized audio-lip
synchronization. Extensive experiments show that our method outperforms
state-of-the-art baselines in terms of subjective and objective evaluation.
Video samples are available at https://genefaceplusplus.github.io .

Comments:
- 18 Pages, 7 figures