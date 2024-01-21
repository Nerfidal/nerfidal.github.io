---
layout: default
title: June
parent: 2023
nav_order: 6
---
<!---metadata--->

## Magic123: One Image to High-Quality 3D Object Generation Using Both 2D  and 3D Diffusion Priors



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-30 | Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, Bernard Ghanem | cs.CV | [PDF](http://arxiv.org/pdf/2306.17843v2){: .btn .btn-green } |

**Abstract**: We present Magic123, a two-stage coarse-to-fine approach for high-quality,
textured 3D meshes generation from a single unposed image in the wild using
both2D and 3D priors. In the first stage, we optimize a neural radiance field
to produce a coarse geometry. In the second stage, we adopt a memory-efficient
differentiable mesh representation to yield a high-resolution mesh with a
visually appealing texture. In both stages, the 3D content is learned through
reference view supervision and novel views guided by a combination of 2D and 3D
diffusion priors. We introduce a single trade-off parameter between the 2D and
3D priors to control exploration (more imaginative) and exploitation (more
precise) of the generated geometry. Additionally, we employ textual inversion
and monocular depth regularization to encourage consistent appearances across
views and to prevent degenerate solutions, respectively. Magic123 demonstrates
a significant improvement over previous image-to-3D techniques, as validated
through extensive experiments on synthetic benchmarks and diverse real-world
images. Our code, models, and generated 3D assets are available at
https://github.com/guochengqian/Magic123.

Comments:
- webpage: https://guochengqian.github.io/project/magic123/

---

## FlipNeRF: Flipped Reflection Rays for Few-shot Novel View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-30 | Seunghyeon Seo, Yeonjin Chang, Nojun Kwak | cs.CV | [PDF](http://arxiv.org/pdf/2306.17723v4){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has been a mainstream in novel view synthesis
with its remarkable quality of rendered images and simple architecture.
Although NeRF has been developed in various directions improving continuously
its performance, the necessity of a dense set of multi-view images still exists
as a stumbling block to progress for practical application. In this work, we
propose FlipNeRF, a novel regularization method for few-shot novel view
synthesis by utilizing our proposed flipped reflection rays. The flipped
reflection rays are explicitly derived from the input ray directions and
estimated normal vectors, and play a role of effective additional training rays
while enabling to estimate more accurate surface normals and learn the 3D
geometry effectively. Since the surface normal and the scene depth are both
derived from the estimated densities along a ray, the accurate surface normal
leads to more exact depth estimation, which is a key factor for few-shot novel
view synthesis. Furthermore, with our proposed Uncertainty-aware Emptiness Loss
and Bottleneck Feature Consistency Loss, FlipNeRF is able to estimate more
reliable outputs with reducing floating artifacts effectively across the
different scene structures, and enhance the feature-level consistency between
the pair of the rays cast toward the photo-consistent pixels without any
additional feature extractor, respectively. Our FlipNeRF achieves the SOTA
performance on the multiple benchmarks across all the scenarios.

Comments:
- ICCV 2023. Project Page: https://shawn615.github.io/flipnerf/

---

## Sphere2Vec: A General-Purpose Location Representation Learning over a  Spherical Surface for Large-Scale Geospatial Predictions

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-30 | Gengchen Mai, Yao Xuan, Wenyun Zuo, Yutong He, Jiaming Song, Stefano Ermon, Krzysztof Janowicz, Ni Lao | cs.CV | [PDF](http://arxiv.org/pdf/2306.17624v2){: .btn .btn-green } |

**Abstract**: Generating learning-friendly representations for points in space is a
fundamental and long-standing problem in ML. Recently, multi-scale encoding
schemes (such as Space2Vec and NeRF) were proposed to directly encode any point
in 2D/3D Euclidean space as a high-dimensional vector, and has been
successfully applied to various geospatial prediction and generative tasks.
However, all current 2D and 3D location encoders are designed to model point
distances in Euclidean space. So when applied to large-scale real-world GPS
coordinate datasets, which require distance metric learning on the spherical
surface, both types of models can fail due to the map projection distortion
problem (2D) and the spherical-to-Euclidean distance approximation error (3D).
To solve these problems, we propose a multi-scale location encoder called
Sphere2Vec which can preserve spherical distances when encoding point
coordinates on a spherical surface. We developed a unified view of
distance-reserving encoding on spheres based on the DFS. We also provide
theoretical proof that the Sphere2Vec preserves the spherical surface distance
between any two points, while existing encoding schemes do not. Experiments on
20 synthetic datasets show that Sphere2Vec can outperform all baseline models
on all these datasets with up to 30.8% error rate reduction. We then apply
Sphere2Vec to three geo-aware image classification tasks - fine-grained species
recognition, Flickr image recognition, and remote sensing image classification.
Results on 7 real-world datasets show the superiority of Sphere2Vec over
multiple location encoders on all three tasks. Further analysis shows that
Sphere2Vec outperforms other location encoder models, especially in the polar
regions and data-sparse areas because of its nature for spherical surface
distance preservation. Code and data are available at
https://gengchenmai.github.io/sphere2vec-website/.

Comments:
- 30 Pages, 16 figures. Accepted to ISPRS Journal of Photogrammetry and
  Remote Sensing

---

## One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape  Optimization



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-29 | Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, Hao Su | cs.CV | [PDF](http://arxiv.org/pdf/2306.16928v1){: .btn .btn-green } |

**Abstract**: Single image 3D reconstruction is an important but challenging task that
requires extensive knowledge of our natural world. Many existing methods solve
this problem by optimizing a neural radiance field under the guidance of 2D
diffusion models but suffer from lengthy optimization time, 3D inconsistency
results, and poor geometry. In this work, we propose a novel method that takes
a single image of any object as input and generates a full 360-degree 3D
textured mesh in a single feed-forward pass. Given a single image, we first use
a view-conditioned 2D diffusion model, Zero123, to generate multi-view images
for the input view, and then aim to lift them up to 3D space. Since traditional
reconstruction methods struggle with inconsistent multi-view predictions, we
build our 3D reconstruction module upon an SDF-based generalizable neural
surface reconstruction method and propose several critical training strategies
to enable the reconstruction of 360-degree meshes. Without costly
optimizations, our method reconstructs 3D shapes in significantly less time
than existing methods. Moreover, our method favors better geometry, generates
more 3D consistent results, and adheres more closely to the input image. We
evaluate our approach on both synthetic data and in-the-wild images and
demonstrate its superiority in terms of both mesh quality and runtime. In
addition, our approach can seamlessly support the text-to-3D task by
integrating with off-the-shelf text-to-image diffusion models.

Comments:
- project website: one-2-3-45.com

---

## Envisioning a Next Generation Extended Reality Conferencing System with  Efficient Photorealistic Human Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-28 | Chuanyue Shen, Letian Zhang, Zhangsihao Yang, Masood Mortazavi, Xiyun Song, Liang Peng, Heather Yu | cs.CV | [PDF](http://arxiv.org/pdf/2306.16541v1){: .btn .btn-green } |

**Abstract**: Meeting online is becoming the new normal. Creating an immersive experience
for online meetings is a necessity towards more diverse and seamless
environments. Efficient photorealistic rendering of human 3D dynamics is the
core of immersive meetings. Current popular applications achieve real-time
conferencing but fall short in delivering photorealistic human dynamics, either
due to limited 2D space or the use of avatars that lack realistic interactions
between participants. Recent advances in neural rendering, such as the Neural
Radiance Field (NeRF), offer the potential for greater realism in metaverse
meetings. However, the slow rendering speed of NeRF poses challenges for
real-time conferencing. We envision a pipeline for a future extended reality
metaverse conferencing system that leverages monocular video acquisition and
free-viewpoint synthesis to enhance data and hardware efficiency. Towards an
immersive conferencing experience, we explore an accelerated NeRF-based
free-viewpoint synthesis algorithm for rendering photorealistic human dynamics
more efficiently. We show that our algorithm achieves comparable rendering
quality while performing training and inference 44.5% and 213% faster than
state-of-the-art methods, respectively. Our exploration provides a design basis
for constructing metaverse conferencing systems that can handle complex
application scenarios, including dynamic scene relighting with customized
themes and multi-user conferencing that harmonizes real-world people into an
extended world.

Comments:
- Accepted to CVPR 2023 ECV Workshop

---

## Unsupervised Polychromatic Neural Representation for CT Metal Artifact  Reduction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-27 | Qing Wu, Lixuan Chen, Ce Wang, Hongjiang Wei, S. Kevin Zhou, Jingyi Yu, Yuyao Zhang | eess.IV | [PDF](http://arxiv.org/pdf/2306.15203v2){: .btn .btn-green } |

**Abstract**: Emerging neural reconstruction techniques based on tomography (e.g., NeRF,
NeAT, and NeRP) have started showing unique capabilities in medical imaging. In
this work, we present a novel Polychromatic neural representation (Polyner) to
tackle the challenging problem of CT imaging when metallic implants exist
within the human body. CT metal artifacts arise from the drastic variation of
metal's attenuation coefficients at various energy levels of the X-ray
spectrum, leading to a nonlinear metal effect in CT measurements. Recovering CT
images from metal-affected measurements hence poses a complicated nonlinear
inverse problem where empirical models adopted in previous metal artifact
reduction (MAR) approaches lead to signal loss and strongly aliased
reconstructions. Polyner instead models the MAR problem from a nonlinear
inverse problem perspective. Specifically, we first derive a polychromatic
forward model to accurately simulate the nonlinear CT acquisition process.
Then, we incorporate our forward model into the implicit neural representation
to accomplish reconstruction. Lastly, we adopt a regularizer to preserve the
physical properties of the CT images across different energy levels while
effectively constraining the solution space. Our Polyner is an unsupervised
method and does not require any external training data. Experimenting with
multiple datasets shows that our Polyner achieves comparable or better
performance than supervised methods on in-domain datasets while demonstrating
significant performance improvements on out-of-domain datasets. To the best of
our knowledge, our Polyner is the first unsupervised MAR method that
outperforms its supervised counterparts. The code for this work is available
at: https://github.com/iwuqing/Polyner.

Comments:
- Accepted by NeurIPS 2023

---

## Toward a Spectral Foundation Model: An Attention-Based Approach with  Domain-Inspired Fine-Tuning and Wavelength Parameterization



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-27 | Tomasz Różański, Yuan-Sen Ting, Maja Jabłońska | astro-ph.IM | [PDF](http://arxiv.org/pdf/2306.15703v1){: .btn .btn-green } |

**Abstract**: Astrophysical explorations are underpinned by large-scale stellar
spectroscopy surveys, necessitating a paradigm shift in spectral fitting
techniques. Our study proposes three enhancements to transcend the limitations
of the current spectral emulation models. We implement an attention-based
emulator, adept at unveiling long-range information between wavelength pixels.
We leverage a domain-specific fine-tuning strategy where the model is
pre-trained on spectra with fixed stellar parameters and variable elemental
abundances, followed by fine-tuning on the entire domain. Moreover, by treating
wavelength as an autonomous model parameter, akin to neural radiance fields,
the model can generate spectra on any wavelength grid. In the case with a
training set of O(1000), our approach exceeds current leading methods by a
factor of 5-10 across all metrics.

Comments:
- 7 pages, 3 figures, accepted to ICML 2023 Workshop on Machine
  Learning for Astrophysics

---

## TaiChi Action Capture and Performance Analysis with Multi-view RGB  Cameras



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-26 | Jianwei Li, Siyu Mo, Yanfei Shen | cs.CV | [PDF](http://arxiv.org/pdf/2306.14490v1){: .btn .btn-green } |

**Abstract**: Recent advances in computer vision and deep learning have influenced the
field of sports performance analysis for researchers to track and reconstruct
freely moving humans without any marker attachment. However, there are few
works for vision-based motion capture and intelligent analysis for professional
TaiChi movement. In this paper, we propose a framework for TaiChi performance
capture and analysis with multi-view geometry and artificial intelligence
technology. The main innovative work is as follows: 1) A multi-camera system
suitable for TaiChi motion capture is built and the multi-view TaiChi data is
collected and processed; 2) A combination of traditional visual method and
implicit neural radiance field is proposed to achieve sparse 3D skeleton fusion
and dense 3D surface reconstruction. 3) The normalization modeling of movement
sequences is carried out based on motion transfer, so as to realize TaiChi
performance analysis for different groups. We have carried out evaluation
experiments, and the experimental results have shown the efficiency of our
method.

---

## Blended-NeRF: Zero-Shot Object Generation and Blending in Existing  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-22 | Ori Gordon, Omri Avrahami, Dani Lischinski | cs.CV | [PDF](http://arxiv.org/pdf/2306.12760v2){: .btn .btn-green } |

**Abstract**: Editing a local region or a specific object in a 3D scene represented by a
NeRF or consistently blending a new realistic object into the scene is
challenging, mainly due to the implicit nature of the scene representation. We
present Blended-NeRF, a robust and flexible framework for editing a specific
region of interest in an existing NeRF scene, based on text prompts, along with
a 3D ROI box. Our method leverages a pretrained language-image model to steer
the synthesis towards a user-provided text prompt, along with a 3D MLP model
initialized on an existing NeRF scene to generate the object and blend it into
a specified region in the original scene. We allow local editing by localizing
a 3D ROI box in the input scene, and blend the content synthesized inside the
ROI with the existing scene using a novel volumetric blending technique. To
obtain natural looking and view-consistent results, we leverage existing and
new geometric priors and 3D augmentations for improving the visual fidelity of
the final result. We test our framework both qualitatively and quantitatively
on a variety of real 3D scenes and text prompts, demonstrating realistic
multi-view consistent results with much flexibility and diversity compared to
the baselines. Finally, we show the applicability of our framework for several
3D editing applications, including adding new objects to a scene,
removing/replacing/altering existing objects, and texture conversion.

Comments:
- 16 pages, 14 figures. Project page:
  https://www.vision.huji.ac.il/blended-nerf/

---

## Local 3D Editing via 3D Distillation of CLIP Knowledge

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-21 | Junha Hyung, Sungwon Hwang, Daejin Kim, Hyunji Lee, Jaegul Choo | cs.CV | [PDF](http://arxiv.org/pdf/2306.12570v1){: .btn .btn-green } |

**Abstract**: 3D content manipulation is an important computer vision task with many
real-world applications (e.g., product design, cartoon generation, and 3D
Avatar editing). Recently proposed 3D GANs can generate diverse photorealistic
3D-aware contents using Neural Radiance fields (NeRF). However, manipulation of
NeRF still remains a challenging problem since the visual quality tends to
degrade after manipulation and suboptimal control handles such as 2D semantic
maps are used for manipulations. While text-guided manipulations have shown
potential in 3D editing, such approaches often lack locality. To overcome these
problems, we propose Local Editing NeRF (LENeRF), which only requires text
inputs for fine-grained and localized manipulation. Specifically, we present
three add-on modules of LENeRF, the Latent Residual Mapper, the Attention Field
Network, and the Deformation Network, which are jointly used for local
manipulations of 3D features by estimating a 3D attention field. The 3D
attention field is learned in an unsupervised way, by distilling the zero-shot
mask generation capability of CLIP to the 3D space with multi-view guidance. We
conduct diverse experiments and thorough evaluations both quantitatively and
qualitatively.

Comments:
- conference: CVPR 2023

---

## Benchmarking and Analyzing 3D-aware Image Synthesis with a Modularized  Codebase

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-21 | Qiuyu Wang, Zifan Shi, Kecheng Zheng, Yinghao Xu, Sida Peng, Yujun Shen | cs.CV | [PDF](http://arxiv.org/pdf/2306.12423v1){: .btn .btn-green } |

**Abstract**: Despite the rapid advance of 3D-aware image synthesis, existing studies
usually adopt a mixture of techniques and tricks, leaving it unclear how each
part contributes to the final performance in terms of generality. Following the
most popular and effective paradigm in this field, which incorporates a neural
radiance field (NeRF) into the generator of a generative adversarial network
(GAN), we build a well-structured codebase, dubbed Carver, through modularizing
the generation process. Such a design allows researchers to develop and replace
each module independently, and hence offers an opportunity to fairly compare
various approaches and recognize their contributions from the module
perspective. The reproduction of a range of cutting-edge algorithms
demonstrates the availability of our modularized codebase. We also perform a
variety of in-depth analyses, such as the comparison across different types of
point feature, the necessity of the tailing upsampler in the generator, the
reliance on the camera pose prior, etc., which deepen our understanding of
existing methods and point out some further directions of the research work. We
release code and models at https://github.com/qiuyu96/Carver to facilitate the
development and evaluation of this field.

Comments:
- Code: https://github.com/qiuyu96/Carver

---

## DreamTime: An Improved Optimization Strategy for Text-to-3D Content  Creation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-21 | Yukun Huang, Jianan Wang, Yukai Shi, Xianbiao Qi, Zheng-Jun Zha, Lei Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2306.12422v1){: .btn .btn-green } |

**Abstract**: Text-to-image diffusion models pre-trained on billions of image-text pairs
have recently enabled text-to-3D content creation by optimizing a randomly
initialized Neural Radiance Fields (NeRF) with score distillation. However, the
resultant 3D models exhibit two limitations: (a) quality concerns such as
saturated color and the Janus problem; (b) extremely low diversity comparing to
text-guided image synthesis. In this paper, we show that the conflict between
NeRF optimization process and uniform timestep sampling in score distillation
is the main reason for these limitations. To resolve this conflict, we propose
to prioritize timestep sampling with monotonically non-increasing functions,
which aligns NeRF optimization with the sampling process of diffusion model.
Extensive experiments show that our simple redesign significantly improves
text-to-3D content creation with higher quality and diversity.

---

## NeRF synthesis with shading guidance

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-20 | Chenbin Li, Yu Xin, Gaoyi Liu, Xiang Zeng, Ligang Liu | cs.CV | [PDF](http://arxiv.org/pdf/2306.11556v1){: .btn .btn-green } |

**Abstract**: The emerging Neural Radiance Field (NeRF) shows great potential in
representing 3D scenes, which can render photo-realistic images from novel view
with only sparse views given. However, utilizing NeRF to reconstruct real-world
scenes requires images from different viewpoints, which limits its practical
application. This problem can be even more pronounced for large scenes. In this
paper, we introduce a new task called NeRF synthesis that utilizes the
structural content of a NeRF patch exemplar to construct a new radiance field
of large size. We propose a two-phase method for synthesizing new scenes that
are continuous in geometry and appearance. We also propose a boundary
constraint method to synthesize scenes of arbitrary size without artifacts.
Specifically, we control the lighting effects of synthesized scenes using
shading guidance instead of decoupling the scene. We have demonstrated that our
method can generate high-quality results with consistent geometry and
appearance, even for scenes with complex lighting. We can also synthesize new
scenes on curved surface with arbitrary lighting effects, which enhances the
practicality of our proposed NeRF synthesis approach.

Comments:
- 16 pages, 16 figures, accepted by CAD/Graphics 2023(poster)

---

## MA-NeRF: Motion-Assisted Neural Radiance Fields for Face Synthesis from  Sparse Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-17 | Weichen Zhang, Xiang Zhou, Yukang Cao, Wensen Feng, Chun Yuan | cs.CV | [PDF](http://arxiv.org/pdf/2306.10350v2){: .btn .btn-green } |

**Abstract**: We address the problem of photorealistic 3D face avatar synthesis from sparse
images. Existing Parametric models for face avatar reconstruction struggle to
generate details that originate from inputs. Meanwhile, although current
NeRF-based avatar methods provide promising results for novel view synthesis,
they fail to generalize well for unseen expressions. We improve from NeRF and
propose a novel framework that, by leveraging the parametric 3DMM models, can
reconstruct a high-fidelity drivable face avatar and successfully handle the
unseen expressions. At the core of our implementation are structured
displacement feature and semantic-aware learning module. Our structured
displacement feature will introduce the motion prior as an additional
constraints and help perform better for unseen expressions, by constructing
displacement volume. Besides, the semantic-aware learning incorporates
multi-level prior, e.g., semantic embedding, learnable latent code, to lift the
performance to a higher level. Thorough experiments have been doen both
quantitatively and qualitatively to demonstrate the design of our framework,
and our method achieves much better results than the current state-of-the-arts.

---

## Edit-DiffNeRF: Editing 3D Neural Radiance Fields using 2D Diffusion  Model

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-15 | Lu Yu, Wei Xiang, Kang Han | cs.CV | [PDF](http://arxiv.org/pdf/2306.09551v1){: .btn .btn-green } |

**Abstract**: Recent research has demonstrated that the combination of pretrained diffusion
models with neural radiance fields (NeRFs) has emerged as a promising approach
for text-to-3D generation. Simply coupling NeRF with diffusion models will
result in cross-view inconsistency and degradation of stylized view syntheses.
To address this challenge, we propose the Edit-DiffNeRF framework, which is
composed of a frozen diffusion model, a proposed delta module to edit the
latent semantic space of the diffusion model, and a NeRF. Instead of training
the entire diffusion for each scene, our method focuses on editing the latent
semantic space in frozen pretrained diffusion models by the delta module. This
fundamental change to the standard diffusion framework enables us to make
fine-grained modifications to the rendered views and effectively consolidate
these instructions in a 3D scene via NeRF training. As a result, we are able to
produce an edited 3D scene that faithfully aligns to input text instructions.
Furthermore, to ensure semantic consistency across different viewpoints, we
propose a novel multi-view semantic consistency loss that extracts a latent
semantic embedding from the input view as a prior, and aim to reconstruct it in
different views. Our proposed method has been shown to effectively edit
real-world 3D scenes, resulting in 25% improvement in the alignment of the
performed 3D edits with text instructions compared to prior work.

---

## UrbanIR: Large-Scale Urban Scene Inverse Rendering from a Single Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-15 | Zhi-Hao Lin, Bohan Liu, Yi-Ting Chen, David Forsyth, Jia-Bin Huang, Anand Bhattad, Shenlong Wang | cs.CV | [PDF](http://arxiv.org/pdf/2306.09349v2){: .btn .btn-green } |

**Abstract**: We show how to build a model that allows realistic, free-viewpoint renderings
of a scene under novel lighting conditions from video. Our method -- UrbanIR:
Urban Scene Inverse Rendering -- computes an inverse graphics representation
from the video. UrbanIR jointly infers shape, albedo, visibility, and sun and
sky illumination from a single video of unbounded outdoor scenes with unknown
lighting. UrbanIR uses videos from cameras mounted on cars (in contrast to many
views of the same points in typical NeRF-style estimation). As a result,
standard methods produce poor geometry estimates (for example, roofs), and
there are numerous ''floaters''. Errors in inverse graphics inference can
result in strong rendering artifacts. UrbanIR uses novel losses to control
these and other sources of error. UrbanIR uses a novel loss to make very good
estimates of shadow volumes in the original scene. The resulting
representations facilitate controllable editing, delivering photorealistic
free-viewpoint renderings of relit scenes and inserted objects. Qualitative
evaluation demonstrates strong improvements over the state-of-the-art.

Comments:
- https://urbaninverserendering.github.io/

---

## DreamHuman: Animatable 3D Avatars from Text



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-15 | Nikos Kolotouros, Thiemo Alldieck, Andrei Zanfir, Eduard Gabriel Bazavan, Mihai Fieraru, Cristian Sminchisescu | cs.CV | [PDF](http://arxiv.org/pdf/2306.09329v1){: .btn .btn-green } |

**Abstract**: We present DreamHuman, a method to generate realistic animatable 3D human
avatar models solely from textual descriptions. Recent text-to-3D methods have
made considerable strides in generation, but are still lacking in important
aspects. Control and often spatial resolution remain limited, existing methods
produce fixed rather than animated 3D human models, and anthropometric
consistency for complex structures like people remains a challenge. DreamHuman
connects large text-to-image synthesis models, neural radiance fields, and
statistical human body models in a novel modeling and optimization framework.
This makes it possible to generate dynamic 3D human avatars with high-quality
textures and learned, instance-specific, surface deformations. We demonstrate
that our method is capable to generate a wide variety of animatable, realistic
3D human models from text. Our 3D models have diverse appearance, clothing,
skin tones and body shapes, and significantly outperform both generic
text-to-3D approaches and previous text-based 3D avatar generators in visual
fidelity. For more results and animations please check our website at
https://dream-human.github.io.

Comments:
- Project website at https://dream-human.github.io/

---

## Parametric Implicit Face Representation for Audio-Driven Facial  Reenactment



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-13 | Ricong Huang, Peiwen Lai, Yipeng Qin, Guanbin Li | cs.CV | [PDF](http://arxiv.org/pdf/2306.07579v1){: .btn .btn-green } |

**Abstract**: Audio-driven facial reenactment is a crucial technique that has a range of
applications in film-making, virtual avatars and video conferences. Existing
works either employ explicit intermediate face representations (e.g., 2D facial
landmarks or 3D face models) or implicit ones (e.g., Neural Radiance Fields),
thus suffering from the trade-offs between interpretability and expressive
power, hence between controllability and quality of the results. In this work,
we break these trade-offs with our novel parametric implicit face
representation and propose a novel audio-driven facial reenactment framework
that is both controllable and can generate high-quality talking heads.
Specifically, our parametric implicit representation parameterizes the implicit
representation with interpretable parameters of 3D face models, thereby taking
the best of both explicit and implicit methods. In addition, we propose several
new techniques to improve the three components of our framework, including i)
incorporating contextual information into the audio-to-expression parameters
encoding; ii) using conditional image synthesis to parameterize the implicit
representation and implementing it with an innovative tri-plane structure for
efficient learning; iii) formulating facial reenactment as a conditional image
inpainting problem and proposing a novel data augmentation technique to improve
model generalizability. Extensive experiments demonstrate that our method can
generate more realistic results than previous methods with greater fidelity to
the identities and talking styles of speakers.

Comments:
- CVPR 2023

---

## Binary Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-13 | Seungjoo Shin, Jaesik Park | cs.CV | [PDF](http://arxiv.org/pdf/2306.07581v2){: .btn .btn-green } |

**Abstract**: In this paper, we propose \textit{binary radiance fields} (BiRF), a
storage-efficient radiance field representation employing binary feature
encoding that encodes local features using binary encoding parameters in a
format of either $+1$ or $-1$. This binarization strategy lets us represent the
feature grid with highly compact feature encoding and a dramatic reduction in
storage size. Furthermore, our 2D-3D hybrid feature grid design enhances the
compactness of feature encoding as the 3D grid includes main components while
2D grids capture details. In our experiments, binary radiance field
representation successfully outperforms the reconstruction performance of
state-of-the-art (SOTA) efficient radiance field models with lower storage
allocation. In particular, our model achieves impressive results in static
scene reconstruction, with a PSNR of 32.03 dB for Synthetic-NeRF scenes, 34.48
dB for Synthetic-NSVF scenes, 28.20 dB for Tanks and Temples scenes while only
utilizing 0.5 MB of storage space, respectively. We hope the proposed binary
radiance field representation will make radiance fields more accessible without
a storage bottleneck.

Comments:
- Accepted to NeurIPS 2023. Project page:
  https://seungjooshin.github.io/BiRF

---

## DORSal: Diffusion for Object-centric Representations of Scenes et al

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-13 | Allan Jabri, Sjoerd van Steenkiste, Emiel Hoogeboom, Mehdi S. M. Sajjadi, Thomas Kipf | cs.CV | [PDF](http://arxiv.org/pdf/2306.08068v2){: .btn .btn-green } |

**Abstract**: Recent progress in 3D scene understanding enables scalable learning of
representations across large datasets of diverse scenes. As a consequence,
generalization to unseen scenes and objects, rendering novel views from just a
single or a handful of input images, and controllable scene generation that
supports editing, is now possible. However, training jointly on a large number
of scenes typically compromises rendering quality when compared to single-scene
optimized models such as NeRFs. In this paper, we leverage recent progress in
diffusion models to equip 3D scene representation learning models with the
ability to render high-fidelity novel views, while retaining benefits such as
object-level scene editing to a large degree. In particular, we propose DORSal,
which adapts a video diffusion architecture for 3D scene generation conditioned
on frozen object-centric slot-based representations of scenes. On both complex
synthetic multi-object scenes and on the real-world large-scale Street View
dataset, we show that DORSal enables scalable neural rendering of 3D scenes
with object-level editing and improves upon existing approaches.

Comments:
- Project page: https://www.sjoerdvansteenkiste.com/dorsal

---

## From NeRFLiX to NeRFLiX++: A General NeRF-Agnostic Restorer Paradigm

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-10 | Kun Zhou, Wenbo Li, Nianjuan Jiang, Xiaoguang Han, Jiangbo Lu | cs.CV | [PDF](http://arxiv.org/pdf/2306.06388v3){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) have shown great success in novel view
synthesis. However, recovering high-quality details from real-world scenes is
still challenging for the existing NeRF-based approaches, due to the potential
imperfect calibration information and scene representation inaccuracy. Even
with high-quality training frames, the synthetic novel views produced by NeRF
models still suffer from notable rendering artifacts, such as noise and blur.
To address this, we propose NeRFLiX, a general NeRF-agnostic restorer paradigm
that learns a degradation-driven inter-viewpoint mixer. Specially, we design a
NeRF-style degradation modeling approach and construct large-scale training
data, enabling the possibility of effectively removing NeRF-native rendering
artifacts for deep neural networks. Moreover, beyond the degradation removal,
we propose an inter-viewpoint aggregation framework that fuses highly related
high-quality training images, pushing the performance of cutting-edge NeRF
models to entirely new levels and producing highly photo-realistic synthetic
views. Based on this paradigm, we further present NeRFLiX++ with a stronger
two-stage NeRF degradation simulator and a faster inter-viewpoint mixer,
achieving superior performance with significantly improved computational
efficiency. Notably, NeRFLiX++ is capable of restoring photo-realistic
ultra-high-resolution outputs from noisy low-resolution NeRF-rendered views.
Extensive experiments demonstrate the excellent restoration ability of
NeRFLiX++ on various novel view synthesis benchmarks.

Comments:
- 17 pages, 17 figures. To appear in TPAMI2023. Project Page:
  https://redrock303.github.io/nerflix_plus/. arXiv admin note: text overlap
  with arXiv:2303.06919

---

## NeRFool: Uncovering the Vulnerability of Generalizable Neural Radiance  Fields against Adversarial Perturbations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-10 | Yonggan Fu, Ye Yuan, Souvik Kundu, Shang Wu, Shunyao Zhang, Yingyan Lin | cs.CV | [PDF](http://arxiv.org/pdf/2306.06359v1){: .btn .btn-green } |

**Abstract**: Generalizable Neural Radiance Fields (GNeRF) are one of the most promising
real-world solutions for novel view synthesis, thanks to their cross-scene
generalization capability and thus the possibility of instant rendering on new
scenes. While adversarial robustness is essential for real-world applications,
little study has been devoted to understanding its implication on GNeRF. We
hypothesize that because GNeRF is implemented by conditioning on the source
views from new scenes, which are often acquired from the Internet or
third-party providers, there are potential new security concerns regarding its
real-world applications. Meanwhile, existing understanding and solutions for
neural networks' adversarial robustness may not be applicable to GNeRF, due to
its 3D nature and uniquely diverse operations. To this end, we present NeRFool,
which to the best of our knowledge is the first work that sets out to
understand the adversarial robustness of GNeRF. Specifically, NeRFool unveils
the vulnerability patterns and important insights regarding GNeRF's adversarial
robustness. Built upon the above insights gained from NeRFool, we further
develop NeRFool+, which integrates two techniques capable of effectively
attacking GNeRF across a wide range of target views, and provide guidelines for
defending against our proposed attacks. We believe that our NeRFool/NeRFool+
lays the initial foundation for future innovations in developing robust
real-world GNeRF solutions. Our codes are available at:
https://github.com/GATECH-EIC/NeRFool.

Comments:
- Accepted by ICML 2023

---

## NERFBK: A High-Quality Benchmark for NERF-Based 3D Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-09 | Ali Karami, Simone Rigon, Gabriele Mazzacca, Ziyang Yan, Fabio Remondino | cs.CV | [PDF](http://arxiv.org/pdf/2306.06300v2){: .btn .btn-green } |

**Abstract**: This paper introduces a new real and synthetic dataset called NeRFBK
specifically designed for testing and comparing NeRF-based 3D reconstruction
algorithms. High-quality 3D reconstruction has significant potential in various
fields, and advancements in image-based algorithms make it essential to
evaluate new advanced techniques. However, gathering diverse data with precise
ground truth is challenging and may not encompass all relevant applications.
The NeRFBK dataset addresses this issue by providing multi-scale, indoor and
outdoor datasets with high-resolution images and videos and camera parameters
for testing and comparing NeRF-based algorithms. This paper presents the design
and creation of the NeRFBK benchmark, various examples and application
scenarios, and highlights its potential for advancing the field of 3D
reconstruction.

Comments:
- paper result has problem

---

## HyP-NeRF: Learning Improved NeRF Priors using a HyperNetwork

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-09 | Bipasha Sen, Gaurav Singh, Aditya Agarwal, Rohith Agaram, K Madhava Krishna, Srinath Sridhar | cs.CV | [PDF](http://arxiv.org/pdf/2306.06093v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have become an increasingly popular
representation to capture high-quality appearance and shape of scenes and
objects. However, learning generalizable NeRF priors over categories of scenes
or objects has been challenging due to the high dimensionality of network
weight space. To address the limitations of existing work on generalization,
multi-view consistency and to improve quality, we propose HyP-NeRF, a latent
conditioning method for learning generalizable category-level NeRF priors using
hypernetworks. Rather than using hypernetworks to estimate only the weights of
a NeRF, we estimate both the weights and the multi-resolution hash encodings
resulting in significant quality gains. To improve quality even further, we
incorporate a denoise and finetune strategy that denoises images rendered from
NeRFs estimated by the hypernetwork and finetunes it while retaining multiview
consistency. These improvements enable us to use HyP-NeRF as a generalizable
prior for multiple downstream tasks including NeRF reconstruction from
single-view or cluttered scenes and text-to-NeRF. We provide qualitative
comparisons and evaluate HyP-NeRF on three tasks: generalization, compression,
and retrieval, demonstrating our state-of-the-art results.

Comments:
- Project Page: https://hyp-nerf.github.io

---

## GANeRF: Leveraging Discriminators to Optimize Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-09 | Barbara Roessle, Norman Müller, Lorenzo Porzi, Samuel Rota Bulò, Peter Kontschieder, Matthias Nießner | cs.CV | [PDF](http://arxiv.org/pdf/2306.06044v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have shown impressive novel view synthesis
results; nonetheless, even thorough recordings yield imperfections in
reconstructions, for instance due to poorly observed areas or minor lighting
changes. Our goal is to mitigate these imperfections from various sources with
a joint solution: we take advantage of the ability of generative adversarial
networks (GANs) to produce realistic images and use them to enhance realism in
3D scene reconstruction with NeRFs. To this end, we learn the patch
distribution of a scene using an adversarial discriminator, which provides
feedback to the radiance field reconstruction, thus improving realism in a
3D-consistent fashion. Thereby, rendering artifacts are repaired directly in
the underlying 3D representation by imposing multi-view path rendering
constraints. In addition, we condition a generator with multi-resolution NeRF
renderings which is adversarially trained to further improve rendering quality.
We demonstrate that our approach significantly improves rendering quality,
e.g., nearly halving LPIPS scores compared to Nerfacto while at the same time
improving PSNR by 1.4dB on the advanced indoor scenes of Tanks and Temples.

Comments:
- SIGGRAPH Asia 2023, project page:
  https://barbararoessle.github.io/ganerf , video: https://youtu.be/352ccXWxQVE

---

## RePaint-NeRF: NeRF Editting via Semantic Masks and Diffusion Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-09 | Xingchen Zhou, Ying He, F. Richard Yu, Jianqiang Li, You Li | cs.CV | [PDF](http://arxiv.org/pdf/2306.05668v2){: .btn .btn-green } |

**Abstract**: The emergence of Neural Radiance Fields (NeRF) has promoted the development
of synthesized high-fidelity views of the intricate real world. However, it is
still a very demanding task to repaint the content in NeRF. In this paper, we
propose a novel framework that can take RGB images as input and alter the 3D
content in neural scenes. Our work leverages existing diffusion models to guide
changes in the designated 3D content. Specifically, we semantically select the
target object and a pre-trained diffusion model will guide the NeRF model to
generate new 3D objects, which can improve the editability, diversity, and
application range of NeRF. Experiment results show that our algorithm is
effective for editing 3D objects in NeRF under different text prompts,
including editing appearance, shape, and more. We validate our method on both
real-world datasets and synthetic-world datasets for these editing tasks.
Please visit https://starstesla.github.io/repaintnerf for a better view of our
results.

Comments:
- IJCAI 2023 Accepted (Main Track)

---

## Variable Radiance Field for Real-Life Category-Specifc Reconstruction  from Single Image



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-08 | Kun Wang, Zhiqiang Yan, Zhenyu Zhang, Xiang Li, Jun Li, Jian Yang | cs.CV | [PDF](http://arxiv.org/pdf/2306.05145v1){: .btn .btn-green } |

**Abstract**: Reconstructing category-specific objects from a single image is a challenging
task that requires inferring the geometry and appearance of an object from a
limited viewpoint. Existing methods typically rely on local feature retrieval
based on re-projection with known camera intrinsic, which are slow and prone to
distortion at viewpoints distant from the input image. In this paper, we
present Variable Radiance Field (VRF), a novel framework that can efficiently
reconstruct category-specific objects from a single image without known camera
parameters. Our key contributions are: (1) We parameterize the geometry and
appearance of the object using a multi-scale global feature extractor, which
avoids frequent point-wise feature retrieval and camera dependency. We also
propose a contrastive learning-based pretraining strategy to improve the
feature extractor. (2) We reduce the geometric complexity of the object by
learning a category template, and use hypernetworks to generate a small neural
radiance field for fast and instance-specific rendering. (3) We align each
training instance to the template space using a learned similarity
transformation, which enables semantic-consistent learning across different
objects. We evaluate our method on the CO3D dataset and show that it
outperforms existing methods in terms of quality and speed. We also demonstrate
its applicability to shape interpolation and object placement tasks.

---

## Enhance-NeRF: Multiple Performance Evaluation for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-08 | Qianqiu Tan, Tao Liu, Yinling Xie, Shuwan Yu, Baohua Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2306.05303v1){: .btn .btn-green } |

**Abstract**: The quality of three-dimensional reconstruction is a key factor affecting the
effectiveness of its application in areas such as virtual reality (VR) and
augmented reality (AR) technologies. Neural Radiance Fields (NeRF) can generate
realistic images from any viewpoint. It simultaneously reconstructs the shape,
lighting, and materials of objects, and without surface defects, which breaks
down the barrier between virtuality and reality. The potential spatial
correspondences displayed by NeRF between reconstructed scenes and real-world
scenes offer a wide range of practical applications possibilities. Despite
significant progress in 3D reconstruction since NeRF were introduced, there
remains considerable room for exploration and experimentation. NeRF-based
models are susceptible to interference issues caused by colored "fog" noise.
Additionally, they frequently encounter instabilities and failures while
attempting to reconstruct unbounded scenes. Moreover, the model takes a
significant amount of time to converge, making it even more challenging to use
in such scenarios. Our approach, coined Enhance-NeRF, which adopts joint color
to balance low and high reflectivity objects display, utilizes a decoding
architecture with prior knowledge to improve recognition, and employs
multi-layer performance evaluation mechanisms to enhance learning capacity. It
achieves reconstruction of outdoor scenes within one hour under single-card
condition. Based on experimental results, Enhance-NeRF partially enhances
fitness capability and provides some support to outdoor scene reconstruction.
The Enhance-NeRF method can be used as a plug-and-play component, making it
easy to integrate with other NeRF-based models. The code is available at:
https://github.com/TANQIanQ/Enhance-NeRF

---

## LU-NeRF: Scene and Pose Estimation by Synchronizing Local Unposed NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-08 | Zezhou Cheng, Carlos Esteves, Varun Jampani, Abhishek Kar, Subhransu Maji, Ameesh Makadia | cs.CV | [PDF](http://arxiv.org/pdf/2306.05410v1){: .btn .btn-green } |

**Abstract**: A critical obstacle preventing NeRF models from being deployed broadly in the
wild is their reliance on accurate camera poses. Consequently, there is growing
interest in extending NeRF models to jointly optimize camera poses and scene
representation, which offers an alternative to off-the-shelf SfM pipelines
which have well-understood failure modes. Existing approaches for unposed NeRF
operate under limited assumptions, such as a prior pose distribution or coarse
pose initialization, making them less effective in a general setting. In this
work, we propose a novel approach, LU-NeRF, that jointly estimates camera poses
and neural radiance fields with relaxed assumptions on pose configuration. Our
approach operates in a local-to-global manner, where we first optimize over
local subsets of the data, dubbed mini-scenes. LU-NeRF estimates local pose and
geometry for this challenging few-shot task. The mini-scene poses are brought
into a global reference frame through a robust pose synchronization step, where
a final global optimization of pose and scene can be performed. We show our
LU-NeRF pipeline outperforms prior attempts at unposed NeRF without making
restrictive assumptions on the pose prior. This allows us to operate in the
general SE(3) pose setting, unlike the baselines. Our results also indicate our
model can be complementary to feature-based SfM pipelines as it compares
favorably to COLMAP on low-texture and low-resolution images.

Comments:
- Project website: https://people.cs.umass.edu/~zezhoucheng/lu-nerf/

---

## BAA-NGP: Bundle-Adjusting Accelerated Neural Graphics Primitives



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-07 | Sainan Liu, Shan Lin, Jingpei Lu, Shreya Saha, Alexey Supikov, Michael Yip | cs.CV | [PDF](http://arxiv.org/pdf/2306.04166v3){: .btn .btn-green } |

**Abstract**: Implicit neural representation has emerged as a powerful method for
reconstructing 3D scenes from 2D images. Given a set of camera poses and
associated images, the models can be trained to synthesize novel, unseen views.
In order to expand the use cases for implicit neural representations, we need
to incorporate camera pose estimation capabilities as part of the
representation learning, as this is necessary for reconstructing scenes from
real-world video sequences where cameras are generally not being tracked.
Existing approaches like COLMAP and, most recently, bundle-adjusting neural
radiance field methods often suffer from lengthy processing times. These delays
ranging from hours to days, arise from laborious feature matching, hardware
limitations, dense point sampling, and long training times required by a
multi-layer perceptron structure with a large number of parameters. To address
these challenges, we propose a framework called bundle-adjusting accelerated
neural graphics primitives (BAA-NGP). Our approach leverages accelerated
sampling and hash encoding to expedite both pose refinement/estimation and 3D
scene reconstruction. Experimental results demonstrate that our method achieves
a more than 10 to 20 $\times$ speed improvement in novel view synthesis
compared to other bundle-adjusting neural radiance field methods without
sacrificing the quality of pose estimation. The github repository can be found
here https://github.com/IntelLabs/baa-ngp.

---

## ATT3D: Amortized Text-to-3D Object Synthesis



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-06 | Jonathan Lorraine, Kevin Xie, Xiaohui Zeng, Chen-Hsuan Lin, Towaki Takikawa, Nicholas Sharp, Tsung-Yi Lin, Ming-Yu Liu, Sanja Fidler, James Lucas | cs.LG | [PDF](http://arxiv.org/pdf/2306.07349v1){: .btn .btn-green } |

**Abstract**: Text-to-3D modelling has seen exciting progress by combining generative
text-to-image models with image-to-3D methods like Neural Radiance Fields.
DreamFusion recently achieved high-quality results but requires a lengthy,
per-prompt optimization to create 3D objects. To address this, we amortize
optimization over text prompts by training on many prompts simultaneously with
a unified model, instead of separately. With this, we share computation across
a prompt set, training in less time than per-prompt optimization. Our framework
- Amortized text-to-3D (ATT3D) - enables knowledge-sharing between prompts to
generalize to unseen setups and smooth interpolations between text for novel
assets and simple animations.

Comments:
- 22 pages, 20 figures

---

## Towards Visual Foundational Models of Physical Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-06 | Chethan Parameshwara, Alessandro Achille, Matthew Trager, Xiaolong Li, Jiawei Mo, Matthew Trager, Ashwin Swaminathan, CJ Taylor, Dheera Venkatraman, Xiaohan Fei, Stefano Soatto | cs.CV | [PDF](http://arxiv.org/pdf/2306.03727v1){: .btn .btn-green } |

**Abstract**: We describe a first step towards learning general-purpose visual
representations of physical scenes using only image prediction as a training
criterion. To do so, we first define "physical scene" and show that, even
though different agents may maintain different representations of the same
scene, the underlying physical scene that can be inferred is unique. Then, we
show that NeRFs cannot represent the physical scene, as they lack extrapolation
mechanisms. Those, however, could be provided by Diffusion Models, at least in
theory. To test this hypothesis empirically, NeRFs can be combined with
Diffusion Models, a process we refer to as NeRF Diffusion, used as unsupervised
representations of the physical scene. Our analysis is limited to visual data,
without external grounding mechanisms that can be provided by independent
sensory modalities.

Comments:
- TLDR: Physical scenes are equivalence classes of sufficient
  statistics, and can be inferred uniquely by any agent measuring the same
  finite data; We formalize and implement an approach to representation
  learning that overturns "naive realism" in favor of an analytical approach of
  Russell and Koenderink. NeRFs cannot capture the physical scenes, but
  combined with Diffusion Models they can

---

## Human 3D Avatar Modeling with Implicit Neural Representation: A Brief  Survey

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-06 | Mingyang Sun, Dingkang Yang, Dongliang Kou, Yang Jiang, Weihua Shan, Zhe Yan, Lihua Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2306.03576v1){: .btn .btn-green } |

**Abstract**: A human 3D avatar is one of the important elements in the metaverse, and the
modeling effect directly affects people's visual experience. However, the human
body has a complex topology and diverse details, so it is often expensive,
time-consuming, and laborious to build a satisfactory model. Recent studies
have proposed a novel method, implicit neural representation, which is a
continuous representation method and can describe objects with arbitrary
topology at arbitrary resolution. Researchers have applied implicit neural
representation to human 3D avatar modeling and obtained more excellent results
than traditional methods. This paper comprehensively reviews the application of
implicit neural representation in human body modeling. First, we introduce
three implicit representations of occupancy field, SDF, and NeRF, and make a
classification of the literature investigated in this paper. Then the
application of implicit modeling methods in the body, hand, and head are
compared and analyzed respectively. Finally, we point out the shortcomings of
current work and provide available suggestions for researchers.

Comments:
- A Brief Survey

---

## H2-Mapping: Real-time Dense Mapping Using Hierarchical Hybrid  Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-05 | Chenxing Jiang, Hanwen Zhang, Peize Liu, Zehuan Yu, Hui Cheng, Boyu Zhou, Shaojie Shen | cs.RO | [PDF](http://arxiv.org/pdf/2306.03207v2){: .btn .btn-green } |

**Abstract**: Constructing a high-quality dense map in real-time is essential for robotics,
AR/VR, and digital twins applications. As Neural Radiance Field (NeRF) greatly
improves the mapping performance, in this paper, we propose a NeRF-based
mapping method that enables higher-quality reconstruction and real-time
capability even on edge computers. Specifically, we propose a novel
hierarchical hybrid representation that leverages implicit multiresolution hash
encoding aided by explicit octree SDF priors, describing the scene at different
levels of detail. This representation allows for fast scene geometry
initialization and makes scene geometry easier to learn. Besides, we present a
coverage-maximizing keyframe selection strategy to address the forgetting issue
and enhance mapping quality, particularly in marginal areas. To the best of our
knowledge, our method is the first to achieve high-quality NeRF-based mapping
on edge computers of handheld devices and quadrotors in real-time. Experiments
demonstrate that our method outperforms existing NeRF-based mapping methods in
geometry accuracy, texture realism, and time consumption. The code will be
released at: https://github.com/SYSU-STAR/H2-Mapping

Comments:
- Accepted by IEEE Robotics and Automation Letters

---

## BeyondPixels: A Comprehensive Review of the Evolution of Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-05 | AKM Shahariar Azad Rabby, Chengcui Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2306.03000v2){: .btn .btn-green } |

**Abstract**: Neural rendering combines ideas from classical computer graphics and machine
learning to synthesize images from real-world observations. NeRF, short for
Neural Radiance Fields, is a recent innovation that uses AI algorithms to
create 3D objects from 2D images. By leveraging an interpolation approach, NeRF
can produce new 3D reconstructed views of complicated scenes. Rather than
directly restoring the whole 3D scene geometry, NeRF generates a volumetric
representation called a ``radiance field,'' which is capable of creating color
and density for every point within the relevant 3D space. The broad appeal and
notoriety of NeRF make it imperative to examine the existing research on the
topic comprehensively. While previous surveys on 3D rendering have primarily
focused on traditional computer vision-based or deep learning-based approaches,
only a handful of them discuss the potential of NeRF. However, such surveys
have predominantly focused on NeRF's early contributions and have not explored
its full potential. NeRF is a relatively new technique continuously being
investigated for its capabilities and limitations. This survey reviews recent
advances in NeRF and categorizes them according to their architectural designs,
especially in the field of novel view synthesis.

Comments:
- 22 page, 1 figure, 5 table

---

## Instruct-Video2Avatar: Video-to-Avatar Generation with Instructions



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-05 | Shaoxu Li | cs.CV | [PDF](http://arxiv.org/pdf/2306.02903v1){: .btn .btn-green } |

**Abstract**: We propose a method for synthesizing edited photo-realistic digital avatars
with text instructions. Given a short monocular RGB video and text
instructions, our method uses an image-conditioned diffusion model to edit one
head image and uses the video stylization method to accomplish the editing of
other head images. Through iterative training and update (three times or more),
our method synthesizes edited photo-realistic animatable 3D neural head avatars
with a deformable neural radiance field head synthesis method. In quantitative
and qualitative studies on various subjects, our method outperforms
state-of-the-art methods.

Comments:
- https://github.com/lsx0101/Instruct-Video2Avatar

---

## ZIGNeRF: Zero-shot 3D Scene Representation with Invertible Generative  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-05 | Kanghyeok Ko, Minhyeok Lee | cs.CV | [PDF](http://arxiv.org/pdf/2306.02741v1){: .btn .btn-green } |

**Abstract**: Generative Neural Radiance Fields (NeRFs) have demonstrated remarkable
proficiency in synthesizing multi-view images by learning the distribution of a
set of unposed images. Despite the aptitude of existing generative NeRFs in
generating 3D-consistent high-quality random samples within data distribution,
the creation of a 3D representation of a singular input image remains a
formidable challenge. In this manuscript, we introduce ZIGNeRF, an innovative
model that executes zero-shot Generative Adversarial Network (GAN) inversion
for the generation of multi-view images from a single out-of-domain image. The
model is underpinned by a novel inverter that maps out-of-domain images into
the latent code of the generator manifold. Notably, ZIGNeRF is capable of
disentangling the object from the background and executing 3D operations such
as 360-degree rotation or depth and horizontal translation. The efficacy of our
model is validated using multiple real-image datasets: Cats, AFHQ, CelebA,
CelebA-HQ, and CompCars.

---

## PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline  Panoramas



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-02 | Zheng Chen, Yan-Pei Cao, Yuan-Chen Guo, Chen Wang, Ying Shan, Song-Hai Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2306.01531v2){: .btn .btn-green } |

**Abstract**: Achieving an immersive experience enabling users to explore virtual
environments with six degrees of freedom (6DoF) is essential for various
applications such as virtual reality (VR). Wide-baseline panoramas are commonly
used in these applications to reduce network bandwidth and storage
requirements. However, synthesizing novel views from these panoramas remains a
key challenge. Although existing neural radiance field methods can produce
photorealistic views under narrow-baseline and dense image captures, they tend
to overfit the training views when dealing with \emph{wide-baseline} panoramas
due to the difficulty in learning accurate geometry from sparse $360^{\circ}$
views. To address this problem, we propose PanoGRF, Generalizable Spherical
Radiance Fields for Wide-baseline Panoramas, which construct spherical radiance
fields incorporating $360^{\circ}$ scene priors. Unlike generalizable radiance
fields trained on perspective images, PanoGRF avoids the information loss from
panorama-to-perspective conversion and directly aggregates geometry and
appearance features of 3D sample points from each panoramic view based on
spherical projection. Moreover, as some regions of the panorama are only
visible from one view while invisible from others under wide baseline settings,
PanoGRF incorporates $360^{\circ}$ monocular depth priors into spherical depth
estimation to improve the geometry features. Experimental results on multiple
panoramic datasets demonstrate that PanoGRF significantly outperforms
state-of-the-art generalizable view synthesis methods for wide-baseline
panoramas (e.g., OmniSyn) and perspective images (e.g., IBRNet, NeuRay).

Comments:
- accepted to NeurIPS2023; Project Page:
  https://thucz.github.io/PanoGRF/

---

## FaceDNeRF: Semantics-Driven Face Reconstruction, Prompt Editing and  Relighting with Diffusion Models

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-01 | Hao Zhang, Yanbo Xu, Tianyuan Dai, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2306.00783v2){: .btn .btn-green } |

**Abstract**: The ability to create high-quality 3D faces from a single image has become
increasingly important with wide applications in video conferencing, AR/VR, and
advanced video editing in movie industries. In this paper, we propose Face
Diffusion NeRF (FaceDNeRF), a new generative method to reconstruct high-quality
Face NeRFs from single images, complete with semantic editing and relighting
capabilities. FaceDNeRF utilizes high-resolution 3D GAN inversion and expertly
trained 2D latent-diffusion model, allowing users to manipulate and construct
Face NeRFs in zero-shot learning without the need for explicit 3D data. With
carefully designed illumination and identity preserving loss, as well as
multi-modal pre-training, FaceDNeRF offers users unparalleled control over the
editing process enabling them to create and edit face NeRFs using just
single-view images, text prompts, and explicit target lighting. The advanced
features of FaceDNeRF have been designed to produce more impressive results
than existing 2D editing approaches that rely on 2D segmentation maps for
editable attributes. Experiments show that our FaceDNeRF achieves exceptionally
realistic results and unprecedented flexibility in editing compared with
state-of-the-art 3D face reconstruction and editing methods. Our code will be
available at https://github.com/BillyXYB/FaceDNeRF.

---

## Analyzing the Internals of Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-01 | Lukas Radl, Andreas Kurz, Markus Steinberger | cs.CV | [PDF](http://arxiv.org/pdf/2306.00696v1){: .btn .btn-green } |

**Abstract**: Modern Neural Radiance Fields (NeRFs) learn a mapping from position to
volumetric density via proposal network samplers. In contrast to the
coarse-to-fine sampling approach with two NeRFs, this offers significant
potential for speedups using lower network capacity as the task of mapping
spatial coordinates to volumetric density involves no view-dependent effects
and is thus much easier to learn. Given that most of the network capacity is
utilized to estimate radiance, NeRFs could store valuable density information
in their parameters or their deep features. To this end, we take one step back
and analyze large, trained ReLU-MLPs used in coarse-to-fine sampling. We find
that trained NeRFs, Mip-NeRFs and proposal network samplers map samples with
high density to local minima along a ray in activation feature space. We show
how these large MLPs can be accelerated by transforming the intermediate
activations to a weight estimate, without any modifications to the parameters
post-optimization. With our approach, we can reduce the computational
requirements of trained NeRFs by up to 50% with only a slight hit in rendering
quality and no changes to the training protocol or architecture. We evaluate
our approach on a variety of architectures and datasets, showing that our
proposition holds in various settings.

Comments:
- project page: nerfinternals.github.io

---

## AvatarStudio: Text-driven Editing of 3D Dynamic Human Head Avatars

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-06-01 | Mohit Mendiratta, Xingang Pan, Mohamed Elgharib, Kartik Teotia, Mallikarjun B R, Ayush Tewari, Vladislav Golyanik, Adam Kortylewski, Christian Theobalt | cs.CV | [PDF](http://arxiv.org/pdf/2306.00547v2){: .btn .btn-green } |

**Abstract**: Capturing and editing full head performances enables the creation of virtual
characters with various applications such as extended reality and media
production. The past few years witnessed a steep rise in the photorealism of
human head avatars. Such avatars can be controlled through different input data
modalities, including RGB, audio, depth, IMUs and others. While these data
modalities provide effective means of control, they mostly focus on editing the
head movements such as the facial expressions, head pose and/or camera
viewpoint. In this paper, we propose AvatarStudio, a text-based method for
editing the appearance of a dynamic full head avatar. Our approach builds on
existing work to capture dynamic performances of human heads using neural
radiance field (NeRF) and edits this representation with a text-to-image
diffusion model. Specifically, we introduce an optimization strategy for
incorporating multiple keyframes representing different camera viewpoints and
time stamps of a video performance into a single diffusion model. Using this
personalized diffusion model, we edit the dynamic NeRF by introducing
view-and-time-aware Score Distillation Sampling (VT-SDS) following a
model-based guidance approach. Our method edits the full head in a canonical
space, and then propagates these edits to remaining time steps via a pretrained
deformation network. We evaluate our method visually and numerically via a user
study, and results show that our method outperforms existing approaches. Our
experiments validate the design choices of our method and highlight that our
edits are genuine, personalized, as well as 3D- and time-consistent.

Comments:
- 17 pages, 17 figures. Project page:
  https://vcai.mpi-inf.mpg.de/projects/AvatarStudio/