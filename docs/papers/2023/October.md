---
layout: default
title: October
parent: 2023
nav_order: 10
---
<!---metadata--->

## FPO++: Efficient Encoding and Rendering of Dynamic Neural Radiance  Fields by Analyzing and Enhancing Fourier PlenOctrees

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-31 | Saskia Rabich, Patrick Stotko, Reinhard Klein | cs.CV | [PDF](http://arxiv.org/pdf/2310.20710v1){: .btn .btn-green } |

**Abstract**: Fourier PlenOctrees have shown to be an efficient representation for
real-time rendering of dynamic Neural Radiance Fields (NeRF). Despite its many
advantages, this method suffers from artifacts introduced by the involved
compression when combining it with recent state-of-the-art techniques for
training the static per-frame NeRF models. In this paper, we perform an
in-depth analysis of these artifacts and leverage the resulting insights to
propose an improved representation. In particular, we present a novel density
encoding that adapts the Fourier-based compression to the characteristics of
the transfer function used by the underlying volume rendering procedure and
leads to a substantial reduction of artifacts in the dynamic model.
Furthermore, we show an augmentation of the training data that relaxes the
periodicity assumption of the compression. We demonstrate the effectiveness of
our enhanced Fourier PlenOctrees in the scope of quantitative and qualitative
evaluations on synthetic and real-world scenes.

---

## NeRF Revisited: Fixing Quadrature Instability in Volume Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-31 | Mikaela Angelina Uy, Kiyohiro Nakayama, Guandao Yang, Rahul Krishna Thomas, Leonidas Guibas, Ke Li | cs.CV | [PDF](http://arxiv.org/pdf/2310.20685v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRF) rely on volume rendering to synthesize novel
views. Volume rendering requires evaluating an integral along each ray, which
is numerically approximated with a finite sum that corresponds to the exact
integral along the ray under piecewise constant volume density. As a
consequence, the rendered result is unstable w.r.t. the choice of samples along
the ray, a phenomenon that we dub quadrature instability. We propose a
mathematically principled solution by reformulating the sample-based rendering
equation so that it corresponds to the exact integral under piecewise linear
volume density. This simultaneously resolves multiple issues: conflicts between
samples along different rays, imprecise hierarchical sampling, and
non-differentiability of quantiles of ray termination distances w.r.t. model
parameters. We demonstrate several benefits over the classical sample-based
rendering equation, such as sharper textures, better geometric reconstruction,
and stronger depth supervision. Our proposed formulation can be also be used as
a drop-in replacement to the volume rendering equation of existing NeRF-based
methods. Our project page can be found at pl-nerf.github.io.

Comments:
- Neurips 2023

---

## Dynamic Gaussian Splatting from Markerless Motion Capture can  Reconstruct Infants Movements

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-30 | R. James Cotton, Colleen Peyton | cs.CV | [PDF](http://arxiv.org/pdf/2310.19441v1){: .btn .btn-green } |

**Abstract**: Easy access to precise 3D tracking of movement could benefit many aspects of
rehabilitation. A challenge to achieving this goal is that while there are many
datasets and pretrained algorithms for able-bodied adults, algorithms trained
on these datasets often fail to generalize to clinical populations including
people with disabilities, infants, and neonates. Reliable movement analysis of
infants and neonates is important as spontaneous movement behavior is an
important indicator of neurological function and neurodevelopmental disability,
which can help guide early interventions. We explored the application of
dynamic Gaussian splatting to sparse markerless motion capture (MMC) data. Our
approach leverages semantic segmentation masks to focus on the infant,
significantly improving the initialization of the scene. Our results
demonstrate the potential of this method in rendering novel views of scenes and
tracking infant movements. This work paves the way for advanced movement
analysis tools that can be applied to diverse clinical populations, with a
particular emphasis on early detection in infants.

---

## Generative Neural Fields by Mixtures of Neural Implicit Functions

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-30 | Tackgeun You, Mijeong Kim, Jungtaek Kim, Bohyung Han | cs.LG | [PDF](http://arxiv.org/pdf/2310.19464v1){: .btn .btn-green } |

**Abstract**: We propose a novel approach to learning the generative neural fields
represented by linear combinations of implicit basis networks. Our algorithm
learns basis networks in the form of implicit neural representations and their
coefficients in a latent space by either conducting meta-learning or adopting
auto-decoding paradigms. The proposed method easily enlarges the capacity of
generative neural fields by increasing the number of basis networks while
maintaining the size of a network for inference to be small through their
weighted model averaging. Consequently, sampling instances using the model is
efficient in terms of latency and memory footprint. Moreover, we customize
denoising diffusion probabilistic model for a target task to sample latent
mixture coefficients, which allows our final model to generate unseen data
effectively. Experiments show that our approach achieves competitive generation
performance on diverse benchmarks for images, voxel data, and NeRF scenes
without sophisticated designs for specific modalities and domains.

---

## SeamlessNeRF: Stitching Part NeRFs with Gradient Propagation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-30 | Bingchen Gong, Yuehao Wang, Xiaoguang Han, Qi Dou | cs.CV | [PDF](http://arxiv.org/pdf/2311.16127v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have emerged as promising digital mediums of
3D objects and scenes, sparking a surge in research to extend the editing
capabilities in this domain. The task of seamless editing and merging of
multiple NeRFs, resembling the ``Poisson blending'' in 2D image editing,
remains a critical operation that is under-explored by existing work. To fill
this gap, we propose SeamlessNeRF, a novel approach for seamless appearance
blending of multiple NeRFs. In specific, we aim to optimize the appearance of a
target radiance field in order to harmonize its merge with a source field. We
propose a well-tailored optimization procedure for blending, which is
constrained by 1) pinning the radiance color in the intersecting boundary area
between the source and target fields and 2) maintaining the original gradient
of the target. Extensive experiments validate that our approach can effectively
propagate the source appearance from the boundary area to the entire target
field through the gradients. To the best of our knowledge, SeamlessNeRF is the
first work that introduces gradient-guided appearance editing to radiance
fields, offering solutions for seamless stitching of 3D objects represented in
NeRFs.

Comments:
- To appear in SIGGRAPH Asia 2023. Project website is accessible at
  https://sites.google.com/view/seamlessnerf

---

## DynPoint: Dynamic Neural Point For View Synthesis



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-29 | Kaichen Zhou, Jia-Xing Zhong, Sangyun Shin, Kai Lu, Yiyuan Yang, Andrew Markham, Niki Trigoni | cs.CV | [PDF](http://arxiv.org/pdf/2310.18999v2){: .btn .btn-green } |

**Abstract**: The introduction of neural radiance fields has greatly improved the
effectiveness of view synthesis for monocular videos. However, existing
algorithms face difficulties when dealing with uncontrolled or lengthy
scenarios, and require extensive training time specific to each new scenario.
To tackle these limitations, we propose DynPoint, an algorithm designed to
facilitate the rapid synthesis of novel views for unconstrained monocular
videos. Rather than encoding the entirety of the scenario information into a
latent representation, DynPoint concentrates on predicting the explicit 3D
correspondence between neighboring frames to realize information aggregation.
Specifically, this correspondence prediction is achieved through the estimation
of consistent depth and scene flow information across frames. Subsequently, the
acquired correspondence is utilized to aggregate information from multiple
reference frames to a target frame, by constructing hierarchical neural point
clouds. The resulting framework enables swift and accurate view synthesis for
desired views of target frames. The experimental results obtained demonstrate
the considerable acceleration of training time achieved - typically an order of
magnitude - by our proposed method while yielding comparable outcomes compared
to prior approaches. Furthermore, our method exhibits strong robustness in
handling long-duration videos without learning a canonical representation of
video content.

---

## TiV-NeRF: Tracking and Mapping via Time-Varying Representation with  Dynamic Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-29 | Chengyao Duan, Zhiliu Yang | cs.CV | [PDF](http://arxiv.org/pdf/2310.18917v2){: .btn .btn-green } |

**Abstract**: Previous attempts to integrate Neural Radiance Fields (NeRF) into
Simultaneous Localization and Mapping (SLAM) framework either rely on the
assumption of static scenes or treat dynamic objects as outliers. However, most
of real-world scenarios is dynamic. In this paper, we propose a time-varying
representation to track and reconstruct the dynamic scenes. Our system
simultaneously maintains two processes, tracking process and mapping process.
For tracking process, the entire input images are uniformly sampled and
training of the RGB images are self-supervised. For mapping process, we
leverage know masks to differentiate dynamic objects and static backgrounds,
and we apply distinct sampling strategies for two types of areas. The
parameters optimization for both processes are made up by two stages, the first
stage associates time with 3D positions to convert the deformation field to the
canonical field. And the second associates time with 3D positions in canonical
field to obtain colors and Signed Distance Function (SDF). Besides, We propose
a novel keyframe selection strategy based on the overlapping rate. We evaluate
our approach on two publicly available synthetic datasets and validate that our
method is more effective compared to current state-of-the-art dynamic mapping
methods.

---

## INCODE: Implicit Neural Conditioning with Prior Knowledge Embeddings

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-28 | Amirhossein Kazerouni, Reza Azad, Alireza Hosseini, Dorit Merhof, Ulas Bagci | cs.CV | [PDF](http://arxiv.org/pdf/2310.18846v1){: .btn .btn-green } |

**Abstract**: Implicit Neural Representations (INRs) have revolutionized signal
representation by leveraging neural networks to provide continuous and smooth
representations of complex data. However, existing INRs face limitations in
capturing fine-grained details, handling noise, and adapting to diverse signal
types. To address these challenges, we introduce INCODE, a novel approach that
enhances the control of the sinusoidal-based activation function in INRs using
deep prior knowledge. INCODE comprises a harmonizer network and a composer
network, where the harmonizer network dynamically adjusts key parameters of the
activation function. Through a task-specific pre-trained model, INCODE adapts
the task-specific parameters to optimize the representation process. Our
approach not only excels in representation, but also extends its prowess to
tackle complex tasks such as audio, image, and 3D shape reconstructions, as
well as intricate challenges such as neural radiance fields (NeRFs), and
inverse problems, including denoising, super-resolution, inpainting, and CT
reconstruction. Through comprehensive experiments, INCODE demonstrates its
superiority in terms of robustness, accuracy, quality, and convergence rate,
broadening the scope of signal representation. Please visit the project's
website for details on the proposed method and access to the code.

Comments:
- Accepted at WACV 2024 conference

---

## ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Real Image

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-27 | Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, Jiajun Wu | cs.CV | [PDF](http://arxiv.org/pdf/2310.17994v1){: .btn .btn-green } |

**Abstract**: We introduce a 3D-aware diffusion model, ZeroNVS, for single-image novel view
synthesis for in-the-wild scenes. While existing methods are designed for
single objects with masked backgrounds, we propose new techniques to address
challenges introduced by in-the-wild multi-object scenes with complex
backgrounds. Specifically, we train a generative prior on a mixture of data
sources that capture object-centric, indoor, and outdoor scenes. To address
issues from data mixture such as depth-scale ambiguity, we propose a novel
camera conditioning parameterization and normalization scheme. Further, we
observe that Score Distillation Sampling (SDS) tends to truncate the
distribution of complex backgrounds during distillation of 360-degree scenes,
and propose "SDS anchoring" to improve the diversity of synthesized novel
views. Our model sets a new state-of-the-art result in LPIPS on the DTU dataset
in the zero-shot setting, even outperforming methods specifically trained on
DTU. We further adapt the challenging Mip-NeRF 360 dataset as a new benchmark
for single-image novel view synthesis, and demonstrate strong performance in
this setting. Our code and data are at http://kylesargent.github.io/zeronvs/

Comments:
- 17 pages

---

## Reconstructive Latent-Space Neural Radiance Fields for Efficient 3D  Scene Representations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-27 | Tristan Aumentado-Armstrong, Ashkan Mirzaei, Marcus A. Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G. Derpanis, Igor Gilitschenski | cs.CV | [PDF](http://arxiv.org/pdf/2310.17880v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have proven to be powerful 3D representations,
capable of high quality novel view synthesis of complex scenes. While NeRFs
have been applied to graphics, vision, and robotics, problems with slow
rendering speed and characteristic visual artifacts prevent adoption in many
use cases. In this work, we investigate combining an autoencoder (AE) with a
NeRF, in which latent features (instead of colours) are rendered and then
convolutionally decoded. The resulting latent-space NeRF can produce novel
views with higher quality than standard colour-space NeRFs, as the AE can
correct certain visual artifacts, while rendering over three times faster. Our
work is orthogonal to other techniques for improving NeRF efficiency. Further,
we can control the tradeoff between efficiency and image quality by shrinking
the AE architecture, achieving over 13 times faster rendering with only a small
drop in performance. We hope that our approach can form the basis of an
efficient, yet high-fidelity, 3D scene representation for downstream tasks,
especially when retaining differentiability is useful, as in many robotics
scenarios requiring continual learning.

---

## HyperFields: Towards Zero-Shot Generation of NeRFs from Text

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-26 | Sudarshan Babu, Richard Liu, Avery Zhou, Michael Maire, Greg Shakhnarovich, Rana Hanocka | cs.CV | [PDF](http://arxiv.org/pdf/2310.17075v2){: .btn .btn-green } |

**Abstract**: We introduce HyperFields, a method for generating text-conditioned Neural
Radiance Fields (NeRFs) with a single forward pass and (optionally) some
fine-tuning. Key to our approach are: (i) a dynamic hypernetwork, which learns
a smooth mapping from text token embeddings to the space of NeRFs; (ii) NeRF
distillation training, which distills scenes encoded in individual NeRFs into
one dynamic hypernetwork. These techniques enable a single network to fit over
a hundred unique scenes. We further demonstrate that HyperFields learns a more
general map between text and NeRFs, and consequently is capable of predicting
novel in-distribution and out-of-distribution scenes -- either zero-shot or
with a few finetuning steps. Finetuning HyperFields benefits from accelerated
convergence thanks to the learned general map, and is capable of synthesizing
novel scenes 5 to 10 times faster than existing neural optimization-based
methods. Our ablation experiments show that both the dynamic architecture and
NeRF distillation are critical to the expressivity of HyperFields.

Comments:
- Project page: https://threedle.github.io/hyperfields/

---

## Open-NeRF: Towards Open Vocabulary NeRF Decomposition

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-25 | Hao Zhang, Fang Li, Narendra Ahuja | cs.CV | [PDF](http://arxiv.org/pdf/2310.16383v1){: .btn .btn-green } |

**Abstract**: In this paper, we address the challenge of decomposing Neural Radiance Fields
(NeRF) into objects from an open vocabulary, a critical task for object
manipulation in 3D reconstruction and view synthesis. Current techniques for
NeRF decomposition involve a trade-off between the flexibility of processing
open-vocabulary queries and the accuracy of 3D segmentation. We present,
Open-vocabulary Embedded Neural Radiance Fields (Open-NeRF), that leverage
large-scale, off-the-shelf, segmentation models like the Segment Anything Model
(SAM) and introduce an integrate-and-distill paradigm with hierarchical
embeddings to achieve both the flexibility of open-vocabulary querying and 3D
segmentation accuracy. Open-NeRF first utilizes large-scale foundation models
to generate hierarchical 2D mask proposals from varying viewpoints. These
proposals are then aligned via tracking approaches and integrated within the 3D
space and subsequently distilled into the 3D field. This process ensures
consistent recognition and granularity of objects from different viewpoints,
even in challenging scenarios involving occlusion and indistinct features. Our
experimental results show that the proposed Open-NeRF outperforms
state-of-the-art methods such as LERF \cite{lerf} and FFD \cite{ffd} in
open-vocabulary scenarios. Open-NeRF offers a promising solution to NeRF
decomposition, guided by open-vocabulary queries, enabling novel applications
in robotics and vision-language interaction in open-world 3D scenes.

Comments:
- Accepted by WACV 2024

---

## UAV-Sim: NeRF-based Synthetic Data Generation for UAV-based Perception

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-25 | Christopher Maxey, Jaehoon Choi, Hyungtae Lee, Dinesh Manocha, Heesung Kwon | cs.CV | [PDF](http://arxiv.org/pdf/2310.16255v1){: .btn .btn-green } |

**Abstract**: Tremendous variations coupled with large degrees of freedom in UAV-based
imaging conditions lead to a significant lack of data in adequately learning
UAV-based perception models. Using various synthetic renderers in conjunction
with perception models is prevalent to create synthetic data to augment the
learning in the ground-based imaging domain. However, severe challenges in the
austere UAV-based domain require distinctive solutions to image synthesis for
data augmentation. In this work, we leverage recent advancements in neural
rendering to improve static and dynamic novelview UAV-based image synthesis,
especially from high altitudes, capturing salient scene attributes. Finally, we
demonstrate a considerable performance boost is achieved when a state-ofthe-art
detection model is optimized primarily on hybrid sets of real and synthetic
data instead of the real or synthetic data separately.

Comments:
- Video Link: https://www.youtube.com/watch?v=ucPzbPLqqpI

---

## PERF: Panoramic Neural Radiance Field from a Single Panorama

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-25 | Guangcong Wang, Peng Wang, Zhaoxi Chen, Wenping Wang, Chen Change Loy, Ziwei Liu | cs.CV | [PDF](http://arxiv.org/pdf/2310.16831v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has achieved substantial progress in novel view
synthesis given multi-view images. Recently, some works have attempted to train
a NeRF from a single image with 3D priors. They mainly focus on a limited field
of view with a few occlusions, which greatly limits their scalability to
real-world 360-degree panoramic scenarios with large-size occlusions. In this
paper, we present PERF, a 360-degree novel view synthesis framework that trains
a panoramic neural radiance field from a single panorama. Notably, PERF allows
3D roaming in a complex scene without expensive and tedious image collection.
To achieve this goal, we propose a novel collaborative RGBD inpainting method
and a progressive inpainting-and-erasing method to lift up a 360-degree 2D
scene to a 3D scene. Specifically, we first predict a panoramic depth map as
initialization given a single panorama and reconstruct visible 3D regions with
volume rendering. Then we introduce a collaborative RGBD inpainting approach
into a NeRF for completing RGB images and depth maps from random views, which
is derived from an RGB Stable Diffusion model and a monocular depth estimator.
Finally, we introduce an inpainting-and-erasing strategy to avoid inconsistent
geometry between a newly-sampled view and reference views. The two components
are integrated into the learning of NeRFs in a unified optimization framework
and achieve promising results. Extensive experiments on Replica and a new
dataset PERF-in-the-wild demonstrate the superiority of our PERF over
state-of-the-art methods. Our PERF can be widely used for real-world
applications, such as panorama-to-3D, text-to-3D, and 3D scene stylization
applications. Project page and code are available at
https://perf-project.github.io/ and https://github.com/perf-project/PeRF.

Comments:
- Project Page: https://perf-project.github.io/ , Code:
  https://github.com/perf-project/PeRF

---

## LightSpeed: Light and Fast Neural Light Fields on Mobile Devices

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-25 | Aarush Gupta, Junli Cao, Chaoyang Wang, Ju Hu, Sergey Tulyakov, Jian Ren, László A Jeni | cs.CV | [PDF](http://arxiv.org/pdf/2310.16832v2){: .btn .btn-green } |

**Abstract**: Real-time novel-view image synthesis on mobile devices is prohibitive due to
the limited computational power and storage. Using volumetric rendering
methods, such as NeRF and its derivatives, on mobile devices is not suitable
due to the high computational cost of volumetric rendering. On the other hand,
recent advances in neural light field representations have shown promising
real-time view synthesis results on mobile devices. Neural light field methods
learn a direct mapping from a ray representation to the pixel color. The
current choice of ray representation is either stratified ray sampling or
Plucker coordinates, overlooking the classic light slab (two-plane)
representation, the preferred representation to interpolate between light field
views. In this work, we find that using the light slab representation is an
efficient representation for learning a neural light field. More importantly,
it is a lower-dimensional ray representation enabling us to learn the 4D ray
space using feature grids which are significantly faster to train and render.
Although mostly designed for frontal views, we show that the light-slab
representation can be further extended to non-frontal scenes using a
divide-and-conquer strategy. Our method offers superior rendering quality
compared to previous light field methods and achieves a significantly improved
trade-off between rendering quality and speed.

Comments:
- Project Page: http://lightspeed-r2l.github.io/ . Add camera ready
  version

---

## 4D-Editor: Interactive Object-level Editing in Dynamic Neural Radiance  Fields via Semantic Distillation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-25 | Dadong Jiang, Zhihui Ke, Xiaobo Zhou, Xidong Shi | cs.CV | [PDF](http://arxiv.org/pdf/2310.16858v2){: .btn .btn-green } |

**Abstract**: This paper targets interactive object-level editing (e.g., deletion,
recoloring, transformation, composition) in dynamic scenes. Recently, some
methods aiming for flexible editing static scenes represented by neural
radiance field (NeRF) have shown impressive synthesis quality, while similar
capabilities in time-variant dynamic scenes remain limited. To solve this
problem, we propose 4D-Editor, an interactive semantic-driven editing
framework, allowing editing multiple objects in a dynamic NeRF with user
strokes on a single frame. We propose an extension to the original dynamic NeRF
by incorporating a hybrid semantic feature distillation to maintain
spatial-temporal consistency after editing. In addition, we design Recursive
Selection Refinement that significantly boosts object segmentation accuracy
within a dynamic NeRF to aid the editing process. Moreover, we develop
Multi-view Reprojection Inpainting to fill holes caused by incomplete scene
capture after editing. Extensive experiments and editing examples on real-world
demonstrate that 4D-Editor achieves photo-realistic editing on dynamic NeRFs.
Project page: https://patrickddj.github.io/4D-Editor

Comments:
- Project page: https://patrickddj.github.io/4D-Editor

---

## Cross-view Self-localization from Synthesized Scene-graphs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-24 | Ryogo Yamamoto, Kanji Tanaka | cs.CV | [PDF](http://arxiv.org/pdf/2310.15504v1){: .btn .btn-green } |

**Abstract**: Cross-view self-localization is a challenging scenario of visual place
recognition in which database images are provided from sparse viewpoints.
Recently, an approach for synthesizing database images from unseen viewpoints
using NeRF (Neural Radiance Fields) technology has emerged with impressive
performance. However, synthesized images provided by these techniques are often
of lower quality than the original images, and furthermore they significantly
increase the storage cost of the database. In this study, we explore a new
hybrid scene model that combines the advantages of view-invariant appearance
features computed from raw images and view-dependent spatial-semantic features
computed from synthesized images. These two types of features are then fused
into scene graphs, and compressively learned and recognized by a graph neural
network. The effectiveness of the proposed method was verified using a novel
cross-view self-localization dataset with many unseen views generated using a
photorealistic Habitat simulator.

Comments:
- 5 pages, 5 figures, technical report

---

## VQ-NeRF: Vector Quantization Enhances Implicit Neural Representations

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-23 | Yiying Yang, Wen Liu, Fukun Yin, Xin Chen, Gang Yu, Jiayuan Fan, Tao Chen | cs.CV | [PDF](http://arxiv.org/pdf/2310.14487v1){: .btn .btn-green } |

**Abstract**: Recent advancements in implicit neural representations have contributed to
high-fidelity surface reconstruction and photorealistic novel view synthesis.
However, the computational complexity inherent in these methodologies presents
a substantial impediment, constraining the attainable frame rates and
resolutions in practical applications. In response to this predicament, we
propose VQ-NeRF, an effective and efficient pipeline for enhancing implicit
neural representations via vector quantization. The essence of our method
involves reducing the sampling space of NeRF to a lower resolution and
subsequently reinstating it to the original size utilizing a pre-trained VAE
decoder, thereby effectively mitigating the sampling time bottleneck
encountered during rendering. Although the codebook furnishes representative
features, reconstructing fine texture details of the scene remains challenging
due to high compression rates. To overcome this constraint, we design an
innovative multi-scale NeRF sampling scheme that concurrently optimizes the
NeRF model at both compressed and original scales to enhance the network's
ability to preserve fine details. Furthermore, we incorporate a semantic loss
function to improve the geometric fidelity and semantic coherence of our 3D
reconstructions. Extensive experiments demonstrate the effectiveness of our
model in achieving the optimal trade-off between rendering quality and
efficiency. Evaluation on the DTU, BlendMVS, and H3DS datasets confirms the
superior performance of our approach.

Comments:
- Submitted to the 38th Annual AAAI Conference on Artificial
  Intelligence

---

## CAwa-NeRF: Instant Learning of Compression-Aware NeRF Features

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-23 | Omnia Mahmoud, Théo Ladune, Matthieu Gendrin | cs.CV | [PDF](http://arxiv.org/pdf/2310.14695v1){: .btn .btn-green } |

**Abstract**: Modeling 3D scenes by volumetric feature grids is one of the promising
directions of neural approximations to improve Neural Radiance Fields (NeRF).
Instant-NGP (INGP) introduced multi-resolution hash encoding from a lookup
table of trainable feature grids which enabled learning high-quality neural
graphics primitives in a matter of seconds. However, this improvement came at
the cost of higher storage size. In this paper, we address this challenge by
introducing instant learning of compression-aware NeRF features (CAwa-NeRF),
that allows exporting the zip compressed feature grids at the end of the model
training with a negligible extra time overhead without changing neither the
storage architecture nor the parameters used in the original INGP paper.
Nonetheless, the proposed method is not limited to INGP but could also be
adapted to any model. By means of extensive simulations, our proposed instant
learning pipeline can achieve impressive results on different kinds of static
scenes such as single object masked background scenes and real-life scenes
captured in our studio. In particular, for single object masked background
scenes CAwa-NeRF compresses the feature grids down to 6% (1.2 MB) of the
original size without any loss in the PSNR (33 dB) or down to 2.4% (0.53 MB)
with a slight virtual loss (32.31 dB).

Comments:
- 10 pages, 9 figures

---

## ManifoldNeRF: View-dependent Image Feature Supervision for Few-shot  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-20 | Daiju Kanaoka, Motoharu Sonogashira, Hakaru Tamukoh, Yasutomo Kawanishi | cs.CV | [PDF](http://arxiv.org/pdf/2310.13670v1){: .btn .btn-green } |

**Abstract**: Novel view synthesis has recently made significant progress with the advent
of Neural Radiance Fields (NeRF). DietNeRF is an extension of NeRF that aims to
achieve this task from only a few images by introducing a new loss function for
unknown viewpoints with no input images. The loss function assumes that a
pre-trained feature extractor should output the same feature even if input
images are captured at different viewpoints since the images contain the same
object. However, while that assumption is ideal, in reality, it is known that
as viewpoints continuously change, also feature vectors continuously change.
Thus, the assumption can harm training. To avoid this harmful training, we
propose ManifoldNeRF, a method for supervising feature vectors at unknown
viewpoints using interpolated features from neighboring known viewpoints. Since
the method provides appropriate supervision for each unknown viewpoint by the
interpolated features, the volume representation is learned better than
DietNeRF. Experimental results show that the proposed method performs better
than others in a complex scene. We also experimented with several subsets of
viewpoints from a set of viewpoints and identified an effective set of
viewpoints for real environments. This provided a basic policy of viewpoint
patterns for real-world application. The code is available at
https://github.com/haganelego/ManifoldNeRF_BMVC2023

Comments:
- Accepted by BMVC2023

---

## Sync-NeRF: Generalizing Dynamic NeRFs to Unsynchronized Videos

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-20 | Seoha Kim, Jeongmin Bae, Youngsik Yun, Hahyun Lee, Gun Bang, Youngjung Uh | cs.CV | [PDF](http://arxiv.org/pdf/2310.13356v2){: .btn .btn-green } |

**Abstract**: Recent advancements in 4D scene reconstruction using neural radiance fields
(NeRF) have demonstrated the ability to represent dynamic scenes from
multi-view videos. However, they fail to reconstruct the dynamic scenes and
struggle to fit even the training views in unsynchronized settings. It happens
because they employ a single latent embedding for a frame while the multi-view
images at the same frame were actually captured at different moments. To
address this limitation, we introduce time offsets for individual
unsynchronized videos and jointly optimize the offsets with NeRF. By design,
our method is applicable for various baselines and improves them with large
margins. Furthermore, finding the offsets naturally works as synchronizing the
videos without manual effort. Experiments are conducted on the common Plenoptic
Video Dataset and a newly built Unsynchronized Dynamic Blender Dataset to
verify the performance of our method. Project page:
https://seoha-kim.github.io/sync-nerf

Comments:
- AAAI 2024, Project page: https://seoha-kim.github.io/sync-nerf

---

## UE4-NeRF:Neural Radiance Field for Real-Time Rendering of Large-Scale  Scene

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-20 | Jiaming Gu, Minchao Jiang, Hongsheng Li, Xiaoyuan Lu, Guangming Zhu, Syed Afaq Ali Shah, Liang Zhang, Mohammed Bennamoun | cs.CV | [PDF](http://arxiv.org/pdf/2310.13263v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) is a novel implicit 3D reconstruction method
that shows immense potential and has been gaining increasing attention. It
enables the reconstruction of 3D scenes solely from a set of photographs.
However, its real-time rendering capability, especially for interactive
real-time rendering of large-scale scenes, still has significant limitations.
To address these challenges, in this paper, we propose a novel neural rendering
system called UE4-NeRF, specifically designed for real-time rendering of
large-scale scenes. We partitioned each large scene into different sub-NeRFs.
In order to represent the partitioned independent scene, we initialize
polygonal meshes by constructing multiple regular octahedra within the scene
and the vertices of the polygonal faces are continuously optimized during the
training process. Drawing inspiration from Level of Detail (LOD) techniques, we
trained meshes of varying levels of detail for different observation levels.
Our approach combines with the rasterization pipeline in Unreal Engine 4 (UE4),
achieving real-time rendering of large-scale scenes at 4K resolution with a
frame rate of up to 43 FPS. Rendering within UE4 also facilitates scene editing
in subsequent stages. Furthermore, through experiments, we have demonstrated
that our method achieves rendering quality comparable to state-of-the-art
approaches. Project page: https://jamchaos.github.io/UE4-NeRF/.

Comments:
- Accepted by NeurIPS2023

---

## VQ-NeRF: Neural Reflectance Decomposition and Editing with Vector  Quantization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-18 | Hongliang Zhong, Jingbo Zhang, Jing Liao | cs.CV | [PDF](http://arxiv.org/pdf/2310.11864v3){: .btn .btn-green } |

**Abstract**: We propose VQ-NeRF, a two-branch neural network model that incorporates
Vector Quantization (VQ) to decompose and edit reflectance fields in 3D scenes.
Conventional neural reflectance fields use only continuous representations to
model 3D scenes, despite the fact that objects are typically composed of
discrete materials in reality. This lack of discretization can result in noisy
material decomposition and complicated material editing. To address these
limitations, our model consists of a continuous branch and a discrete branch.
The continuous branch follows the conventional pipeline to predict decomposed
materials, while the discrete branch uses the VQ mechanism to quantize
continuous materials into individual ones. By discretizing the materials, our
model can reduce noise in the decomposition process and generate a segmentation
map of discrete materials. Specific materials can be easily selected for
further editing by clicking on the corresponding area of the segmentation
outcomes. Additionally, we propose a dropout-based VQ codeword ranking strategy
to predict the number of materials in a scene, which reduces redundancy in the
material segmentation process. To improve usability, we also develop an
interactive interface to further assist material editing. We evaluate our model
on both computer-generated and real-world scenes, demonstrating its superior
performance. To the best of our knowledge, our model is the first to enable
discrete material editing in 3D scenes.

Comments:
- Accepted by TVCG. Project Page:
  https://jtbzhl.github.io/VQ-NeRF.github.io/

---

## Towards Abdominal 3-D Scene Rendering from Laparoscopy Surgical Videos  using NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-18 | Khoa Tuan Nguyen, Francesca Tozzi, Nikdokht Rashidian, Wouter Willaert, Joris Vankerschaver, Wesley De Neve | cs.CV | [PDF](http://arxiv.org/pdf/2310.11645v1){: .btn .btn-green } |

**Abstract**: Given that a conventional laparoscope only provides a two-dimensional (2-D)
view, the detection and diagnosis of medical ailments can be challenging. To
overcome the visual constraints associated with laparoscopy, the use of
laparoscopic images and videos to reconstruct the three-dimensional (3-D)
anatomical structure of the abdomen has proven to be a promising approach.
Neural Radiance Fields (NeRFs) have recently gained attention thanks to their
ability to generate photorealistic images from a 3-D static scene, thus
facilitating a more comprehensive exploration of the abdomen through the
synthesis of new views. This distinguishes NeRFs from alternative methods such
as Simultaneous Localization and Mapping (SLAM) and depth estimation. In this
paper, we present a comprehensive examination of NeRFs in the context of
laparoscopy surgical videos, with the goal of rendering abdominal scenes in
3-D. Although our experimental results are promising, the proposed approach
encounters substantial challenges, which require further exploration in future
research.

Comments:
- The Version of Record of this contribution is published in MLMI 2023
  Part I, and is available online at
  https://doi.org/10.1007/978-3-031-45673-2_9

---

## TraM-NeRF: Tracing Mirror and Near-Perfect Specular Reflections through  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-16 | Leif Van Holland, Ruben Bliersbach, Jan U. Müller, Patrick Stotko, Reinhard Klein | cs.CV | [PDF](http://arxiv.org/pdf/2310.10650v1){: .btn .btn-green } |

**Abstract**: Implicit representations like Neural Radiance Fields (NeRF) showed impressive
results for photorealistic rendering of complex scenes with fine details.
However, ideal or near-perfectly specular reflecting objects such as mirrors,
which are often encountered in various indoor scenes, impose ambiguities and
inconsistencies in the representation of the reconstructed scene leading to
severe artifacts in the synthesized renderings. In this paper, we present a
novel reflection tracing method tailored for the involved volume rendering
within NeRF that takes these mirror-like objects into account while avoiding
the cost of straightforward but expensive extensions through standard path
tracing. By explicitly modeling the reflection behavior using physically
plausible materials and estimating the reflected radiance with Monte-Carlo
methods within the volume rendering formulation, we derive efficient strategies
for importance sampling and the transmittance computation along rays from only
few samples. We show that our novel method enables the training of consistent
representations of such challenging scenes and achieves superior results in
comparison to previous state-of-the-art approaches.

---

## Real-time Photorealistic Dynamic Scene Representation and Rendering with  4D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-16 | Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, Li Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2310.10642v2){: .btn .btn-green } |

**Abstract**: Reconstructing dynamic 3D scenes from 2D images and generating diverse views
over time is challenging due to scene complexity and temporal dynamics. Despite
advancements in neural implicit models, limitations persist: (i) Inadequate
Scene Structure: Existing methods struggle to reveal the spatial and temporal
structure of dynamic scenes from directly learning the complex 6D plenoptic
function. (ii) Scaling Deformation Modeling: Explicitly modeling scene element
deformation becomes impractical for complex dynamics. To address these issues,
we consider the spacetime as an entirety and propose to approximate the
underlying spatio-temporal 4D volume of a dynamic scene by optimizing a
collection of 4D primitives, with explicit geometry and appearance modeling.
Learning to optimize the 4D primitives enables us to synthesize novel views at
any desired time with our tailored rendering routine. Our model is conceptually
simple, consisting of a 4D Gaussian parameterized by anisotropic ellipses that
can rotate arbitrarily in space and time, as well as view-dependent and
time-evolved appearance represented by the coefficient of 4D spherindrical
harmonics. This approach offers simplicity, flexibility for variable-length
video and end-to-end training, and efficient real-time rendering, making it
suitable for capturing complex dynamic scene motions. Experiments across
various benchmarks, including monocular and multi-view scenarios, demonstrate
our 4DGS model's superior visual quality and efficiency.

Comments:
- ICLR 2024

---

## DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and  View-Change Human-Centric Video Editing

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-16 | Jia-Wei Liu, Yan-Pei Cao, Jay Zhangjie Wu, Weijia Mao, Yuchao Gu, Rui Zhao, Jussi Keppo, Ying Shan, Mike Zheng Shou | cs.CV | [PDF](http://arxiv.org/pdf/2310.10624v2){: .btn .btn-green } |

**Abstract**: Despite recent progress in diffusion-based video editing, existing methods
are limited to short-length videos due to the contradiction between long-range
consistency and frame-wise editing. Prior attempts to address this challenge by
introducing video-2D representations encounter significant difficulties with
large-scale motion- and view-change videos, especially in human-centric
scenarios. To overcome this, we propose to introduce the dynamic Neural
Radiance Fields (NeRF) as the innovative video representation, where the
editing can be performed in the 3D spaces and propagated to the entire video
via the deformation field. To provide consistent and controllable editing, we
propose the image-based video-NeRF editing pipeline with a set of innovative
designs, including multi-view multi-pose Score Distillation Sampling (SDS) from
both the 2D personalized diffusion prior and 3D diffusion prior, reconstruction
losses, text-guided local parts super-resolution, and style transfer. Extensive
experiments demonstrate that our method, dubbed as DynVideo-E, significantly
outperforms SOTA approaches on two challenging datasets by a large margin of
50% ~ 95% for human preference. Code will be released at
https://showlab.github.io/DynVideo-E/.

Comments:
- Project Page: https://showlab.github.io/DynVideo-E/

---

## Self-supervised Fetal MRI 3D Reconstruction Based on Radiation Diffusion  Generation Model

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-16 | Junpeng Tan, Xin Zhang, Yao Lv, Xiangmin Xu, Gang Li | eess.IV | [PDF](http://arxiv.org/pdf/2310.10209v1){: .btn .btn-green } |

**Abstract**: Although the use of multiple stacks can handle slice-to-volume motion
correction and artifact removal problems, there are still several problems: 1)
The slice-to-volume method usually uses slices as input, which cannot solve the
problem of uniform intensity distribution and complementarity in regions of
different fetal MRI stacks; 2) The integrity of 3D space is not considered,
which adversely affects the discrimination and generation of globally
consistent information in fetal MRI; 3) Fetal MRI with severe motion artifacts
in the real-world cannot achieve high-quality super-resolution reconstruction.
To address these issues, we propose a novel fetal brain MRI high-quality volume
reconstruction method, called the Radiation Diffusion Generation Model (RDGM).
It is a self-supervised generation method, which incorporates the idea of
Neural Radiation Field (NeRF) based on the coordinate generation and diffusion
model based on super-resolution generation. To solve regional intensity
heterogeneity in different directions, we use a pre-trained transformer model
for slice registration, and then, a new regionally Consistent Implicit Neural
Representation (CINR) network sub-module is proposed. CINR can generate the
initial volume by combining a coordinate association map of two different
coordinate mapping spaces. To enhance volume global consistency and
discrimination, we introduce the Volume Diffusion Super-resolution Generation
(VDSG) mechanism. The global intensity discriminant generation from
volume-to-volume is carried out using the idea of diffusion generation, and
CINR becomes the deviation intensity generation network of the volume-to-volume
diffusion model. Finally, the experimental results on real-world fetal brain
MRI stacks demonstrate the state-of-the-art performance of our method.

---

## ProteusNeRF: Fast Lightweight NeRF Editing using 3D-Aware Image Context

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-15 | Binglun Wang, Niladri Shekhar Dutt, Niloy J. Mitra | cs.CV | [PDF](http://arxiv.org/pdf/2310.09965v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have recently emerged as a popular option for
photo-realistic object capture due to their ability to faithfully capture
high-fidelity volumetric content even from handheld video input. Although much
research has been devoted to efficient optimization leading to real-time
training and rendering, options for interactive editing NeRFs remain limited.
We present a very simple but effective neural network architecture that is fast
and efficient while maintaining a low memory footprint. This architecture can
be incrementally guided through user-friendly image-based edits. Our
representation allows straightforward object selection via semantic feature
distillation at the training stage. More importantly, we propose a local
3D-aware image context to facilitate view-consistent image editing that can
then be distilled into fine-tuned NeRFs, via geometric and appearance
adjustments. We evaluate our setup on a variety of examples to demonstrate
appearance and geometric edits and report 10-30x speedup over concurrent work
focusing on text-guided NeRF editing. Video results can be seen on our project
webpage at https://proteusnerf.github.io.

---

## Active Perception using Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-15 | Siming He, Christopher D. Hsu, Dexter Ong, Yifei Simon Shao, Pratik Chaudhari | cs.RO | [PDF](http://arxiv.org/pdf/2310.09892v1){: .btn .btn-green } |

**Abstract**: We study active perception from first principles to argue that an autonomous
agent performing active perception should maximize the mutual information that
past observations posses about future ones. Doing so requires (a) a
representation of the scene that summarizes past observations and the ability
to update this representation to incorporate new observations (state estimation
and mapping), (b) the ability to synthesize new observations of the scene (a
generative model), and (c) the ability to select control trajectories that
maximize predictive information (planning). This motivates a neural radiance
field (NeRF)-like representation which captures photometric, geometric and
semantic properties of the scene grounded. This representation is well-suited
to synthesizing new observations from different viewpoints. And thereby, a
sampling-based planner can be used to calculate the predictive information from
synthetic observations along dynamically-feasible trajectories. We use active
perception for exploring cluttered indoor environments and employ a notion of
semantic uncertainty to check for the successful completion of an exploration
task. We demonstrate these ideas via simulation in realistic 3D indoor
environments.

---

## CBARF: Cascaded Bundle-Adjusting Neural Radiance Fields from Imperfect  Camera Poses

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-15 | Hongyu Fu, Xin Yu, Lincheng Li, Li Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2310.09776v1){: .btn .btn-green } |

**Abstract**: Existing volumetric neural rendering techniques, such as Neural Radiance
Fields (NeRF), face limitations in synthesizing high-quality novel views when
the camera poses of input images are imperfect. To address this issue, we
propose a novel 3D reconstruction framework that enables simultaneous
optimization of camera poses, dubbed CBARF (Cascaded Bundle-Adjusting NeRF).In
a nutshell, our framework optimizes camera poses in a coarse-to-fine manner and
then reconstructs scenes based on the rectified poses. It is observed that the
initialization of camera poses has a significant impact on the performance of
bundle-adjustment (BA). Therefore, we cascade multiple BA modules at different
scales to progressively improve the camera poses. Meanwhile, we develop a
neighbor-replacement strategy to further optimize the results of BA in each
stage. In this step, we introduce a novel criterion to effectively identify
poorly estimated camera poses. Then we replace them with the poses of
neighboring cameras, thus further eliminating the impact of inaccurate camera
poses. Once camera poses have been optimized, we employ a density voxel grid to
generate high-quality 3D reconstructed scenes and images in novel views.
Experimental results demonstrate that our CBARF model achieves state-of-the-art
performance in both pose optimization and novel view synthesis, especially in
the existence of large camera pose noise.

---

## GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging  2D and 3D Diffusion Models

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-12 | Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, Xinggang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2310.08529v2){: .btn .btn-green } |

**Abstract**: In recent times, the generation of 3D assets from text prompts has shown
impressive results. Both 2D and 3D diffusion models can help generate decent 3D
objects based on prompts. 3D diffusion models have good 3D consistency, but
their quality and generalization are limited as trainable 3D data is expensive
and hard to obtain. 2D diffusion models enjoy strong abilities of
generalization and fine generation, but 3D consistency is hard to guarantee.
This paper attempts to bridge the power from the two types of diffusion models
via the recent explicit and efficient 3D Gaussian splatting representation. A
fast 3D object generation framework, named as GaussianDreamer, is proposed,
where the 3D diffusion model provides priors for initialization and the 2D
diffusion model enriches the geometry and appearance. Operations of noisy point
growing and color perturbation are introduced to enhance the initialized
Gaussians. Our GaussianDreamer can generate a high-quality 3D instance or 3D
avatar within 15 minutes on one GPU, much faster than previous methods, while
the generated instances can be directly rendered in real time. Demos and code
are available at https://taoranyi.com/gaussiandreamer/.

Comments:
- Project page: https://taoranyi.com/gaussiandreamer/

---

## 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-12 | Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, Xinggang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2310.08528v2){: .btn .btn-green } |

**Abstract**: Representing and rendering dynamic scenes has been an important but
challenging task. Especially, to accurately model complex motions, high
efficiency is usually hard to guarantee. To achieve real-time dynamic scene
rendering while also enjoying high training and storage efficiency, we propose
4D Gaussian Splatting (4D-GS) as a holistic representation for dynamic scenes
rather than applying 3D-GS for each individual frame. In 4D-GS, a novel
explicit representation containing both 3D Gaussians and 4D neural voxels is
proposed. A decomposed neural voxel encoding algorithm inspired by HexPlane is
proposed to efficiently build Gaussian features from 4D neural voxels and then
a lightweight MLP is applied to predict Gaussian deformations at novel
timestamps. Our 4D-GS method achieves real-time rendering under high
resolutions, 82 FPS at an 800$\times$800 resolution on an RTX 3090 GPU while
maintaining comparable or better quality than previous state-of-the-art
methods. More demos and code are available at
https://guanjunwu.github.io/4dgs/.

Comments:
- Project page: https://guanjunwu.github.io/4dgs/

---

## rpcPRF: Generalizable MPI Neural Radiance Field for Satellite Camera

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-11 | Tongtong Zhang, Yuanxiang Li | cs.CV | [PDF](http://arxiv.org/pdf/2310.07179v1){: .btn .btn-green } |

**Abstract**: Novel view synthesis of satellite images holds a wide range of practical
applications. While recent advances in the Neural Radiance Field have
predominantly targeted pin-hole cameras, and models for satellite cameras often
demand sufficient input views. This paper presents rpcPRF, a Multiplane Images
(MPI) based Planar neural Radiance Field for Rational Polynomial Camera (RPC).
Unlike coordinate-based neural radiance fields in need of sufficient views of
one scene, our model is applicable to single or few inputs and performs well on
images from unseen scenes. To enable generalization across scenes, we propose
to use reprojection supervision to induce the predicted MPI to learn the
correct geometry between the 3D coordinates and the images. Moreover, we remove
the stringent requirement of dense depth supervision from deep
multiview-stereo-based methods by introducing rendering techniques of radiance
fields. rpcPRF combines the superiority of implicit representations and the
advantages of the RPC model, to capture the continuous altitude space while
learning the 3D structure. Given an RGB image and its corresponding RPC, the
end-to-end model learns to synthesize the novel view with a new RPC and
reconstruct the altitude of the scene. When multiple views are provided as
inputs, rpcPRF exerts extra supervision provided by the extra views. On the TLC
dataset from ZY-3, and the SatMVS3D dataset with urban scenes from WV-3, rpcPRF
outperforms state-of-the-art nerf-based methods by a significant margin in
terms of image fidelity, reconstruction accuracy, and efficiency, for both
single-view and multiview task.

---

## PoRF: Pose Residual Field for Accurate Neural Surface Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-11 | Jia-Wang Bian, Wenjing Bian, Victor Adrian Prisacariu, Philip Torr | cs.CV | [PDF](http://arxiv.org/pdf/2310.07449v2){: .btn .btn-green } |

**Abstract**: Neural surface reconstruction is sensitive to the camera pose noise, even if
state-of-the-art pose estimators like COLMAP or ARKit are used. More
importantly, existing Pose-NeRF joint optimisation methods have struggled to
improve pose accuracy in challenging real-world scenarios. To overcome the
challenges, we introduce the pose residual field (\textbf{PoRF}), a novel
implicit representation that uses an MLP for regressing pose updates. This is
more robust than the conventional pose parameter optimisation due to parameter
sharing that leverages global information over the entire sequence.
Furthermore, we propose an epipolar geometry loss to enhance the supervision
that leverages the correspondences exported from COLMAP results without the
extra computational overhead. Our method yields promising results. On the DTU
dataset, we reduce the rotation error by 78\% for COLMAP poses, leading to the
decreased reconstruction Chamfer distance from 3.48mm to 0.85mm. On the
MobileBrick dataset that contains casually captured unbounded 360-degree
videos, our method refines ARKit poses and improves the reconstruction F1 score
from 69.18 to 75.67, outperforming that with the dataset provided ground-truth
pose (75.14). These achievements demonstrate the efficacy of our approach in
refining camera poses and improving the accuracy of neural surface
reconstruction in real-world scenarios.

Comments:
- Under review

---

## Dynamic Appearance Particle Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-11 | Ancheng Lin, Jun Li | cs.CV | [PDF](http://arxiv.org/pdf/2310.07916v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have shown great potential in modelling 3D
scenes. Dynamic NeRFs extend this model by capturing time-varying elements,
typically using deformation fields. The existing dynamic NeRFs employ a similar
Eulerian representation for both light radiance and deformation fields. This
leads to a close coupling of appearance and motion and lacks a physical
interpretation. In this work, we propose Dynamic Appearance Particle Neural
Radiance Field (DAP-NeRF), which introduces particle-based representation to
model the motions of visual elements in a dynamic 3D scene. DAP-NeRF consists
of superposition of a static field and a dynamic field. The dynamic field is
quantised as a collection of {\em appearance particles}, which carries the
visual information of a small dynamic element in the scene and is equipped with
a motion model. All components, including the static field, the visual features
and motion models of the particles, are learned from monocular videos without
any prior geometric knowledge of the scene. We develop an efficient
computational framework for the particle-based model. We also construct a new
dataset to evaluate motion modelling. Experimental results show that DAP-NeRF
is an effective technique to capture not only the appearance but also the
physically meaningful motions in a 3D dynamic scene.

---

## Leveraging Neural Radiance Fields for Uncertainty-Aware Visual  Localization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-10 | Le Chen, Weirong Chen, Rui Wang, Marc Pollefeys | cs.CV | [PDF](http://arxiv.org/pdf/2310.06984v1){: .btn .btn-green } |

**Abstract**: As a promising fashion for visual localization, scene coordinate regression
(SCR) has seen tremendous progress in the past decade. Most recent methods
usually adopt neural networks to learn the mapping from image pixels to 3D
scene coordinates, which requires a vast amount of annotated training data. We
propose to leverage Neural Radiance Fields (NeRF) to generate training samples
for SCR. Despite NeRF's efficiency in rendering, many of the rendered data are
polluted by artifacts or only contain minimal information gain, which can
hinder the regression accuracy or bring unnecessary computational costs with
redundant data. These challenges are addressed in three folds in this paper:
(1) A NeRF is designed to separately predict uncertainties for the rendered
color and depth images, which reveal data reliability at the pixel level. (2)
SCR is formulated as deep evidential learning with epistemic uncertainty, which
is used to evaluate information gain and scene coordinate quality. (3) Based on
the three arts of uncertainties, a novel view selection policy is formed that
significantly improves data efficiency. Experiments on public datasets
demonstrate that our method could select the samples that bring the most
information gain and promote the performance with the highest efficiency.

Comments:
- 8 pages, 5 figures

---

## High-Fidelity 3D Head Avatars Reconstruction through Spatially-Varying  Expression Conditioned Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-10 | Minghan Qin, Yifan Liu, Yuelang Xu, Xiaochen Zhao, Yebin Liu, Haoqian Wang | cs.CV | [PDF](http://arxiv.org/pdf/2310.06275v1){: .btn .btn-green } |

**Abstract**: One crucial aspect of 3D head avatar reconstruction lies in the details of
facial expressions. Although recent NeRF-based photo-realistic 3D head avatar
methods achieve high-quality avatar rendering, they still encounter challenges
retaining intricate facial expression details because they overlook the
potential of specific expression variations at different spatial positions when
conditioning the radiance field. Motivated by this observation, we introduce a
novel Spatially-Varying Expression (SVE) conditioning. The SVE can be obtained
by a simple MLP-based generation network, encompassing both spatial positional
features and global expression information. Benefiting from rich and diverse
information of the SVE at different positions, the proposed SVE-conditioned
neural radiance field can deal with intricate facial expressions and achieve
realistic rendering and geometry details of high-fidelity 3D head avatars.
Additionally, to further elevate the geometric and rendering quality, we
introduce a new coarse-to-fine training strategy, including a geometry
initialization strategy at the coarse stage and an adaptive importance sampling
strategy at the fine stage. Extensive experiments indicate that our method
outperforms other state-of-the-art (SOTA) methods in rendering and geometry
quality on mobile phone-collected and public datasets.

Comments:
- 9 pages, 5 figures

---

## A Real-time Method for Inserting Virtual Objects into Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-09 | Keyang Ye, Hongzhi Wu, Xin Tong, Kun Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2310.05837v1){: .btn .btn-green } |

**Abstract**: We present the first real-time method for inserting a rigid virtual object
into a neural radiance field, which produces realistic lighting and shadowing
effects, as well as allows interactive manipulation of the object. By
exploiting the rich information about lighting and geometry in a NeRF, our
method overcomes several challenges of object insertion in augmented reality.
For lighting estimation, we produce accurate, robust and 3D spatially-varying
incident lighting that combines the near-field lighting from NeRF and an
environment lighting to account for sources not covered by the NeRF. For
occlusion, we blend the rendered virtual object with the background scene using
an opacity map integrated from the NeRF. For shadows, with a precomputed field
of spherical signed distance field, we query the visibility term for any point
around the virtual object, and cast soft, detailed shadows onto 3D surfaces.
Compared with state-of-the-art techniques, our approach can insert virtual
object into scenes with superior fidelity, and has a great potential to be
further applied to augmented reality systems.

---

## Neural Impostor: Editing Neural Radiance Fields with Explicit Shape  Manipulation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-09 | Ruiyang Liu, Jinxu Xiang, Bowen Zhao, Ran Zhang, Jingyi Yu, Changxi Zheng | cs.GR | [PDF](http://arxiv.org/pdf/2310.05391v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) have significantly advanced the generation of
highly realistic and expressive 3D scenes. However, the task of editing NeRF,
particularly in terms of geometry modification, poses a significant challenge.
This issue has obstructed NeRF's wider adoption across various applications. To
tackle the problem of efficiently editing neural implicit fields, we introduce
Neural Impostor, a hybrid representation incorporating an explicit tetrahedral
mesh alongside a multigrid implicit field designated for each tetrahedron
within the explicit mesh. Our framework bridges the explicit shape manipulation
and the geometric editing of implicit fields by utilizing multigrid barycentric
coordinate encoding, thus offering a pragmatic solution to deform, composite,
and generate neural implicit fields while maintaining a complex volumetric
appearance. Furthermore, we propose a comprehensive pipeline for editing neural
implicit fields based on a set of explicit geometric editing operations. We
show the robustness and adaptability of our system through diverse examples and
experiments, including the editing of both synthetic objects and real captured
data. Finally, we demonstrate the authoring process of a hybrid
synthetic-captured object utilizing a variety of editing operations,
underlining the transformative potential of Neural Impostor in the field of 3D
content creation and manipulation.

Comments:
- Accepted at Pacific Graphics 2023 and Computer Graphics Forum

---

## LocoNeRF: A NeRF-based Approach for Local Structure from Motion for  Precise Localization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-08 | Artem Nenashev, Mikhail Kurenkov, Andrei Potapov, Iana Zhura, Maksim Katerishich, Dzmitry Tsetserukou | cs.CV | [PDF](http://arxiv.org/pdf/2310.05134v1){: .btn .btn-green } |

**Abstract**: Visual localization is a critical task in mobile robotics, and researchers
are continuously developing new approaches to enhance its efficiency. In this
article, we propose a novel approach to improve the accuracy of visual
localization using Structure from Motion (SfM) techniques. We highlight the
limitations of global SfM, which suffers from high latency, and the challenges
of local SfM, which requires large image databases for accurate reconstruction.
To address these issues, we propose utilizing Neural Radiance Fields (NeRF), as
opposed to image databases, to cut down on the space required for storage. We
suggest that sampling reference images around the prior query position can lead
to further improvements. We evaluate the accuracy of our proposed method
against ground truth obtained using LIDAR and Advanced Lidar Odometry and
Mapping in Real-time (A-LOAM), and compare its storage usage against local SfM
with COLMAP in the conducted experiments. Our proposed method achieves an
accuracy of 0.068 meters compared to the ground truth, which is slightly lower
than the most advanced method COLMAP, which has an accuracy of 0.022 meters.
However, the size of the database required for COLMAP is 400 megabytes, whereas
the size of our NeRF model is only 160 megabytes. Finally, we perform an
ablation study to assess the impact of using reference images from the NeRF
reconstruction.

---

## Geometry Aware Field-to-field Transformations for 3D Semantic  Segmentation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-08 | Dominik Hollidt, Clinton Wang, Polina Golland, Marc Pollefeys | cs.CV | [PDF](http://arxiv.org/pdf/2310.05133v1){: .btn .btn-green } |

**Abstract**: We present a novel approach to perform 3D semantic segmentation solely from
2D supervision by leveraging Neural Radiance Fields (NeRFs). By extracting
features along a surface point cloud, we achieve a compact representation of
the scene which is sample-efficient and conducive to 3D reasoning. Learning
this feature space in an unsupervised manner via masked autoencoding enables
few-shot segmentation. Our method is agnostic to the scene parameterization,
working on scenes fit with any type of NeRF.

Comments:
- 8 pages

---

## Improving Neural Radiance Field using Near-Surface Sampling with Point  Cloud Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-06 | Hye Bin Yoo, Hyun Min Han, Sung Soo Hwang, Il Yong Chun | cs.CV | [PDF](http://arxiv.org/pdf/2310.04152v1){: .btn .btn-green } |

**Abstract**: Neural radiance field (NeRF) is an emerging view synthesis method that
samples points in a three-dimensional (3D) space and estimates their existence
and color probabilities. The disadvantage of NeRF is that it requires a long
training time since it samples many 3D points. In addition, if one samples
points from occluded regions or in the space where an object is unlikely to
exist, the rendering quality of NeRF can be degraded. These issues can be
solved by estimating the geometry of 3D scene. This paper proposes a
near-surface sampling framework to improve the rendering quality of NeRF. To
this end, the proposed method estimates the surface of a 3D object using depth
images of the training set and sampling is performed around there only. To
obtain depth information on a novel view, the paper proposes a 3D point cloud
generation method and a simple refining method for projected depth from a point
cloud. Experimental results show that the proposed near-surface sampling NeRF
framework can significantly improve the rendering quality, compared to the
original NeRF and a state-of-the-art depth-based NeRF method. In addition, one
can significantly accelerate the training time of a NeRF model with the
proposed near-surface sampling framework.

Comments:
- 13 figures, 2 tables

---

## Point-Based Radiance Fields for Controllable Human Motion Synthesis



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-05 | Haitao Yu, Deheng Zhang, Peiyuan Xie, Tianyi Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2310.03375v1){: .btn .btn-green } |

**Abstract**: This paper proposes a novel controllable human motion synthesis method for
fine-level deformation based on static point-based radiance fields. Although
previous editable neural radiance field methods can generate impressive results
on novel-view synthesis and allow naive deformation, few algorithms can achieve
complex 3D human editing such as forward kinematics. Our method exploits the
explicit point cloud to train the static 3D scene and apply the deformation by
encoding the point cloud translation using a deformation MLP. To make sure the
rendering result is consistent with the canonical space training, we estimate
the local rotation using SVD and interpolate the per-point rotation to the
query view direction of the pre-trained radiance field. Extensive experiments
show that our approach can significantly outperform the state-of-the-art on
fine-level complex deformation which can be generalized to other 3D characters
besides humans.

---

## Targeted Adversarial Attacks on Generalizable Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-05 | Andras Horvath, Csaba M. Jozsa | cs.LG | [PDF](http://arxiv.org/pdf/2310.03578v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have recently emerged as a powerful tool for
3D scene representation and rendering. These data-driven models can learn to
synthesize high-quality images from sparse 2D observations, enabling realistic
and interactive scene reconstructions. However, the growing usage of NeRFs in
critical applications such as augmented reality, robotics, and virtual
environments could be threatened by adversarial attacks.
  In this paper we present how generalizable NeRFs can be attacked by both
low-intensity adversarial attacks and adversarial patches, where the later
could be robust enough to be used in real world applications. We also
demonstrate targeted attacks, where a specific, predefined output scene is
generated by these attack with success.

---

## BID-NeRF: RGB-D image pose estimation with inverted Neural Radiance  Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-05 | Ágoston István Csehi, Csaba Máté Józsa | cs.CV | [PDF](http://arxiv.org/pdf/2310.03563v1){: .btn .btn-green } |

**Abstract**: We aim to improve the Inverted Neural Radiance Fields (iNeRF) algorithm which
defines the image pose estimation problem as a NeRF based iterative linear
optimization. NeRFs are novel neural space representation models that can
synthesize photorealistic novel views of real-world scenes or objects. Our
contributions are as follows: we extend the localization optimization objective
with a depth-based loss function, we introduce a multi-image based loss
function where a sequence of images with known relative poses are used without
increasing the computational complexity, we omit hierarchical sampling during
volumetric rendering, meaning only the coarse model is used for pose
estimation, and we how that by extending the sampling interval convergence can
be achieved even or higher initial pose estimate errors. With the proposed
modifications the convergence speed is significantly improved, and the basin of
convergence is substantially extended.

Comments:
- Accepted to Nerf4ADR workshop of ICCV23 conference

---

## Shielding the Unseen: Privacy Protection through Poisoning NeRF with  Spatial Deformation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-04 | Yihan Wu, Brandon Y. Feng, Heng Huang | cs.CV | [PDF](http://arxiv.org/pdf/2310.03125v1){: .btn .btn-green } |

**Abstract**: In this paper, we introduce an innovative method of safeguarding user privacy
against the generative capabilities of Neural Radiance Fields (NeRF) models.
Our novel poisoning attack method induces changes to observed views that are
imperceptible to the human eye, yet potent enough to disrupt NeRF's ability to
accurately reconstruct a 3D scene. To achieve this, we devise a bi-level
optimization algorithm incorporating a Projected Gradient Descent (PGD)-based
spatial deformation. We extensively test our approach on two common NeRF
benchmark datasets consisting of 29 real-world scenes with high-quality images.
Our results compellingly demonstrate that our privacy-preserving method
significantly impairs NeRF's performance across these benchmark datasets.
Additionally, we show that our method is adaptable and versatile, functioning
across various perturbation strengths and NeRF architectures. This work offers
valuable insights into NeRF's vulnerabilities and emphasizes the need to
account for such potential privacy risks when developing robust 3D scene
reconstruction algorithms. Our study contributes to the larger conversation
surrounding responsible AI and generative machine learning, aiming to protect
user privacy and respect creative ownership in the digital age.

---

## Efficient-3DiM: Learning a Generalizable Single-image Novel-view  Synthesizer in One Day



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-04 | Yifan Jiang, Hao Tang, Jen-Hao Rick Chang, Liangchen Song, Zhangyang Wang, Liangliang Cao | cs.CV | [PDF](http://arxiv.org/pdf/2310.03015v1){: .btn .btn-green } |

**Abstract**: The task of novel view synthesis aims to generate unseen perspectives of an
object or scene from a limited set of input images. Nevertheless, synthesizing
novel views from a single image still remains a significant challenge in the
realm of computer vision. Previous approaches tackle this problem by adopting
mesh prediction, multi-plain image construction, or more advanced techniques
such as neural radiance fields. Recently, a pre-trained diffusion model that is
specifically designed for 2D image synthesis has demonstrated its capability in
producing photorealistic novel views, if sufficiently optimized on a 3D
finetuning task. Although the fidelity and generalizability are greatly
improved, training such a powerful diffusion model requires a vast volume of
training data and model parameters, resulting in a notoriously long time and
high computational costs. To tackle this issue, we propose Efficient-3DiM, a
simple but effective framework to learn a single-image novel-view synthesizer.
Motivated by our in-depth analysis of the inference process of diffusion
models, we propose several pragmatic strategies to reduce the training overhead
to a manageable scale, including a crafted timestep sampling strategy, a
superior 3D feature extractor, and an enhanced training scheme. When combined,
our framework is able to reduce the total training time from 10 days to less
than 1 day, significantly accelerating the training process under the same
computational platform (one instance with 8 Nvidia A100 GPUs). Comprehensive
experiments are conducted to demonstrate the efficiency and generalizability of
our proposed method.

---

## T$^3$Bench: Benchmarking Current Progress in Text-to-3D Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-04 | Yuze He, Yushi Bai, Matthieu Lin, Wang Zhao, Yubin Hu, Jenny Sheng, Ran Yi, Juanzi Li, Yong-Jin Liu | cs.CV | [PDF](http://arxiv.org/pdf/2310.02977v1){: .btn .btn-green } |

**Abstract**: Recent methods in text-to-3D leverage powerful pretrained diffusion models to
optimize NeRF. Notably, these methods are able to produce high-quality 3D
scenes without training on 3D data. Due to the open-ended nature of the task,
most studies evaluate their results with subjective case studies and user
experiments, thereby presenting a challenge in quantitatively addressing the
question: How has current progress in Text-to-3D gone so far? In this paper, we
introduce T$^3$Bench, the first comprehensive text-to-3D benchmark containing
diverse text prompts of three increasing complexity levels that are specially
designed for 3D generation. To assess both the subjective quality and the text
alignment, we propose two automatic metrics based on multi-view images produced
by the 3D contents. The quality metric combines multi-view text-image scores
and regional convolution to detect quality and view inconsistency. The
alignment metric uses multi-view captioning and Large Language Model (LLM)
evaluation to measure text-3D consistency. Both metrics closely correlate with
different dimensions of human judgments, providing a paradigm for efficiently
evaluating text-to-3D models. The benchmarking results, shown in Fig. 1, reveal
performance differences among six prevalent text-to-3D methods. Our analysis
further highlights the common struggles for current methods on generating
surroundings and multi-object scenes, as well as the bottleneck of leveraging
2D guidance for 3D generation. Our project page is available at:
https://t3bench.com.

Comments:
- 16 pages, 11 figures

---

## ED-NeRF: Efficient Text-Guided Editing of 3D Scene using Latent Space  NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-04 | Jangho Park, Gihyun Kwon, Jong Chul Ye | cs.CV | [PDF](http://arxiv.org/pdf/2310.02712v1){: .btn .btn-green } |

**Abstract**: Recently, there has been a significant advancement in text-to-image diffusion
models, leading to groundbreaking performance in 2D image generation. These
advancements have been extended to 3D models, enabling the generation of novel
3D objects from textual descriptions. This has evolved into NeRF editing
methods, which allow the manipulation of existing 3D objects through textual
conditioning. However, existing NeRF editing techniques have faced limitations
in their performance due to slow training speeds and the use of loss functions
that do not adequately consider editing. To address this, here we present a
novel 3D NeRF editing approach dubbed ED-NeRF by successfully embedding
real-world scenes into the latent space of the latent diffusion model (LDM)
through a unique refinement layer. This approach enables us to obtain a NeRF
backbone that is not only faster but also more amenable to editing compared to
traditional image space NeRF editing. Furthermore, we propose an improved loss
function tailored for editing by migrating the delta denoising score (DDS)
distillation loss, originally used in 2D image editing to the three-dimensional
domain. This novel loss function surpasses the well-known score distillation
sampling (SDS) loss in terms of suitability for editing purposes. Our
experimental results demonstrate that ED-NeRF achieves faster editing speed
while producing improved output quality compared to state-of-the-art 3D editing
models.

---

## USB-NeRF: Unrolling Shutter Bundle Adjusted Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-04 | Moyang Li, Peng Wang, Lingzhe Zhao, Bangyan Liao, Peidong Liu | cs.CV | [PDF](http://arxiv.org/pdf/2310.02687v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) has received much attention recently due to its
impressive capability to represent 3D scene and synthesize novel view images.
Existing works usually assume that the input images are captured by a global
shutter camera. Thus, rolling shutter (RS) images cannot be trivially applied
to an off-the-shelf NeRF algorithm for novel view synthesis. Rolling shutter
effect would also affect the accuracy of the camera pose estimation (e.g. via
COLMAP), which further prevents the success of NeRF algorithm with RS images.
In this paper, we propose Unrolling Shutter Bundle Adjusted Neural Radiance
Fields (USB-NeRF). USB-NeRF is able to correct rolling shutter distortions and
recover accurate camera motion trajectory simultaneously under the framework of
NeRF, by modeling the physical image formation process of a RS camera.
Experimental results demonstrate that USB-NeRF achieves better performance
compared to prior works, in terms of RS effect removal, novel view image
synthesis as well as camera motion estimation. Furthermore, our algorithm can
also be used to recover high-fidelity high frame-rate global shutter video from
a sequence of RS images.

---

## Adaptive Multi-NeRF: Exploit Efficient Parallelism in Adaptive Multiple  Scale Neural Radiance Field Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-03 | Tong Wang, Shuichi Kurabayashi | cs.CV | [PDF](http://arxiv.org/pdf/2310.01881v1){: .btn .btn-green } |

**Abstract**: Recent advances in Neural Radiance Fields (NeRF) have demonstrated
significant potential for representing 3D scene appearances as implicit neural
networks, enabling the synthesis of high-fidelity novel views. However, the
lengthy training and rendering process hinders the widespread adoption of this
promising technique for real-time rendering applications. To address this
issue, we present an effective adaptive multi-NeRF method designed to
accelerate the neural rendering process for large scenes with unbalanced
workloads due to varying scene complexities.
  Our method adaptively subdivides scenes into axis-aligned bounding boxes
using a tree hierarchy approach, assigning smaller NeRFs to different-sized
subspaces based on the complexity of each scene portion. This ensures the
underlying neural representation is specific to a particular part of the scene.
We optimize scene subdivision by employing a guidance density grid, which
balances representation capability for each Multilayer Perceptron (MLP).
Consequently, samples generated by each ray can be sorted and collected for
parallel inference, achieving a balanced workload suitable for small MLPs with
consistent dimensions for regular and GPU-friendly computations. We aosl
demonstrated an efficient NeRF sampling strategy that intrinsically adapts to
increase parallelism, utilization, and reduce kernel calls, thereby achieving
much higher GPU utilization and accelerating the rendering process.

---

## MIMO-NeRF: Fast Neural Rendering with Multi-input Multi-output Neural  Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-03 | Takuhiro Kaneko | cs.CV | [PDF](http://arxiv.org/pdf/2310.01821v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) have shown impressive results for novel view
synthesis. However, they depend on the repetitive use of a single-input
single-output multilayer perceptron (SISO MLP) that maps 3D coordinates and
view direction to the color and volume density in a sample-wise manner, which
slows the rendering. We propose a multi-input multi-output NeRF (MIMO-NeRF)
that reduces the number of MLPs running by replacing the SISO MLP with a MIMO
MLP and conducting mappings in a group-wise manner. One notable challenge with
this approach is that the color and volume density of each point can differ
according to a choice of input coordinates in a group, which can lead to some
notable ambiguity. We also propose a self-supervised learning method that
regularizes the MIMO MLP with multiple fast reformulated MLPs to alleviate this
ambiguity without using pretrained models. The results of a comprehensive
experimental evaluation including comparative and ablation studies are
presented to show that MIMO-NeRF obtains a good trade-off between speed and
quality with a reasonable training time. We then demonstrate that MIMO-NeRF is
compatible with and complementary to previous advancements in NeRFs by applying
it to two representative fast NeRFs, i.e., a NeRF with sample reduction
(DONeRF) and a NeRF with alternative representations (TensoRF).

Comments:
- Accepted to ICCV 2023. Project page:
  https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/mimo-nerf/

---

## EvDNeRF: Reconstructing Event Data with Dynamic Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-03 | Anish Bhattacharya, Ratnesh Madaan, Fernando Cladera, Sai Vemprala, Rogerio Bonatti, Kostas Daniilidis, Ashish Kapoor, Vijay Kumar, Nikolai Matni, Jayesh K. Gupta | cs.CV | [PDF](http://arxiv.org/pdf/2310.02437v2){: .btn .btn-green } |

**Abstract**: We present EvDNeRF, a pipeline for generating event data and training an
event-based dynamic NeRF, for the purpose of faithfully reconstructing
eventstreams on scenes with rigid and non-rigid deformations that may be too
fast to capture with a standard camera. Event cameras register asynchronous
per-pixel brightness changes at MHz rates with high dynamic range, making them
ideal for observing fast motion with almost no motion blur. Neural radiance
fields (NeRFs) offer visual-quality geometric-based learnable rendering, but
prior work with events has only considered reconstruction of static scenes. Our
EvDNeRF can predict eventstreams of dynamic scenes from a static or moving
viewpoint between any desired timestamps, thereby allowing it to be used as an
event-based simulator for a given scene. We show that by training on varied
batch sizes of events, we can improve test-time predictions of events at fine
time resolutions, outperforming baselines that pair standard dynamic NeRFs with
event generators. We release our simulated and real datasets, as well as code
for multi-view event-based data generation and the training and evaluation of
EvDNeRF models (https://github.com/anish-bhattacharya/EvDNeRF).

Comments:
- 16 pages, 20 figures, 2 tables

---

## PC-NeRF: Parent-Child Neural Radiance Fields under Partial Sensor Data  Loss in Autonomous Driving Environments

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-02 | Xiuzhong Hu, Guangming Xiong, Zheng Zang, Peng Jia, Yuxuan Han, Junyi Ma | cs.CV | [PDF](http://arxiv.org/pdf/2310.00874v1){: .btn .btn-green } |

**Abstract**: Reconstructing large-scale 3D scenes is essential for autonomous vehicles,
especially when partial sensor data is lost. Although the recently developed
neural radiance fields (NeRF) have shown compelling results in implicit
representations, the large-scale 3D scene reconstruction using partially lost
LiDAR point cloud data still needs to be explored. To bridge this gap, we
propose a novel 3D scene reconstruction framework called parent-child neural
radiance field (PC-NeRF). The framework comprises two modules, the parent NeRF
and the child NeRF, to simultaneously optimize scene-level, segment-level, and
point-level scene representations. Sensor data can be utilized more efficiently
by leveraging the segment-level representation capabilities of child NeRFs, and
an approximate volumetric representation of the scene can be quickly obtained
even with limited observations. With extensive experiments, our proposed
PC-NeRF is proven to achieve high-precision 3D reconstruction in large-scale
scenes. Moreover, PC-NeRF can effectively tackle situations where partial
sensor data is lost and has high deployment efficiency with limited training
time. Our approach implementation and the pre-trained models will be available
at https://github.com/biter0088/pc-nerf.

---

## How Many Views Are Needed to Reconstruct an Unknown Object Using NeRF?

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-01 | Sicong Pan, Liren Jin, Hao Hu, Marija Popović, Maren Bennewitz | cs.RO | [PDF](http://arxiv.org/pdf/2310.00684v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) are gaining significant interest for online
active object reconstruction due to their exceptional memory efficiency and
requirement for only posed RGB inputs. Previous NeRF-based view planning
methods exhibit computational inefficiency since they rely on an iterative
paradigm, consisting of (1) retraining the NeRF when new images arrive; and (2)
planning a path to the next best view only. To address these limitations, we
propose a non-iterative pipeline based on the Prediction of the Required number
of Views (PRV). The key idea behind our approach is that the required number of
views to reconstruct an object depends on its complexity. Therefore, we design
a deep neural network, named PRVNet, to predict the required number of views,
allowing us to tailor the data acquisition based on the object complexity and
plan a globally shortest path. To train our PRVNet, we generate supervision
labels using the ShapeNet dataset. Simulated experiments show that our
PRV-based view planning method outperforms baselines, achieving good
reconstruction quality while significantly reducing movement cost and planning
time. We further justify the generalization ability of our approach in a
real-world experiment.

Comments:
- Submitted to ICRA 2024

---

## Multi-tiling Neural Radiance Field (NeRF) -- Geometric Assessment on  Large-scale Aerial Datasets

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-10-01 | Ningli Xu, Rongjun Qin, Debao Huang, Fabio Remondino | cs.CV | [PDF](http://arxiv.org/pdf/2310.00530v3){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) offer the potential to benefit 3D
reconstruction tasks, including aerial photogrammetry. However, the scalability
and accuracy of the inferred geometry are not well-documented for large-scale
aerial assets,since such datasets usually result in very high memory
consumption and slow convergence.. In this paper, we aim to scale the NeRF on
large-scael aerial datasets and provide a thorough geometry assessment of NeRF.
Specifically, we introduce a location-specific sampling technique as well as a
multi-camera tiling (MCT) strategy to reduce memory consumption during image
loading for RAM, representation training for GPU memory, and increase the
convergence rate within tiles. MCT decomposes a large-frame image into multiple
tiled images with different camera models, allowing these small-frame images to
be fed into the training process as needed for specific locations without a
loss of accuracy. We implement our method on a representative approach,
Mip-NeRF, and compare its geometry performance with threephotgrammetric MVS
pipelines on two typical aerial datasets against LiDAR reference data. Both
qualitative and quantitative results suggest that the proposed NeRF approach
produces better completeness and object details than traditional approaches,
although as of now, it still falls short in terms of accuracy.

Comments:
- 9 Figure