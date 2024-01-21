---
layout: default
title: November
parent: 2023
nav_order: 11
---
<!---metadata--->

## Contrastive Denoising Score for Text-guided Latent Diffusion Image  Editing

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Hyelin Nam, Gihyun Kwon, Geon Yeong Park, Jong Chul Ye | cs.CV | [PDF](http://arxiv.org/pdf/2311.18608v1){: .btn .btn-green } |

**Abstract**: With the remarkable advent of text-to-image diffusion models, image editing
methods have become more diverse and continue to evolve. A promising recent
approach in this realm is Delta Denoising Score (DDS) - an image editing
technique based on Score Distillation Sampling (SDS) framework that leverages
the rich generative prior of text-to-image diffusion models. However, relying
solely on the difference between scoring functions is insufficient for
preserving specific structural elements from the original image, a crucial
aspect of image editing. Inspired by the similarity and importance differences
between DDS and the contrastive learning for unpaired image-to-image
translation (CUT), here we present an embarrassingly simple yet very powerful
modification of DDS, called Contrastive Denoising Score (CDS), for latent
diffusion models (LDM). Specifically, to enforce structural correspondence
between the input and output while maintaining the controllability of contents,
we introduce a straightforward approach to regulate structural consistency
using CUT loss within the DDS framework. To calculate this loss, instead of
employing auxiliary networks, we utilize the intermediate features of LDM, in
particular, those from the self-attention layers, which possesses rich spatial
information. Our approach enables zero-shot image-to-image translation and
neural radiance field (NeRF) editing, achieving a well-balanced interplay
between maintaining the structural details and transforming content.
Qualitative results and comparisons demonstrates the effectiveness of our
proposed method. Project page with code is available at
https://hyelinnam.github.io/CDS/.

Comments:
- Project page: https://hyelinnam.github.io/CDS/

---

## Compact3D: Compressing Gaussian Splat Radiance Field Models with Vector  Quantization

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | KL Navaneet, Kossar Pourahmadi Meibodi, Soroush Abbasi Koohpayegani, Hamed Pirsiavash | cs.CV | [PDF](http://arxiv.org/pdf/2311.18159v1){: .btn .btn-green } |

**Abstract**: 3D Gaussian Splatting is a new method for modeling and rendering 3D radiance
fields that achieves much faster learning and rendering time compared to SOTA
NeRF methods. However, it comes with a drawback in the much larger storage
demand compared to NeRF methods since it needs to store the parameters for
several 3D Gaussians. We notice that many Gaussians may share similar
parameters, so we introduce a simple vector quantization method based on
\kmeans algorithm to quantize the Gaussian parameters. Then, we store the small
codebook along with the index of the code for each Gaussian. Moreover, we
compress the indices further by sorting them and using a method similar to
run-length encoding. We do extensive experiments on standard benchmarks as well
as a new benchmark which is an order of magnitude larger than the standard
benchmarks. We show that our simple yet effective method can reduce the storage
cost for the original 3D Gaussian Splatting method by a factor of almost
$20\times$ with a very small drop in the quality of rendered images.

Comments:
- Code is available at https://github.com/UCDvision/compact3d

---

## CosAvatar: Consistent and Animatable Portrait Video Tuning with Text  Prompt

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Haiyao Xiao, Chenglai Zhong, Xuan Gao, Yudong Guo, Juyong Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2311.18288v1){: .btn .btn-green } |

**Abstract**: Recently, text-guided digital portrait editing has attracted more and more
attentions. However, existing methods still struggle to maintain consistency
across time, expression, and view or require specific data prerequisites. To
solve these challenging problems, we propose CosAvatar, a high-quality and
user-friendly framework for portrait tuning. With only monocular video and text
instructions as input, we can produce animatable portraits with both temporal
and 3D consistency. Different from methods that directly edit in the 2D domain,
we employ a dynamic NeRF-based 3D portrait representation to model both the
head and torso. We alternate between editing the video frames' dataset and
updating the underlying 3D portrait until the edited frames reach 3D
consistency. Additionally, we integrate the semantic portrait priors to enhance
the edited results, allowing precise modifications in specified semantic areas.
Extensive results demonstrate that our proposed method can not only accurately
edit portrait styles or local attributes based on text instructions but also
support expressive animation driven by a source video.

Comments:
- Project page: https://ustc3dv.github.io/CosAvatar/

---

## ZeST-NeRF: Using temporal aggregation for Zero-Shot Temporal NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Violeta Menéndez González, Andrew Gilbert, Graeme Phillipson, Stephen Jolly, Simon Hadfield | cs.CV | [PDF](http://arxiv.org/pdf/2311.18491v1){: .btn .btn-green } |

**Abstract**: In the field of media production, video editing techniques play a pivotal
role. Recent approaches have had great success at performing novel view image
synthesis of static scenes. But adding temporal information adds an extra layer
of complexity. Previous models have focused on implicitly representing static
and dynamic scenes using NeRF. These models achieve impressive results but are
costly at training and inference time. They overfit an MLP to describe the
scene implicitly as a function of position. This paper proposes ZeST-NeRF, a
new approach that can produce temporal NeRFs for new scenes without retraining.
We can accurately reconstruct novel views using multi-view synthesis techniques
and scene flow-field estimation, trained only with unrelated scenes. We
demonstrate how existing state-of-the-art approaches from a range of fields
cannot adequately solve this new task and demonstrate the efficacy of our
solution. The resulting network improves quantitatively by 15% and produces
significantly better visual results.

Comments:
- VUA BMVC 2023

---

## Periodic Vibration Gaussian: Dynamic Urban Scene Reconstruction and  Real-time Rendering

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Yurui Chen, Chun Gu, Junzhe Jiang, Xiatian Zhu, Li Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2311.18561v1){: .btn .btn-green } |

**Abstract**: Modeling dynamic, large-scale urban scenes is challenging due to their highly
intricate geometric structures and unconstrained dynamics in both space and
time. Prior methods often employ high-level architectural priors, separating
static and dynamic elements, resulting in suboptimal capture of their
synergistic interactions. To address this challenge, we present a unified
representation model, called Periodic Vibration Gaussian (PVG). PVG builds upon
the efficient 3D Gaussian splatting technique, originally designed for static
scene representation, by introducing periodic vibration-based temporal
dynamics. This innovation enables PVG to elegantly and uniformly represent the
characteristics of various objects and elements in dynamic urban scenes. To
enhance temporally coherent representation learning with sparse training data,
we introduce a novel flow-based temporal smoothing mechanism and a
position-aware adaptive control strategy. Extensive experiments on Waymo Open
Dataset and KITTI benchmarks demonstrate that PVG surpasses state-of-the-art
alternatives in both reconstruction and novel view synthesis for both dynamic
and static scenes. Notably, PVG achieves this without relying on manually
labeled object bounding boxes or expensive optical flow estimation. Moreover,
PVG exhibits 50/6000-fold acceleration in training/rendering over the best
alternative.

Comments:
- Project page: https://fudan-zvg.github.io/PVG/

---

## Redefining Recon: Bridging Gaps with UAVs, 360 degree Cameras, and  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Hartmut Surmann, Niklas Digakis, Jan-Nicklas Kremer, Julien Meine, Max Schulte, Niklas Voigt | cs.CV | [PDF](http://arxiv.org/pdf/2401.06143v1){: .btn .btn-green } |

**Abstract**: In the realm of digital situational awareness during disaster situations,
accurate digital representations, like 3D models, play an indispensable role.
To ensure the safety of rescue teams, robotic platforms are often deployed to
generate these models. In this paper, we introduce an innovative approach that
synergizes the capabilities of compact Unmaned Arial Vehicles (UAVs), smaller
than 30 cm, equipped with 360 degree cameras and the advances of Neural
Radiance Fields (NeRFs). A NeRF, a specialized neural network, can deduce a 3D
representation of any scene using 2D images and then synthesize it from various
angles upon request. This method is especially tailored for urban environments
which have experienced significant destruction, where the structural integrity
of buildings is compromised to the point of barring entry-commonly observed
post-earthquakes and after severe fires. We have tested our approach through
recent post-fire scenario, underlining the efficacy of NeRFs even in
challenging outdoor environments characterized by water, snow, varying light
conditions, and reflective surfaces.

Comments:
- 6 pages, published at IEEE International Symposium on
  Safety,Security,and Rescue Robotics SSRR2023 in FUKUSHIMA, November 13-15
  2023

---

## Anisotropic Neural Representation Learning for High-Quality Neural  Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Y. Wang, J. Xu, Y. Zeng, Y. Gong | cs.CV | [PDF](http://arxiv.org/pdf/2311.18311v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) have achieved impressive view synthesis
results by learning an implicit volumetric representation from multi-view
images. To project the implicit representation into an image, NeRF employs
volume rendering that approximates the continuous integrals of rays as an
accumulation of the colors and densities of the sampled points. Although this
approximation enables efficient rendering, it ignores the direction information
in point intervals, resulting in ambiguous features and limited reconstruction
quality. In this paper, we propose an anisotropic neural representation
learning method that utilizes learnable view-dependent features to improve
scene representation and reconstruction. We model the volumetric function as
spherical harmonic (SH)-guided anisotropic features, parameterized by
multilayer perceptrons, facilitating ambiguity elimination while preserving the
rendering efficiency. To achieve robust scene reconstruction without anisotropy
overfitting, we regularize the energy of the anisotropic features during
training. Our method is flexiable and can be plugged into NeRF-based
frameworks. Extensive experiments show that the proposed representation can
boost the rendering quality of various NeRFs and achieve state-of-the-art
rendering performance on both synthetic and real-world scenes.

---

## Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, Bo Dai | cs.CV | [PDF](http://arxiv.org/pdf/2312.00109v1){: .btn .btn-green } |

**Abstract**: Neural rendering methods have significantly advanced photo-realistic 3D scene
rendering in various academic and industrial applications. The recent 3D
Gaussian Splatting method has achieved the state-of-the-art rendering quality
and speed combining the benefits of both primitive-based representations and
volumetric representations. However, it often leads to heavily redundant
Gaussians that try to fit every training view, neglecting the underlying scene
geometry. Consequently, the resulting model becomes less robust to significant
view changes, texture-less area and lighting effects. We introduce Scaffold-GS,
which uses anchor points to distribute local 3D Gaussians, and predicts their
attributes on-the-fly based on viewing direction and distance within the view
frustum. Anchor growing and pruning strategies are developed based on the
importance of neural Gaussians to reliably improve the scene coverage. We show
that our method effectively reduces redundant Gaussians while delivering
high-quality rendering. We also demonstrates an enhanced capability to
accommodate scenes with varying levels-of-detail and view-dependent
observations, without sacrificing the rendering speed.

Comments:
- Project page: https://city-super.github.io/scaffold-gs/

---

## LucidDreaming: Controllable Object-Centric 3D Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Zhaoning Wang, Ming Li, Chen Chen | cs.CV | [PDF](http://arxiv.org/pdf/2312.00588v1){: .btn .btn-green } |

**Abstract**: With the recent development of generative models, Text-to-3D generations have
also seen significant growth. Nonetheless, achieving precise control over 3D
generation continues to be an arduous task, as using text to control often
leads to missing objects and imprecise locations. Contemporary strategies for
enhancing controllability in 3D generation often entail the introduction of
additional parameters, such as customized diffusion models. This often induces
hardness in adapting to different diffusion models or creating distinct
objects.
  In this paper, we present LucidDreaming as an effective pipeline capable of
fine-grained control over 3D generation. It requires only minimal input of 3D
bounding boxes, which can be deduced from a simple text prompt using a Large
Language Model. Specifically, we propose clipped ray sampling to separately
render and optimize objects with user specifications. We also introduce
object-centric density blob bias, fostering the separation of generated
objects. With individual rendering and optimizing of objects, our method excels
not only in controlled content generation from scratch but also within the
pre-trained NeRF scenes. In such scenarios, existing generative approaches
often disrupt the integrity of the original scene, and current editing methods
struggle to synthesize new content in empty spaces. We show that our method
exhibits remarkable adaptability across a spectrum of mainstream Score
Distillation Sampling-based 3D generation frameworks, and achieves superior
alignment of 3D content when compared to baseline approaches. We also provide a
dataset of prompts with 3D bounding boxes, benchmarking 3D spatial
controllability.

---

## MD-Splatting: Learning Metric Deformation from 4D Gaussians in Highly  Deformable Scenes

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Bardienus P. Duisterhof, Zhao Mandi, Yunchao Yao, Jia-Wei Liu, Mike Zheng Shou, Shuran Song, Jeffrey Ichnowski | cs.CV | [PDF](http://arxiv.org/pdf/2312.00583v1){: .btn .btn-green } |

**Abstract**: Accurate 3D tracking in highly deformable scenes with occlusions and shadows
can facilitate new applications in robotics, augmented reality, and generative
AI. However, tracking under these conditions is extremely challenging due to
the ambiguity that arises with large deformations, shadows, and occlusions. We
introduce MD-Splatting, an approach for simultaneous 3D tracking and novel view
synthesis, using video captures of a dynamic scene from various camera poses.
MD-Splatting builds on recent advances in Gaussian splatting, a method that
learns the properties of a large number of Gaussians for state-of-the-art and
fast novel view synthesis. MD-Splatting learns a deformation function to
project a set of Gaussians with non-metric, thus canonical, properties into
metric space. The deformation function uses a neural-voxel encoding and a
multilayer perceptron (MLP) to infer Gaussian position, rotation, and a shadow
scalar. We enforce physics-inspired regularization terms based on local
rigidity, conservation of momentum, and isometry, which leads to trajectories
with smaller trajectory errors. MD-Splatting achieves high-quality 3D tracking
on highly deformable scenes with shadows and occlusions. Compared to
state-of-the-art, we improve 3D tracking by an average of 23.9 %, while
simultaneously achieving high-quality novel view synthesis. With sufficient
texture such as in scene 6, MD-Splatting achieves a median tracking error of
3.39 mm on a cloth of 1 x 1 meters in size. Project website:
https://md-splatting.github.io/.

---

## DynMF: Neural Motion Factorization for Real-time Dynamic View Synthesis  with 3D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Agelos Kratimenos, Jiahui Lei, Kostas Daniilidis | cs.CV | [PDF](http://arxiv.org/pdf/2312.00112v1){: .btn .btn-green } |

**Abstract**: Accurately and efficiently modeling dynamic scenes and motions is considered
so challenging a task due to temporal dynamics and motion complexity. To
address these challenges, we propose DynMF, a compact and efficient
representation that decomposes a dynamic scene into a few neural trajectories.
We argue that the per-point motions of a dynamic scene can be decomposed into a
small set of explicit or learned trajectories. Our carefully designed neural
framework consisting of a tiny set of learned basis queried only in time allows
for rendering speed similar to 3D Gaussian Splatting, surpassing 120 FPS, while
at the same time, requiring only double the storage compared to static scenes.
Our neural representation adequately constrains the inherently underconstrained
motion field of a dynamic scene leading to effective and fast optimization.
This is done by biding each point to motion coefficients that enforce the
per-point sharing of basis trajectories. By carefully applying a sparsity loss
to the motion coefficients, we are able to disentangle the motions that
comprise the scene, independently control them, and generate novel motion
combinations that have never been seen before. We can reach state-of-the-art
render quality within just 5 minutes of training and in less than half an hour,
we can synthesize novel views of dynamic scenes with superior photorealistic
quality. Our representation is interpretable, efficient, and expressive enough
to offer real-time view synthesis of complex dynamic scene motions, in
monocular and multi-view scenarios.

Comments:
- Project page: https://agelosk.github.io/dynmf/

---

## SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian  Splatting

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay, Pradyumna Chari, Achuta Kadambi | cs.CV | [PDF](http://arxiv.org/pdf/2312.00206v1){: .btn .btn-green } |

**Abstract**: The problem of novel view synthesis has grown significantly in popularity
recently with the introduction of Neural Radiance Fields (NeRFs) and other
implicit scene representation methods. A recent advance, 3D Gaussian Splatting
(3DGS), leverages an explicit representation to achieve real-time rendering
with high-quality results. However, 3DGS still requires an abundance of
training views to generate a coherent scene representation. In few shot
settings, similar to NeRF, 3DGS tends to overfit to training views, causing
background collapse and excessive floaters, especially as the number of
training views are reduced. We propose a method to enable training coherent
3DGS-based radiance fields of 360 scenes from sparse training views. We find
that using naive depth priors is not sufficient and integrate depth priors with
generative and explicit constraints to reduce background collapse, remove
floaters, and enhance consistency from unseen viewpoints. Experiments show that
our method outperforms base 3DGS by up to 30.5% and NeRF-based methods by up to
15.6% in LPIPS on the MipNeRF-360 dataset with substantially less training and
inference cost.

Comments:
- The main text spans eight pages, followed by two pages of references
  and four pages of supplementary materials

---

## PyNeRF: Pyramidal Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-30 | Haithem Turki, Michael Zollhöfer, Christian Richardt, Deva Ramanan | cs.CV | [PDF](http://arxiv.org/pdf/2312.00252v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) can be dramatically accelerated by spatial
grid representations. However, they do not explicitly reason about scale and so
introduce aliasing artifacts when reconstructing scenes captured at different
camera distances. Mip-NeRF and its extensions propose scale-aware renderers
that project volumetric frustums rather than point samples but such approaches
rely on positional encodings that are not readily compatible with grid methods.
We propose a simple modification to grid-based models by training model heads
at different spatial grid resolutions. At render time, we simply use coarser
grids to render samples that cover larger volumes. Our method can be easily
applied to existing accelerated NeRF methods and significantly improves
rendering quality (reducing error rates by 20-90% across synthetic and
unbounded real-world scenes) while incurring minimal performance overhead (as
each model head is quick to evaluate). Compared to Mip-NeRF, we reduce error
rates by 20% while training over 60x faster.

Comments:
- Neurips 2023 Project page: https://haithemturki.com/pynerf/

---

## GaussianShader: 3D Gaussian Splatting with Shading Functions for  Reflective Surfaces

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-29 | Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, Yuexin Ma | cs.CV | [PDF](http://arxiv.org/pdf/2311.17977v1){: .btn .btn-green } |

**Abstract**: The advent of neural 3D Gaussians has recently brought about a revolution in
the field of neural rendering, facilitating the generation of high-quality
renderings at real-time speeds. However, the explicit and discrete
representation encounters challenges when applied to scenes featuring
reflective surfaces. In this paper, we present GaussianShader, a novel method
that applies a simplified shading function on 3D Gaussians to enhance the
neural rendering in scenes with reflective surfaces while preserving the
training and rendering efficiency. The main challenge in applying the shading
function lies in the accurate normal estimation on discrete 3D Gaussians.
Specifically, we proposed a novel normal estimation framework based on the
shortest axis directions of 3D Gaussians with a delicately designed loss to
make the consistency between the normals and the geometries of Gaussian
spheres. Experiments show that GaussianShader strikes a commendable balance
between efficiency and visual quality. Our method surpasses Gaussian Splatting
in PSNR on specular object datasets, exhibiting an improvement of 1.57dB. When
compared to prior works handling reflective surfaces, such as Ref-NeRF, our
optimization time is significantly accelerated (23h vs. 0.58h). Please click on
our project website to see more results.

Comments:
- 13 pages, 11 figures, refrences added

---

## NeRFTAP: Enhancing Transferability of Adversarial Patches on Face  Recognition using Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-29 | Xiaoliang Liu, Furao Shen, Feng Han, Jian Zhao, Changhai Nie | cs.CV | [PDF](http://arxiv.org/pdf/2311.17332v1){: .btn .btn-green } |

**Abstract**: Face recognition (FR) technology plays a crucial role in various
applications, but its vulnerability to adversarial attacks poses significant
security concerns. Existing research primarily focuses on transferability to
different FR models, overlooking the direct transferability to victim's face
images, which is a practical threat in real-world scenarios. In this study, we
propose a novel adversarial attack method that considers both the
transferability to the FR model and the victim's face image, called NeRFTAP.
Leveraging NeRF-based 3D-GAN, we generate new view face images for the source
and target subjects to enhance transferability of adversarial patches. We
introduce a style consistency loss to ensure the visual similarity between the
adversarial UV map and the target UV map under a 0-1 mask, enhancing the
effectiveness and naturalness of the generated adversarial face images.
Extensive experiments and evaluations on various FR models demonstrate the
superiority of our approach over existing attack techniques. Our work provides
valuable insights for enhancing the robustness of FR systems in practical
adversarial settings.

---

## SyncTalk: The Devil is in the Synchronization for Talking Head Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-29 | Ziqiao Peng, Wentao Hu, Yue Shi, Xiangyu Zhu, Xiaomei Zhang, Hao Zhao, Jun He, Hongyan Liu, Zhaoxin Fan | cs.CV | [PDF](http://arxiv.org/pdf/2311.17590v1){: .btn .btn-green } |

**Abstract**: Achieving high synchronization in the synthesis of realistic, speech-driven
talking head videos presents a significant challenge. Traditional Generative
Adversarial Networks (GAN) struggle to maintain consistent facial identity,
while Neural Radiance Fields (NeRF) methods, although they can address this
issue, often produce mismatched lip movements, inadequate facial expressions,
and unstable head poses. A lifelike talking head requires synchronized
coordination of subject identity, lip movements, facial expressions, and head
poses. The absence of these synchronizations is a fundamental flaw, leading to
unrealistic and artificial outcomes. To address the critical issue of
synchronization, identified as the "devil" in creating realistic talking heads,
we introduce SyncTalk. This NeRF-based method effectively maintains subject
identity, enhancing synchronization and realism in talking head synthesis.
SyncTalk employs a Face-Sync Controller to align lip movements with speech and
innovatively uses a 3D facial blendshape model to capture accurate facial
expressions. Our Head-Sync Stabilizer optimizes head poses, achieving more
natural head movements. The Portrait-Sync Generator restores hair details and
blends the generated head with the torso for a seamless visual experience.
Extensive experiments and user studies demonstrate that SyncTalk outperforms
state-of-the-art methods in synchronization and realism. We recommend watching
the supplementary video: https://ziqiaopeng.github.io/synctalk

Comments:
- 11 pages, 5 figures

---

## Cinematic Behavior Transfer via NeRF-based Differentiable Filming

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-29 | Xuekun Jiang, Anyi Rao, Jingbo Wang, Dahua Lin, Bo Dai | cs.CV | [PDF](http://arxiv.org/pdf/2311.17754v1){: .btn .btn-green } |

**Abstract**: In the evolving landscape of digital media and video production, the precise
manipulation and reproduction of visual elements like camera movements and
character actions are highly desired. Existing SLAM methods face limitations in
dynamic scenes and human pose estimation often focuses on 2D projections,
neglecting 3D statuses. To address these issues, we first introduce a reverse
filming behavior estimation technique. It optimizes camera trajectories by
leveraging NeRF as a differentiable renderer and refining SMPL tracks. We then
introduce a cinematic transfer pipeline that is able to transfer various shot
types to a new 2D video or a 3D virtual environment. The incorporation of 3D
engine workflow enables superior rendering and control abilities, which also
achieves a higher rating in the user study.

Comments:
- Project Page:
  https://virtualfilmstudio.github.io/projects/cinetransfer

---

## HUGS: Human Gaussian Splats

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-29 | Muhammed Kocabas, Jen-Hao Rick Chang, James Gabriel, Oncel Tuzel, Anurag Ranjan | cs.CV | [PDF](http://arxiv.org/pdf/2311.17910v1){: .btn .btn-green } |

**Abstract**: Recent advances in neural rendering have improved both training and rendering
times by orders of magnitude. While these methods demonstrate state-of-the-art
quality and speed, they are designed for photogrammetry of static scenes and do
not generalize well to freely moving humans in the environment. In this work,
we introduce Human Gaussian Splats (HUGS) that represents an animatable human
together with the scene using 3D Gaussian Splatting (3DGS). Our method takes
only a monocular video with a small number of (50-100) frames, and it
automatically learns to disentangle the static scene and a fully animatable
human avatar within 30 minutes. We utilize the SMPL body model to initialize
the human Gaussians. To capture details that are not modeled by SMPL (e.g.
cloth, hairs), we allow the 3D Gaussians to deviate from the human body model.
Utilizing 3D Gaussians for animated humans brings new challenges, including the
artifacts created when articulating the Gaussians. We propose to jointly
optimize the linear blend skinning weights to coordinate the movements of
individual Gaussians during animation. Our approach enables novel-pose
synthesis of human and novel view synthesis of both the human and the scene. We
achieve state-of-the-art rendering quality with a rendering speed of 60 FPS
while being ~100x faster to train over previous work. Our code will be
announced here: https://github.com/apple/ml-hugs

---

## FisherRF: Active View Selection and Uncertainty Quantification for  Radiance Fields using Fisher Information

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-29 | Wen Jiang, Boshu Lei, Kostas Daniilidis | cs.CV | [PDF](http://arxiv.org/pdf/2311.17874v1){: .btn .btn-green } |

**Abstract**: This study addresses the challenging problem of active view selection and
uncertainty quantification within the domain of Radiance Fields. Neural
Radiance Fields (NeRF) have greatly advanced image rendering and
reconstruction, but the limited availability of 2D images poses uncertainties
stemming from occlusions, depth ambiguities, and imaging errors. Efficiently
selecting informative views becomes crucial, and quantifying NeRF model
uncertainty presents intricate challenges. Existing approaches either depend on
model architecture or are based on assumptions regarding density distributions
that are not generally applicable. By leveraging Fisher Information, we
efficiently quantify observed information within Radiance Fields without ground
truth data. This can be used for the next best view selection and pixel-wise
uncertainty quantification. Our method overcomes existing limitations on model
architecture and effectiveness, achieving state-of-the-art results in both view
selection and uncertainty quantification, demonstrating its potential to
advance the field of Radiance Fields. Our method with the 3D Gaussian Splatting
backend could perform view selections at 70 fps.

Comments:
- Project page: https://jiangwenpl.github.io/FisherRF/

---

## AvatarStudio: High-fidelity and Animatable 3D Avatar Creation from Text

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-29 | Jianfeng Zhang, Xuanmeng Zhang, Huichao Zhang, Jun Hao Liew, Chenxu Zhang, Yi Yang, Jiashi Feng | cs.GR | [PDF](http://arxiv.org/pdf/2311.17917v1){: .btn .btn-green } |

**Abstract**: We study the problem of creating high-fidelity and animatable 3D avatars from
only textual descriptions. Existing text-to-avatar methods are either limited
to static avatars which cannot be animated or struggle to generate animatable
avatars with promising quality and precise pose control. To address these
limitations, we propose AvatarStudio, a coarse-to-fine generative model that
generates explicit textured 3D meshes for animatable human avatars.
Specifically, AvatarStudio begins with a low-resolution NeRF-based
representation for coarse generation, followed by incorporating SMPL-guided
articulation into the explicit mesh representation to support avatar animation
and high resolution rendering. To ensure view consistency and pose
controllability of the resulting avatars, we introduce a 2D diffusion model
conditioned on DensePose for Score Distillation Sampling supervision. By
effectively leveraging the synergy between the articulated mesh representation
and the DensePose-conditional diffusion model, AvatarStudio can create
high-quality avatars from text that are ready for animation, significantly
outperforming previous methods. Moreover, it is competent for many
applications, e.g., multimodal avatar animations and style-guided avatar
creation. For more results, please refer to our project page:
http://jeff95.me/projects/avatarstudio.html

Comments:
- Project page at http://jeff95.me/projects/avatarstudio.html

---

## CG3D: Compositional Generation for Text-to-3D via Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-29 | Alexander Vilesov, Pradyumna Chari, Achuta Kadambi | cs.CV | [PDF](http://arxiv.org/pdf/2311.17907v1){: .btn .btn-green } |

**Abstract**: With the onset of diffusion-based generative models and their ability to
generate text-conditioned images, content generation has received a massive
invigoration. Recently, these models have been shown to provide useful guidance
for the generation of 3D graphics assets. However, existing work in
text-conditioned 3D generation faces fundamental constraints: (i) inability to
generate detailed, multi-object scenes, (ii) inability to textually control
multi-object configurations, and (iii) physically realistic scene composition.
In this work, we propose CG3D, a method for compositionally generating scalable
3D assets that resolves these constraints. We find that explicit Gaussian
radiance fields, parameterized to allow for compositions of objects, possess
the capability to enable semantically and physically consistent scenes. By
utilizing a guidance framework built around this explicit representation, we
show state of the art results, capable of even exceeding the guiding diffusion
model in terms of object combinations and physics accuracy.

---

## Point'n Move: Interactive Scene Object Manipulation on Gaussian  Splatting Radiance Fields

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Jiajun Huang, Hongchuan Yu | cs.CV | [PDF](http://arxiv.org/pdf/2311.16737v1){: .btn .btn-green } |

**Abstract**: We propose Point'n Move, a method that achieves interactive scene object
manipulation with exposed region inpainting. Interactivity here further comes
from intuitive object selection and real-time editing. To achieve this, we
adopt Gaussian Splatting Radiance Field as the scene representation and fully
leverage its explicit nature and speed advantage. Its explicit representation
formulation allows us to devise a 2D prompt points to 3D mask dual-stage
self-prompting segmentation algorithm, perform mask refinement and merging,
minimize change as well as provide good initialization for scene inpainting and
perform editing in real-time without per-editing training, all leads to
superior quality and performance. We test our method by performing editing on
both forward-facing and 360 scenes. We also compare our method against existing
scene object removal methods, showing superior quality despite being more
capable and having a speed advantage.

---

## Multi-Scale 3D Gaussian Splatting for Anti-Aliased Rendering

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Zhiwen Yan, Weng Fei Low, Yu Chen, Gim Hee Lee | cs.CV | [PDF](http://arxiv.org/pdf/2311.17089v1){: .btn .btn-green } |

**Abstract**: 3D Gaussians have recently emerged as a highly efficient representation for
3D reconstruction and rendering. Despite its high rendering quality and speed
at high resolutions, they both deteriorate drastically when rendered at lower
resolutions or from far away camera position. During low resolution or far away
rendering, the pixel size of the image can fall below the Nyquist frequency
compared to the screen size of each splatted 3D Gaussian and leads to aliasing
effect. The rendering is also drastically slowed down by the sequential alpha
blending of more splatted Gaussians per pixel. To address these issues, we
propose a multi-scale 3D Gaussian splatting algorithm, which maintains
Gaussians at different scales to represent the same scene. Higher-resolution
images are rendered with more small Gaussians, and lower-resolution images are
rendered with fewer larger Gaussians. With similar training time, our algorithm
can achieve 13\%-66\% PSNR and 160\%-2400\% rendering speed improvement at
4$\times$-128$\times$ scale rendering on Mip-NeRF360 dataset compared to the
single scale 3D Gaussian splatting.

---

## RGBGrasp: Image-based Object Grasping by Capturing Multiple Views during  Robot Arm Movement with Neural Radiance Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Chang Liu, Kejian Shi, Kaichen Zhou, Haoxiao Wang, Jiyao Zhang, Hao Dong | cs.RO | [PDF](http://arxiv.org/pdf/2311.16592v1){: .btn .btn-green } |

**Abstract**: Robotic research encounters a significant hurdle when it comes to the
intricate task of grasping objects that come in various shapes, materials, and
textures. Unlike many prior investigations that heavily leaned on specialized
point-cloud cameras or abundant RGB visual data to gather 3D insights for
object-grasping missions, this paper introduces a pioneering approach called
RGBGrasp. This method depends on a limited set of RGB views to perceive the 3D
surroundings containing transparent and specular objects and achieve accurate
grasping. Our method utilizes pre-trained depth prediction models to establish
geometry constraints, enabling precise 3D structure estimation, even under
limited view conditions. Finally, we integrate hash encoding and a proposal
sampler strategy to significantly accelerate the 3D reconstruction process.
These innovations significantly enhance the adaptability and effectiveness of
our algorithm in real-world scenarios. Through comprehensive experimental
validation, we demonstrate that RGBGrasp achieves remarkable success across a
wide spectrum of object-grasping scenarios, establishing it as a promising
solution for real-world robotic manipulation tasks. The demo of our method can
be found on: https://sites.google.com/view/rgbgrasp

---

## SCALAR-NeRF: SCAlable LARge-scale Neural Radiance Fields for Scene  Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Yu Chen, Gim Hee Lee | cs.CV | [PDF](http://arxiv.org/pdf/2311.16657v1){: .btn .btn-green } |

**Abstract**: In this work, we introduce SCALAR-NeRF, a novel framework tailored for
scalable large-scale neural scene reconstruction. We structure the neural
representation as an encoder-decoder architecture, where the encoder processes
3D point coordinates to produce encoded features, and the decoder generates
geometric values that include volume densities of signed distances and colors.
Our approach first trains a coarse global model on the entire image dataset.
Subsequently, we partition the images into smaller blocks using KMeans with
each block being modeled by a dedicated local model. We enhance the overlapping
regions across different blocks by scaling up the bounding boxes of each local
block. Notably, the decoder from the global model is shared across distinct
blocks and therefore promoting alignment in the feature space of local
encoders. We propose an effective and efficient methodology to fuse the outputs
from these local models to attain the final reconstruction. Employing this
refined coarse-to-fine strategy, our method outperforms state-of-the-art NeRF
methods and demonstrates scalability for large-scale scene reconstruction. The
code will be available on our project page at
https://aibluefisher.github.io/SCALAR-NeRF/

Comments:
- Project Page: https://aibluefisher.github.io/SCALAR-NeRF

---

## SplitNeRF: Split Sum Approximation Neural Field for Joint Geometry,  Illumination, and Material Estimation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Jesus Zarzar, Bernard Ghanem | cs.CV | [PDF](http://arxiv.org/pdf/2311.16671v1){: .btn .btn-green } |

**Abstract**: We present a novel approach for digitizing real-world objects by estimating
their geometry, material properties, and environmental lighting from a set of
posed images with fixed lighting. Our method incorporates into Neural Radiance
Field (NeRF) pipelines the split sum approximation used with image-based
lighting for real-time physical-based rendering. We propose modeling the
scene's lighting with a single scene-specific MLP representing pre-integrated
image-based lighting at arbitrary resolutions. We achieve accurate modeling of
pre-integrated lighting by exploiting a novel regularizer based on efficient
Monte Carlo sampling. Additionally, we propose a new method of supervising
self-occlusion predictions by exploiting a similar regularizer based on Monte
Carlo sampling. Experimental results demonstrate the efficiency and
effectiveness of our approach in estimating scene geometry, material
properties, and lighting. Our method is capable of attaining state-of-the-art
relighting quality after only ${\sim}1$ hour of training in a single NVIDIA
A100 GPU.

---

## Human Gaussian Splatting: Real-time Rendering of Animatable Avatars

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Arthur Moreau, Jifei Song, Helisa Dhamo, Richard Shaw, Yiren Zhou, Eduardo Pérez-Pellitero | cs.CV | [PDF](http://arxiv.org/pdf/2311.17113v1){: .btn .btn-green } |

**Abstract**: This work addresses the problem of real-time rendering of photorealistic
human body avatars learned from multi-view videos. While the classical
approaches to model and render virtual humans generally use a textured mesh,
recent research has developed neural body representations that achieve
impressive visual quality. However, these models are difficult to render in
real-time and their quality degrades when the character is animated with body
poses different than the training observations. We propose the first animatable
human model based on 3D Gaussian Splatting, that has recently emerged as a very
efficient alternative to neural radiance fields. Our body is represented by a
set of gaussian primitives in a canonical space which are deformed in a coarse
to fine approach that combines forward skinning and local non-rigid refinement.
We describe how to learn our Human Gaussian Splatting (\OURS) model in an
end-to-end fashion from multi-view observations, and evaluate it against the
state-of-the-art approaches for novel pose synthesis of clothed body. Our
method presents a PSNR 1.5dbB better than the state-of-the-art on THuman4
dataset while being able to render at 20fps or more.

---

## REF$^2$-NeRF: Reflection and Refraction aware Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Wooseok Kim, Taiki Fukiage, Takeshi Oishi | cs.CV | [PDF](http://arxiv.org/pdf/2311.17116v3){: .btn .btn-green } |

**Abstract**: Recently, significant progress has been made in the study of methods for 3D
reconstruction from multiple images using implicit neural representations,
exemplified by the neural radiance field (NeRF) method. Such methods, which are
based on volume rendering, can model various light phenomena, and various
extended methods have been proposed to accommodate different scenes and
situations. However, when handling scenes with multiple glass objects, e.g.,
objects in a glass showcase, modeling the target scene accurately has been
challenging due to the presence of multiple reflection and refraction effects.
Thus, this paper proposes a NeRF-based modeling method for scenes containing a
glass case. In the proposed method, refraction and reflection are modeled using
elements that are dependent and independent of the viewer's perspective. This
approach allows us to estimate the surfaces where refraction occurs, i.e.,
glass surfaces, and enables the separation and modeling of both direct and
reflected light components. Compared to existing methods, the proposed method
enables more accurate modeling of both glass refraction and the overall scene.

Comments:
- 10 pages, 8 figures, 2 tables

---

## DGNR: Density-Guided Neural Point Rendering of Large Driving Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Zhuopeng Li, Chenming Wu, Liangjun Zhang, Jianke Zhu | cs.CV | [PDF](http://arxiv.org/pdf/2311.16664v1){: .btn .btn-green } |

**Abstract**: Despite the recent success of Neural Radiance Field (NeRF), it is still
challenging to render large-scale driving scenes with long trajectories,
particularly when the rendering quality and efficiency are in high demand.
Existing methods for such scenes usually involve with spatial warping,
geometric supervision from zero-shot normal or depth estimation, or scene
division strategies, where the synthesized views are often blurry or fail to
meet the requirement of efficient rendering. To address the above challenges,
this paper presents a novel framework that learns a density space from the
scenes to guide the construction of a point-based renderer, dubbed as DGNR
(Density-Guided Neural Rendering). In DGNR, geometric priors are no longer
needed, which can be intrinsically learned from the density space through
volumetric rendering. Specifically, we make use of a differentiable renderer to
synthesize images from the neural density features obtained from the learned
density space. A density-based fusion module and geometric regularization are
proposed to optimize the density space. By conducting experiments on a widely
used autonomous driving dataset, we have validated the effectiveness of DGNR in
synthesizing photorealistic driving scenes and achieving real-time capable
rendering.

---

## Continuous Pose for Monocular Cameras in Neural Implicit Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Qi Ma, Danda Pani Paudel, Ajad Chhatkuli, Luc Van Gool | cs.CV | [PDF](http://arxiv.org/pdf/2311.17119v1){: .btn .btn-green } |

**Abstract**: In this paper, we showcase the effectiveness of optimizing monocular camera
poses as a continuous function of time. The camera poses are represented using
an implicit neural function which maps the given time to the corresponding
camera pose. The mapped camera poses are then used for the downstream tasks
where joint camera pose optimization is also required. While doing so, the
network parameters -- that implicitly represent camera poses -- are optimized.
We exploit the proposed method in four diverse experimental settings, namely,
(1) NeRF from noisy poses; (2) NeRF from asynchronous Events; (3) Visual
Simultaneous Localization and Mapping (vSLAM); and (4) vSLAM with IMUs. In all
four settings, the proposed method performs significantly better than the
compared baselines and the state-of-the-art methods. Additionally, using the
assumption of continuous motion, changes in pose may actually live in a
manifold that has lower than 6 degrees of freedom (DOF) is also realized. We
call this low DOF motion representation as the \emph{intrinsic motion} and use
the approach in vSLAM settings, showing impressive camera tracking performance.

---

## Rethinking Directional Integration in Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Congyue Deng, Jiawei Yang, Leonidas Guibas, Yue Wang | cs.CV | [PDF](http://arxiv.org/pdf/2311.16504v1){: .btn .btn-green } |

**Abstract**: Recent works use the Neural radiance field (NeRF) to perform multi-view 3D
reconstruction, providing a significant leap in rendering photorealistic
scenes. However, despite its efficacy, NeRF exhibits limited capability of
learning view-dependent effects compared to light field rendering or
image-based view synthesis. To that end, we introduce a modification to the
NeRF rendering equation which is as simple as a few lines of code change for
any NeRF variations, while greatly improving the rendering quality of
view-dependent effects. By swapping the integration operator and the direction
decoder network, we only integrate the positional features along the ray and
move the directional terms out of the integration, resulting in a
disentanglement of the view-dependent and independent components. The modified
equation is equivalent to the classical volumetric rendering in ideal cases on
object surfaces with Dirac densities. Furthermore, we prove that with the
errors caused by network approximation and numerical integration, our rendering
equation exhibits better convergence properties with lower error accumulations
compared to the classical NeRF. We also show that the modified equation can be
interpreted as light field rendering with learned ray embeddings. Experiments
on different NeRF variations show consistent improvements in the quality of
view-dependent effects with our simple modification.

---

## LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and  200+ FPS

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang | cs.CV | [PDF](http://arxiv.org/pdf/2311.17245v3){: .btn .btn-green } |

**Abstract**: Recent advancements in real-time neural rendering using point-based
techniques have paved the way for the widespread adoption of 3D
representations. However, foundational approaches like 3D Gaussian Splatting
come with a substantial storage overhead caused by growing the SfM points to
millions, often demanding gigabyte-level disk space for a single unbounded
scene, posing significant scalability challenges and hindering the splatting
efficiency.
  To address this challenge, we introduce LightGaussian, a novel method
designed to transform 3D Gaussians into a more efficient and compact format.
Drawing inspiration from the concept of Network Pruning, LightGaussian
identifies Gaussians that are insignificant in contributing to the scene
reconstruction and adopts a pruning and recovery process, effectively reducing
redundancy in Gaussian counts while preserving visual effects. Additionally,
LightGaussian employs distillation and pseudo-view augmentation to distill
spherical harmonics to a lower degree, allowing knowledge transfer to more
compact representations while maintaining reflectance. Furthermore, we propose
a hybrid scheme, VecTree Quantization, to quantize all attributes, resulting in
lower bitwidth representations with minimal accuracy losses.
  In summary, LightGaussian achieves an averaged compression rate over 15x
while boosting the FPS from 139 to 215, enabling an efficient representation of
complex scenes on Mip-NeRF 360, Tank and Temple datasets.
  Project website: https://lightgaussian.github.io/

Comments:
- 16pages, 8figures

---

## HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Xian Liu, Xiaohang Zhan, Jiaxiang Tang, Ying Shan, Gang Zeng, Dahua Lin, Xihui Liu, Ziwei Liu | cs.CV | [PDF](http://arxiv.org/pdf/2311.17061v1){: .btn .btn-green } |

**Abstract**: Realistic 3D human generation from text prompts is a desirable yet
challenging task. Existing methods optimize 3D representations like mesh or
neural fields via score distillation sampling (SDS), which suffers from
inadequate fine details or excessive training time. In this paper, we propose
an efficient yet effective framework, HumanGaussian, that generates
high-quality 3D humans with fine-grained geometry and realistic appearance. Our
key insight is that 3D Gaussian Splatting is an efficient renderer with
periodic Gaussian shrinkage or growing, where such adaptive density control can
be naturally guided by intrinsic human structures. Specifically, 1) we first
propose a Structure-Aware SDS that simultaneously optimizes human appearance
and geometry. The multi-modal score function from both RGB and depth space is
leveraged to distill the Gaussian densification and pruning process. 2)
Moreover, we devise an Annealed Negative Prompt Guidance by decomposing SDS
into a noisier generative score and a cleaner classifier score, which well
addresses the over-saturation issue. The floating artifacts are further
eliminated based on Gaussian size in a prune-only phase to enhance generation
smoothness. Extensive experiments demonstrate the superior efficiency and
competitive quality of our framework, rendering vivid 3D humans under diverse
scenarios. Project Page: https://alvinliu0.github.io/projects/HumanGaussian

Comments:
- Project Page: https://alvinliu0.github.io/projects/HumanGaussian

---

## A Unified Approach for Text- and Image-guided 4D Scene Generation



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Yufeng Zheng, Xueting Li, Koki Nagano, Sifei Liu, Karsten Kreis, Otmar Hilliges, Shalini De Mello | cs.CV | [PDF](http://arxiv.org/pdf/2311.16854v2){: .btn .btn-green } |

**Abstract**: Large-scale diffusion generative models are greatly simplifying image, video
and 3D asset creation from user-provided text prompts and images. However, the
challenging problem of text-to-4D dynamic 3D scene generation with diffusion
guidance remains largely unexplored. We propose Dream-in-4D, which features a
novel two-stage approach for text-to-4D synthesis, leveraging (1) 3D and 2D
diffusion guidance to effectively learn a high-quality static 3D asset in the
first stage; (2) a deformable neural radiance field that explicitly
disentangles the learned static asset from its deformation, preserving quality
during motion learning; and (3) a multi-resolution feature grid for the
deformation field with a displacement total variation loss to effectively learn
motion with video diffusion guidance in the second stage. Through a user
preference study, we demonstrate that our approach significantly advances image
and motion quality, 3D consistency and text fidelity for text-to-4D generation
compared to baseline approaches. Thanks to its motion-disentangled
representation, Dream-in-4D can also be easily adapted for controllable
generation where appearance is defined by one or multiple images, without the
need to modify the motion learning stage. Thus, our method offers, for the
first time, a unified approach for text-to-4D, image-to-4D and personalized 4D
generation tasks.

Comments:
- Project page: https://research.nvidia.com/labs/nxp/dream-in-4d/

---

## UC-NeRF: Neural Radiance Field for Under-Calibrated Multi-view Cameras  in Autonomous Driving

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | Kai Cheng, Xiaoxiao Long, Wei Yin, Jin Wang, Zhiqiang Wu, Yuexin Ma, Kaixuan Wang, Xiaozhi Chen, Xuejin Chen | cs.CV | [PDF](http://arxiv.org/pdf/2311.16945v2){: .btn .btn-green } |

**Abstract**: Multi-camera setups find widespread use across various applications, such as
autonomous driving, as they greatly expand sensing capabilities. Despite the
fast development of Neural radiance field (NeRF) techniques and their wide
applications in both indoor and outdoor scenes, applying NeRF to multi-camera
systems remains very challenging. This is primarily due to the inherent
under-calibration issues in multi-camera setup, including inconsistent imaging
effects stemming from separately calibrated image signal processing units in
diverse cameras, and system errors arising from mechanical vibrations during
driving that affect relative camera poses. In this paper, we present UC-NeRF, a
novel method tailored for novel view synthesis in under-calibrated multi-view
camera systems. Firstly, we propose a layer-based color correction to rectify
the color inconsistency in different image regions. Second, we propose virtual
warping to generate more viewpoint-diverse but color-consistent virtual views
for color correction and 3D recovery. Finally, a spatiotemporally constrained
pose refinement is designed for more robust and accurate pose calibration in
multi-camera systems. Our method not only achieves state-of-the-art performance
of novel view synthesis in multi-camera setups, but also effectively
facilitates depth estimation in large-scale outdoor scenes with the synthesized
novel views.

Comments:
- See the project page for code, data:
  https://kcheng1021.github.io/ucnerf.github.io

---

## The Sky's the Limit: Re-lightable Outdoor Scenes via a Sky-pixel  Constrained Illumination Prior and Outside-In Visibility

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-28 | James A. D. Gardner, Evgenii Kashin, Bernhard Egger, William A. P. Smith | cs.CV | [PDF](http://arxiv.org/pdf/2311.16937v1){: .btn .btn-green } |

**Abstract**: Inverse rendering of outdoor scenes from unconstrained image collections is a
challenging task, particularly illumination/albedo ambiguities and occlusion of
the illumination environment (shadowing) caused by geometry. However, there are
many cues in an image that can aid in the disentanglement of geometry, albedo
and shadows. We exploit the fact that any sky pixel provides a direct
measurement of distant lighting in the corresponding direction and, via a
neural illumination prior, a statistical cue as to the remaining illumination
environment. We also introduce a novel `outside-in' method for computing
differentiable sky visibility based on a neural directional distance function.
This is efficient and can be trained in parallel with the neural scene
representation, allowing gradients from appearance loss to flow from shadows to
influence estimation of illumination and geometry. Our method estimates
high-quality albedo, geometry, illumination and sky visibility, achieving
state-of-the-art results on the NeRF-OSR relighting benchmark. Our code and
models can be found https://github.com/JADGardner/neusky

---

## Animatable Gaussians: Learning Pose-dependent Gaussian Maps for  High-fidelity Human Avatar Modeling

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-27 | Zhe Li, Zerong Zheng, Lizhen Wang, Yebin Liu | cs.CV | [PDF](http://arxiv.org/pdf/2311.16096v1){: .btn .btn-green } |

**Abstract**: Modeling animatable human avatars from RGB videos is a long-standing and
challenging problem. Recent works usually adopt MLP-based neural radiance
fields (NeRF) to represent 3D humans, but it remains difficult for pure MLPs to
regress pose-dependent garment details. To this end, we introduce Animatable
Gaussians, a new avatar representation that leverages powerful 2D CNNs and 3D
Gaussian splatting to create high-fidelity avatars. To associate 3D Gaussians
with the animatable avatar, we learn a parametric template from the input
videos, and then parameterize the template on two front \& back canonical
Gaussian maps where each pixel represents a 3D Gaussian. The learned template
is adaptive to the wearing garments for modeling looser clothes like dresses.
Such template-guided 2D parameterization enables us to employ a powerful
StyleGAN-based CNN to learn the pose-dependent Gaussian maps for modeling
detailed dynamic appearances. Furthermore, we introduce a pose projection
strategy for better generalization given novel poses. Overall, our method can
create lifelike avatars with dynamic, realistic and generalized appearances.
Experiments show that our method outperforms other state-of-the-art approaches.
Code: https://github.com/lizhe00/AnimatableGaussians

Comments:
- Projectpage: https://animatable-gaussians.github.io/, Code:
  https://github.com/lizhe00/AnimatableGaussians

---

## GaussianEditor: Editing 3D Gaussians Delicately with Text Instructions

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-27 | Jiemin Fang, Junjie Wang, Xiaopeng Zhang, Lingxi Xie, Qi Tian | cs.CV | [PDF](http://arxiv.org/pdf/2311.16037v1){: .btn .btn-green } |

**Abstract**: Recently, impressive results have been achieved in 3D scene editing with text
instructions based on a 2D diffusion model. However, current diffusion models
primarily generate images by predicting noise in the latent space, and the
editing is usually applied to the whole image, which makes it challenging to
perform delicate, especially localized, editing for 3D scenes. Inspired by
recent 3D Gaussian splatting, we propose a systematic framework, named
GaussianEditor, to edit 3D scenes delicately via 3D Gaussians with text
instructions. Benefiting from the explicit property of 3D Gaussians, we design
a series of techniques to achieve delicate editing. Specifically, we first
extract the region of interest (RoI) corresponding to the text instruction,
aligning it to 3D Gaussians. The Gaussian RoI is further used to control the
editing process. Our framework can achieve more delicate and precise editing of
3D scenes than previous methods while enjoying much faster training speed, i.e.
within 20 minutes on a single V100 GPU, more than twice as fast as
Instruct-NeRF2NeRF (45 minutes -- 2 hours).

Comments:
- Project page: https://GaussianEditor.github.io

---

## Deceptive-Human: Prompt-to-NeRF 3D Human Generation with 3D-Consistent  Synthetic Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-27 | Shiu-hong Kao, Xinhang Liu, Yu-Wing Tai, Chi-Keung Tang | cs.CV | [PDF](http://arxiv.org/pdf/2311.16499v1){: .btn .btn-green } |

**Abstract**: This paper presents Deceptive-Human, a novel Prompt-to-NeRF framework
capitalizing state-of-the-art control diffusion models (e.g., ControlNet) to
generate a high-quality controllable 3D human NeRF. Different from direct 3D
generative approaches, e.g., DreamFusion and DreamHuman, Deceptive-Human
employs a progressive refinement technique to elevate the reconstruction
quality. This is achieved by utilizing high-quality synthetic human images
generated through the ControlNet with view-consistent loss. Our method is
versatile and readily extensible, accommodating multimodal inputs, including a
text prompt and additional data such as 3D mesh, poses, and seed images. The
resulting 3D human NeRF model empowers the synthesis of highly photorealistic
novel views from 360-degree perspectives. The key to our Deceptive-Human for
hallucinating multi-view consistent synthetic human images lies in our
progressive finetuning strategy. This strategy involves iteratively enhancing
views using the provided multimodal inputs at each intermediate step to improve
the human NeRF model. Within this iterative refinement process, view-dependent
appearances are systematically eliminated to prevent interference with the
underlying density estimation. Extensive qualitative and quantitative
experimental comparison shows that our deceptive human models achieve
state-of-the-art application quality.

Comments:
- Github project: https://github.com/DanielSHKao/DeceptiveHuman

---

## SOAC: Spatio-Temporal Overlap-Aware Multi-Sensor Calibration using  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-27 | Quentin Herau, Nathan Piasco, Moussab Bennehar, Luis Roldão, Dzmitry Tsishkou, Cyrille Migniot, Pascal Vasseur, Cédric Demonceaux | cs.CV | [PDF](http://arxiv.org/pdf/2311.15803v2){: .btn .btn-green } |

**Abstract**: In rapidly-evolving domains such as autonomous driving, the use of multiple
sensors with different modalities is crucial to ensure high operational
precision and stability. To correctly exploit the provided information by each
sensor in a single common frame, it is essential for these sensors to be
accurately calibrated. In this paper, we leverage the ability of Neural
Radiance Fields (NeRF) to represent different sensors modalities in a common
volumetric representation to achieve robust and accurate spatio-temporal sensor
calibration. By designing a partitioning approach based on the visible part of
the scene for each sensor, we formulate the calibration problem using only the
overlapping areas. This strategy results in a more robust and accurate
calibration that is less prone to failure. We demonstrate that our approach
works on outdoor urban scenes by validating it on multiple established driving
datasets. Results show that our method is able to get better accuracy and
robustness compared to existing methods.

Comments:
- Paper + Supplementary, under review. Project page:
  https://qherau.github.io/SOAC/

---

## Mip-Splatting: Alias-free 3D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-27 | Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, Andreas Geiger | cs.CV | [PDF](http://arxiv.org/pdf/2311.16493v1){: .btn .btn-green } |

**Abstract**: Recently, 3D Gaussian Splatting has demonstrated impressive novel view
synthesis results, reaching high fidelity and efficiency. However, strong
artifacts can be observed when changing the sampling rate, \eg, by changing
focal length or camera distance. We find that the source for this phenomenon
can be attributed to the lack of 3D frequency constraints and the usage of a 2D
dilation filter. To address this problem, we introduce a 3D smoothing filter
which constrains the size of the 3D Gaussian primitives based on the maximal
sampling frequency induced by the input views, eliminating high-frequency
artifacts when zooming in. Moreover, replacing 2D dilation with a 2D Mip
filter, which simulates a 2D box filter, effectively mitigates aliasing and
dilation issues. Our evaluation, including scenarios such a training on
single-scale images and testing on multiple scales, validates the effectiveness
of our approach.

Comments:
- Project page: https://niujinshuchong.github.io/mip-splatting/

---

## PaintNeSF: Artistic Creation of Stylized Scenes with Vectorized 3D  Strokes



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-27 | Hao-Bin Duan, Miao Wang, Yan-Xun Li, Yong-Liang Yang | cs.CV | [PDF](http://arxiv.org/pdf/2311.15637v1){: .btn .btn-green } |

**Abstract**: We present Paint Neural Stroke Field (PaintNeSF), a novel technique to
generate stylized images of a 3D scene at arbitrary novel views from multi-view
2D images. Different from existing methods which apply stylization to trained
neural radiance fields at the voxel level, our approach draws inspiration from
image-to-painting methods, simulating the progressive painting process of human
artwork with vector strokes. We develop a palette of stylized 3D strokes from
basic primitives and splines, and consider the 3D scene stylization task as a
multi-view reconstruction process based on these 3D stroke primitives. Instead
of directly searching for the parameters of these 3D strokes, which would be
too costly, we introduce a differentiable renderer that allows optimizing
stroke parameters using gradient descent, and propose a training scheme to
alleviate the vanishing gradient issue. The extensive evaluation demonstrates
that our approach effectively synthesizes 3D scenes with significant geometric
and aesthetic stylization while maintaining a consistent appearance across
different views. Our method can be further integrated with style loss and
image-text contrastive models to extend its applications, including color
transfer and text-driven 3D scene drawing.

---

## Animatable 3D Gaussian: Fast and High-Quality Reconstruction of Multiple  Human Avatars



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-27 | Yang Liu, Xiang Huang, Minghan Qin, Qinwei Lin, Haoqian Wang | cs.CV | [PDF](http://arxiv.org/pdf/2311.16482v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields are capable of reconstructing high-quality drivable
human avatars but are expensive to train and render. To reduce consumption, we
propose Animatable 3D Gaussian, which learns human avatars from input images
and poses. We extend 3D Gaussians to dynamic human scenes by modeling a set of
skinned 3D Gaussians and a corresponding skeleton in canonical space and
deforming 3D Gaussians to posed space according to the input poses. We
introduce hash-encoded shape and appearance to speed up training and propose
time-dependent ambient occlusion to achieve high-quality reconstructions in
scenes containing complex motions and dynamic shadows. On both novel view
synthesis and novel pose synthesis tasks, our method outperforms existing
methods in terms of training time, rendering speed, and reconstruction quality.
Our method can be easily extended to multi-human scenes and achieve comparable
novel view synthesis results on a scene with ten people in only 25 seconds of
training.

---

## CaesarNeRF: Calibrated Semantic Representation for Few-shot  Generalizable Neural Rendering

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-27 | Haidong Zhu, Tianyu Ding, Tianyi Chen, Ilya Zharkov, Ram Nevatia, Luming Liang | cs.CV | [PDF](http://arxiv.org/pdf/2311.15510v1){: .btn .btn-green } |

**Abstract**: Generalizability and few-shot learning are key challenges in Neural Radiance
Fields (NeRF), often due to the lack of a holistic understanding in pixel-level
rendering. We introduce CaesarNeRF, an end-to-end approach that leverages
scene-level CAlibratEd SemAntic Representation along with pixel-level
representations to advance few-shot, generalizable neural rendering,
facilitating a holistic understanding without compromising high-quality
details. CaesarNeRF explicitly models pose differences of reference views to
combine scene-level semantic representations, providing a calibrated holistic
understanding. This calibration process aligns various viewpoints with precise
location and is further enhanced by sequential refinement to capture varying
details. Extensive experiments on public datasets, including LLFF, Shiny,
mip-NeRF 360, and MVImgNet, show that CaesarNeRF delivers state-of-the-art
performance across varying numbers of reference views, proving effective even
with a single reference image. The project page of this work can be found at
https://haidongz-usc.github.io/project/caesarnerf.

---

## Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF  Decomposition and Ray Tracing

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-27 | Jian Gao, Chun Gu, Youtian Lin, Hao Zhu, Xun Cao, Li Zhang, Yao Yao | cs.CV | [PDF](http://arxiv.org/pdf/2311.16043v1){: .btn .btn-green } |

**Abstract**: We present a novel differentiable point-based rendering framework for
material and lighting decomposition from multi-view images, enabling editing,
ray-tracing, and real-time relighting of the 3D point cloud. Specifically, a 3D
scene is represented as a set of relightable 3D Gaussian points, where each
point is additionally associated with a normal direction, BRDF parameters, and
incident lights from different directions. To achieve robust lighting
estimation, we further divide incident lights of each point into global and
local components, as well as view-dependent visibilities. The 3D scene is
optimized through the 3D Gaussian Splatting technique while BRDF and lighting
are decomposed by physically-based differentiable rendering. Moreover, we
introduce an innovative point-based ray-tracing approach based on the bounding
volume hierarchy for efficient visibility baking, enabling real-time rendering
and relighting of 3D Gaussian points with accurate shadow effects. Extensive
experiments demonstrate improved BRDF estimation and novel view rendering
results compared to state-of-the-art material estimation approaches. Our
framework showcases the potential to revolutionize the mesh-based graphics
pipeline with a relightable, traceable, and editable rendering pipeline solely
based on point cloud. Project
page:https://nju-3dv.github.io/projects/Relightable3DGaussian/.

---

## GS-IR: 3D Gaussian Splatting for Inverse Rendering

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-26 | Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, Kui Jia | cs.CV | [PDF](http://arxiv.org/pdf/2311.16473v2){: .btn .btn-green } |

**Abstract**: We propose GS-IR, a novel inverse rendering approach based on 3D Gaussian
Splatting (GS) that leverages forward mapping volume rendering to achieve
photorealistic novel view synthesis and relighting results. Unlike previous
works that use implicit neural representations and volume rendering (e.g.
NeRF), which suffer from low expressive power and high computational
complexity, we extend GS, a top-performance representation for novel view
synthesis, to estimate scene geometry, surface material, and environment
illumination from multi-view images captured under unknown lighting conditions.
There are two main problems when introducing GS to inverse rendering: 1) GS
does not support producing plausible normal natively; 2) forward mapping (e.g.
rasterization and splatting) cannot trace the occlusion like backward mapping
(e.g. ray tracing). To address these challenges, our GS-IR proposes an
efficient optimization scheme that incorporates a depth-derivation-based
regularization for normal estimation and a baking-based occlusion to model
indirect lighting. The flexible and expressive GS representation allows us to
achieve fast and compact geometry reconstruction, photorealistic novel view
synthesis, and effective physically-based rendering. We demonstrate the
superiority of our method over baseline methods through qualitative and
quantitative evaluations on various challenging scenes.

---

## NeuRAD: Neural Rendering for Autonomous Driving

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-26 | Adam Tonderski, Carl Lindström, Georg Hess, William Ljungbergh, Lennart Svensson, Christoffer Petersson | cs.CV | [PDF](http://arxiv.org/pdf/2311.15260v2){: .btn .btn-green } |

**Abstract**: Neural radiance fields (NeRFs) have gained popularity in the autonomous
driving (AD) community. Recent methods show NeRFs' potential for closed-loop
simulation, enabling testing of AD systems, and as an advanced training data
augmentation technique. However, existing methods often require long training
times, dense semantic supervision, or lack generalizability. This, in turn,
hinders the application of NeRFs for AD at scale. In this paper, we propose
NeuRAD, a robust novel view synthesis method tailored to dynamic AD data. Our
method features simple network design, extensive sensor modeling for both
camera and lidar -- including rolling shutter, beam divergence and ray dropping
-- and is applicable to multiple datasets out of the box. We verify its
performance on five popular AD datasets, achieving state-of-the-art performance
across the board. To encourage further development, we will openly release the
NeuRAD source code. See https://github.com/georghess/NeuRAD .

---

## Obj-NeRF: Extract Object NeRFs from Multi-view Images

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-26 | Zhiyi Li, Lihe Ding, Tianfan Xue | cs.CV | [PDF](http://arxiv.org/pdf/2311.15291v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have demonstrated remarkable effectiveness in
novel view synthesis within 3D environments. However, extracting a radiance
field of one specific object from multi-view images encounters substantial
challenges due to occlusion and background complexity, thereby presenting
difficulties in downstream applications such as NeRF editing and 3D mesh
extraction. To solve this problem, in this paper, we propose Obj-NeRF, a
comprehensive pipeline that recovers the 3D geometry of a specific object from
multi-view images using a single prompt. This method combines the 2D
segmentation capabilities of the Segment Anything Model (SAM) in conjunction
with the 3D reconstruction ability of NeRF. Specifically, we first obtain
multi-view segmentation for the indicated object using SAM with a single
prompt. Then, we use the segmentation images to supervise NeRF construction,
integrating several effective techniques. Additionally, we construct a large
object-level NeRF dataset containing diverse objects, which can be useful in
various downstream tasks. To demonstrate the practicality of our method, we
also apply Obj-NeRF to various applications, including object removal,
rotation, replacement, and recoloring.

---

## Efficient Encoding of Graphics Primitives with Simplex-based Structures

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-26 | Yibo Wen, Yunfan Yang | cs.CV | [PDF](http://arxiv.org/pdf/2311.15439v1){: .btn .btn-green } |

**Abstract**: Grid-based structures are commonly used to encode explicit features for
graphics primitives such as images, signed distance functions (SDF), and neural
radiance fields (NeRF) due to their simple implementation. However, in
$n$-dimensional space, calculating the value of a sampled point requires
interpolating the values of its $2^n$ neighboring vertices. The exponential
scaling with dimension leads to significant computational overheads. To address
this issue, we propose a simplex-based approach for encoding graphics
primitives. The number of vertices in a simplex-based structure increases
linearly with dimension, making it a more efficient and generalizable
alternative to grid-based representations. Using the non-axis-aligned
simplicial structure property, we derive and prove a coordinate transformation,
simplicial subdivision, and barycentric interpolation scheme for efficient
sampling, which resembles transformation procedures in the simplex noise
algorithm. Finally, we use hash tables to store multiresolution features of all
interest points in the simplicial grid, which are passed into a tiny fully
connected neural network to parameterize graphics primitives. We implemented a
detailed simplex-based structure encoding algorithm in C++ and CUDA using the
methods outlined in our approach. In the 2D image fitting task, the proposed
method is capable of fitting a giga-pixel image with 9.4% less time compared to
the baseline method proposed by instant-ngp, while maintaining the same quality
and compression rate. In the volumetric rendering setup, we observe a maximum
41.2% speedup when the samples are dense enough.

Comments:
- 10 pages, 8 figures

---

## GaussianEditor: Swift and Controllable 3D Editing with Gaussian  Splatting

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-24 | Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, Guosheng Lin | cs.CV | [PDF](http://arxiv.org/pdf/2311.14521v4){: .btn .btn-green } |

**Abstract**: 3D editing plays a crucial role in many areas such as gaming and virtual
reality. Traditional 3D editing methods, which rely on representations like
meshes and point clouds, often fall short in realistically depicting complex
scenes. On the other hand, methods based on implicit 3D representations, like
Neural Radiance Field (NeRF), render complex scenes effectively but suffer from
slow processing speeds and limited control over specific scene areas. In
response to these challenges, our paper presents GaussianEditor, an innovative
and efficient 3D editing algorithm based on Gaussian Splatting (GS), a novel 3D
representation. GaussianEditor enhances precision and control in editing
through our proposed Gaussian semantic tracing, which traces the editing target
throughout the training process. Additionally, we propose Hierarchical Gaussian
splatting (HGS) to achieve stabilized and fine results under stochastic
generative guidance from 2D diffusion models. We also develop editing
strategies for efficient object removal and integration, a challenging task for
existing methods. Our comprehensive experiments demonstrate GaussianEditor's
superior control, efficacy, and rapid performance, marking a significant
advancement in 3D editing. Project Page:
https://buaacyw.github.io/gaussian-editor/

Comments:
- Project Page: https://buaacyw.github.io/gaussian-editor/ Code:
  https://github.com/buaacyw/GaussianEditor

---

## Animate124: Animating One Image to 4D Dynamic Scene

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-24 | Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, Gim Hee Lee | cs.CV | [PDF](http://arxiv.org/pdf/2311.14603v1){: .btn .btn-green } |

**Abstract**: We introduce Animate124 (Animate-one-image-to-4D), the first work to animate
a single in-the-wild image into 3D video through textual motion descriptions,
an underexplored problem with significant applications. Our 4D generation
leverages an advanced 4D grid dynamic Neural Radiance Field (NeRF) model,
optimized in three distinct stages using multiple diffusion priors. Initially,
a static model is optimized using the reference image, guided by 2D and 3D
diffusion priors, which serves as the initialization for the dynamic NeRF.
Subsequently, a video diffusion model is employed to learn the motion specific
to the subject. However, the object in the 3D videos tends to drift away from
the reference image over time. This drift is mainly due to the misalignment
between the text prompt and the reference image in the video diffusion model.
In the final stage, a personalized diffusion prior is therefore utilized to
address the semantic drift. As the pioneering image-text-to-4D generation
framework, our method demonstrates significant advancements over existing
baselines, evidenced by comprehensive quantitative and qualitative assessments.

Comments:
- Project Page: https://animate124.github.io

---

## ECRF: Entropy-Constrained Neural Radiance Fields Compression with  Frequency Domain Optimization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-23 | Soonbin Lee, Fangwen Shu, Yago Sanchez, Thomas Schierl, Cornelius Hellge | cs.CV | [PDF](http://arxiv.org/pdf/2311.14208v1){: .btn .btn-green } |

**Abstract**: Explicit feature-grid based NeRF models have shown promising results in terms
of rendering quality and significant speed-up in training. However, these
methods often require a significant amount of data to represent a single scene
or object. In this work, we present a compression model that aims to minimize
the entropy in the frequency domain in order to effectively reduce the data
size. First, we propose using the discrete cosine transform (DCT) on the
tensorial radiance fields to compress the feature-grid. This feature-grid is
transformed into coefficients, which are then quantized and entropy encoded,
following a similar approach to the traditional video coding pipeline.
Furthermore, to achieve a higher level of sparsity, we propose using an entropy
parameterization technique for the frequency domain, specifically for DCT
coefficients of the feature-grid. Since the transformed coefficients are
optimized during the training phase, the proposed model does not require any
fine-tuning or additional information. Our model only requires a lightweight
compression pipeline for encoding and decoding, making it easier to apply
volumetric radiance field methods for real-world applications. Experimental
results demonstrate that our proposed frequency domain entropy model can
achieve superior compression performance across various datasets. The source
code will be made publicly available.

Comments:
- 10 pages, 6 figures, 4 tables

---

## Tube-NeRF: Efficient Imitation Learning of Visuomotor Policies from MPC  using Tube-Guided Data Augmentation and NeRFs

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-23 | Andrea Tagliabue, Jonathan P. How | cs.RO | [PDF](http://arxiv.org/pdf/2311.14153v1){: .btn .btn-green } |

**Abstract**: Imitation learning (IL) can train computationally-efficient sensorimotor
policies from a resource-intensive Model Predictive Controller (MPC), but it
often requires many samples, leading to long training times or limited
robustness. To address these issues, we combine IL with a variant of robust MPC
that accounts for process and sensing uncertainties, and we design a data
augmentation (DA) strategy that enables efficient learning of vision-based
policies. The proposed DA method, named Tube-NeRF, leverages Neural Radiance
Fields (NeRFs) to generate novel synthetic images, and uses properties of the
robust MPC (the tube) to select relevant views and to efficiently compute the
corresponding actions. We tailor our approach to the task of localization and
trajectory tracking on a multirotor, by learning a visuomotor policy that
generates control actions using images from the onboard camera as only source
of horizontal position. Our evaluations numerically demonstrate learning of a
robust visuomotor policy with an 80-fold increase in demonstration efficiency
and a 50% reduction in training time over current IL methods. Additionally, our
policies successfully transfer to a real multirotor, achieving accurate
localization and low tracking errors despite large disturbances, with an
onboard inference time of only 1.5 ms.

Comments:
- Video: https://youtu.be/_W5z33ZK1m4. Evolved paper from our previous
  work: arXiv:2210.10127

---

## Posterior Distillation Sampling



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-23 | Juil Koo, Chanho Park, Minhyuk Sung | cs.CV | [PDF](http://arxiv.org/pdf/2311.13831v1){: .btn .btn-green } |

**Abstract**: We introduce Posterior Distillation Sampling (PDS), a novel optimization
method for parametric image editing based on diffusion models. Existing
optimization-based methods, which leverage the powerful 2D prior of diffusion
models to handle various parametric images, have mainly focused on generation.
Unlike generation, editing requires a balance between conforming to the target
attribute and preserving the identity of the source content. Recent 2D image
editing methods have achieved this balance by leveraging the stochastic latent
encoded in the generative process of diffusion models. To extend the editing
capabilities of diffusion models shown in pixel space to parameter space, we
reformulate the 2D image editing method into an optimization form named PDS.
PDS matches the stochastic latents of the source and the target, enabling the
sampling of targets in diverse parameter spaces that align with a desired
attribute while maintaining the source's identity. We demonstrate that this
optimization resembles running a generative process with the target attribute,
but aligning this process with the trajectory of the source's generative
process. Extensive editing results in Neural Radiance Fields and Scalable
Vector Graphics representations demonstrate that PDS is capable of sampling
targets to fulfill the aforementioned balance across various parameter spaces.

Comments:
- Project page: https://posterior-distillation-sampling.github.io/

---

## Towards Transferable Multi-modal Perception Representation Learning for  Autonomy: NeRF-Supervised Masked AutoEncoder

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-23 | Xiaohao Xu | cs.CV | [PDF](http://arxiv.org/pdf/2311.13750v2){: .btn .btn-green } |

**Abstract**: This work proposes a unified self-supervised pre-training framework for
transferable multi-modal perception representation learning via masked
multi-modal reconstruction in Neural Radiance Field (NeRF), namely
NeRF-Supervised Masked AutoEncoder (NS-MAE). Specifically, conditioned on
certain view directions and locations, multi-modal embeddings extracted from
corrupted multi-modal input signals, i.e., Lidar point clouds and images, are
rendered into projected multi-modal feature maps via neural rendering. Then,
original multi-modal signals serve as reconstruction targets for the rendered
multi-modal feature maps to enable self-supervised representation learning.
Extensive experiments show that the representation learned via NS-MAE shows
promising transferability for diverse multi-modal and single-modal (camera-only
and Lidar-only) perception models on diverse 3D perception downstream tasks (3D
object detection and BEV map segmentation) with diverse amounts of fine-tuning
labeled data. Moreover, we empirically find that NS-MAE enjoys the synergy of
both the mechanism of masked autoencoder and neural radiance field. We hope
this study can inspire exploration of more general multi-modal representation
learning for autonomous agents.

---

## Compact 3D Gaussian Representation for Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-22 | Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, Eunbyung Park | cs.CV | [PDF](http://arxiv.org/pdf/2311.13681v1){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRFs) have demonstrated remarkable potential in
capturing complex 3D scenes with high fidelity. However, one persistent
challenge that hinders the widespread adoption of NeRFs is the computational
bottleneck due to the volumetric rendering. On the other hand, 3D Gaussian
splatting (3DGS) has recently emerged as an alternative representation that
leverages a 3D Gaussisan-based representation and adopts the rasterization
pipeline to render the images rather than volumetric rendering, achieving very
fast rendering speed and promising image quality. However, a significant
drawback arises as 3DGS entails a substantial number of 3D Gaussians to
maintain the high fidelity of the rendered images, which requires a large
amount of memory and storage. To address this critical issue, we place a
specific emphasis on two key objectives: reducing the number of Gaussian points
without sacrificing performance and compressing the Gaussian attributes, such
as view-dependent color and covariance. To this end, we propose a learnable
mask strategy that significantly reduces the number of Gaussians while
preserving high performance. In addition, we propose a compact but effective
representation of view-dependent color by employing a grid-based neural field
rather than relying on spherical harmonics. Finally, we learn codebooks to
compactly represent the geometric attributes of Gaussian by vector
quantization. In our extensive experiments, we consistently show over
10$\times$ reduced storage and enhanced rendering speed, while maintaining the
quality of the scene representation, compared to 3DGS. Our work provides a
comprehensive framework for 3D scene representation, achieving high
performance, fast training, compactness, and real-time rendering. Our project
page is available at https://maincold2.github.io/c3dgs/.

Comments:
- Project page: http://maincold2.github.io/c3dgs/

---

## Animatable 3D Gaussians for High-fidelity Synthesis of Human Motions

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-22 | Keyang Ye, Tianjia Shao, Kun Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2311.13404v2){: .btn .btn-green } |

**Abstract**: We present a novel animatable 3D Gaussian model for rendering high-fidelity
free-view human motions in real time. Compared to existing NeRF-based methods,
the model owns better capability in synthesizing high-frequency details without
the jittering problem across video frames. The core of our model is a novel
augmented 3D Gaussian representation, which attaches each Gaussian with a
learnable code. The learnable code serves as a pose-dependent appearance
embedding for refining the erroneous appearance caused by geometric
transformation of Gaussians, based on which an appearance refinement model is
learned to produce residual Gaussian properties to match the appearance in
target pose. To force the Gaussians to learn the foreground human only without
background interference, we further design a novel alpha loss to explicitly
constrain the Gaussians within the human body. We also propose to jointly
optimize the human joint parameters to improve the appearance accuracy. The
animatable 3D Gaussian model can be learned with shallow MLPs, so new human
motions can be synthesized in real time (66 fps on avarage). Experiments show
that our model has superior performance over NeRF-based methods.

Comments:
- Some experiment data is wrong. The expression of the paper in
  introduction and abstract is incorrect. Some graphs have inappropriate
  descriptions

---

## LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-22 | Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, Kyoung Mu Lee | cs.CV | [PDF](http://arxiv.org/pdf/2311.13384v2){: .btn .btn-green } |

**Abstract**: With the widespread usage of VR devices and contents, demands for 3D scene
generation techniques become more popular. Existing 3D scene generation models,
however, limit the target scene to specific domain, primarily due to their
training strategies using 3D scan dataset that is far from the real-world. To
address such limitation, we propose LucidDreamer, a domain-free scene
generation pipeline by fully leveraging the power of existing large-scale
diffusion-based generative model. Our LucidDreamer has two alternate steps:
Dreaming and Alignment. First, to generate multi-view consistent images from
inputs, we set the point cloud as a geometrical guideline for each image
generation. Specifically, we project a portion of point cloud to the desired
view and provide the projection as a guidance for inpainting using the
generative model. The inpainted images are lifted to 3D space with estimated
depth maps, composing a new points. Second, to aggregate the new points into
the 3D scene, we propose an aligning algorithm which harmoniously integrates
the portions of newly generated 3D scenes. The finally obtained 3D scene serves
as initial points for optimizing Gaussian splats. LucidDreamer produces
Gaussian splats that are highly-detailed compared to the previous 3D scene
generation methods, with no constraint on domain of the target scene. Project
page: https://luciddreamer-cvlab.github.io/

Comments:
- Project page: https://luciddreamer-cvlab.github.io/

---

## Retargeting Visual Data with Deformation Fields



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-22 | Tim Elsner, Julia Berger, Tong Wu, Victor Czech, Lin Gao, Leif Kobbelt | cs.CV | [PDF](http://arxiv.org/pdf/2311.13297v1){: .btn .btn-green } |

**Abstract**: Seam carving is an image editing method that enable content-aware resizing,
including operations like removing objects. However, the seam-finding strategy
based on dynamic programming or graph-cut limits its applications to broader
visual data formats and degrees of freedom for editing. Our observation is that
describing the editing and retargeting of images more generally by a
displacement field yields a generalisation of content-aware deformations. We
propose to learn a deformation with a neural network that keeps the output
plausible while trying to deform it only in places with low information
content. This technique applies to different kinds of visual data, including
images, 3D scenes given as neural radiance fields, or even polygon meshes.
Experiments conducted on different visual data show that our method achieves
better content-aware retargeting compared to previous methods.

---

## Boosting3D: High-Fidelity Image-to-3D by Boosting 2D Diffusion Prior to  3D Prior with Progressive Learning

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-22 | Kai Yu, Jinlin Liu, Mengyang Feng, Miaomiao Cui, Xuansong Xie | cs.CV | [PDF](http://arxiv.org/pdf/2311.13617v1){: .btn .btn-green } |

**Abstract**: We present Boosting3D, a multi-stage single image-to-3D generation method
that can robustly generate reasonable 3D objects in different data domains. The
point of this work is to solve the view consistency problem in single
image-guided 3D generation by modeling a reasonable geometric structure. For
this purpose, we propose to utilize better 3D prior to training the NeRF. More
specifically, we train an object-level LoRA for the target object using
original image and the rendering output of NeRF. And then we train the LoRA and
NeRF using a progressive training strategy. The LoRA and NeRF will boost each
other while training. After the progressive training, the LoRA learns the 3D
information of the generated object and eventually turns to an object-level 3D
prior. In the final stage, we extract the mesh from the trained NeRF and use
the trained LoRA to optimize the structure and appearance of the mesh. The
experiments demonstrate the effectiveness of the proposed method. Boosting3D
learns object-specific 3D prior which is beyond the ability of pre-trained
diffusion priors and achieves state-of-the-art performance in the single
image-to-3d generation task.

Comments:
- 8 pages, 7 figures, 1 table

---

## 3D Face Style Transfer with a Hybrid Solution of NeRF and Mesh  Rasterization

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-22 | Jianwei Feng, Prateek Singhal | cs.CV | [PDF](http://arxiv.org/pdf/2311.13168v1){: .btn .btn-green } |

**Abstract**: Style transfer for human face has been widely researched in recent years.
Majority of the existing approaches work in 2D image domain and have 3D
inconsistency issue when applied on different viewpoints of the same face. In
this paper, we tackle the problem of 3D face style transfer which aims at
generating stylized novel views of a 3D human face with multi-view consistency.
We propose to use a neural radiance field (NeRF) to represent 3D human face and
combine it with 2D style transfer to stylize the 3D face. We find that directly
training a NeRF on stylized images from 2D style transfer brings in 3D
inconsistency issue and causes blurriness. On the other hand, training a NeRF
jointly with 2D style transfer objectives shows poor convergence due to the
identity and head pose gap between style image and content image. It also poses
challenge in training time and memory due to the need of volume rendering for
full image to apply style transfer loss functions. We therefore propose a
hybrid framework of NeRF and mesh rasterization to combine the benefits of high
fidelity geometry reconstruction of NeRF and fast rendering speed of mesh. Our
framework consists of three stages: 1. Training a NeRF model on input face
images to learn the 3D geometry; 2. Extracting a mesh from the trained NeRF
model and optimizing it with style transfer objectives via differentiable
rasterization; 3. Training a new color network in NeRF conditioned on a style
embedding to enable arbitrary style transfer to the 3D face. Experiment results
show that our approach generates high quality face style transfer with great 3D
consistency, while also enabling a flexible style control.

---

## PIE-NeRF: Physics-based Interactive Elastodynamics with NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-22 | Yutao Feng, Yintong Shang, Xuan Li, Tianjia Shao, Chenfanfu Jiang, Yin Yang | cs.CV | [PDF](http://arxiv.org/pdf/2311.13099v1){: .btn .btn-green } |

**Abstract**: We show that physics-based simulations can be seamlessly integrated with NeRF
to generate high-quality elastodynamics of real-world objects. Unlike existing
methods, we discretize nonlinear hyperelasticity in a meshless way, obviating
the necessity for intermediate auxiliary shape proxies like a tetrahedral mesh
or voxel grid. A quadratic generalized moving least square (Q-GMLS) is employed
to capture nonlinear dynamics and large deformation on the implicit model. Such
meshless integration enables versatile simulations of complex and codimensional
shapes. We adaptively place the least-square kernels according to the NeRF
density field to significantly reduce the complexity of the nonlinear
simulation. As a result, physically realistic animations can be conveniently
synthesized using our method for a wide range of hyperelastic materials at an
interactive rate. For more information, please visit our project page at
https://fytalon.github.io/pienerf/.

---

## Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot  Images

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-22 | Jaeyoung Chung, Jeongtaek Oh, Kyoung Mu Lee | cs.CV | [PDF](http://arxiv.org/pdf/2311.13398v3){: .btn .btn-green } |

**Abstract**: In this paper, we present a method to optimize Gaussian splatting with a
limited number of images while avoiding overfitting. Representing a 3D scene by
combining numerous Gaussian splats has yielded outstanding visual quality.
However, it tends to overfit the training views when only a small number of
images are available. To address this issue, we introduce a dense depth map as
a geometry guide to mitigate overfitting. We obtained the depth map using a
pre-trained monocular depth estimation model and aligning the scale and offset
using sparse COLMAP feature points. The adjusted depth aids in the color-based
optimization of 3D Gaussian splatting, mitigating floating artifacts, and
ensuring adherence to geometric constraints. We verify the proposed method on
the NeRF-LLFF dataset with varying numbers of few images. Our approach
demonstrates robust geometry compared to the original method that relies solely
on images. Project page: robot0321.github.io/DepthRegGS

Comments:
- 10 pages, 5 figures; Project page: robot0321.github.io/DepthRegGS

---

## SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh  Reconstruction and High-Quality Mesh Rendering

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-21 | Antoine Guédon, Vincent Lepetit | cs.GR | [PDF](http://arxiv.org/pdf/2311.12775v3){: .btn .btn-green } |

**Abstract**: We propose a method to allow precise and extremely fast mesh extraction from
3D Gaussian Splatting. Gaussian Splatting has recently become very popular as
it yields realistic rendering while being significantly faster to train than
NeRFs. It is however challenging to extract a mesh from the millions of tiny 3D
gaussians as these gaussians tend to be unorganized after optimization and no
method has been proposed so far. Our first key contribution is a regularization
term that encourages the gaussians to align well with the surface of the scene.
We then introduce a method that exploits this alignment to extract a mesh from
the Gaussians using Poisson reconstruction, which is fast, scalable, and
preserves details, in contrast to the Marching Cubes algorithm usually applied
to extract meshes from Neural SDFs. Finally, we introduce an optional
refinement strategy that binds gaussians to the surface of the mesh, and
jointly optimizes these Gaussians and the mesh through Gaussian splatting
rendering. This enables easy editing, sculpting, rigging, animating,
compositing and relighting of the Gaussians using traditional softwares by
manipulating the mesh instead of the gaussians themselves. Retrieving such an
editable mesh for realistic rendering is done within minutes with our method,
compared to hours with the state-of-the-art methods on neural SDFs, while
providing a better rendering quality. Our project page is the following:
https://anttwo.github.io/sugar/

Comments:
- We identified a minor typographical error in Equation 6; We updated
  the paper accordingly. Project Webpage: https://anttwo.github.io/sugar/

---

## Hyb-NeRF: A Multiresolution Hybrid Encoding for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-21 | Yifan Wang, Yi Gong, Yuan Zeng | cs.CV | [PDF](http://arxiv.org/pdf/2311.12490v1){: .btn .btn-green } |

**Abstract**: Recent advances in Neural radiance fields (NeRF) have enabled high-fidelity
scene reconstruction for novel view synthesis. However, NeRF requires hundreds
of network evaluations per pixel to approximate a volume rendering integral,
making it slow to train. Caching NeRFs into explicit data structures can
effectively enhance rendering speed but at the cost of higher memory usage. To
address these issues, we present Hyb-NeRF, a novel neural radiance field with a
multi-resolution hybrid encoding that achieves efficient neural modeling and
fast rendering, which also allows for high-quality novel view synthesis. The
key idea of Hyb-NeRF is to represent the scene using different encoding
strategies from coarse-to-fine resolution levels. Hyb-NeRF exploits
memory-efficiency learnable positional features at coarse resolutions and the
fast optimization speed and local details of hash-based feature grids at fine
resolutions. In addition, to further boost performance, we embed cone
tracing-based features in our learnable positional encoding that eliminates
encoding ambiguity and reduces aliasing artifacts. Extensive experiments on
both synthetic and real-world datasets show that Hyb-NeRF achieves faster
rendering speed with better rending quality and even a lower memory footprint
in comparison to previous state-of-the-art methods.

Comments:
- WACV2024

---

## An Efficient 3D Gaussian Representation for Monocular/Multi-view Dynamic  Scenes



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-21 | Kai Katsumata, Duc Minh Vo, Hideki Nakayama | cs.GR | [PDF](http://arxiv.org/pdf/2311.12897v1){: .btn .btn-green } |

**Abstract**: In novel view synthesis of scenes from multiple input views, 3D Gaussian
splatting emerges as a viable alternative to existing radiance field
approaches, delivering great visual quality and real-time rendering. While
successful in static scenes, the present advancement of 3D Gaussian
representation, however, faces challenges in dynamic scenes in terms of memory
consumption and the need for numerous observations per time step, due to the
onus of storing 3D Gaussian parameters per time step. In this study, we present
an efficient 3D Gaussian representation tailored for dynamic scenes in which we
define positions and rotations as functions of time while leaving other
time-invariant properties of the static 3D Gaussian unchanged. Notably, our
representation reduces memory usage, which is consistent regardless of the
input sequence length. Additionally, it mitigates the risk of overfitting
observed frames by accounting for temporal changes. The optimization of our
Gaussian representation based on image and flow reconstruction results in a
powerful framework for dynamic scene view synthesis in both monocular and
multi-view cases. We obtain the highest rendering speed of $118$ frames per
second (FPS) at a resolution of $1352 \times 1014$ with a single GPU, showing
the practical usability and effectiveness of our proposed method in dynamic
scene rendering scenarios.

Comments:
- 10 pages, 10 figures

---

## GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-20 | Chi Yan, Delin Qu, Dong Wang, Dan Xu, Zhigang Wang, Bin Zhao, Xuelong Li | cs.CV | [PDF](http://arxiv.org/pdf/2311.11700v3){: .btn .btn-green } |

**Abstract**: In this paper, we introduce $\textbf{GS-SLAM}$ that first utilizes 3D
Gaussian representation in the Simultaneous Localization and Mapping (SLAM)
system. It facilitates a better balance between efficiency and accuracy.
Compared to recent SLAM methods employing neural implicit representations, our
method utilizes a real-time differentiable splatting rendering pipeline that
offers significant speedup to map optimization and RGB-D re-rendering.
Specifically, we propose an adaptive expansion strategy that adds new or
deletes noisy 3D Gaussian in order to efficiently reconstruct new observed
scene geometry and improve the mapping of previously observed areas. This
strategy is essential to extend 3D Gaussian representation to reconstruct the
whole scene rather than synthesize a static object in existing methods.
Moreover, in the pose tracking process, an effective coarse-to-fine technique
is designed to select reliable 3D Gaussian representations to optimize camera
pose, resulting in runtime reduction and robust estimation. Our method achieves
competitive performance compared with existing state-of-the-art real-time
methods on the Replica, TUM-RGBD datasets. The source code will be released
soon.

---

## GP-NeRF: Generalized Perception NeRF for Context-Aware 3D Scene  Understanding

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-20 | Hao Li, Dingwen Zhang, Yalun Dai, Nian Liu, Lechao Cheng, Jingfeng Li, Jingdong Wang, Junwei Han | cs.CV | [PDF](http://arxiv.org/pdf/2311.11863v1){: .btn .btn-green } |

**Abstract**: Applying NeRF to downstream perception tasks for scene understanding and
representation is becoming increasingly popular. Most existing methods treat
semantic prediction as an additional rendering task, \textit{i.e.}, the "label
rendering" task, to build semantic NeRFs. However, by rendering
semantic/instance labels per pixel without considering the contextual
information of the rendered image, these methods usually suffer from unclear
boundary segmentation and abnormal segmentation of pixels within an object. To
solve this problem, we propose Generalized Perception NeRF (GP-NeRF), a novel
pipeline that makes the widely used segmentation model and NeRF work compatibly
under a unified framework, for facilitating context-aware 3D scene perception.
To accomplish this goal, we introduce transformers to aggregate radiance as
well as semantic embedding fields jointly for novel views and facilitate the
joint volumetric rendering of both fields. In addition, we propose two
self-distillation mechanisms, i.e., the Semantic Distill Loss and the
Depth-Guided Semantic Distill Loss, to enhance the discrimination and quality
of the semantic field and the maintenance of geometric consistency. In
evaluation, we conduct experimental comparisons under two perception tasks
(\textit{i.e.} semantic and instance segmentation) using both synthetic and
real-world datasets. Notably, our method outperforms SOTA approaches by 6.94\%,
11.76\%, and 8.47\% on generalized semantic segmentation, finetuning semantic
segmentation, and instance segmentation, respectively.

---

## Entangled View-Epipolar Information Aggregation for Generalizable Neural  Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-20 | Zhiyuan Min, Yawei Luo, Wei Yang, Yuesong Wang, Yi Yang | cs.CV | [PDF](http://arxiv.org/pdf/2311.11845v1){: .btn .btn-green } |

**Abstract**: Generalizable NeRF can directly synthesize novel views across new scenes,
eliminating the need for scene-specific retraining in vanilla NeRF. A critical
enabling factor in these approaches is the extraction of a generalizable 3D
representation by aggregating source-view features. In this paper, we propose
an Entangled View-Epipolar Information Aggregation method dubbed EVE-NeRF.
Different from existing methods that consider cross-view and along-epipolar
information independently, EVE-NeRF conducts the view-epipolar feature
aggregation in an entangled manner by injecting the scene-invariant appearance
continuity and geometry consistency priors to the aggregation process. Our
approach effectively mitigates the potential lack of inherent geometric and
appearance constraint resulting from one-dimensional interactions, thus further
boosting the 3D representation generalizablity. EVE-NeRF attains
state-of-the-art performance across various evaluation scenarios. Extensive
experiments demonstate that, compared to prevailing single-dimensional
aggregation, the entangled network excels in the accuracy of 3D scene geometry
and appearance reconstruction.Our project page is
https://github.com/tatakai1/EVENeRF.

---

## GaussianDiffusion: 3D Gaussian Splatting for Denoising Diffusion  Probabilistic Models with Structured Noise

gaussian splatting
{: .label .label-blue }

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-19 | Xinhai Li, Huaibin Wang, Kuo-Kun Tseng | cs.CV | [PDF](http://arxiv.org/pdf/2311.11221v1){: .btn .btn-green } |

**Abstract**: Text-to-3D, known for its efficient generation methods and expansive creative
potential, has garnered significant attention in the AIGC domain. However, the
amalgamation of Nerf and 2D diffusion models frequently yields oversaturated
images, posing severe limitations on downstream industrial applications due to
the constraints of pixelwise rendering method. Gaussian splatting has recently
superseded the traditional pointwise sampling technique prevalent in NeRF-based
methodologies, revolutionizing various aspects of 3D reconstruction. This paper
introduces a novel text to 3D content generation framework based on Gaussian
splatting, enabling fine control over image saturation through individual
Gaussian sphere transparencies, thereby producing more realistic images. The
challenge of achieving multi-view consistency in 3D generation significantly
impedes modeling complexity and accuracy. Taking inspiration from SJC, we
explore employing multi-view noise distributions to perturb images generated by
3D Gaussian splatting, aiming to rectify inconsistencies in multi-view
geometry. We ingeniously devise an efficient method to generate noise that
produces Gaussian noise from diverse viewpoints, all originating from a shared
noise source. Furthermore, vanilla 3D Gaussian-based generation tends to trap
models in local minima, causing artifacts like floaters, burrs, or
proliferative elements. To mitigate these issues, we propose the variational
Gaussian splatting technique to enhance the quality and stability of 3D
appearance. To our knowledge, our approach represents the first comprehensive
utilization of Gaussian splatting across the entire spectrum of 3D content
generation processes.

---

## LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval  Score Matching

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-19 | Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, Yingcong Chen | cs.CV | [PDF](http://arxiv.org/pdf/2311.11284v3){: .btn .btn-green } |

**Abstract**: The recent advancements in text-to-3D generation mark a significant milestone
in generative models, unlocking new possibilities for creating imaginative 3D
assets across various real-world scenarios. While recent advancements in
text-to-3D generation have shown promise, they often fall short in rendering
detailed and high-quality 3D models. This problem is especially prevalent as
many methods base themselves on Score Distillation Sampling (SDS). This paper
identifies a notable deficiency in SDS, that it brings inconsistent and
low-quality updating direction for the 3D model, causing the over-smoothing
effect. To address this, we propose a novel approach called Interval Score
Matching (ISM). ISM employs deterministic diffusing trajectories and utilizes
interval-based score matching to counteract over-smoothing. Furthermore, we
incorporate 3D Gaussian Splatting into our text-to-3D generation pipeline.
Extensive experiments show that our model largely outperforms the
state-of-the-art in quality and training efficiency.

Comments:
- The first two authors contributed equally to this work. Our code will
  be available at: https://github.com/EnVision-Research/LucidDreamer

---

## Towards Function Space Mesh Watermarking: Protecting the Copyright of  Signed Distance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-18 | Xingyu Zhu, Guanhui Ye, Chengdong Dong, Xiapu Luo, Xuetao Wei | cs.CV | [PDF](http://arxiv.org/pdf/2311.12059v1){: .btn .btn-green } |

**Abstract**: The signed distance field (SDF) represents 3D geometries in continuous
function space. Due to its continuous nature, explicit 3D models (e.g., meshes)
can be extracted from it at arbitrary resolution, which means losing the SDF is
equivalent to losing the mesh. Recent research has shown meshes can also be
extracted from SDF-enhanced neural radiance fields (NeRF). Such a signal raises
an alarm that any implicit neural representation with SDF enhancement can
extract the original mesh, which indicates identifying the SDF's intellectual
property becomes an urgent issue. This paper proposes FuncMark, a robust and
invisible watermarking method to protect the copyright of signed distance
fields by leveraging analytic on-surface deformations to embed binary watermark
messages. Such deformation can survive isosurfacing and thus be inherited by
the extracted meshes for further watermark message decoding. Our method can
recover the message with high-resolution meshes extracted from SDFs and detect
the watermark even when mesh vertices are extremely sparse. Furthermore, our
method is robust even when various distortions (including remeshing) are
encountered. Extensive experiments demonstrate that our \tool significantly
outperforms state-of-the-art approaches and the message is still detectable
even when only 50 vertex samples are given.

---

## SNI-SLAM: Semantic Neural Implicit SLAM

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-18 | Siting Zhu, Guangming Wang, Hermann Blum, Jiuming Liu, Liang Song, Marc Pollefeys, Hesheng Wang | cs.RO | [PDF](http://arxiv.org/pdf/2311.11016v1){: .btn .btn-green } |

**Abstract**: We propose SNI-SLAM, a semantic SLAM system utilizing neural implicit
representation, that simultaneously performs accurate semantic mapping,
high-quality surface reconstruction, and robust camera tracking. In this
system, we introduce hierarchical semantic representation to allow multi-level
semantic comprehension for top-down structured semantic mapping of the scene.
In addition, to fully utilize the correlation between multiple attributes of
the environment, we integrate appearance, geometry and semantic features
through cross-attention for feature collaboration. This strategy enables a more
multifaceted understanding of the environment, thereby allowing SNI-SLAM to
remain robust even when single attribute is defective. Then, we design an
internal fusion-based decoder to obtain semantic, RGB, Truncated Signed
Distance Field (TSDF) values from multi-level features for accurate decoding.
Furthermore, we propose a feature loss to update the scene representation at
the feature level. Compared with low-level losses such as RGB loss and depth
loss, our feature loss is capable of guiding the network optimization on a
higher-level. Our SNI-SLAM method demonstrates superior performance over all
recent NeRF-based SLAM methods in terms of mapping and tracking accuracy on
Replica and ScanNet datasets, while also showing excellent capabilities in
accurate semantic segmentation and real-time semantic mapping.

---

## Structure-Aware Sparse-View X-ray 3D Reconstruction

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-18 | Yuanhao Cai, Jiahao Wang, Alan Yuille, Zongwei Zhou, Angtian Wang | eess.IV | [PDF](http://arxiv.org/pdf/2311.10959v1){: .btn .btn-green } |

**Abstract**: X-ray, known for its ability to reveal internal structures of objects, is
expected to provide richer information for 3D reconstruction than visible
light. Yet, existing neural radiance fields (NeRF) algorithms overlook this
important nature of X-ray, leading to their limitations in capturing structural
contents of imaged objects. In this paper, we propose a framework,
Structure-Aware X-ray Neural Radiodensity Fields (SAX-NeRF), for sparse-view
X-ray 3D reconstruction. Firstly, we design a Line Segment-based Transformer
(Lineformer) as the backbone of SAX-NeRF. Linefomer captures internal
structures of objects in 3D space by modeling the dependencies within each line
segment of an X-ray. Secondly, we present a Masked Local-Global (MLG) ray
sampling strategy to extract contextual and geometric information in 2D
projection. Plus, we collect a larger-scale dataset X3D covering wider X-ray
applications. Experiments on X3D show that SAX-NeRF surpasses previous
NeRF-based methods by 12.56 and 2.49 dB on novel view synthesis and CT
reconstruction. Code, models, and data will be released at
https://github.com/caiyuanhao1998/SAX-NeRF

---

## Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-17 | Simon Niedermayr, Josef Stumpfegger, Rüdiger Westermann | cs.CV | [PDF](http://arxiv.org/pdf/2401.02436v1){: .btn .btn-green } |

**Abstract**: Recently, high-fidelity scene reconstruction with an optimized 3D Gaussian
splat representation has been introduced for novel view synthesis from sparse
image sets. Making such representations suitable for applications like network
streaming and rendering on low-power devices requires significantly reduced
memory consumption as well as improved rendering efficiency. We propose a
compressed 3D Gaussian splat representation that utilizes sensitivity-aware
vector clustering with quantization-aware training to compress directional
colors and Gaussian parameters. The learned codebooks have low bitrates and
achieve a compression rate of up to $31\times$ on real-world scenes with only
minimal degradation of visual quality. We demonstrate that the compressed splat
representation can be efficiently rendered with hardware rasterization on
lightweight GPUs at up to $4\times$ higher framerates than reported via an
optimized GPU compute pipeline. Extensive experiments across multiple datasets
demonstrate the robustness and rendering speed of the proposed approach.

---

## Removing Adverse Volumetric Effects From Trained Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-17 | Andreas L. Teigen, Mauhing Yip, Victor P. Hamran, Vegard Skui, Annette Stahl, Rudolf Mester | cs.CV | [PDF](http://arxiv.org/pdf/2311.10523v1){: .btn .btn-green } |

**Abstract**: While the use of neural radiance fields (NeRFs) in different challenging
settings has been explored, only very recently have there been any
contributions that focus on the use of NeRF in foggy environments. We argue
that the traditional NeRF models are able to replicate scenes filled with fog
and propose a method to remove the fog when synthesizing novel views. By
calculating the global contrast of a scene, we can estimate a density threshold
that, when applied, removes all visible fog. This makes it possible to use NeRF
as a way of rendering clear views of objects of interest located in fog-filled
environments. Additionally, to benchmark performance on such scenes, we
introduce a new dataset that expands some of the original synthetic NeRF scenes
through the addition of fog and natural environments. The code, dataset, and
video results can be found on our project page: https://vegardskui.com/fognerf/

Comments:
- This work has been submitted to the IEEE for possible publication.
  Copyright may be transferred without notice, after which this version may no
  longer be accessible

---

## SplatArmor: Articulated Gaussian splatting for animatable humans from  monocular RGB videos

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-17 | Rohit Jena, Ganesh Subramanian Iyer, Siddharth Choudhary, Brandon Smith, Pratik Chaudhari, James Gee | cs.CV | [PDF](http://arxiv.org/pdf/2311.10812v1){: .btn .btn-green } |

**Abstract**: We propose SplatArmor, a novel approach for recovering detailed and
animatable human models by `armoring' a parameterized body model with 3D
Gaussians. Our approach represents the human as a set of 3D Gaussians within a
canonical space, whose articulation is defined by extending the skinning of the
underlying SMPL geometry to arbitrary locations in the canonical space. To
account for pose-dependent effects, we introduce a SE(3) field, which allows us
to capture both the location and anisotropy of the Gaussians. Furthermore, we
propose the use of a neural color field to provide color regularization and 3D
supervision for the precise positioning of these Gaussians. We show that
Gaussian splatting provides an interesting alternative to neural rendering
based methods by leverging a rasterization primitive without facing any of the
non-differentiability and optimization challenges typically faced in such
approaches. The rasterization paradigms allows us to leverage forward skinning,
and does not suffer from the ambiguities associated with inverse skinning and
warping. We show compelling results on the ZJU MoCap and People Snapshot
datasets, which underscore the effectiveness of our method for controllable
human synthesis.

---

## Adaptive Shells for Efficient Neural Radiance Field Rendering



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-16 | Zian Wang, Tianchang Shen, Merlin Nimier-David, Nicholas Sharp, Jun Gao, Alexander Keller, Sanja Fidler, Thomas Müller, Zan Gojcic | cs.CV | [PDF](http://arxiv.org/pdf/2311.10091v1){: .btn .btn-green } |

**Abstract**: Neural radiance fields achieve unprecedented quality for novel view
synthesis, but their volumetric formulation remains expensive, requiring a huge
number of samples to render high-resolution images. Volumetric encodings are
essential to represent fuzzy geometry such as foliage and hair, and they are
well-suited for stochastic optimization. Yet, many scenes ultimately consist
largely of solid surfaces which can be accurately rendered by a single sample
per pixel. Based on this insight, we propose a neural radiance formulation that
smoothly transitions between volumetric- and surface-based rendering, greatly
accelerating rendering speed and even improving visual fidelity. Our method
constructs an explicit mesh envelope which spatially bounds a neural volumetric
representation. In solid regions, the envelope nearly converges to a surface
and can often be rendered with a single sample. To this end, we generalize the
NeuS formulation with a learned spatially-varying kernel size which encodes the
spread of the density, fitting a wide kernel to volume-like regions and a tight
kernel to surface-like regions. We then extract an explicit mesh of a narrow
band around the surface, with width determined by the kernel size, and
fine-tune the radiance field within this band. At inference time, we cast rays
against the mesh and evaluate the radiance field only within the enclosed
region, greatly reducing the number of samples required. Experiments show that
our approach enables efficient rendering at very high fidelity. We also
demonstrate that the extracted envelope enables downstream applications such as
animation and simulation.

Comments:
- SIGGRAPH Asia 2023. Project page:
  research.nvidia.com/labs/toronto-ai/adaptive-shells/

---

## EvaSurf: Efficient View-Aware Implicit Textured Surface Reconstruction  on Mobile Devices

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-16 | Jingnan Gao, Zhuo Chen, Yichao Yan, Bowen Pan, Zhe Wang, Jiangjing Lyu, Xiaokang Yang | cs.CV | [PDF](http://arxiv.org/pdf/2311.09806v2){: .btn .btn-green } |

**Abstract**: Reconstructing real-world 3D objects has numerous applications in computer
vision, such as virtual reality, video games, and animations. Ideally, 3D
reconstruction methods should generate high-fidelity results with 3D
consistency in real-time. Traditional methods match pixels between images using
photo-consistency constraints or learned features, while differentiable
rendering methods like Neural Radiance Fields (NeRF) use differentiable volume
rendering or surface-based representation to generate high-fidelity scenes.
However, these methods require excessive runtime for rendering, making them
impractical for daily applications. To address these challenges, we present
$\textbf{EvaSurf}$, an $\textbf{E}$fficient $\textbf{V}$iew-$\textbf{A}$ware
implicit textured $\textbf{Surf}$ace reconstruction method on mobile devices.
In our method, we first employ an efficient surface-based model with a
multi-view supervision module to ensure accurate mesh reconstruction. To enable
high-fidelity rendering, we learn an implicit texture embedded with a set of
Gaussian lobes to capture view-dependent information. Furthermore, with the
explicit geometry and the implicit texture, we can employ a lightweight neural
shader to reduce the expense of computation and further support real-time
rendering on common mobile devices. Extensive experiments demonstrate that our
method can reconstruct high-quality appearance and accurate mesh on both
synthetic and real-world datasets. Moreover, our method can be trained in just
1-2 hours using a single GPU and run on mobile devices at over 40 FPS (Frames
Per Second), with a final package required for rendering taking up only 40-50
MB.

Comments:
- Project Page: http://g-1nonly.github.io/EvaSurf-Website/

---

## Reconstructing Continuous Light Field From Single Coded Image

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-16 | Yuya Ishikawa, Keita Takahashi, Chihiro Tsutake, Toshiaki Fujii | cs.CV | [PDF](http://arxiv.org/pdf/2311.09646v1){: .btn .btn-green } |

**Abstract**: We propose a method for reconstructing a continuous light field of a target
scene from a single observed image. Our method takes the best of two worlds:
joint aperture-exposure coding for compressive light-field acquisition, and a
neural radiance field (NeRF) for view synthesis. Joint aperture-exposure coding
implemented in a camera enables effective embedding of 3-D scene information
into an observed image, but in previous works, it was used only for
reconstructing discretized light-field views. NeRF-based neural rendering
enables high quality view synthesis of a 3-D scene from continuous viewpoints,
but when only a single image is given as the input, it struggles to achieve
satisfactory quality. Our method integrates these two techniques into an
efficient and end-to-end trainable pipeline. Trained on a wide variety of
scenes, our method can reconstruct continuous light fields accurately and
efficiently without any test time optimization. To our knowledge, this is the
first work to bridge two worlds: camera design for efficiently acquiring 3-D
information and neural rendering.

---

## Single-Image 3D Human Digitization with Shape-Guided Diffusion

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-15 | Badour AlBahar, Shunsuke Saito, Hung-Yu Tseng, Changil Kim, Johannes Kopf, Jia-Bin Huang | cs.CV | [PDF](http://arxiv.org/pdf/2311.09221v1){: .btn .btn-green } |

**Abstract**: We present an approach to generate a 360-degree view of a person with a
consistent, high-resolution appearance from a single input image. NeRF and its
variants typically require videos or images from different viewpoints. Most
existing approaches taking monocular input either rely on ground-truth 3D scans
for supervision or lack 3D consistency. While recent 3D generative models show
promise of 3D consistent human digitization, these approaches do not generalize
well to diverse clothing appearances, and the results lack photorealism. Unlike
existing work, we utilize high-capacity 2D diffusion models pretrained for
general image synthesis tasks as an appearance prior of clothed humans. To
achieve better 3D consistency while retaining the input identity, we
progressively synthesize multiple views of the human in the input image by
inpainting missing regions with shape-guided diffusion conditioned on
silhouette and surface normal. We then fuse these synthesized multi-view images
via inverse rendering to obtain a fully textured high-resolution 3D mesh of the
given person. Experiments show that our approach outperforms prior methods and
achieves photorealistic 360-degree synthesis of a wide range of clothed humans
with complex textures from a single image.

Comments:
- SIGGRAPH Asia 2023. Project website: https://human-sgd.github.io/

---

## DMV3D: Denoising Multi-View Diffusion using 3D Large Reconstruction  Model

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-15 | Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, Kai Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2311.09217v1){: .btn .btn-green } |

**Abstract**: We propose \textbf{DMV3D}, a novel 3D generation approach that uses a
transformer-based 3D large reconstruction model to denoise multi-view
diffusion. Our reconstruction model incorporates a triplane NeRF representation
and can denoise noisy multi-view images via NeRF reconstruction and rendering,
achieving single-stage 3D generation in $\sim$30s on single A100 GPU. We train
\textbf{DMV3D} on large-scale multi-view image datasets of highly diverse
objects using only image reconstruction losses, without accessing 3D assets. We
demonstrate state-of-the-art results for the single-image reconstruction
problem where probabilistic modeling of unseen object parts is required for
generating diverse reconstructions with sharp textures. We also show
high-quality text-to-3D generation results outperforming previous 3D diffusion
models. Our project website is at: https://justimyhxu.github.io/projects/dmv3d/ .

Comments:
- Project Page: https://justimyhxu.github.io/projects/dmv3d/

---

## Spiking NeRF: Representing the Real-World Geometry by a Discontinuous  Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-15 | Zhanfeng Liao, Qian Zheng, Yan Liu, Gang Pan | cs.CV | [PDF](http://arxiv.org/pdf/2311.09077v2){: .btn .btn-green } |

**Abstract**: A crucial reason for the success of existing NeRF-based methods is to build a
neural density field for the geometry representation via multiple perceptron
layers (MLPs). MLPs are continuous functions, however, real geometry or density
field is frequently discontinuous at the interface between the air and the
surface. Such a contrary brings the problem of unfaithful geometry
representation. To this end, this paper proposes spiking NeRF, which leverages
spiking neurons and a hybrid Artificial Neural Network (ANN)-Spiking Neural
Network (SNN) framework to build a discontinuous density field for faithful
geometry representation. Specifically, we first demonstrate the reason why
continuous density fields will bring inaccuracy. Then, we propose to use the
spiking neurons to build a discontinuous density field. We conduct a
comprehensive analysis for the problem of existing spiking neuron models and
then provide the numerical relationship between the parameter of the spiking
neuron and the theoretical accuracy of geometry. Based on this, we propose a
bounded spiking neuron to build the discontinuous density field. Our method
achieves SOTA performance. The source code and the supplementary material are
available at https://github.com/liaozhanfeng/Spiking-NeRF.

---

## Drivable 3D Gaussian Avatars

gaussian splatting
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-14 | Wojciech Zielonka, Timur Bagautdinov, Shunsuke Saito, Michael Zollhöfer, Justus Thies, Javier Romero | cs.CV | [PDF](http://arxiv.org/pdf/2311.08581v1){: .btn .btn-green } |

**Abstract**: We present Drivable 3D Gaussian Avatars (D3GA), the first 3D controllable
model for human bodies rendered with Gaussian splats. Current photorealistic
drivable avatars require either accurate 3D registrations during training,
dense input images during testing, or both. The ones based on neural radiance
fields also tend to be prohibitively slow for telepresence applications. This
work uses the recently presented 3D Gaussian Splatting (3DGS) technique to
render realistic humans at real-time framerates, using dense calibrated
multi-view videos as input. To deform those primitives, we depart from the
commonly used point deformation method of linear blend skinning (LBS) and use a
classic volumetric deformation method: cage deformations. Given their smaller
size, we drive these deformations with joint angles and keypoints, which are
more suitable for communication applications. Our experiments on nine subjects
with varied body shapes, clothes, and motions obtain higher-quality results
than state-of-the-art methods when using the same training and test data.

Comments:
- Website: https://zielon.github.io/d3ga/

---

## $L_0$-Sampler: An $L_{0}$ Model Guided Volume Sampling for NeRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-13 | Liangchen Li, Juyong Zhang | cs.CV | [PDF](http://arxiv.org/pdf/2311.07044v1){: .btn .btn-green } |

**Abstract**: Since being proposed, Neural Radiance Fields (NeRF) have achieved great
success in related tasks, mainly adopting the hierarchical volume sampling
(HVS) strategy for volume rendering. However, the HVS of NeRF approximates
distributions using piecewise constant functions, which provides a relatively
rough estimation. Based on the observation that a well-trained weight function
$w(t)$ and the $L_0$ distance between points and the surface have very high
similarity, we propose $L_0$-Sampler by incorporating the $L_0$ model into
$w(t)$ to guide the sampling process. Specifically, we propose to use piecewise
exponential functions rather than piecewise constant functions for
interpolation, which can not only approximate quasi-$L_0$ weight distributions
along rays quite well but also can be easily implemented with few lines of code
without additional computational burden. Stable performance improvements can be
achieved by applying $L_0$-Sampler to NeRF and its related tasks like 3D
reconstruction. Code is available at https://ustc3dv.github.io/L0-Sampler/ .

Comments:
- Project page: https://ustc3dv.github.io/L0-Sampler/

---

## Aria-NeRF: Multimodal Egocentric View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-11 | Jiankai Sun, Jianing Qiu, Chuanyang Zheng, John Tucker, Javier Yu, Mac Schwager | cs.CV | [PDF](http://arxiv.org/pdf/2311.06455v1){: .btn .btn-green } |

**Abstract**: We seek to accelerate research in developing rich, multimodal scene models
trained from egocentric data, based on differentiable volumetric ray-tracing
inspired by Neural Radiance Fields (NeRFs). The construction of a NeRF-like
model from an egocentric image sequence plays a pivotal role in understanding
human behavior and holds diverse applications within the realms of VR/AR. Such
egocentric NeRF-like models may be used as realistic simulations, contributing
significantly to the advancement of intelligent agents capable of executing
tasks in the real-world. The future of egocentric view synthesis may lead to
novel environment representations going beyond today's NeRFs by augmenting
visual data with multimodal sensors such as IMU for egomotion tracking, audio
sensors to capture surface texture and human language context, and eye-gaze
trackers to infer human attention patterns in the scene. To support and
facilitate the development and evaluation of egocentric multimodal scene
modeling, we present a comprehensive multimodal egocentric video dataset. This
dataset offers a comprehensive collection of sensory data, featuring RGB
images, eye-tracking camera footage, audio recordings from a microphone,
atmospheric pressure readings from a barometer, positional coordinates from
GPS, connectivity details from Wi-Fi and Bluetooth, and information from
dual-frequency IMU datasets (1kHz and 800Hz) paired with a magnetometer. The
dataset was collected with the Meta Aria Glasses wearable device platform. The
diverse data modalities and the real-world context captured within this dataset
serve as a robust foundation for furthering our understanding of human behavior
and enabling more immersive and intelligent experiences in the realms of VR,
AR, and robotics.

---

## Instant3D: Fast Text-to-3D with Sparse-View Generation and Large  Reconstruction Model

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-10 | Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, Sai Bi | cs.CV | [PDF](http://arxiv.org/pdf/2311.06214v2){: .btn .btn-green } |

**Abstract**: Text-to-3D with diffusion models has achieved remarkable progress in recent
years. However, existing methods either rely on score distillation-based
optimization which suffer from slow inference, low diversity and Janus
problems, or are feed-forward methods that generate low-quality results due to
the scarcity of 3D training data. In this paper, we propose Instant3D, a novel
method that generates high-quality and diverse 3D assets from text prompts in a
feed-forward manner. We adopt a two-stage paradigm, which first generates a
sparse set of four structured and consistent views from text in one shot with a
fine-tuned 2D text-to-image diffusion model, and then directly regresses the
NeRF from the generated images with a novel transformer-based sparse-view
reconstructor. Through extensive experiments, we demonstrate that our method
can generate diverse 3D assets of high visual quality within 20 seconds, which
is two orders of magnitude faster than previous optimization-based methods that
can take 1 to 10 hours. Our project webpage: https://jiahao.ai/instant3d/.

Comments:
- Project webpage: https://jiahao.ai/instant3d/

---

## ASSIST: Interactive Scene Nodes for Scalable and Realistic Indoor  Simulation



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-10 | Zhide Zhong, Jiakai Cao, Songen Gu, Sirui Xie, Weibo Gao, Liyi Luo, Zike Yan, Hao Zhao, Guyue Zhou | cs.CV | [PDF](http://arxiv.org/pdf/2311.06211v1){: .btn .btn-green } |

**Abstract**: We present ASSIST, an object-wise neural radiance field as a panoptic
representation for compositional and realistic simulation. Central to our
approach is a novel scene node data structure that stores the information of
each object in a unified fashion, allowing online interaction in both intra-
and cross-scene settings. By incorporating a differentiable neural network
along with the associated bounding box and semantic features, the proposed
structure guarantees user-friendly interaction on independent objects to scale
up novel view simulation. Objects in the scene can be queried, added,
duplicated, deleted, transformed, or swapped simply through mouse/keyboard
controls or language instructions. Experiments demonstrate the efficacy of the
proposed method, where scaled realistic simulation can be achieved through
interactive editing and compositional rendering, with color images, depth
images, and panoptic segmentation masks generated in a 3D consistent manner.

---

## A Neural Height-Map Approach for the Binocular Photometric Stereo  Problem

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-10 | Fotios Logothetis, Ignas Budvytis, Roberto Cipolla | cs.CV | [PDF](http://arxiv.org/pdf/2311.05958v1){: .btn .btn-green } |

**Abstract**: In this work we propose a novel, highly practical, binocular photometric
stereo (PS) framework, which has same acquisition speed as single view PS,
however significantly improves the quality of the estimated geometry.
  As in recent neural multi-view shape estimation frameworks such as NeRF,
SIREN and inverse graphics approaches to multi-view photometric stereo (e.g.
PS-NeRF) we formulate shape estimation task as learning of a differentiable
surface and texture representation by minimising surface normal discrepancy for
normals estimated from multiple varying light images for two views as well as
discrepancy between rendered surface intensity and observed images. Our method
differs from typical multi-view shape estimation approaches in two key ways.
First, our surface is represented not as a volume but as a neural heightmap
where heights of points on a surface are computed by a deep neural network.
Second, instead of predicting an average intensity as PS-NeRF or introducing
lambertian material assumptions as Guo et al., we use a learnt BRDF and perform
near-field per point intensity rendering.
  Our method achieves the state-of-the-art performance on the DiLiGenT-MV
dataset adapted to binocular stereo setup as well as a new binocular
photometric stereo dataset - LUCES-ST.

Comments:
- WACV 2024

---

## UMedNeRF: Uncertainty-aware Single View Volumetric Rendering for Medical  Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-10 | Jing Hu, Qinrui Fan, Shu Hu, Siwei Lyu, Xi Wu, Xin Wang | eess.IV | [PDF](http://arxiv.org/pdf/2311.05836v4){: .btn .btn-green } |

**Abstract**: In the field of clinical medicine, computed tomography (CT) is an effective
medical imaging modality for the diagnosis of various pathologies. Compared
with X-ray images, CT images can provide more information, including
multi-planar slices and three-dimensional structures for clinical diagnosis.
However, CT imaging requires patients to be exposed to large doses of ionizing
radiation for a long time, which may cause irreversible physical harm. In this
paper, we propose an Uncertainty-aware MedNeRF (UMedNeRF) network based on
generated radiation fields. The network can learn a continuous representation
of CT projections from 2D X-ray images by obtaining the internal structure and
depth information and using adaptive loss weights to ensure the quality of the
generated images. Our model is trained on publicly available knee and chest
datasets, and we show the results of CT projection rendering with a single
X-ray and compare our method with other methods based on generated radiation
fields.

---

## BakedAvatar: Baking Neural Fields for Real-Time Head Avatar Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-09 | Hao-Bin Duan, Miao Wang, Jin-Chuan Shi, Xu-Chuan Chen, Yan-Pei Cao | cs.GR | [PDF](http://arxiv.org/pdf/2311.05521v2){: .btn .btn-green } |

**Abstract**: Synthesizing photorealistic 4D human head avatars from videos is essential
for VR/AR, telepresence, and video game applications. Although existing Neural
Radiance Fields (NeRF)-based methods achieve high-fidelity results, the
computational expense limits their use in real-time applications. To overcome
this limitation, we introduce BakedAvatar, a novel representation for real-time
neural head avatar synthesis, deployable in a standard polygon rasterization
pipeline. Our approach extracts deformable multi-layer meshes from learned
isosurfaces of the head and computes expression-, pose-, and view-dependent
appearances that can be baked into static textures for efficient rasterization.
We thus propose a three-stage pipeline for neural head avatar synthesis, which
includes learning continuous deformation, manifold, and radiance fields,
extracting layered meshes and textures, and fine-tuning texture details with
differential rasterization. Experimental results demonstrate that our
representation generates synthesis results of comparable quality to other
state-of-the-art methods while significantly reducing the inference time
required. We further showcase various head avatar synthesis results from
monocular videos, including view synthesis, face reenactment, expression
editing, and pose editing, all at interactive frame rates.

Comments:
- ACM Transactions on Graphics (SIGGRAPH Asia 2023). Project Page:
  https://buaavrcg.github.io/BakedAvatar

---

## Control3D: Towards Controllable Text-to-3D Generation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-09 | Yang Chen, Yingwei Pan, Yehao Li, Ting Yao, Tao Mei | cs.CV | [PDF](http://arxiv.org/pdf/2311.05461v1){: .btn .btn-green } |

**Abstract**: Recent remarkable advances in large-scale text-to-image diffusion models have
inspired a significant breakthrough in text-to-3D generation, pursuing 3D
content creation solely from a given text prompt. However, existing text-to-3D
techniques lack a crucial ability in the creative process: interactively
control and shape the synthetic 3D contents according to users' desired
specifications (e.g., sketch). To alleviate this issue, we present the first
attempt for text-to-3D generation conditioning on the additional hand-drawn
sketch, namely Control3D, which enhances controllability for users. In
particular, a 2D conditioned diffusion model (ControlNet) is remoulded to guide
the learning of 3D scene parameterized as NeRF, encouraging each view of 3D
scene aligned with the given text prompt and hand-drawn sketch. Moreover, we
exploit a pre-trained differentiable photo-to-sketch model to directly estimate
the sketch of the rendered image over synthetic 3D scene. Such estimated sketch
along with each sampled view is further enforced to be geometrically consistent
with the given sketch, pursuing better controllable text-to-3D generation.
Through extensive experiments, we demonstrate that our proposal can generate
accurate and faithful 3D scenes that align closely with the input text prompts
and sketches.

Comments:
- ACM Multimedia 2023

---

## VoxNeRF: Bridging Voxel Representation and Neural Radiance Fields for  Enhanced Indoor View Synthesis

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-09 | Sen Wang, Wei Zhang, Stefano Gasperini, Shun-Cheng Wu, Nassir Navab | cs.CV | [PDF](http://arxiv.org/pdf/2311.05289v1){: .btn .btn-green } |

**Abstract**: Creating high-quality view synthesis is essential for immersive applications
but continues to be problematic, particularly in indoor environments and for
real-time deployment. Current techniques frequently require extensive
computational time for both training and rendering, and often produce
less-than-ideal 3D representations due to inadequate geometric structuring. To
overcome this, we introduce VoxNeRF, a novel approach that leverages volumetric
representations to enhance the quality and efficiency of indoor view synthesis.
Firstly, VoxNeRF constructs a structured scene geometry and converts it into a
voxel-based representation. We employ multi-resolution hash grids to adaptively
capture spatial features, effectively managing occlusions and the intricate
geometry of indoor scenes. Secondly, we propose a unique voxel-guided efficient
sampling technique. This innovation selectively focuses computational resources
on the most relevant portions of ray segments, substantially reducing
optimization time. We validate our approach against three public indoor
datasets and demonstrate that VoxNeRF outperforms state-of-the-art methods.
Remarkably, it achieves these gains while reducing both training and rendering
times, surpassing even Instant-NGP in speed and bringing the technology closer
to real-time.

Comments:
- 8 pages, 4 figures

---

## ConRad: Image Constrained Radiance Fields for 3D Generation from a  Single Image



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-09 | Senthil Purushwalkam, Nikhil Naik | cs.CV | [PDF](http://arxiv.org/pdf/2311.05230v1){: .btn .btn-green } |

**Abstract**: We present a novel method for reconstructing 3D objects from a single RGB
image. Our method leverages the latest image generation models to infer the
hidden 3D structure while remaining faithful to the input image. While existing
methods obtain impressive results in generating 3D models from text prompts,
they do not provide an easy approach for conditioning on input RGB data.
Na\"ive extensions of these methods often lead to improper alignment in
appearance between the input image and the 3D reconstructions. We address these
challenges by introducing Image Constrained Radiance Fields (ConRad), a novel
variant of neural radiance fields. ConRad is an efficient 3D representation
that explicitly captures the appearance of an input image in one viewpoint. We
propose a training algorithm that leverages the single RGB image in conjunction
with pretrained Diffusion Models to optimize the parameters of a ConRad
representation. Extensive experiments show that ConRad representations can
simplify preservation of image details while producing a realistic 3D
reconstruction. Compared to existing state-of-the-art baselines, we show that
our 3D reconstructions remain more faithful to the input and produce more
consistent 3D models while demonstrating significantly improved quantitative
performance on a ShapeNet object benchmark.

Comments:
- Advances in Neural Information Processing Systems (NeurIPS 2023)

---

## LRM: Large Reconstruction Model for Single Image to 3D

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-08 | Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, Hao Tan | cs.CV | [PDF](http://arxiv.org/pdf/2311.04400v1){: .btn .btn-green } |

**Abstract**: We propose the first Large Reconstruction Model (LRM) that predicts the 3D
model of an object from a single input image within just 5 seconds. In contrast
to many previous methods that are trained on small-scale datasets such as
ShapeNet in a category-specific fashion, LRM adopts a highly scalable
transformer-based architecture with 500 million learnable parameters to
directly predict a neural radiance field (NeRF) from the input image. We train
our model in an end-to-end manner on massive multi-view data containing around
1 million objects, including both synthetic renderings from Objaverse and real
captures from MVImgNet. This combination of a high-capacity model and
large-scale training data empowers our model to be highly generalizable and
produce high-quality 3D reconstructions from various testing inputs including
real-world in-the-wild captures and images from generative models. Video demos
and interactable 3D meshes can be found on this website:
https://yiconghong.me/LRM/.

Comments:
- 23 pages

---

## Learning Robust Multi-Scale Representation for Neural Radiance Fields  from Unposed Images



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-08 | Nishant Jain, Suryansh Kumar, Luc Van Gool | cs.CV | [PDF](http://arxiv.org/pdf/2311.04521v1){: .btn .btn-green } |

**Abstract**: We introduce an improved solution to the neural image-based rendering problem
in computer vision. Given a set of images taken from a freely moving camera at
train time, the proposed approach could synthesize a realistic image of the
scene from a novel viewpoint at test time. The key ideas presented in this
paper are (i) Recovering accurate camera parameters via a robust pipeline from
unposed day-to-day images is equally crucial in neural novel view synthesis
problem; (ii) It is rather more practical to model object's content at
different resolutions since dramatic camera motion is highly likely in
day-to-day unposed images. To incorporate the key ideas, we leverage the
fundamentals of scene rigidity, multi-scale neural scene representation, and
single-image depth prediction. Concretely, the proposed approach makes the
camera parameters as learnable in a neural fields-based modeling framework. By
assuming per view depth prediction is given up to scale, we constrain the
relative pose between successive frames. From the relative poses, absolute
camera pose estimation is modeled via a graph-neural network-based multiple
motion averaging within the multi-scale neural-fields network, leading to a
single loss function. Optimizing the introduced loss function provides camera
intrinsic, extrinsic, and image rendering from unposed images. We demonstrate,
with examples, that for a unified framework to accurately model multiscale
neural scene representation from day-to-day acquired unposed multi-view images,
it is equally essential to have precise camera-pose estimates within the scene
representation framework. Without considering robustness measures in the camera
pose estimation pipeline, modeling for multi-scale aliasing artifacts can be
counterproductive. We present extensive experiments on several benchmark
datasets to demonstrate the suitability of our approach.

Comments:
- Accepted for publication at International Journal of Computer Vision
  (IJCV). Draft info: 22 pages, 12 figures and 14 tables

---

## High-fidelity 3D Reconstruction of Plants using Neural Radiance Field

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-07 | Kewei Hu, Ying Wei, Yaoqiang Pan, Hanwen Kang, Chao Chen | cs.CV | [PDF](http://arxiv.org/pdf/2311.04154v1){: .btn .btn-green } |

**Abstract**: Accurate reconstruction of plant phenotypes plays a key role in optimising
sustainable farming practices in the field of Precision Agriculture (PA).
Currently, optical sensor-based approaches dominate the field, but the need for
high-fidelity 3D reconstruction of crops and plants in unstructured
agricultural environments remains challenging. Recently, a promising
development has emerged in the form of Neural Radiance Field (NeRF), a novel
method that utilises neural density fields. This technique has shown impressive
performance in various novel vision synthesis tasks, but has remained
relatively unexplored in the agricultural context. In our study, we focus on
two fundamental tasks within plant phenotyping: (1) the synthesis of 2D
novel-view images and (2) the 3D reconstruction of crop and plant models. We
explore the world of neural radiance fields, in particular two SOTA methods:
Instant-NGP, which excels in generating high-quality images with impressive
training and inference speed, and Instant-NSR, which improves the reconstructed
geometry by incorporating the Signed Distance Function (SDF) during training.
In particular, we present a novel plant phenotype dataset comprising real plant
images from production environments. This dataset is a first-of-its-kind
initiative aimed at comprehensively exploring the advantages and limitations of
NeRF in agricultural contexts. Our experimental results show that NeRF
demonstrates commendable performance in the synthesis of novel-view images and
is able to achieve reconstruction results that are competitive with Reality
Capture, a leading commercial software for 3D Multi-View Stereo (MVS)-based
reconstruction. However, our study also highlights certain drawbacks of NeRF,
including relatively slow training speeds, performance limitations in cases of
insufficient sampling, and challenges in obtaining geometry quality in complex
setups.

---

## Fast Sun-aligned Outdoor Scene Relighting based on TensoRF

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-07 | Yeonjin Chang, Yearim Kim, Seunghyeon Seo, Jung Yi, Nojun Kwak | cs.CV | [PDF](http://arxiv.org/pdf/2311.03965v1){: .btn .btn-green } |

**Abstract**: In this work, we introduce our method of outdoor scene relighting for Neural
Radiance Fields (NeRF) named Sun-aligned Relighting TensoRF (SR-TensoRF).
SR-TensoRF offers a lightweight and rapid pipeline aligned with the sun,
thereby achieving a simplified workflow that eliminates the need for
environment maps. Our sun-alignment strategy is motivated by the insight that
shadows, unlike viewpoint-dependent albedo, are determined by light direction.
We directly use the sun direction as an input during shadow generation,
simplifying the requirements of the inference process significantly. Moreover,
SR-TensoRF leverages the training efficiency of TensoRF by incorporating our
proposed cubemap concept, resulting in notable acceleration in both training
and rendering processes compared to existing methods.

Comments:
- WACV 2024

---

## UP-NeRF: Unconstrained Pose-Prior-Free Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-07 | Injae Kim, Minhyuk Choi, Hyunwoo J. Kim | cs.CV | [PDF](http://arxiv.org/pdf/2311.03784v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Field (NeRF) has enabled novel view synthesis with high
fidelity given images and camera poses. Subsequent works even succeeded in
eliminating the necessity of pose priors by jointly optimizing NeRF and camera
pose. However, these works are limited to relatively simple settings such as
photometrically consistent and occluder-free image collections or a sequence of
images from a video. So they have difficulty handling unconstrained images with
varying illumination and transient occluders. In this paper, we propose
$\textbf{UP-NeRF}$ ($\textbf{U}$nconstrained $\textbf{P}$ose-prior-free
$\textbf{Ne}$ural $\textbf{R}$adiance $\textbf{F}$ields) to optimize NeRF with
unconstrained image collections without camera pose prior. We tackle these
challenges with surrogate tasks that optimize color-insensitive feature fields
and a separate module for transient occluders to block their influence on pose
estimation. In addition, we introduce a candidate head to enable more robust
pose estimation and transient-aware depth supervision to minimize the effect of
incorrect prior. Our experiments verify the superior performance of our method
compared to the baselines including BARF and its variants in a challenging
internet photo collection, $\textit{Phototourism}$ dataset.

Comments:
- Neural Information Processing Systems (NeurIPS), 2023. The code is
  available at https://github.com/mlvlab/UP-NeRF

---

## ADFactory: An Effective Framework for Generalizing Optical Flow with  Nerf

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-07 | Han Ling | cs.CV | [PDF](http://arxiv.org/pdf/2311.04246v2){: .btn .btn-green } |

**Abstract**: A significant challenge facing current optical flow methods is the difficulty
in generalizing them well to the real world. This is mainly due to the high
cost of hand-crafted datasets, and existing self-supervised methods are limited
by indirect loss and occlusions, resulting in fuzzy outcomes. To address this
challenge, we introduce a novel optical flow training framework: automatic data
factory (ADF). ADF only requires RGB images as input to effectively train the
optical flow network on the target data domain. Specifically, we use advanced
Nerf technology to reconstruct scenes from photo groups collected by a
monocular camera, and then calculate optical flow labels between camera pose
pairs based on the rendering results. To eliminate erroneous labels caused by
defects in the scene reconstructed by Nerf, we screened the generated labels
from multiple aspects, such as optical flow matching accuracy, radiation field
confidence, and depth consistency. The filtered labels can be directly used for
network supervision. Experimentally, the generalization ability of ADF on KITTI
surpasses existing self-supervised optical flow and monocular scene flow
algorithms. In addition, ADF achieves impressive results in real-world
zero-point generalization evaluations and surpasses most supervised methods.

Comments:
- 8 pages

---

## Osprey: Multi-Session Autonomous Aerial Mapping with LiDAR-based SLAM  and Next Best View Planning

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-06 | Rowan Border, Nived Chebrolu, Yifu Tao, Jonathan D. Gammell, Maurice Fallon | cs.RO | [PDF](http://arxiv.org/pdf/2311.03484v1){: .btn .btn-green } |

**Abstract**: Aerial mapping systems are important for many surveying applications (e.g.,
industrial inspection or agricultural monitoring). Semi-autonomous mapping with
GPS-guided aerial platforms that fly preplanned missions is already widely
available but fully autonomous systems can significantly improve efficiency.
Autonomously mapping complex 3D structures requires a system that performs
online mapping and mission planning. This paper presents Osprey, an autonomous
aerial mapping system with state-of-the-art multi-session mapping capabilities.
It enables a non-expert operator to specify a bounded target area that the
aerial platform can then map autonomously, over multiple flights if necessary.
Field experiments with Osprey demonstrate that this system can achieve greater
map coverage of large industrial sites than manual surveys with a pilot-flown
aerial platform or a terrestrial laser scanner (TLS). Three sites, with a total
ground coverage of $7085$ m$^2$ and a maximum height of $27$ m, were mapped in
separate missions using $112$ minutes of autonomous flight time. True colour
maps were created from images captured by Osprey using pointcloud and NeRF
reconstruction methods. These maps provide useful data for structural
inspection tasks.

Comments:
- Submitted to Field Robotics, Manuscript #FR-23-0016. 25 pages, 15
  figures, 3 tables. Video available at
  https://www.youtube.com/watch?v=CVIXu2qUQJ8

---

## Long-Term Invariant Local Features via Implicit Cross-Domain  Correspondences



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-06 | Zador Pataki, Mohammad Altillawi, Menelaos Kanakis, Rémi Pautrat, Fengyi Shen, Ziyuan Liu, Luc Van Gool, Marc Pollefeys | cs.CV | [PDF](http://arxiv.org/pdf/2311.03345v1){: .btn .btn-green } |

**Abstract**: Modern learning-based visual feature extraction networks perform well in
intra-domain localization, however, their performance significantly declines
when image pairs are captured across long-term visual domain variations, such
as different seasonal and daytime variations. In this paper, our first
contribution is a benchmark to investigate the performance impact of long-term
variations on visual localization. We conduct a thorough analysis of the
performance of current state-of-the-art feature extraction networks under
various domain changes and find a significant performance gap between intra-
and cross-domain localization. We investigate different methods to close this
gap by improving the supervision of modern feature extractor networks. We
propose a novel data-centric method, Implicit Cross-Domain Correspondences
(iCDC). iCDC represents the same environment with multiple Neural Radiance
Fields, each fitting the scene under individual visual domains. It utilizes the
underlying 3D representations to generate accurate correspondences across
different long-term visual conditions. Our proposed method enhances
cross-domain localization performance, significantly reducing the performance
gap. When evaluated on popular long-term localization benchmarks, our trained
networks consistently outperform existing methods. This work serves as a
substantial stride toward more robust visual localization pipelines for
long-term deployments, and opens up research avenues in the development of
long-term invariant descriptors.

Comments:
- 14 pages + 5 pages appendix, 13 figures

---

## Animating NeRFs from Texture Space: A Framework for Pose-Dependent  Rendering of Human Performances

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-06 | Paul Knoll, Wieland Morgenstern, Anna Hilsmann, Peter Eisert | cs.CV | [PDF](http://arxiv.org/pdf/2311.03140v1){: .btn .btn-green } |

**Abstract**: Creating high-quality controllable 3D human models from multi-view RGB videos
poses a significant challenge. Neural radiance fields (NeRFs) have demonstrated
remarkable quality in reconstructing and free-viewpoint rendering of static as
well as dynamic scenes. The extension to a controllable synthesis of dynamic
human performances poses an exciting research question. In this paper, we
introduce a novel NeRF-based framework for pose-dependent rendering of human
performances. In our approach, the radiance field is warped around an SMPL body
mesh, thereby creating a new surface-aligned representation. Our representation
can be animated through skeletal joint parameters that are provided to the NeRF
in addition to the viewpoint for pose dependent appearances. To achieve this,
our representation includes the corresponding 2D UV coordinates on the mesh
texture map and the distance between the query point and the mesh. To enable
efficient learning despite mapping ambiguities and random visual variations, we
introduce a novel remapping process that refines the mapped coordinates.
Experiments demonstrate that our approach results in high-quality renderings
for novel-view and novel-pose synthesis.

---

## Consistent4D: Consistent 360° Dynamic Object Generation from  Monocular Video

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-06 | Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, Yao Yao | cs.CV | [PDF](http://arxiv.org/pdf/2311.02848v1){: .btn .btn-green } |

**Abstract**: In this paper, we present Consistent4D, a novel approach for generating 4D
dynamic objects from uncalibrated monocular videos. Uniquely, we cast the
360-degree dynamic object reconstruction as a 4D generation problem,
eliminating the need for tedious multi-view data collection and camera
calibration. This is achieved by leveraging the object-level 3D-aware image
diffusion model as the primary supervision signal for training Dynamic Neural
Radiance Fields (DyNeRF). Specifically, we propose a Cascade DyNeRF to
facilitate stable convergence and temporal continuity under the supervision
signal which is discrete along the time axis. To achieve spatial and temporal
consistency, we further introduce an Interpolation-driven Consistency Loss. It
is optimized by minimizing the discrepancy between rendered frames from DyNeRF
and interpolated frames from a pre-trained video interpolation model. Extensive
experiments show that our Consistent4D can perform competitively to prior art
alternatives, opening up new possibilities for 4D dynamic object generation
from monocular videos, whilst also demonstrating advantage for conventional
text-to-3D generation tasks. Our project page is
https://consistent4d.github.io/.

Comments:
- Technique report. Project page: https://consistent4d.github.io/

---

## InstructPix2NeRF: Instructed 3D Portrait Editing from a Single Image

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-06 | Jianhui Li, Shilong Liu, Zidong Liu, Yikai Wang, Kaiwen Zheng, Jinghui Xu, Jianmin Li, Jun Zhu | cs.CV | [PDF](http://arxiv.org/pdf/2311.02826v1){: .btn .btn-green } |

**Abstract**: With the success of Neural Radiance Field (NeRF) in 3D-aware portrait
editing, a variety of works have achieved promising results regarding both
quality and 3D consistency. However, these methods heavily rely on per-prompt
optimization when handling natural language as editing instructions. Due to the
lack of labeled human face 3D datasets and effective architectures, the area of
human-instructed 3D-aware editing for open-world portraits in an end-to-end
manner remains under-explored. To solve this problem, we propose an end-to-end
diffusion-based framework termed InstructPix2NeRF, which enables instructed
3D-aware portrait editing from a single open-world image with human
instructions. At its core lies a conditional latent 3D diffusion process that
lifts 2D editing to 3D space by learning the correlation between the paired
images' difference and the instructions via triplet data. With the help of our
proposed token position randomization strategy, we could even achieve
multi-semantic editing through one single pass with the portrait identity
well-preserved. Besides, we further propose an identity consistency module that
directly modulates the extracted identity signals into our diffusion process,
which increases the multi-view 3D identity consistency. Extensive experiments
verify the effectiveness of our method and show its superiority against strong
baselines quantitatively and qualitatively.

Comments:
- https://github.com/mybabyyh/InstructPix2NeRF

---

## VR-NeRF: High-Fidelity Virtualized Walkable Spaces

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-05 | Linning Xu, Vasu Agrawal, William Laney, Tony Garcia, Aayush Bansal, Changil Kim, Samuel Rota Bulò, Lorenzo Porzi, Peter Kontschieder, Aljaž Božič, Dahua Lin, Michael Zollhöfer, Christian Richardt | cs.CV | [PDF](http://arxiv.org/pdf/2311.02542v1){: .btn .btn-green } |

**Abstract**: We present an end-to-end system for the high-fidelity capture, model
reconstruction, and real-time rendering of walkable spaces in virtual reality
using neural radiance fields. To this end, we designed and built a custom
multi-camera rig to densely capture walkable spaces in high fidelity and with
multi-view high dynamic range images in unprecedented quality and density. We
extend instant neural graphics primitives with a novel perceptual color space
for learning accurate HDR appearance, and an efficient mip-mapping mechanism
for level-of-detail rendering with anti-aliasing, while carefully optimizing
the trade-off between quality and speed. Our multi-GPU renderer enables
high-fidelity volume rendering of our neural radiance field model at the full
VR resolution of dual 2K$\times$2K at 36 Hz on our custom demo machine. We
demonstrate the quality of our results on our challenging high-fidelity
datasets, and compare our method and datasets to existing baselines. We release
our dataset on our project website.

Comments:
- SIGGRAPH Asia 2023; Project page: https://vr-nerf.github.io

---

## A Neural Radiance Field-Based Architecture for Intelligent Multilayered  View Synthesis



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-03 | D. Dhinakaran, S. M. Udhaya Sankar, G. Elumalai, N. Jagadish kumar | cs.NI | [PDF](http://arxiv.org/pdf/2311.01842v1){: .btn .btn-green } |

**Abstract**: A mobile ad hoc network is made up of a number of wireless portable nodes
that spontaneously come together en route for establish a transitory network
with no need for any central management. A mobile ad hoc network (MANET) is
made up of a sizable and reasonably dense community of mobile nodes that travel
across any terrain and rely solely on wireless interfaces for communication,
not on any well before centralized management. Furthermore, routing be supposed
to offer a method for instantly delivering data across a network between any
two nodes. Finding the best packet routing from across infrastructure is the
major issue, though. The proposed protocol's major goal is to identify the
least-expensive nominal capacity acquisition that assures the transportation of
realistic transport that ensures its durability in the event of any node
failure. This study suggests the Optimized Route Selection via Red Imported
Fire Ants (RIFA) Strategy as a way to improve on-demand source routing systems.
Predicting Route Failure and energy Utilization is used to pick the path during
the routing phase. Proposed work assess the results of the comparisons based on
performance parameters like as energy usage, packet delivery rate (PDR), and
end-to-end (E2E) delay. The outcome demonstrates that the proposed strategy is
preferable and increases network lifetime while lowering node energy
consumption and typical E2E delay under the majority of network performance
measures and factors.

---

## Estimating 3D Uncertainty Field: Quantifying Uncertainty for Neural  Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-03 | Jianxiong Shen, Ruijie Ren, Adria Ruiz, Francesc Moreno-Noguer | cs.CV | [PDF](http://arxiv.org/pdf/2311.01815v2){: .btn .btn-green } |

**Abstract**: Current methods based on Neural Radiance Fields (NeRF) significantly lack the
capacity to quantify uncertainty in their predictions, particularly on the
unseen space including the occluded and outside scene content. This limitation
hinders their extensive applications in robotics, where the reliability of
model predictions has to be considered for tasks such as robotic exploration
and planning in unknown environments. To address this, we propose a novel
approach to estimate a 3D Uncertainty Field based on the learned incomplete
scene geometry, which explicitly identifies these unseen regions. By
considering the accumulated transmittance along each camera ray, our
Uncertainty Field infers 2D pixel-wise uncertainty, exhibiting high values for
rays directly casting towards occluded or outside the scene content. To
quantify the uncertainty on the learned surface, we model a stochastic radiance
field. Our experiments demonstrate that our approach is the only one that can
explicitly reason about high uncertainty both on 3D unseen regions and its
involved 2D rendered pixels, compared with recent methods. Furthermore, we
illustrate that our designed uncertainty field is ideally suited for real-world
robotics tasks, such as next-best-view selection.

---

## PDF: Point Diffusion Implicit Function for Large-scale Scene Neural  Representation

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-03 | Yuhan Ding, Fukun Yin, Jiayuan Fan, Hui Li, Xin Chen, Wen Liu, Chongshan Lu, Gang YU, Tao Chen | cs.CV | [PDF](http://arxiv.org/pdf/2311.01773v1){: .btn .btn-green } |

**Abstract**: Recent advances in implicit neural representations have achieved impressive
results by sampling and fusing individual points along sampling rays in the
sampling space. However, due to the explosively growing sampling space, finely
representing and synthesizing detailed textures remains a challenge for
unbounded large-scale outdoor scenes. To alleviate the dilemma of using
individual points to perceive the entire colossal space, we explore learning
the surface distribution of the scene to provide structural priors and reduce
the samplable space and propose a Point Diffusion implicit Function, PDF, for
large-scale scene neural representation. The core of our method is a
large-scale point cloud super-resolution diffusion module that enhances the
sparse point cloud reconstructed from several training images into a dense
point cloud as an explicit prior. Then in the rendering stage, only sampling
points with prior points within the sampling radius are retained. That is, the
sampling space is reduced from the unbounded space to the scene surface.
Meanwhile, to fill in the background of the scene that cannot be provided by
point clouds, the region sampling based on Mip-NeRF 360 is employed to model
the background representation. Expensive experiments have demonstrated the
effectiveness of our method for large-scale scene novel view synthesis, which
outperforms relevant state-of-the-art baselines.

Comments:
- Accepted to NeurIPS 2023

---

## Efficient Cloud Pipelines for Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-03 | Derek Jacoby, Donglin Xu, Weder Ribas, Minyi Xu, Ting Liu, Vishwanath Jayaraman, Mengdi Wei, Emma De Blois, Yvonne Coady | cs.CV | [PDF](http://arxiv.org/pdf/2311.01659v1){: .btn .btn-green } |

**Abstract**: Since their introduction in 2020, Neural Radiance Fields (NeRFs) have taken
the computer vision community by storm. They provide a multi-view
representation of a scene or object that is ideal for eXtended Reality (XR)
applications and for creative endeavors such as virtual production, as well as
change detection operations in geospatial analytics. The computational cost of
these generative AI models is quite high, however, and the construction of
cloud pipelines to generate NeRFs is neccesary to realize their potential in
client applications. In this paper, we present pipelines on a high performance
academic computing cluster and compare it with a pipeline implemented on
Microsoft Azure. Along the way, we describe some uses of NeRFs in enabling
novel user interaction scenarios.

---

## INeAT: Iterative Neural Adaptive Tomography



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-03 | Bo Xiong, Changqing Su, Zihan Lin, You Zhou, Zhaofei Yu | eess.IV | [PDF](http://arxiv.org/pdf/2311.01653v1){: .btn .btn-green } |

**Abstract**: Computed Tomography (CT) with its remarkable capability for three-dimensional
imaging from multiple projections, enjoys a broad range of applications in
clinical diagnosis, scientific observation, and industrial detection. Neural
Adaptive Tomography (NeAT) is a recently proposed 3D rendering method based on
neural radiance field for CT, and it demonstrates superior performance compared
to traditional methods. However, it still faces challenges when dealing with
the substantial perturbations and pose shifts encountered in CT scanning
processes. Here, we propose a neural rendering method for CT reconstruction,
named Iterative Neural Adaptive Tomography (INeAT), which incorporates
iterative posture optimization to effectively counteract the influence of
posture perturbations in data, particularly in cases involving significant
posture variations. Through the implementation of a posture feedback
optimization strategy, INeAT iteratively refines the posture corresponding to
the input images based on the reconstructed 3D volume. We demonstrate that
INeAT achieves artifact-suppressed and resolution-enhanced reconstruction in
scenarios with significant pose disturbances. Furthermore, we show that our
INeAT maintains comparable reconstruction performance to stable-state
acquisitions even using data from unstable-state acquisitions, which
significantly reduces the time required for CT scanning and relaxes the
stringent requirements on imaging hardware systems, underscoring its immense
potential for applications in short-time and low-cost CT technology.

---

## Novel View Synthesis from a Single RGBD Image for Indoor Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2023-11-02 | Congrui Hetang, Yuping Wang | cs.CV | [PDF](http://arxiv.org/pdf/2311.01065v1){: .btn .btn-green } |

**Abstract**: In this paper, we propose an approach for synthesizing novel view images from
a single RGBD (Red Green Blue-Depth) input. Novel view synthesis (NVS) is an
interesting computer vision task with extensive applications. Methods using
multiple images has been well-studied, exemplary ones include training
scene-specific Neural Radiance Fields (NeRF), or leveraging multi-view stereo
(MVS) and 3D rendering pipelines. However, both are either computationally
intensive or non-generalizable across different scenes, limiting their
practical value. Conversely, the depth information embedded in RGBD images
unlocks 3D potential from a singular view, simplifying NVS. The widespread
availability of compact, affordable stereo cameras, and even LiDARs in
contemporary devices like smartphones, makes capturing RGBD images more
accessible than ever. In our method, we convert an RGBD image into a point
cloud and render it from a different viewpoint, then formulate the NVS task
into an image translation problem. We leveraged generative adversarial networks
to style-transfer the rendered image, achieving a result similar to a
photograph taken from the new perspective. We explore both unsupervised
learning using CycleGAN and supervised learning with Pix2Pix, and demonstrate
the qualitative results. Our method circumvents the limitations of traditional
multi-image techniques, holding significant promise for practical, real-time
applications in NVS.

Comments:
- 2nd International Conference on Image Processing, Computer Vision and
  Machine Learning, November 2023