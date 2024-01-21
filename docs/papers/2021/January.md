---
layout: default
title: January
parent: 2021
nav_order: 1
---
<!---metadata--->

## PVA: Pixel-aligned Volumetric Avatars



| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-01-07 | Amit Raj, Michael Zollhoefer, Tomas Simon, Jason Saragih, Shunsuke Saito, James Hays, Stephen Lombardi | cs.CV | [PDF](http://arxiv.org/pdf/2101.02697v1){: .btn .btn-green } |

**Abstract**: Acquisition and rendering of photo-realistic human heads is a highly
challenging research problem of particular importance for virtual telepresence.
Currently, the highest quality is achieved by volumetric approaches trained in
a person specific manner on multi-view data. These models better represent fine
structure, such as hair, compared to simpler mesh-based models. Volumetric
models typically employ a global code to represent facial expressions, such
that they can be driven by a small set of animation parameters. While such
architectures achieve impressive rendering quality, they can not easily be
extended to the multi-identity setting. In this paper, we devise a novel
approach for predicting volumetric avatars of the human head given just a small
number of inputs. We enable generalization across identities by a novel
parameterization that combines neural radiance fields with local, pixel-aligned
features extracted directly from the inputs, thus sidestepping the need for
very deep or complex networks. Our approach is trained in an end-to-end manner
solely based on a photometric re-rendering loss without requiring explicit 3D
supervision.We demonstrate that our approach outperforms the existing state of
the art in terms of quality and is able to generate faithful facial expressions
in a multi-identity setting.

Comments:
- Project page located at https://volumetric-avatars.github.io/

---

## Non-line-of-Sight Imaging via Neural Transient Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2021-01-02 | Siyuan Shen, Zi Wang, Ping Liu, Zhengqing Pan, Ruiqian Li, Tian Gao, Shiying Li, Jingyi Yu | eess.IV | [PDF](http://arxiv.org/pdf/2101.00373v3){: .btn .btn-green } |

**Abstract**: We present a neural modeling framework for Non-Line-of-Sight (NLOS) imaging.
Previous solutions have sought to explicitly recover the 3D geometry (e.g., as
point clouds) or voxel density (e.g., within a pre-defined volume) of the
hidden scene. In contrast, inspired by the recent Neural Radiance Field (NeRF)
approach, we use a multi-layer perceptron (MLP) to represent the neural
transient field or NeTF. However, NeTF measures the transient over spherical
wavefronts rather than the radiance along lines. We therefore formulate a
spherical volume NeTF reconstruction pipeline, applicable to both confocal and
non-confocal setups. Compared with NeRF, NeTF samples a much sparser set of
viewpoints (scanning spots) and the sampling is highly uneven. We thus
introduce a Monte Carlo technique to improve the robustness in the
reconstruction. Comprehensive experiments on synthetic and real datasets
demonstrate NeTF provides higher quality reconstruction and preserves fine
details largely missing in the state-of-the-art.