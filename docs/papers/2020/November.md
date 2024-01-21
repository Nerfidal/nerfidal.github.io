---
layout: default
title: November
parent: 2020
nav_order: 11
---
<!---metadata--->

## D-NeRF: Neural Radiance Fields for Dynamic Scenes

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-11-27 | Albert Pumarola, Enric Corona, Gerard Pons-Moll, Francesc Moreno-Noguer | cs.CV | [PDF](http://arxiv.org/pdf/2011.13961v1){: .btn .btn-green } |

**Abstract**: Neural rendering techniques combining machine learning with geometric
reasoning have arisen as one of the most promising approaches for synthesizing
novel views of a scene from a sparse set of images. Among these, stands out the
Neural radiance fields (NeRF), which trains a deep network to map 5D input
coordinates (representing spatial location and viewing direction) into a volume
density and view-dependent emitted radiance. However, despite achieving an
unprecedented level of photorealism on the generated images, NeRF is only
applicable to static scenes, where the same spatial location can be queried
from different images. In this paper we introduce D-NeRF, a method that extends
neural radiance fields to a dynamic domain, allowing to reconstruct and render
novel images of objects under rigid and non-rigid motions from a \emph{single}
camera moving around the scene. For this purpose we consider time as an
additional input to the system, and split the learning process in two main
stages: one that encodes the scene into a canonical space and another that maps
this canonical representation into the deformed scene at a particular time.
Both mappings are simultaneously learned using fully-connected networks. Once
the networks are trained, D-NeRF can render novel images, controlling both the
camera view and the time variable, and thus, the object movement. We
demonstrate the effectiveness of our approach on scenes with objects under
rigid, articulated and non-rigid motions. Code, model weights and the dynamic
scenes dataset will be released.

---

## Nerfies: Deformable Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-11-25 | Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman, Steven M. Seitz, Ricardo Martin-Brualla | cs.CV | [PDF](http://arxiv.org/pdf/2011.12948v5){: .btn .btn-green } |

**Abstract**: We present the first method capable of photorealistically reconstructing
deformable scenes using photos/videos captured casually from mobile phones. Our
approach augments neural radiance fields (NeRF) by optimizing an additional
continuous volumetric deformation field that warps each observed point into a
canonical 5D NeRF. We observe that these NeRF-like deformation fields are prone
to local minima, and propose a coarse-to-fine optimization method for
coordinate-based models that allows for more robust optimization. By adapting
principles from geometry processing and physical simulation to NeRF-like
models, we propose an elastic regularization of the deformation field that
further improves robustness. We show that our method can turn casually captured
selfie photos/videos into deformable NeRF models that allow for photorealistic
renderings of the subject from arbitrary viewpoints, which we dub "nerfies." We
evaluate our method by collecting time-synchronized data using a rig with two
mobile phones, yielding train/validation images of the same pose at different
viewpoints. We show that our method faithfully reconstructs non-rigidly
deforming scenes and reproduces unseen views with high fidelity.

Comments:
- ICCV 2021, Project page with videos: https://nerfies.github.io/

---

## DeRF: Decomposed Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-11-25 | Daniel Rebain, Wei Jiang, Soroosh Yazdani, Ke Li, Kwang Moo Yi, Andrea Tagliasacchi | cs.CV | [PDF](http://arxiv.org/pdf/2011.12490v1){: .btn .btn-green } |

**Abstract**: With the advent of Neural Radiance Fields (NeRF), neural networks can now
render novel views of a 3D scene with quality that fools the human eye. Yet,
generating these images is very computationally intensive, limiting their
applicability in practical scenarios. In this paper, we propose a technique
based on spatial decomposition capable of mitigating this issue. Our key
observation is that there are diminishing returns in employing larger (deeper
and/or wider) networks. Hence, we propose to spatially decompose a scene and
dedicate smaller networks for each decomposed part. When working together,
these networks can render the whole scene. This allows us near-constant
inference time regardless of the number of decomposed parts. Moreover, we show
that a Voronoi spatial decomposition is preferable for this purpose, as it is
provably compatible with the Painter's Algorithm for efficient and GPU-friendly
rendering. Our experiments show that for real-world scenes, our method provides
up to 3x more efficient inference than NeRF (with the same rendering quality),
or an improvement of up to 1.0~dB in PSNR (for the same inference cost).