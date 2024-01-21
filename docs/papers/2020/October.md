---
layout: default
title: October
parent: 2020
nav_order: 10
---
<!---metadata--->

## NeRF++: Analyzing and Improving Neural Radiance Fields

nerf
{: .label .label-blue }

| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| 2020-10-15 | Kai Zhang, Gernot Riegler, Noah Snavely, Vladlen Koltun | cs.CV | [PDF](http://arxiv.org/pdf/2010.07492v2){: .btn .btn-green } |

**Abstract**: Neural Radiance Fields (NeRF) achieve impressive view synthesis results for a
variety of capture settings, including 360 capture of bounded scenes and
forward-facing capture of bounded and unbounded scenes. NeRF fits multi-layer
perceptrons (MLPs) representing view-invariant opacity and view-dependent color
volumes to a set of training images, and samples novel views based on volume
rendering techniques. In this technical report, we first remark on radiance
fields and their potential ambiguities, namely the shape-radiance ambiguity,
and analyze NeRF's success in avoiding such ambiguities. Second, we address a
parametrization issue involved in applying NeRF to 360 captures of objects
within large-scale, unbounded 3D scenes. Our method improves view synthesis
fidelity in this challenging scenario. Code is available at
https://github.com/Kai-46/nerfplusplus.

Comments:
- Code is available at https://github.com/Kai-46/nerfplusplus; fix a
  minor formatting issue in Fig. 4