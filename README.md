# ExpandNet

Create HDR images from LDR inputs:

Training and inference code for:

_ExpandNet: A Deep Convolutional Neural Network for High Dynamic Range Expansion from Low Dynamic Range Content_

[Demetris Marnerides](https://github.com/dmarnerides),
[Thomas Bashford-Rogers](http://thomasbashfordrogers.com/),
[Jonathan Hatchett](http://hatchett.me.uk/)
and [Kurt Debattista](https://warwick.ac.uk/fac/sci/wmg/people/profile/?wmgid=518)

Paper was presented at Eurographics 2018 and published in Computer Graphics Forum.

([arxiv version](https://arxiv.org/abs/1803.02266))

---

## Prerequisites

Requires the PyTorch library along with OpenCV. Python >= 3.6 supported only.

First follow the [instructions for installing PyTorch](http://pytorch.org/).

To install OpenCV use:

```bash
conda install opencv3 -c menpo
```

**NOTE** There might be training issues when using latest versions of OpenCV
for the tone mapping operators (issue #14)

---

## Usage

Open GUI and select an Image and tonemap Algorithm

## Viewing HDR Images

There is a very easy to use online viewer at [openhdr.org](https://viewer.openhdr.org/) which also has tone mapping functionality.

---
