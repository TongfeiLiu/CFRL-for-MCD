# CFRL-for-MCD
The paper [Commonality Feature Representation Learning for Unsupervised Multimodal Change Detection](https://ieeexplore.ieee.org/document/10891329) has been published by **IEEE Transactions on Image Processing in 2025**. 

This repository is the official PyTorch implementation of CFRL.

## Outline
<ul>
  <li>Introduction</li>
  <li>Results Preview</li>
  <li>Requirements</li>
  <li>Installation</li>
  <li>Usage</li>
  <li>Citation</li>
  <li>Acknowledgements</li>
</ul>

## Introduction
The main challenge of multimodal change detection (MCD) is that multimodal bitemporal images (MBIs) cannot be compared directly to identify changes. To overcome this problem, this paper proposes a novel commonality feature representation learning (CFRL) and constructs a CFRL-based unsupervised MCD framework. The CFRL is composed of a Siamese-based encoder and two decoders. First, the Siamese-based encoder can map original MBIs in the same feature space for extracting the representative features of each modality. Then, the two decoders are used to reconstruct the original MBIs by regressing themselves, respectively. Meanwhile, we swap the decoders to reconstruct the pseudo-MBIs to conduct modality alignment. Subsequently, all reconstructed images are input to the Siamese-based encoder again to map them in the same feature space, by which representative features are obtained. On this basis, latent commonality features between MBIs can be extracted by minimizing the distance between these representative features. These latent commonality features are comparable and can be used to identify changes. Notably, the proposed CFRL can be performed simultaneously in two modalities corresponding to MBIs. Therefore, two change magnitude images (CMIs) can be generated simultaneously by measuring the difference between the commonality features of MBIs. Finally, a simple threshold algorithm or a clustering algorithm can be employed to divide CMIs into binary change maps. Extensive experiments on six publicly available MCD datasets show that the proposed CFRL-based framework can achieve superior performance compared with other state-of-the-art approaches.
The framework of the proposed CFRL is presented as follows:

