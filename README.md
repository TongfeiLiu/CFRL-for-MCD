# TIP 2025: Commonality Feature Representation Learning for Unsupervised Multimodal Change Detection

![Paper](https://img.shields.io/badge/Paper-TIP-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)

This repository provides the official implementation of the paper:

Tongfei Liu, Mingyang Zhang, Maoguo Gong, Qingfu Zhang, Fenlong Jiang, Hanhong Zheng, and Di Lu [J]. [Commonality Feature Representation Learning for Unsupervised Multimodal Change Detection](https://ieeexplore.ieee.org/document/10891329), **IEEE Transactions on Image Processing, 2025**, 34:1219-1233. 

---

## 📖 Outline
<ul>
  <li>Introduction</li>
  <li>Results Preview</li>
  <li>Usage</li>
  <li>Citation</li>
  <li>Acknowledgements</li>
  <li>Contact us</li>
</ul>

## 📖 Abstract
The main challenge of multimodal change detection (MCD) is that multimodal bitemporal images (MBIs) cannot be compared directly to identify changes. To overcome this problem, this paper proposes a novel commonality feature representation learning (CFRL) and constructs a CFRL-based unsupervised MCD framework. The CFRL is composed of a Siamese-based encoder and two decoders. First, the Siamese-based encoder can map original MBIs in the same feature space for extracting the representative features of each modality. Then, the two decoders are used to reconstruct the original MBIs by regressing themselves, respectively. Meanwhile, we swap the decoders to reconstruct the pseudo-MBIs to conduct modality alignment. Subsequently, all reconstructed images are input to the Siamese-based encoder again to map them in the same feature space, by which representative features are obtained. On this basis, latent commonality features between MBIs can be extracted by minimizing the distance between these representative features. These latent commonality features are comparable and can be used to identify changes. Notably, the proposed CFRL can be performed simultaneously in two modalities corresponding to MBIs. Therefore, two change magnitude images (CMIs) can be generated simultaneously by measuring the difference between the commonality features of MBIs. Finally, a simple threshold algorithm or a clustering algorithm can be employed to divide CMIs into binary change maps. Extensive experiments on six publicly available MCD datasets show that the proposed CFRL-based framework can achieve superior performance compared with other state-of-the-art approaches.
The framework of the proposed CFRL is presented as follows:
![Framework of our proposed CFRL)](https://github.com/TongfeiLiu/CFRL-for-MCD/blob/main/Figs/Fig1-Framework.jpg)

## 📊 Results Preview
**Accuracy**
![Accuracy of our proposed CFRL)](https://github.com/TongfeiLiu/CFRL-for-MCD/blob/main/Figs/accuracy.png)

**Change Maps**
![Change maps of our proposed CFRL)](https://github.com/TongfeiLiu/CFRL-for-MCD/blob/main/Figs/visual%20result.jpg)

## 🛠 Usage
The following takes the yellow river data as an example.
### 1. Prepare your data: 
* --data_name: e.g., yellow.
* --t1_path: e.g., './data/Yellow/yellow_C_1.bmp'.
* --t2_path: e.g., './data/Yellow/yellow_C_2.bmp'.
* --gt_path: e.g., './data/Yellow/yellow_C_gt.png'.(only evaluation)

### 2. Parameters Setup

* t1_nc: The number of bands used by t1 image (default: 1)
* t2_nc: The number of bands used by t2 image (default: 1)
* patch_size: Patch size during training (default: 9)
* test_ps: Patch size during testing (default: 9)
* epoch: Number of training epochs (Suggestion: 10~20)

**Recommended Hyperparameters**

| Datasets | n_seg | com |
|---------|-------|-----|
| #1 (italy)      |  9  | 11  |
| #2 (yellow)     |  9  | 9 |
| #3 (gloucester2)     |  9  | 9  |
| #4 (bastrop)     |  9  | 3  |
| #5 (california)     |  9  | 3  |
| #6 (France)     |  9  | 9 |

---

### 3. Run the script:
```
python train.py
```
**Note:** Here, we provide the difference maps of the six datasets involved in the paper in the 'MCD DI Results' folder. Based on this, it is easy to use Otsu or FLICM to get the final change map.

## 📜 Citation
If you find our work useful for your research, please consider citing our paper:

```bibtex
@ARTICLE{TIP2025CFRL,
  author={Liu, Tongfei and Zhang, Mingyang and Gong, Maoguo and Zhang, Qingfu and Jiang, Fenlong and Zheng, Hanhong and Lu, Di},
  journal={IEEE Transactions on Image Processing}, 
  title={Commonality Feature Representation Learning for Unsupervised Multimodal Change Detection}, 
  year={2025},
  volume={34},
  number={},
  pages={1219-1233},
  keywords={Feature extraction;Image reconstruction;Training;Data mining;Autoencoders;Representation learning;Image sensors;Electronic mail;Decoding;Clustering algorithms;Multimodal change detection;unsupervised change detection;heterogeneous images;representation learning;commonality feature},
  doi={10.1109/TIP.2025.3539461}}

@ARTICLE{TGRS2025AEKAN,
  author={Liu, Tongfei and Xu, Jianjian and Lei, Tao and Wang, Yingbo and Du, Xiaogang and Zhang, Weichuan and Lv, Zhiyong and Gong, Maoguo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={AEKAN: Exploring Superpixel-Based AutoEncoder Kolmogorov-Arnold Network for Unsupervised Multimodal Change Detection}, 
  year={2025},
  volume={63},
  number={},
  pages={1-14},
  keywords={Feature extraction;Sensors;Image sensors;Training;Remote sensing;Sensor phenomena and characterization;Land surface;Analytical models;Sun;Radar imaging;Commonality features;heterogeneous images;Kolmogorov-Arnold Network (KAN);multimodal change detection (MCD)},
  doi={10.1109/TGRS.2024.3515258}
}
```

## 🙏Acknowledgement
First of all, we would like to thank [CACD](https://ieeexplore.ieee.org/document/9357940/)'s authors for inspiring our approach and Prof. Wu for helping us. We also provide a redeployed Pytorch version of CACD. (The code is available at this [link](https://github.com/TongfeiLiu/CACD-for-MCD)). If this article is helpful to you, please cite [CACD](https://ieeexplore.ieee.org/document/9357940/) and [CFRL](https://ieeexplore.ieee.org/document/10891329).
Secondly, we would like to thank Dr. Luigi Tommaso Luppino for his help in solving some comparative method problems.
In addition, we are also very grateful for the outstanding contributions of the publicly available MCD datasets [1,2,3]

```
[1] https://sites.google.com/view/luppino/data.
[2] Professor Michele Volpi's webpage at https://sites.google.com/site/michelevolpiresearch/home.
[3] Professor Max Mignotte's webpage (http://www-labs.iro.umontreal.ca/~mignotte/).
[4] https://github.com/yulisun.
```

---

## 📮Contact us 
Although the current version can provide a good result, it is not stable enough due to the lack of supervision information. In the future, we will continue to conduct research and strive to innovate more stable and robust algorithms.

If you have any problems when running the code, please do not hesitate to contact us. Thanks.  
Tongfei Liu: liutongfei_home@hotmail.com

Date: Apr. 16, 2025  
