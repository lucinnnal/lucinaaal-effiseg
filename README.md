# EffiSeg: A High-Performance Lightweight Segmentation Framework via Transformer-Aware Optimization and Targeted Distillation

**Authors:** Dain Kim, Ki Pyo Kim, Kyungjun Oh  
**Date:** June 12, 2025  

## ABSTRACT
**Semantic segmentation** is a fundamental task in computer vision, widely used in applica-
tions such as autonomous driving, medical imaging, and mobile robotics. However, deploying
high-performance segmentation models in resource-constrained environments remains a signifi-
cant challenge due to their heavy computational and memory requirements. To address this,
this work proposes a novel training strategies for enhancing efficient semantic segmentation
by exploring three distinct approaches:. This work improve lightweight segmentation model per-
formance via: (1) **model architecture optimization** to reduce training time and computa-
tional load through structural modification; (2) **knowledge distillation** to transfer representa-
tional power from a large teacher network to a compact student model using a combination of
feature-based, logit-based, and patch-based model aware knowledge distillation techniques; and
(3) **loss function refinement** to address class imbalance. The proposed training strategies
achieve competitive segmentation accuracy while maintaining a lightweight architecture, thereby
opening avenues for deployment in real-time and edge-device scenarios. One of our methods achieves up to **+5.69%** mIoU improvement over the SegFormer-B0 baseline on the Cityscapes dataset, without compromising inference speed. The code is available
at https://github.com/lucinnnal/lucinal_EffiSeg.

## 🧩 Distillation Strategy Overview
![KD Architecture](./src/pics/Figure.png)<br>
**Overview of KD** Four types of loss terms are used for knowledge distillation. Class-
Weighted Cross-Entropy Loss is applied for standard label supervision. Cosine Similarity Guided
Feature Distillation (CSKD) is also employed stage-wise to align the spatial and structural represen-
tations, enhancing intermediate feature alignment. Patch Embedding Knowledge Distillation (PEKD)
is also used at each stage to transfer low-level spatial representations from the teacher’s patch em-
beddings to the student. Additionally, KL Divergence-Based Knowledge Distillation (KLKD) is used
in the logit space to guide the student toward the teacher’s output distribution, encouraging better
semantic consistency.

## 📁 Dataset
[Cityscapes Dataset](https://www.cityscapes-dataset.com/) for training and evaluation. Make sure to follow their license and guidelines when using the data.

**Details**
1. Download Cityscapes data : [Download](https://www.cityscapes-dataset.com/downloads/)
2. leftImg8bit_trainvaltest.zip (11GB) -> Data , gtFine_trainvaltest.zip (241MB) -> Labels to a folder cityscapes/
3. Put data folder into src so that the data is in src/data/cityscapes directory !

## Setting Up the Virtual Environment

To create the conda virtual environment, run:
```bash
conda env create -f environment.yml
conda activate EffiSeg
```

# Segformer pretrained weights on Cityscapes
Segformer b2 : [Segformer B2 Cityscapes Weights](https://drive.google.com/file/d/1mixZrRm-nSOhIjM4ltI_wegc14iciZZS/view)

## 💡 Citation
If you use this framework in your research, please cite:
```bibtex
@misc{EffiSeg2025,
  title={EffiSeg: A High-Performance Lightweight Segmentation Framework via Transformer-Aware Optimization and Targeted Distillation},
  author={Dain Kim and Ki Pyo Kim and Kyungjun Oh},
  year={2025},
  url={https://github.com/lucinnnal/lucinal_EffiSeg}
}
```

## 📬 Contact
For questions or feedback, feel free to open an issue or contact the authors.

- Dain Kim : dk7518@g.skku.edu
- Kipyo Kim : kipyo39@gmail.com or kkp0606@skku.edu
- Kyungjun Oh : rudwns55@g.skku.edu

Made by Hackathona Team in SKKU AAI undergraduate students.