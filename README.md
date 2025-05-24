# Edge Awareness Network with Large Kernel Attention for Small Target Segmentation from Intrapartum Ultrasound Images

Official pytorch code for "Edge Awareness Network with Large Kernel Attention for Small Target Segmentation from Intrapartum Ultrasound Images"

- [x] Code release
- [ ] Paper release

## Abstract
In intrapartum ultrasound image segmentation, accurately segment the small target, especially its boundary, is crucial for diagnosis in the emergency ward but still faces challenges. In this paper, we propose the Edge Aware Network with Large Kernel Attention, named EAN\_LKA, for intrapartum ultrasound image segmentation, especially for small but important target-pubic symphysis. Specifically, the architecture is composed of Shift Vision Transformer encoder, Large Kernel Attention decoder and Spatial Channel Cross Transform refinement module, which can respectively model the long-range dependence of tokens representing target boundary, allow the model to focus on relevant features of the small target across a wider spatial area of the input image and enable the network to effectively reinforce semantic differences between the target and clutter at full scales. Quantitative and qualitative experimental results on two MICCAI challenge datasets demonstrate that our proposed EAN\_LKA outperforms the other methods on the small target with an increase the average dice score (2.03%) and a decrease of the average surface distance (0.3). 


## Environment

- GPU: NVIDIA GeForce RTX3090 GPU
- Pytorch: 1.10.0 cuda 11.4
- cudatoolkit: 11.3.1



