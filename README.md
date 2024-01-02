# KSL
Dynamic Korean Sign Language Recognition Using Pose Estimation Based and Attention-Based Neural Network"
Abstract:
Sign language recognition is crucial for improving communication accessibility for the hearing impaired community and reducing dependence on human interpreters. Notably, while significant research efforts have been devoted to many prevalent languages, Korean Sign Language (KSL) remains relatively underexplored, particularly concerning dynamic signs and generalizability. The scarcity of KSL datasets has exacerbated this limitation, hindering progress. Furthermore, most KSL research predominantly relies on static image-based datasets for recognition, leading to diminished accuracy and the inability to detect dynamic sign words. Furthermore, most KSL research predominantly relies on static image-based datasets for recognition, leading to diminished accuracy and the inability to detect dynamic sign words. Additionally, existing KSL recognition systems grapple with suboptimal performance accuracy and heightened computational complexity, further emphasizing the existing research gap. To address these formidable challenges, we propose a robust dynamic KSL recognition system that combines a skeleton-based Graph Convolution network with an attention-based neural network, effectively bridging the gap. Our solution employs a two-stream deep learning network to navigate the intricacies of dynamic signs, enhancing accuracy by effectively handling non-connected joint skeleton features. In this system, the first stream meticulously processes 47 pose landmarks using the Graph Convolutional Network (GCN) to extract graph-based features. These features are meticulously refined through a channel attention module and a general CNN, enhancing their temporal context. Concurrently, the second stream focuses on joint motion-based features, employing a similar approach. Subsequently, these distinct features from both streams are harmoniously integrated and channelled through a classification module to achieve precise sign-word recognition.

Thank you very much to  Mr Chen for opening their repository, we inspired by yan code and thought to update and modifying it "[https://github.com/yuxiaochen1103/DG-STA](https://github.com/yysijie/st-gcn)"
# Dynamic Korean Sign Language Recognition Using Pose Estimation Based and Attention-Based Neural Network
    IEEE Access. 
    LicenseCC BY 4.0
    Labs: Pattern Processing LabAbu Saleh Musa Miah's Lab
## Introduction
Coming soon .....................

## Prerequisites
We implement and run the code in two environments:
GPU Pytorch
Collaborator

I experimented for 
a.KSL dataset at HPE GPU Machine
b.Hirooka PC



## Training
1. KSL-77:"[Automatic mexican sign language recognition using normalized moments and artificial neural networks" https://www.scirp.org/journal/paperinformation.aspx?paperid=71592](https://dl.acm.org/doi/10.1007/978-3-030-37731-1_43) https://github.com/Yangseung/KSL
2. KSL-Lab-Dataset: Under Processing

### Citation
this code come theme come from below paper
```
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={32},
  number={1},
  year={2018}
}
@ARTICLE{10360810,
  author={Shin, Jungpil and Miah, Abu Saleh Musa and Suzuki, Kota and Hirooka, Koki and Hasan, Md. Al Mehedi},
  journal={IEEE Access}, 
  title={Dynamic Korean Sign Language Recognition Using Pose Estimation Based and Attention-Based Neural Network}, 
  year={2023},
  volume={11},
  number={},
  pages={143501-143513},
  doi={10.1109/ACCESS.2023.3343404}}


@article{miah2023dynamic,
  title={Dynamic Hand Gesture Recognition using Multi-Branch Attention Based Graph and General Deep Learning Model},
  author={Miah, Abu Saleh Musa and Hasan, Md Al Mehedi and Shin, Jungpil},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}

@article{miah2023multistage,
  title={Multistage Spatial Attention-Based Neural Network for Hand Gesture Recognition},
  author={Miah, Abu Saleh Musa and Hasan, Md Al Mehedi and Shin, Jungpil and Okuyama, Yuichi and Tomioka, Yoichi},
  journal={Computers},
  volume={12},
  number={1},
  pages={13},
  year={2023},
  publisher={MDPI}
}
@article{miah2022bensignnet,
  title={BenSignNet: Bengali Sign Language Alphabet Recognition Using Concatenated Segmentation and Convolutional Neural Network},
  author={Miah, Abu Saleh Musa and Shin, Jungpil and Hasan, Md Al Mehedi and Rahim, Md Abdur},
  journal={Applied Sciences},
  volume={12},
  number={8},
  pages={3933},
  year={2022},
  publisher={MDPI}
}




```
## Acknowledgement

Part of our code is borrowed from the (Yan, Sijie, Yuanjun Xiong, and Dahua Lin. "Spatial temporal graph convolutional networks for skeleton-based action recognition.) We thank to the authors for releasing their codes.
