# Grammatically Recognizing Images with Tree Convolution


This is a re-implementation of our KDD 2020 paper "Grammatically Recognizing Images with Tree Convolution" (https://dl.acm.org/doi/abs/10.1145/3394486.3403133).

[Guangrun Wang](https://wanggrun.github.io), [Guangcong Wang](https://wanggcong.github.io), Keze Wang, Xiaodan Liang, and Liang Lin*

Sun Yat-sen University (SYSU)


![intro](https://github.com/wanggrun/TreeConv/blob/master/images/intro1.png)


![intro](https://github.com/wanggrun/TreeConv/blob/master/images/intro2.png)






# Table of Contents
0. [Introduction](#introduction)
0. [Requirement](#Requirement)
0. [Pretrained model on ImageNet](#imagenet)
0. [Training on ImageNet](#imagenet)
0. [Citation](#citation)

# Introduction

This repository contains the training & testing code of "Grammatically Recognizing Images with Tree Convolution" (TreeConv) on [ImageNet](http://image-net.org/challenges/LSVRC/2015/).


# Pretrained model on ImageNet

This code was tested on：


+  Python 3.6.7
+ TensorFlow 1.15.0
+ [Tensorpack](https://github.com/ppwwyyxx/tensorpack)
   The code depends on Yuxin Wu's Tensorpack. For convenience, we provide a stable version 'tensorpack-installed' in this repository. 
   ```
   # install tensorpack locally:
   cd tensorpack-installed
   python setup.py install --user
   ```


# Pretrained model on ImageNet

+ ImageNet accuracy and pretrained model (baidu pan code: ow9z):

| Model            | Top 5 Error | Top 1 Error | Download                                                                          |
|:-----------------|:------------|:-----------:|:---------------------------------------------------------------------------------:|
| ResNet50         | 6.9%       | 23.6%      | [:arrow_down:](http://models.tensorpack.com/ResNet/ImageNet-ResNet50.npz)         |
| ResNet50-TreeConv   | 6.16%       | 22.08%      | google drive: [:arrow_down:](https://drive.google.com/open?id=1M0Nb6IKiGdlHy8hOOG_Rcbh861Ve1OeE)   and baidu pan: [:arrow_down:](https://pan.baidu.com/s/1KoaBmK_dr35zkmXXDlyDdA)   |


+ Testing script:
```
cd TreeConv

python imagenet-resnet.py  --gpu 0,1,2,3,4,5,6,7   --data [ROOT-OF-IMAGENET-DATASET]  --log_dir  [ROOT-OF-TEST-LOG] --load   [ROOT-TO-LOAD-MODEL]  --eval --data-format NHWC
```


# Training on ImageNet


+ Training script:
```
cd TreeConv

python imagenet-resnet.py  --gpu 0,1,2,3,4,5,6,7   --data [ROOT-OF-IMAGENET-DATASET]  --log_dir  [ROOT-OF-TRAINING-LOG-AND-MODEL]  --data-format NHWC
```



# Citation

If you use these models in your research, please cite:
```
@inproceedings{Wang2020Grammatically_KDD,
  author    = {Guangrun Wang and
               Guangcong Wang and
               Keze Wang and
               Xiaodan Liang and
               Liang Lin},
  title     = {Grammatically Recognizing Images with Tree Convolution},
  booktitle = {{KDD} '20: The 26th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Virtual Event, CA, USA, August 23-27, 2020},
  pages     = {903--912},
  year      = {2020},
}
      
```

