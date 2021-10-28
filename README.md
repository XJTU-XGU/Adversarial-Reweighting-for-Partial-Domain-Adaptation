# [Adversarial Reweighting for Partial Domain Adaptation](xx)
Code for paper "Xiang Gu, Xi Yu, Yan Yang, Jian Sun, Zongben Xu, **Adversarial Reweighting for Partial Domain Adaptation**, Conference on Neural Information Processing Systems (NeurIPS), 2021".
## Prerequisites:
python==3.6.13 <br>
pytorch ==1.5.1 <br>
torchvision ==0.6.1 <br>
numpy==1.19.2 <br>
cvxpy ==1.1.14 <br>
tqdm ==4.1.2 <br>
Pillow == 8.3.1
## Datasets:
Download the datasets of <br>
[VisDA-2017](http://ai.bu.edu/visda-2017/) <br> 
[DomainNet](http://ai.bu.edu/M3SDA/) <br>
[Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) <br>
[Office](https://www.cc.gatech.edu/~judy/domainadapt/) <br> 
[ImageNet](https://www.image-net.org/) <br>
[Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) <br>
and put them into the folder "./data/" and modify the path of images in each '.txt' under the folder './data/'.
## Domain ID:
**VisDA-2017**: train (synthetic), validation (real) ==> 0,1 <br>
**DomainNet**: clipart, painting, real, sketch ==> 0,1,2,3 <br>
**Office-Home**: Art, Clipart, Product, RealWorld ==> 0,1,2,3 <br>
**Office**: amazon, dslr, webcam  ==> 0,1,2 <br>
**ImageNet-Caltech**: imagenet, caltech ==> 0,1 <br>
## Training
VisDA-2017:
```
python train.py --dset visda-2017 --s 0 --t 1
```
DomainNet:
```
python train.py --dset domainnet --s 0 --t 1
```
Office-Home:
```
#for AR
python train.py --dset office_home --s 0 --t 1
#for AR+LS
python train.py --dset office_home --s 0 --t 1 --label_smooth
```
Office:
```
python train.py --dset office --s 0 --t 1
```
ImageNet-Caltech:
```
python train.py --dset imagenet_caltech --s 0 --t 1
```
## Citation:
```
@InProceedings{Gu_2020_CVPR,
author = {Xiang Gu, Xi Yu, Yan Yang, Jian Sun, Zongben Xu},
title = {Adversarial Reweighting for Partial Domain Adaptation},
booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
month = {xxx},
year = {2021}
}
```
## Reference code:
https://github.com/thuml/CDAN <br>
https://github.com/tim-learn/BA3US <br>
https://github.com/XJTU-XGU/RSDA
## Contact：
If you have any problem, free to contect xianggu@stu.xjtu.edu.cn.
## Note
We are checking the code. It will be finished soon.
