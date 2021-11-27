# Adversarial Reweighting for Partial Domain Adaptation
Code for paper "Xiang Gu, Xi Yu, Yan Yang, Jian Sun, Zongben Xu, [**Adversarial Reweighting for Partial Domain Adaptation**](https://papers.nips.cc/paper/2021/hash/7ce3284b743aefde80ffd9aec500e085-Abstract.html), Conference on Neural Information Processing Systems (NeurIPS), 2021".
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
and put them into the folder "./data/" and modify the path of images in each '.txt' under the folder './data/'. Note the full list of ImageNet (imagenet.txt) is too big. Please download it [here](https://drive.google.com/file/d/1aZGNVO4-6yl7L0ulinDPxo11-RDozeBP/view?usp=sharing) and put it into './data/imagenet_caltech/'. 
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
@inproceedings{
gu2021adversarial,
title={Adversarial Reweighting for Partial Domain Adaptation},
author={Xiang Gu and Xi Yu and Yan Yang and Jian Sun and Zongben Xu},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=f5liPryFRoA}
}
```
## Reference code:
https://github.com/thuml/CDAN <br>
https://github.com/tim-learn/BA3US <br>
https://github.com/XJTU-XGU/RSDA
## Contact：
If you have any problem, feel free to contect xianggu@stu.xjtu.edu.cn.
