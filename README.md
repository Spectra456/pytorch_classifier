
# Multi-task learning Pytorch
Simple multi-task learning example on 2 different datasets.
For all our test we using EfficentNetB0 with input size 3x32x32

# Cifar-10 (Single model)
*Best Accuracy(top-1):* 88.7%
**Accuracy(top-1) and Loss function:**
![Top-1 Accuracy](https://i.imgur.com/QQHZSy7.png)
![Loss](https://i.imgur.com/skHlZlU.png)

**Confusion matrix**
![Confusion matrix](https://i.imgur.com/BXZGaw4.png)

## FashionRGB (Single model)
It's a dataset from my previous test task.
Here we have 5 classes: *Blouse, Dress,Jeans Skirt, Tank*
5k rgb images in train set and 5k images in val set

*Bloose*:
![Bloose](https://i.imgur.com/tXmYd1h.jpg)

*Dress*:
![Dress](https://i.imgur.com/XIY11zh.jpg)

*Jeans*:
![Jeans](https://i.imgur.com/YFFSPhs.jpg)

*Skirt*:
![Skirt](https://i.imgur.com/p37IxCg.jpg)

*Tank*:
![Tank](https://i.imgur.com/6sLSHsL.jpg)

*Best Accuracy(top-1):* 77.4%
**Accuracy(top-1) and Loss function:**
![Accuracy(top-1)](https://i.imgur.com/kpTzril.png)
![Loss](https://i.imgur.com/r71lPgJ.png)
**Confusion Matrix**
![Confusion matrix](https://i.imgur.com/UopLddi.png)

## Multi-Task Model
*Best Accuracy(top-1) FashionRGB:* 80.3%
*Best Accuracy(top-1) Cifar10:* 80.6%

*I tried several methods of MLT, but there always some disbalance between datasets. Here i used my own solution, where:*

    for i in range(len(train_loader_cifar):
	    if (bool(random.getrandbits(1)) ==  True):
		    then train Cifar10 layer
	    if (bool(random.getrandbits(1)) ==  True):
		    then train FashionRGB layer
*It's help us keeping balance between metrics of our datasets*
**Accuracy(top-1) and Loss function:**

![Accuracy(top-1)](https://i.imgur.com/Psmtl5G.png)

![Loss](https://i.imgur.com/UFPpTAa.png)

**Confusion matrix**
Fashion-RGB
![Fashion-RGB Confusion matrix](https://i.imgur.com/jBJVRB9.png)
Cifar-10
![Cifar-10 Confusion matrix](https://i.imgur.com/qaI1eh9.png)

## Conclusion
| Model|Accuracy|
|--|--|
| Cifar-10 |88.7 %|
|FashionRGB|77.4 %|
|MLT(Cifar-10)|80.6 %(⬇️8.1% ⬇️)|
|MLT(FashionRGB)|80.3 %(⬆️2.9 % ⬆️)|

## Run
1. Download FashionRGB dataset from [Google Drive](https://drive.google.com/open?id=14X2KGG_ov0jG04DM2e8xK0AXeBOtnmSS).

2. Unzip archive into ***dataset*** folder of this project.
3. `python train_simple.py --dataset FashionRGB`
 4. `python train_simple.py --dataset FashionRGB`
 5. `python train_multi.py`
You can check all arguments inside this two python scripts.
5. Launch tensorboard (optional)
If you want to check my experiments in tensorboard
``` 
tensorboard --logdir logs/old
```
