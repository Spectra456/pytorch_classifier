# Pytorch classifier

It's a simple CNN on Pytorch for classification type of clothing.

## Requirments

- torch 1.2.0
- torchvision 0.4.0
- tqdm 4.31.1
- pandas 0.24.2
- numpy 1.16.1
- tensorboard 1.14.0

## Run

1. Download this repository.
```bash
git clone https://github.com/Spectra456/pytorch_classifier.git
```

2. Download dataset from [Google Drive](https://drive.google.com/open?id=14X2KGG_ov0jG04DM2e8xK0AXeBOtnmSS).

3. Unzip archive to folder of project.

4. Launch training script.
```bash
python train.py
```
5. Launch tensorboard (optional)
```bashpython train.py
tensorboard --logdir logs
```
## Results
* Input size - 32x32
* Best loss - 1.149
* Best accuracy - 0.753

![Accuracy](https://github.com/Spectra456/pytorch_classifier/blob/master/images/IMG_6036.png)
![Loss](https://github.com/Spectra456/pytorch_classifier/blob/master/images/IMG_6037.png)
