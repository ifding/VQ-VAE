# CT-Lung-Segmentation

This repository contains a Pytorch implementation of Lung CT image segmentation Using U-net

## Requirement

segmentation_models_pytorch

jupyter

numpy

opencv-python

Pillow

torch==1.8.1

torchvision==0.9.1

tqdm==4.61.0

```
    pip install -r requirements.txt
```

## Dataset

1. Download the data from [Kaggle/Finding and Measuring Lungs in CT Data](https://www.kaggle.com/kmader/finding-lungs-in-ct-data)
2. With totally 267 CT slices, I randomly select 200 slices for training and 67 for testing