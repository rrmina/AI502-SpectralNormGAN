# SpectralNorm GAN Implementation

## How to run
```
python train.py
```

## Parameters
* `D_NORM_LAYER`: "SN" Discriminator uses SpectralNorm. "BN" - BaselineModel Discriminator uses batch norm instead of SpectralNorm

## Things to Note

Before running, please make these 2 directories

**Directories**
* `results`
* `pretrained_models`