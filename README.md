# MindSpore-DUCKNet
This repo is the MindSpore implementation of DUCK-Net for my class work **(not perfect)**.

The detailed paper about DUCK-Net can be found at [DUCK-Net paper](https://www.nature.com/articles/s41598-023-36940-5), 
and the author's open-source code can be found at [DUCK-Net code](https://github.com/RazvanDu/DUCK-Net).

Thanks the authors of DUCK-Net for their great work!!

## DUCK-Net Architecture
You can find the code of DUCK-Net based on the MindSpore in [DUCKNet_ms.py](ModelArchitecture/DUCKNet_ms.py).

## DUCK-Block
You can find the code of DUCK-Block based on the MindSpore in [ConvBlock2D_ms.py](CustomLayers/ConvBlock2D_ms.py).

## Installation
You can create the environment by the command:

```
conda env create -f environment.yml
```

## Running the project

The project can be run using Jupyter Notebook on the file [DUCKNet_ms.ipynb](DUCKNet_ms.ipynb).

## Citation

```
@article{article,
author = {Dumitru, Razvan-Gabriel and Peteleaza, Darius},
year = {2023},
month = {06},
pages = {},
title = {Using DUCK-Net for polyp image segmentation},
volume = {13},
journal = {Scientific Reports},
doi = {10.1038/s41598-023-36940-5}
}
```
