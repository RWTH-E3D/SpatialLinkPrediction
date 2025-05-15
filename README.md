<div align="center">

# Spatial Link Prediction

This project is based on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template). See [template docs](template-doc.md) for more information.

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

This repository contains the code for the experiments in our paper [Spatial Link Prediction: Learning topological relationships in MEP systems](https://www.sciencedirect.com/science/article/pii/S1474034625003076).

## How to run
This code was developed and tested with Python 3.9.15, PyTorch 1.13, and torch-geometric 2.2.0.

Install dependencies

```bash
# clone project
git clone https://github.com/RWTH-E3D/SpatialLinkPrediction
cd SpatialLinkPrediction

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch 1.13 according to instructions
# https://pytorch.org/get-started/
# For Linux and Windows
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# install torch-geometric
conda install pyg -c pyg

# install other requirements
pip install -r requirements.txt
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py experiment=experiment_name.yaml trainer.max_epochs=20 datamodule.batch_size=64
```
