# AnaCP Model Repository

Welcome to the AnaCP Model Repository!

This repository contains code and resources to reproduce the experiments presented in our paper *"AnaCP: Toward Upper-Bound Continual Learning via Analytic Contrastive Projection"*. Follow the instructions below to set up the environment.

## Dependencies
- It is recommended to create a fresh conda environment before installing dependencies.
- Install the dependencies using the provided `requirements.txt` file:
  pip install -r requirements.txt

## Quick Start
- The `anacp.py` file in the `models` folder is a standalone module that can be used to incrementally train a model on input features.
- You can pass features `X` and labels `Y` incrementally by calling the `update(X, Y)` method.
- You can optionally apply FSA (First Session Adaptation) before feature extraction to further improve accuracy.
- To evaluate the model and other baselines, use the provided scripts in the repository.

## Datasets
- **Tiny-ImageNet** must be downloaded manually and placed in the `data/` folder. You can download it from the official source: http://cs231n.stanford.edu/tiny-imagenet-200.zip
- The other datasets will be downloaded automatically by the code unless the URLs provided become outdated.

## Pre-trained Checkpoint
- You should download the **MoCo v3** checkpoint from the official repository: https://github.com/facebookresearch/moco-v3. Place the checkpoint in the main folder.

Happy experimenting!


## Citation
If you find this repository useful, please cite our paper:

```
@inproceedings{momeni2025anacp,
  title={AnaCP: Toward Upper-Bound Continual Learning via Analytic Contrastive Projection},
  author={Saleh Momeni, Changnan Xiao, and Bing Liu},
  booktitle={Proceedings of The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

