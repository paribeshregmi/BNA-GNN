# BNA-GNN

This repository contains the official implementation of BNA-GNN from the paper:

> ### Bayesian Neighborhood Adaptation for Graph Neural Networks
> Paper link: https://openreview.net/forum?id=2zEemRib3a
---

## ⚙️ Installation

The code is written in **Python** and requires the following packages:

- `torch`
- `numpy`
- `dgl`

Install dependencies using:

```bash
pip install torch numpy dgl
```

## 🚀 Running Experiments

Use the following commands to run an experiment:
```python
python run.py --fs True --dataset cora --truncation 8 --kld_weight 1.0
python run.py --fs True --dataset citeseer --truncation 8 --kld_weight 2.0
```
**Key Arguments**
```bash
--dataset: Dataset name (cora, citeseer, pubmed)
--fs: enables full supervised learning
--truncation: Maximum neighborhood scope truncation
--kld_weight: Weight for KL-divergence regularization
```

## 📚 References
Parts of the implementation are adapted or modified from the following repositories:
```
https://github.com/kckishan/Depth_and_Dropout/
```
We thank the authors for making their code publicly available.

## 📄 Citation

If you use this code, please cite:
```
@article{Regmi2025:BNA-GNN,
title={Bayesian Neighborhood Adaptation for Graph Neural Networks},
author={Paribesh Regmi and Rui Li and Kishan K C},
journal={Transactions on Machine Learning Research (TMLR)},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=2zEemRib3a},
}
```
