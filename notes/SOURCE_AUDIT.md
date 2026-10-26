# Source audit | 文献与 R 源码核对

Date: 2026-10-26

## Source 1: peer-reviewed paper

Jing, Bing-Yi; Ting Li; Zhongyuan Lyu; Dong Xia (2021), *Community detection on mixture multilayer networks via regularized tensor decomposition*, The Annals of Statistics 49(6):3181–3205. DOI: https://doi.org/10.1214/21-AOS2079 . Author manuscript: https://arxiv.org/pdf/2002.04457 .

Inspected manuscript equations (2.1)–(2.6), Algorithm 1 (regularized power iterations, PDF p.11), equation (3.1) (row clipping plus SVD, PDF p.13). The original data model has symmetric Bernoulli adjacency matrices with layer-specific latent network-class assignments, and shared vertices across layers.

## Source 2: related R implementation

`https://github.com/ChenyuzZZ73/rMultiNet`
Commit SHA inspected: `0d922beb9abdff7255c65d0378c3097fea2cdf69` (2022-11-30).

Relevant R sources:
- [`R/function_GenerateMMSBM.R`](https://github.com/ChenyuzZZ73/rMultiNet/blob/0d922beb9abdff7255c65d0378c3097fea2cdf69/R/function_GenerateMMSBM.R): samples symmetric Bernoulli SBM layers and random type memberships.
- [`R/function_InitializationMMSBM.R`](https://github.com/ChenyuzZZ73/rMultiNet/blob/0d922beb9abdff7255c65d0378c3097fea2cdf69/R/function_InitializationMMSBM.R): pooled adjacency eigenvector and layer-mode SVD warm start.
- [`R/function_PowerIteration.R`](https://github.com/ChenyuzZZ73/rMultiNet/blob/0d922beb9abdff7255c65d0378c3097fea2cdf69/R/function_PowerIteration.R): row clipping, tensor contractions, SVD factor updates.
- [`R/function_speclustering.R`](https://github.com/ChenyuzZZ73/rMultiNet/blob/0d922beb9abdff7255c65d0378c3097fea2cdf69/R/function_speclustering.R): a separate spectral clustering reference.

The inspected root directory had **no explicit LICENSE** file (`.gitignore`, `DESCRIPTION`, `NAMESPACE`, `R`, `README.md`, `data`, `framework.png`, `man`). Thus source code is not imported/copied into this implementation, and the MIT license of this *new* package does not purport to re-license rMultiNet.

## Verification and boundary

The new `twist` solver is based on paper-level matrix/tensor operations and an independent Python implementation. Distinct design decisions include: reorthogonalizing clipped rows as in the paper rather than only clipping as in R; projecting the final core from the new factors; and extending to a rectangular, role-separated log-weight tensor. No exact numerical parity against the original R code has been established because an R execution environment with rTensor and identical fixtures was not available during this build. Binary and continuous simulation unit tests are available locally.
