# QuaDFormer

This study proposes QuaDFormer, a Transformer-based solver for the Traveling Salesman Problem (TSP). QuaDFormer aims to enhance the model’s representational capacity through a more geometrically enriched embedding scheme. To this end, we introduce three key methodological innovations. First, the spatial representation is constructed using Recursive Fuzzy Quadrisection, which captures the geographic layout of nodes through diverse holographic resolutions. Second, a density embedding mechanism is employed to integrate geographical information regarding congestion around each vertex; this is specifically decomposed into a Directionless Density Embedding and a Directional Density Embedding to concurrently encapsulate both density and directional cues. Third, a Context Binding Matrix is incorporated subsequent to the final decoder layer to refine and smooth the information flow from the encoder to the decoder. The introduction of this matrix is designed to address the issue of weak token identity inherent in the TSP domain, where the representation of a node (token) lacks consistent meaning outside its corresponding problem instance. This matrix is derived by processing the encoder’s final output, thereby enforcing a stronger binding between encoder and decoder representations.

## Requirements

- CUDA 12.1
- Python 3.10.12
- Torch 2.2.0+cu121

## Pretrained Checkpoints

다운로드 링크 제공

## Datasets

[Dataset Download Link](https://www.dropbox.com/scl/fo/tw6rjngvn6sxo1k75ddej/ALJNMyz8S5OVXyXEhftbom4?rlkey=7sn8wcsll3cl1oane9cqqzqmc&st=4ntd149x&dl=0)
