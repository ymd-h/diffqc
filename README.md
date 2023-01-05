# diffq: Differentiable Quantum Circuit Simulator for Quantum Machine Learning

## 1. Overview

diffq is a python package providing differentiable quantum circuit simulator.
The main target is quantum machine learning.

diffq is built on [JAX](https://jax.readthedocs.io/en/latest/), so
that supported environments are limited to those supported by
JAX. (Ref.
["Installation" section at JAX README](https://github.com/google/jax#installation))

## 2. Features
diffq provides 2 types of operations, `dense` and `sparse`. Both have
same operations and only internal representations are different.

### 2.1 `dense` operation
In `dense` operation, complex coefficients of all possible
`2**nqubits` states are traced. This is simple matrix calculation but
requires exponentially large memory when `nqubits` is large.

### 2.2 `sparse` operation

> **Warning**  
> `sparse` module is under depelopment, and is not ready to use.

In `sparse` operation, only neccessary states are traced. This might
reduce memory requirements at large `nqubits` system, but it can be
computationally inefficient.


### 2.3 PennyLane Plugin

> **Warning**  
> PennyLane plugin is planned, but is still under development, and is not ready yet.

[PennyLane](https://pennylane.ai/) is a quantum machine learning
framework. By using PennyLane, we can choose machine learning
framework (e.g. [TensorFlow](https://www.tensorflow.org/),
[PyTorch](https://pytorch.org/)) and real/simulation quantum device
independently, and can switch relatively easy.

## 3. Example Usage
- example/00-circuit-basics.py
  - Basic Usage of diffq
- example/01-qcl-flax.py
  - QCL[1] Classification of Iris with [Flax](https://flax.readthedocs.io/en/latest/index.html)


## 4. References
- JAX
  - [Official Site](https://jax.readthedocs.io/en/latest/)
  - [Repository at GitHub](https://github.com/google/jax)
- PennyLane
  - [Official Site](https://pennylane.ai/)
  - [Repository at GitHub](https://github.com/PennyLaneAI/pennylane)
- TensorFlow
  - [Official Site](https://www.tensorflow.org/)
  - [Repository at GitHub](https://github.com/tensorflow/tensorflow)
- PyTorch
  - [Official Site](https://pytorch.org/)
  - [Repository at GitHub](https://github.com/pytorch/pytorch)
- Flax
  - [Official Site](https://flax.readthedocs.io/en/latest/index.html)
  - [Repository at GitHub](https://github.com/google/flax)
- [1] K. Mitarai et al. "Quantum Circuit Learning", Phys. Rev. A 98, 032309 (2018)
  - DOI: https://doi.org/10.1103/PhysRevA.98.032309
  - ArXiv: https://arxiv.org/abs/1803.00745
