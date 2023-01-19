# diffqc: Differentiable Quantum Circuit Simulator for Quantum Machine Learning

## 1. Overview
> **Note**  
> This project started as diffq, but because of accidental name conflict,
> we changed name to diffqc.

diffqc is a python package providing differentiable quantum circuit simulator.
The main target is quantum machine learning.

diffqc is built on [JAX](https://jax.readthedocs.io/en/latest/), so
that it is
* GPU friendly,
* easily vectorized,
* differentiable, but
* supported environments are limited. (Ref.
["Installation" section at JAX README](https://github.com/google/jax#installation))

## 2. Features
diffqc provides 2 types of operations, `dense` and `sparse`. Both have
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

### 2.3 Builtin Algorithm `lib`
Builtin algorithms are implemented at `diffqc.lib`. To support both
`dense` and `sparse` operation, operation module is passed to 1st
argument.


* `GHZ(op, c: jnp.ndarray, wires: Tuple[int])`
  * Create Greenberger-Horne-Zeilinger state [2]
  * `|00...0>` -> `(|00...0> + |11...1>)/sqrt(2)`
* `QFT(op, c: jnp.ndarray, wires: Tuple[int])`
  * Quantum Fourier Transform (without last swap) [3]
* `QPE(op, c: jnp.ndarray, wires: Tuple[int], U: jnp.ndarray, aux: Tuple[int])`
  * Quantum Phase Estimation [4]
  * `wires`: Eigen Vector
  * `U`: Unitary Matrix
  * `aux`: Auxiliary qubits. These should be `|00...0>`.


### 2.4 PennyLane Plugin

> **Warning**  
> PennyLane plugin is planned, but is still under development, and is not ready yet.

[PennyLane](https://pennylane.ai/) is a quantum machine learning
framework. By using PennyLane, we can choose machine learning
framework (e.g. [TensorFlow](https://www.tensorflow.org/),
[PyTorch](https://pytorch.org/)) and real/simulation quantum device
independently, and can switch relatively easy.

## 3. Example Usage
- example/00-circuit-basics.py
  - Basic Usage of diffqc
- example/01-qcl-flax.py
  - QCL[1] Classification of [Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) with [Flax](https://flax.readthedocs.io/en/latest/index.html)
- example/02-cnn-like-qcl-flax.py
  - CNN-like QCL[1] Classification of [Digits](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset) with [Flax](https://flax.readthedocs.io/en/latest/index.html)
- example/03-pennylane.py
  - PennyLane Plugin
- example/04-builtin-variational-cercuit-centric.py
  - Builtin Variational Circuit: Circuit Centric Block described at [5]
  - According to [6], this is one of the best circuit.
- example/05-builtin-variational-josephson-sampler.py
  - Builtin Variational Circuit: Josephson Sampler described at [7]
  - According to [6], this is one of the best circuit.

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
  - arXiv: https://arxiv.org/abs/1803.00745
- [2] D. M. Greenberger et al., "Going Beyond Bell's Theorem", arXiv:0712.0921
  - arXiv: https://arxiv.org/abs/0712.0921
- [3] D. Coppersmith, "An approximate Fourier transform useful in quantum factoring",
  IBM Research Report RC19642
  - arXiv: https://arxiv.org/abs/quant-ph/0201067
- [4] A. Kitaev, "Quantum measurements and the Abelian Stabilizer Problem",
  arXiv:quant-ph/9511026
  - arXiv: https://arxiv.org/abs/quant-ph/9511026
- [5] M. Schuld et al., "Circuit-centric quantum classifiers",
  Phys. Rev. A 101, 032308 (2020)
  - DOI: https://doi.org/10.1103/PhysRevA.101.032308
  - arXiv: https://arxiv.org/abs/1804.00633
- [6] S. Sim et al., "Expressibility and entangling capability of
  parameterized quantum circuits for hybrid quantum-classical algorithms",
  Adv. Quantum Technol. 2 (2019) 1900070
  - DOI: https://doi.org/10.1002/qute.201900070
  - arXiv: https://arxiv.org/abs/1905.10876
- [7] M. R. Geller, "Sampling and scrambling on a chain of superconducting qubits",
  Phys. Rev. Applied 10, 024052 (2018)
  - DOI: https://doi.org/10.1103/PhysRevApplied.10.024052
  - arXiv: https://arxiv.org/abs/1711.11026
