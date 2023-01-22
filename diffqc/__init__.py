"""
diffqc: Differentiable Quantum Circuit Simulator
================================================

See Also
--------
diffqc.dense : Densely Represented Operations
diffqc.sparse : Sparsely Represented Operations
diffqc.lib : Builtin Algorithms
diffqc.nn : Builtin Neural Network Modules
diffqc.util : Utility Functions
diffqc.pennylane : PennyLane Plugin


Examples
--------
>>> import jax.numpy as jnp
>>> from diffqc import dense as op

>>> q0 = op.zeros(2, jnp.complex64) # |00>
>>> op.expectZ(q0, (0,))
1

>>> q1 = op.PauliX(q0, (0,))        # |10>
>>> op.expectZ(q1, (0,))
-1

>>> qh = op.Hadamard(q0, (0,))      # (|00> + |10>)/sqrt(2)
>>> qhcnot = op.CNOT(qh, (0, 1))    # (|00> + |11>)/sqrt(2)
>>> op.expectZ(qhcnot, (1,))
0
"""
from . import nn, sparse, dense, lib, util
from .common import *
