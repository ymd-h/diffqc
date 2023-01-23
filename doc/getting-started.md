# Getting Started


## Choose Operation Module

diffqc provides multiple operation modules (aka. `dense` and `sparse`).
These modules provide same functionality with different internal implementation.

If you don't have special needs, we recommend to use `dense` module
and to rename it for possible future replacement.

```python
from diffqc import dense as op
```


## Create Quantum State and Apply Quantum Gate

Initial `|00...0>` state can be created by `op.zeros(nqubits, dtype)`.

```python
import jax.numpy as jnp
from diffqc import dense as op

nqubits = 5
q = op.zeros(nqubits, jnp.complex64)
```


Quantum gate operations (e.g. `PauliX`, `Hadamard`, `CNOT`) take such
quantum state, wire positions, and parameters (if exists), then
return the new quantum state.


```python
q = op.PauliX(q, (0,))
# |10000>

q = op.Hadamard(q, (1,))
# (|10000> + |11000>)/sqrt(2)

q = op.CNOT(q, (1, 2))
# (|10000> + |11100>)/sqrt(2)
```

```{note}
These quantum gate operations are executed immediately.
Gate decomposition and/or optimization is out of scope
in this project.
```


## Expectation of Measurement

Expectation value at a wire position can be taken with corresponding
function;

```python
x = op.expectX(q, (0,)) # <q|X|q> at wire 0
y = op.expectY(q, (1,)) # <q|Y|q> at wire 1
z = op.expectZ(q, (2,)) # <q|Z|q> at wire 2
```


## Convert to State-Vector Representation

State-Vector representation is a 1d vector with size of `2**(qubits)`.
For 2 qubits, the values are probability amplitudes of `[|00>, |01>, |10>, |11>]`.


```python
sv = op.to_state(q)
```
