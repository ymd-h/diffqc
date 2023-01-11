from typing import Callable

import jax
import jax.numpy as jnp


def CreatePossibleState(op, nqubits: int, dtype: jnp.dtype) -> jnp.ndarray:
    """
    Create all possible state

    Parameters
    ----------
    op
        `dense` or `sparse` module
    nqubits : int
        Number of qubits
    dtype : jnp.dtype
        dtype

    Returns
    -------
    jnp.ndarray
        [|00...0>, |00...1>, ..., |11...1>]
    """
    cs = jnp.expand_dims(op.zeros(nqubits, dtype), 0)

    for i in range(nqubits):
        ith = nqubits - 1 - i
        @jax.vmap
        def flip(c):
            return op.PauliX(c, (ith,))
        cs = jnp.concatenate((cs, flip(cs)), axis=0)

    return cs
