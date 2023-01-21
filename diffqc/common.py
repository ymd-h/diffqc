from typing import Tuple, Union

import jax
import jax.numpy as jnp

__all__ = [
    "prob",
    "marginal_prob",
    "sample",
    "expval"
]


# jax.jit compatible jnp.reshape(s, (2,) * int(jnp.log2(s.shape[0])))
def _qubit_shape(s):
    if s.shape[0] <= 2:
        return s

    s = jnp.reshape(s, (2, -1))
    return jax.vmap(_qubit_shape)(s)


def prob(state: jnp.ndarray, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """
    Calculate probabilities for n-qubit state

    Parameters
    ----------
    state : 1d jnp.ndarray
        n-qubit state e.g. ``[|000>, |001>, |010>, ..., |111>]``
    dtype : jnp.dtype
        dtype of probabilities. Default is jnp.float32.

    Returns
    -------
    jnp.ndarray
        Probabilities
    """
    return jnp.asarray(jnp.square(jnp.abs(state)), dtype=dtype)


def marginal_prob(probs: jnp.ndarray,
                  integrate_wires: Union[int, Tuple[int]]) -> jnp.ndarray:
    """
    Calculate marginal probabilities

    Parameters
    ----------
    probs : 1d jnp.ndarray
        Probabilities of n-qubit state
    integrate_wires : int or tuple of ints
        Wires which will be marginalized out

    Returns
    -------
    jnp.ndarray
        Marginalized probabilities
    """
    p = _qubit_shape(probs)
    return jnp.reshape(jnp.sum(p, axis=integrate_wires), (-1,))


def sample(key: jax.random.PRNGKeyArray,
           probs: jnp.ndarray,
           shape: Tuple[int]) -> jnp.ndarray:
    """
    Sample proportional to probabilities

    Parameters
    ----------
    key : jax.random.PRNGKeyArray
        Random Key.
    probs : 1d jnp.ndarray
        Probabilities of n-qubit state
    shape : tuple of ints
        Sample shape

    Returns
    -------
    jnp.ndarray
        Sampled index. e.g. ``0 -> |000>, 1 -> |001>`` for 3-qubits.
    """
    return jax.random.choice(key,
                             jnp.arange(probs.shape[0]),
                             shape=shape,
                             p=probs)


def expval(probs: jnp.ndarray, wire: int):
    """
    Expectation of ``|1>``

    Parameters
    ----------
    probs : jnp.ndarray
        Probabilities of n-qubit state
    wire : int
        Expectation measurement wire

    Returns
    -------
    float like
        Expectation of ``|1>``
    """
    p = _qubit_shape(probs)
    return jnp.sum(jnp.take(p, jnp.ones((1,), dtype=jnp.int32), axis=wire))
