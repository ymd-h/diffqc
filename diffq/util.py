from typing import Callable, Tuple

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

def CreateMatrix(op, nqubits: int, dtype: jnp.dtype,
                 f: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """
    Create Transition Matrix from function

    Parameters
    ----------
    op
        `dense` or `sparse`
    nqubits : int
        Number of qubits
    dtype : jnp.dtype
        dtype
    f : Callable[[jnp.ndarray], jnp.ndarray]

    Returns
    -------
    jnp.ndarray
        Transition Matrix
    """
    cs = CreatePossibleState(op, nqubits, dtype)

    @jax.vmap
    def F(c):
        return op.to_state(f(c))

    return jnp.moveaxis(F(cs), (0,), (1,))


def Convolution(op,
                kernel_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                kernel_shape: Tuple[int],
                slide: Tuple[int],
                padding: Tuple[int],
                ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Create Convolution Function

    Parameters
    ----------
    op
        `dense` or `sparse`
    kernel_func : Callable
        kernel function of ``f(x, w)``
    kernel_shape : tuple of ints
        kernel shape. ``len(kernel_shape) == 2``.
    slide : tuple of ints
        slides. ``len(slide) == 2``.
    padding : tuple of ints
        padding. ``len(padding) == 2``

    Returns
    -------
    Callable
        convoluted funcion of ``F(x, w)``
    """
    assert (
        len(kernel_shape) == len(slide) == len(padding) == 2
    ), f"BUG: k: {kernel_shape}, s: {slide}, p: {padding}"
    k0 = kernel_shape[0]
    k1 = kernel_shape[1]
    s0 = slide[0]
    s1 = slide[1]
    p0 = padding[0]
    p1 = padding[1]

    def F(x, w):
        x0 = x.shape[0]
        x1 = x.shape[1]

        # X: Padded x
        X0 = x0 + 2 * p0
        X1 = x1 + 2 * p1
        X = jnp.zeros(
            (X0, X1, *x.shape[2:]), dtype=x.dtype
        ).at[p0:X0-p0, p1:X1-p1].set(x)

        x0_idx = jnp.arange(0, x0 + 2 * p0 - k0, s0)
        x1_idx = jnp.arange(0, x1 + 2 * p1 - k1, s1)

        @jax.vmap
        def x0_loop(_x0):
            @jax.vmap
            def x1_loop(_x1):
                return kernel_func(jax.lax.dynamic_slice(X, (_x0, _x1), (k0, k1)), w)
            return x1_loop(x1_idx)

        x = x0_loop(x0_idx)
        assert (
            x.shape[:2] == ((x0 + 2 * p0 - k0) / s0,
                            (x1 + 2 * p1 - k1) / s1)
        ), f"BUG: Output Shape: {x.shape}"
        return x

    return F


def MaxPooling(x: jnp.ndarray, shape: Tuple[int]) -> jnp.ndarray:
    """
    Max Pooling for 2D

    Parameters
    ----------
    x : jnp.ndarray
        values
    shape : tuple of ints
        shape of taking max. ``len(shape) == 2``.

    Returns
    -------
    jnp.ndarray
        maxed values
    """
    assert len(shape) == 2, f"Bug: shape: {shape}"

    x0 = jnp.arange(0, x.shape[0], shape[0])
    x1 = jnp.arange(0, x.shape[1], shape[1])

    @jax.vmap
    def x0_loop(_x0):
        @jax.vmap
        def x1_loop(_x1):
            # dynamic_slice() is clipped for overrun index.
            return jnp.max(jax.lax.dynamic_slice(x, (_x0, _x1), shape))
        return x1_loop(x1)

    x = x0_loop(x0)
    return x
