from typing import Callable, Tuple

import jax
import jax.numpy as jnp

def CircuitCentricBlock(op, c: jnp.ndarray, wires: Tuple[int],
                        weights: jnp.ndarray) -> jnp.ndarray:
    """
    Apply Circuit Centric Block as Parameterized Quantum Circuit (PQC)

    Parameters
    ----------
    op
        `dense` or `sparse`
    c : jnp.ndarray
        qubits
    wires : tuple of ints
        wires. Usually, `(0, ..., qubits-1)`
    weights : jnp.ndarray
        parameters for rotation angle with shape of `(layers, 3 * qubits)`.

    Returns
    -------
    jnp.ndarray
        applied circuit

    Notes
    -----
    Code Block with range = 1 described at [1]_.
    According to [2]_, for middle scale circuit (4, 6, and 8 qubits)
    three layers have enough expressivity. (Circuit 19)

    References
    ----------
    .. [1] M. Schuld /et al/., "Circuit-centric quantum classifiers",
       Phys. Rev. A 101, 032308 (2020) (arXiv:1804.00633)
    .. [2] S. Sim /et al/., "Expressibility and entangling capability of
       parameterized quantum circuits for hybrid quantum-classical algorithms",
       Adv. Quantum Technol. 2 (2019) 1900070 (arXiv:1905.10876)
    """
    assert (weights.ndim == 2) and (weights.shape[1] == 3 * len(wires))

    n = len(wires)
    def Layer(ci, w):
        for i in wires:
            ci = op.RX(ci, (i,), w.at[i].get())
            ci = op.RZ(ci, (i,), w.at[i+n].get())

        for i in range(n):
            ci = op.CRX(ci, (wires[i], wires[(i+1) % n]), w.at[i+2*n].get())

        return ci, None

    c, _ = jax.lax.scan(Layer, c, weights)
    return c


def JosephsonSampler(op, c: jnp.ndarray, wires: Tuple[int],
                      weights: jnp.ndarray) -> jnp.ndarray:
    """
    Apply Josephson Sampler as Parameterized Quantum Circuit (PQC)

    Parameters
    ----------
    op
        `dense` or `sparse`
    c : jnp.ndarray
        qubits
    wires : tuple of ints
        wires. Ususally, `(0, ..., qubits-1)`
    weights : jnp.ndarray
        parameters of rotation with shape of `(layers, )`

    Returns
    -------
    jnp.ndarray
        applied circuit

    Notes
    -----
    Josephson Sampler circuit described at [1]_.
    According to [2]_, for middle scale circuit (4, 6, and 8 qubits)
    three layers have enough expressivity. (Circuit 11)

    References
    ----------
    .. [1] M. R. Geller, "Sampling and scrambling on a chain of superconducting
       qubits", Phys. Rev. Applied 10, 024052 (2018) (arXiv:1711.11026)
    .. [2] S. Sim /et al/., "Expressibility and entangling capability of
       parameterized quantum circuits for hybrid quantum-classical algorithms",
       Adv. Quantum Technol. 2 (2019) 1900070 (arXiv:1905.10876)
    """
    assert (weights.ndim == 2) and (weights.shape[1] == 4 * (len(wires) - 1))

    # qubits
    #   even => n1 = qubits
    #           n2 = qubits - 2
    #   odd  => n1 = qubits - 1
    #           n2 = qubits - 1
    #
    # => n1 + n2 = 2 * (qubits - 1)
    n  = len(wires)
    n1 = n - (n % 2)

    # RY: [   0   ,   n1     )
    # RZ: [  n1   , 2*n1     )
    # RY: [2*n1   , 2*n1+  n2)
    # RZ: [2*n1+n2, 2*n1+2*n2)
    def Layer(ci, w):
        for i in range(0, n-1, 2):
            ci = op.RY(ci, (wires[i],), w.at[i   ].get())
            ci = op.RZ(ci, (wires[i],), w.at[i+n1].get())

            ci = op.RY(ci, (wires[i+1],), w.at[i+1   ].get())
            ci = op.RZ(ci, (wires[i+1],), w.at[i+1+n1].get())

            ci = op.CX(ci, (wires[i], wires[i+1]))

        for i in range(1, n-1, 2):
            ci = op.RY(ci, (wires[i],), w.at[i+2*n1   ].get())
            ci = op.RZ(ci, (wires[i],), w.at[i+2*n1+n2].get())

            ci = op.RY(ci, (wires[i+1],), w.at[i+1+2*n1   ].get())
            ci = op.RZ(ci, (wires[i+1],), w.at[i+1+2*n1+n2].get())

            ci = op.CX(ci, (wires[i], wires[i+1]))

        return ci, None

    c, _ = jax.lax.scan(Layer, c, weights)
    return c


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

        x0_idx = jnp.arange(0, X0 - k0 + 1, s0)
        x1_idx = jnp.arange(0, X0 - k1 + 1, s1)

        @jax.vmap
        def x0_loop(_x0):
            @jax.vmap
            def x1_loop(_x1):
                return kernel_func(jax.lax.dynamic_slice(X, (_x0, _x1), (k0, k1)), w)
            return x1_loop(x1_idx)

        x = x0_loop(x0_idx)
        assert (
            x.shape[:2] == ((X0 - k0 + 1) // s0,
                            (X1 - k1 + 1) // s1)
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
