from typing import Tuple

import jax
import jax.numpy as jnp


def GHZ(op, c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Create Greenberger–Horne–Zeilinger state from |00...0> state

    |00...0> -> (|00...0> + |11...1>)/sqrt(2)

    Parameters
    ----------
    op
        `dense` or `sparse` module
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply.

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    assert len(wires) >= 2, f"BUG: GHZ needs len(wires) >= 2, got {len(wires)}."

    c = op.Hadamard(c, (wires[0],))
    for i in range(len(wires)-1):
        c = op.CNOT(c, (wires[i], wires[i+1]))

    return c


def QFT(op, c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Quantum Fourier Transform

    Parameters
    ----------
    op
        `dense` or `sparse` module
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply.

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    for i in range(len(wires)):
        c = op.Hadamard(c, (i,))

        for j in range(i+1, len(wires)):
            c = op.ControlledPhaseShift(c, (j, i), 2 * jnp.pi / (2 ** (j-i+1)))

    return c
