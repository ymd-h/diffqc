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

    Warnings
    --------
    This QFT doesn't swap wires, so that endian is inverted.
    """
    for i in range(len(wires)):
        c = op.Hadamard(c, (wires[i],))

        for j in range(i+1, len(wires)):
            c = op.ControlledPhaseShift(c,
                                        (wires[j], wires[i]),
                                        2 * jnp.pi / (2 ** (j-i+1)))

    return c

def QPE(op, c: jnp.ndarray, wires: Tuple[int],
        U: jnp.ndarray, aux: Tuple[int]) -> jnp.ndarray:
    """
    Quantum Phase Estimation

    Parameters
    ----------
    op
        `dense` or `sparse` module
    c : jnp.ndarray
        qubits state
    wires : tuples of ints
        wires. Eigen vector of U has been encoded.
    U: jnp.ndarray
        unitary matrix of which eigen value phase is estimated
    aux : tuple of ints
        auxiliary qubits. These should be |00...0>

    Returns
    -------
    jnp.ndarray
        applied qubits state. Phase is encoded at auxiliary qubits.
    """
    naux = len(aux)
    for a in aux:
        c = op.Hadamard(c, (a,))

    for i, a in enumerate(aux):
        c = op.ControlledQubitUnitary(c, (a, *wires), U)
        if i != naux - 1:
            U = U @ U

    # Inverse QFT
    for i in range(naux):
        h = naux - 1 - i
        for j in range(i):
            cnt = naux - 1 - j
            c = op.ControlledPhaseShift(c, (aux[cnt], aux[h]),
                                        -2 * jnp.pi / (2 ** (abs(h-cnt)+1)))
        c = op.Hadamard(c, (aux[h],))

    return c
