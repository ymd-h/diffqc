"""
Builtin Algorithms (:mod:`diffqc.lib`)
======================================

Notes
-----
To support multiple internal representations,
operation module (aka. :mod:`diffqc.dense` or :mod:`diffqc.sparse`) is passed.
"""

from typing import Tuple

import jax.numpy as jnp


def GHZ(op, c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Create Greenberger-Horne-Zeilinger state from ``|00...0>`` state

    ``|00...0>`` -> ``(|00...0> + |11...1>)/sqrt(2)``

    Parameters
    ----------
    op
        ``dense`` or ``sparse`` module
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
        ``dense`` or ``sparse`` module
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
        ``dense`` or ``sparse`` module
    c : jnp.ndarray
        qubits state
    wires : tuples of ints
        wires. Eigen vector of U has been encoded.
    U: jnp.ndarray
        unitary matrix of which eigen value phase is estimated
    aux : tuple of ints
        auxiliary qubits. These should be ``|00...0>``

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


def HHL(op, c: jnp.ndarray, wires: Tuple[int],
        U: jnp.ndarray, aux: Tuple[int], anc: int) -> jnp.ndarray:
    """
    Solving Linear Equation with Harrow-Hassidim-Lloyd Algorithm

    Solve ``A|x> = |b>`` and get ``|x> = A^(-1)|b>`` .

    Parameters
    ----------
    op
        ``dense`` or ``sparse`` module
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wires. ``|b>``
    U : jnp.ndarray
        unitary matrix of exp(iA)
    aux : tuple of ints
        auxiliary qubits for QPE. These should be ``|00...0>``
    anc : int
        ancilla qubits for HHL

    Returns
    -------
    jnp.ndarray
        applied qubits state.
    """
    c = QPE(op, c, wires, U, aux)

    for i, a in enumerate(aux):
        # Lambda = 2pi / (2^(i+1))
        # C = 2^(-len(aux))
        C_over_lambda = (2 ** (i+1 - len(aux))) / (2 * jnp.pi)
        c = op.CRY(c, (a, anc), 2 * jnp.arcsin(C_over_lambda))

    # Uncompute QPE
    c = QFT(op, c, wires)
    c = op.ControlledQubitUnitary(c, aux, jnp.conj(jnp.transpose(U)))
    for i in wires:
        c = op.Hadamard(c, (i,))

    return c
