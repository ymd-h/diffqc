"""
diffqc.sparse
"""

import math
from typing import Tuple

import jax
import jax.numpy as jnp

from . import _operators as _op

__all__ = [
    # Util
    "zeros",
    "to_state",

    # Expectation
    "expectX",
    "expectY",
    "expectZ",
    "expectUnitary",

    # Non-Parametric Operation
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "S",
    "T",
    "SX",
    "CNOT",
    "CZ",
    "CY",
    "SWAP",
    "ISWAP",
    "ECR",
    "SISWAP",
    "SQISWAP",
    "CSWAP",
    "Toffoli",

    # Parametric Operation
    "Rot",
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "ControlledPhaseShift",
    "CPhase",
    "CPhaseShift00",
    "CPhaseShift01",
    "CPhaseShift10",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "U1",
    "U2",
    "U3",
    "PSWAP",
    "RXX",
    "RYY",
    "RZZ",

    # General Matrix Operation
    "QubitUnitary",
    "ControlledQubitUnitary",
]

# StateVec Shape: [possibility, nqubit, qubit]
def zeros(nqubits: int, dtype: jnp.dtype) -> jnp.ndarray:
    """
    Create ``|00...0>`` state

    Parameters
    ----------
    nqubits : int
        Number of qubits
    dtype : jnp.dtype
        dtype.

    Returns
    -------
    jnp.ndarray
        Zero state aka. ``|00...0>``
    """
    return jnp.zeros((1, nqubits, 2), dtype=dtype).at[:,:,0].set(1)

def to_state(x: jnp.ndarray) -> jnp.ndarray:
    """
    Convert qubits internal representation to state vector

    Parameters
    ----------
    x : jnp.ndarray
        qubits state

    Returns
    -------
    jnp.ndarray
        State vector, e.g. ``[|000>, |001>, |010>, ..., |111>]``
    """
    if x.shape[0] > (2 ** x.shape[1]):
        jax.debug.print(
            "Warning: This circuit used required more memory than `dense` does. " +
            "Please consider to use `dense` instead."
        )

    n = x.shape[1]
    idx = jnp.arange(2 ** n)
    qubit = jnp.arange(n)

    def f(c, xi):
        @jax.vmap
        def ps(i):
            @jax.vmap
            def p(q):
                return xi.at[q, (i >> (n-1-q)) % 2].get()

            return jnp.prod(p(qubit))

        c = c.at[:].add(ps(idx))
        return c, None

    state, _ = jax.lax.scan(f, jnp.zeros(2 ** n, dtype=x.dtype), x)
    return state

def _expect(c1, c2):
    return jnp.real(jnp.dot(jnp.conj(to_state(c1)), to_state(c2)))

def expectX(c: jnp.ndarray, wires: Tuple[int]) -> float:
    """
    Expectation of X measurement

    Parameters
    ----------
    c : jnp.ndarray
        qubit state
    wire : tuple of ints
        wires to measure. ``len(wires)`` must be ``1``
    """
    return _expect(c, PauliX(c, wires))

def expectY(c: jnp.ndarray, wires: Tuple[int]) -> float:
    """
    Expectation of Y measurement

    Parameters
    ----------
    c : jnp.ndarray
        qubit state
    wire : tuple of ints
        wires to measure. ``len(wires)`` must be ``1``
    """
    return _expect(c, PauliY(c, wires))

def expectZ(c: jnp.ndarray, wires: Tuple[int]) -> float:
    """
    Expectation of Z measurement

    Parameters
    ----------
    c : jnp.ndarray
        qubit state
    wire : tuple of ints
        wires to measure. ``len(wires)`` must be ``1``
    """
    return _expect(c, PauliZ(c, wires))


def expectUnitary(c: jnp.ndarray, wires: Tuple[int], U: jnp.ndarray) -> float:
    """
    Expectation of Unitary measurement

    Parameters
    ----------
    c : jnp.ndarray
        qubit state
    wire : tuple of ints
        wires to measure.
    U : jnp.ndarray
        Unitary matrix
    """
    return _expect(c, QubitUnitary(c, wires, U))


# Internal Functions
BUG = "BUG: {} quantum operation is called with wrong wires: {}"

def op1(c, wires, op):
    assert len(wires) == 1, BUG.format(1, wires)
    i = wires[0]
    @jax.vmap
    def set(ci):
        q = ci.at[i, :]
        ci = q.set(op @ q.get())
        return ci
    return set(c)

def opN(c, wires, opf):
    assert len(wires) > 1, f"BUG: opN with wrong wires: {len(wires)}"
    assert c.ndim == 3, f"BUG: opN with wrong ndim: {c.ndim}"
    @jax.vmap
    def set(ci):
        assert ci.ndim == 2, f"BUG: {ci.ndim}"
        q = opf(ci.at[wires,:].get())
        ci = jnp.broadcast_to(ci, (q.shape[0], *ci.shape))
        ci = ci.at[:, wires, :].set(q)
        return ci
    return jnp.reshape(set(c), (-1, *c.shape[1:]))

def control_op(op):
    def COP(q12):
        assert q12.shape == (2, 2), f"BUG: control_op with wrong shape: {q12.shape}"
        zero = q12.at[0, 0].get()
        one  = q12.at[0, 1].get()

        ret = jnp.zeros((2, 2, 2), dtype=q12.dtype)
        ret = ret.at[0, 0, 0].set(zero).at[1, 0, 1].set(one)
        ret = ret.at[0, 1, :].set(q12.at[1, :].get())
        ret = ret.at[1, 1, :].set(op @ q12.at[1, :].get())
        return ret
    return COP

def entangle_op2(op):
    assert op.shape == (4, 4), f"BUG: entangle_op2 with wrong op shape: {op.shape}"
    def EOP(q12):
        assert q12.shape == (2, 2), f"BUG: entangle_op2 with wrong shape: {q12.shape}"
        q = jnp.asarray([q12.at[0, 0].get() * q12.at[1, 0].get(), # |00>
                         q12.at[0, 0].get() * q12.at[1, 1].get(), # |01>
                         q12.at[0, 1].get() * q12.at[1, 0].get(), # |10>
                         q12.at[0, 1].get() * q12.at[1, 1].get()],# |11>
                        dtype=q12.dtype)
        coef = jnp.expand_dims(op @ q, (1,))
        return jnp.asarray([[[1, 0], [1, 0]],
                            [[1, 0], [0, 1]],
                            [[0, 1], [1, 0]],
                            [[0, 1], [0, 1]]],
                           dtype=q12.dtype).at[:,0,:].multiply(coef)
    return EOP


# Quantum Operators
def Hadamard(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply Hadamard Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.H(c.dtype))

def PauliX(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply Pauli X Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.sigmaX(c.dtype))

def PauliY(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply Pauli Y Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.sigmaY(c.dtype))

def PauliZ(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply Pauli Z Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.sigmaZ(c.dtype))

def S(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply S Gate (Single qubite Phase Gate)

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.phaseS(c.dtype))

def T(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply T Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.phaseT(c.dtype))

def SX(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply Square Root X Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.sqrtX(c.dtype))

def CNOT(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply controlled-NOT Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, control_op(_op.sigmaX(c.dtype)))

def CZ(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply controlled-Z Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, control_op(_op.sigmaZ(c.dtype)))

def CY(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply controlled-Y Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, control_op(_op.sigmaY(c.dtype)))

def SWAP(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply SWAP Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, lambda q12: jnp.expand_dims(q12.at[(1, 0), :].get(), 0))

def ISWAP(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply i-SWAP Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, entangle_op2(_op.ISWAP(c.dtype)))

def ECR(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply Echoed RZX(pi/2) Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, entangle_op2(_op.ECR(c.dtype)))

def SISWAP(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply Square Root i-SWAP Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, entangle_op2(_op.SISWAP(c.dtype)))

SQISWAP = SISWAP

def CSWAP(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply controlled SWAP Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``3``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    def cswap(q123):
        s0 = q123.at[0, 0].get()
        s1 = q123.at[0, 1].get()
        return (jnp.zeros((2, 3, 2), dtype=c.dtype)
                .at[0, 0, 0].set(s0)
                .at[1, 0, 1].set(s1)
                .at[0, 1:3, :].set(s0 * q123.at[1:3, :].get())
                .at[1, 1:3, :].set(s1 * q123.at[(2,1), :].get()))
    return opN(c, wires, cswap)

def Toffoli(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply Toffoli Gate (controlled controlled X Gate)

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``3``

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    def toffoli(q123):
        s0 = q123.at[0, 0].get()
        s10 = q123.at[0, 1].get() * q123.at[1, 0].get()
        s11 = q123.at[0, 1].get() * q123.at[1, 1].get()
        return (jnp.zeros((3, 3, 2), dtype=c.dtype)
                .at[0, 0, 0].set(s0)
                .at[0, 1:3, :].set(s0 * q123.at[1:3, :].get())
                .at[1, 0, 1].set(s10)
                .at[1, 1, 0].set(s10)
                .at[1, 2, :].set(s10 * q123.at[2, :].get())
                .at[2, 0:2, 1].set(s11)
                .at[2, 2, :].set(s11 * q123.at[2, (1, 0)].get()))
    return opN(c, wires, toffoli)


def Rot(c: jnp.ndarray, wires: Tuple[int],
        phi: float, theta: float, omega: float) -> jnp.ndarray:
    """
    Apply Rotation Gate

    Rot(phi, theta, omega) = RZ(omega)RY(theta)RZ(phi)

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``
    phi, theta, omega : float
        rotation angles

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.Rot(c.dtype, phi, theta, omega))

def RX(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply Rotation X Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.RX(c.dtype, phi))

def RY(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply Rotation Y Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.RY(c.dtype, phi))

def RZ(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply Rotation Z Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.RZ(c.dtype, phi))

def PhaseShift(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply Local Phase Shift Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.phaseShift(c.dtype, phi))

def ControlledPhaseShift(c: jnp.ndarray, wires: Tuple[int],
                         phi: float) -> jnp.ndarray:
    """
    Apply controlled Phase Shift Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, control_op(_op.phaseShift(c.dtype, phi)))

CPhase = ControlledPhaseShift

def CPhaseShift00(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply Phase Shift Gate for ``|00>``

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, entangle_op2(_op.phaseShift00(c.dtype, phi)))

def CPhaseShift01(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply Phase Shift Gate for ``|01>``

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, entangle_op2(_op.phaseShift01(c.dtype, phi)))

def CPhaseShift10(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply Phase Shift Gate for ``|10>``

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, entangle_op2(_op.phaseShift10(c.dtype, phi)))

def CRX(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply controlled RX Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, control_op(_op.RX(c.dtype, phi)))

def CRY(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply controlled RY Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, control_op(_op.RY(c.dtype, phi)))

def CRZ(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply controlled RZ Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, control_op(_op.RZ(c.dtype, phi)))

def CRot(c: jnp.ndarray, wires: Tuple[int],
         phi: float, theta: float, omega: float) -> jnp.ndarray:
    """
    Apply controlled Rotation Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``
    phi, theta, omega : float
        rotation angles

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, control_op(_op.Rot(c.dtype, phi, theta, omega)))

U1 = PhaseShift

def U2(c: jnp.ndarray, wires: Tuple[int], phi: float, delta: float) -> jnp.ndarray:
    """
    Apply controlled U2 Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``
    phi, delta : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.U2(c.dtype, phi, delta))

def U3(c: jnp.ndarray, wires: Tuple[int],
       theta: float, phi: float, delta: float) -> jnp.ndarray:
    """
    Apply U3 Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``
    theta, phi, delta : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.U3(c.dtype, theta, phi, delta))

def PSWAP(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply Phase SWAP Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return opN(c, wires, entangle_op2(_op.PSWAP(c.dtype, phi)))


def mat_opf(c, U):
    assert c.ndim == 3
    nqubits = int(math.log2(U.shape[0]))
    assert c.shape[1] == nqubits

    q = U @ to_state(c)

    c1 = jnp.asarray([[1, 0],
                      [0, 1]], dtype=c.dtype)
    c = jnp.expand_dims(c1.copy(), 1)
    assert c.shape == (2, 1, 2)

    for i in range(nqubits-1):
        @jax.vmap
        def stack(ci):
            ci = jnp.broadcast_to(jnp.reshape(ci, (1, 1, 2)), (c.shape[0], 1, 2))
            return jnp.concatenate((ci, c), axis=1)

        c = jnp.reshape(stack(c1), (-1, i+2, 2))
        assert c.shape == (2**(i+2), i+2, 2), f"BUG: i: {i}, shape: {c.shape}"

    c = c.at[:,0,:].multiply(jnp.expand_dims(q, 1))

    assert c.shape == (U.shape[0], nqubits, 2)
    return c

def QubitUnitary(c: jnp.ndarray, wires: Tuple[int],
                 U: jnp.ndarray) -> jnp.ndarray:
    """
    Unitary Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``log2(U.ndim)``
    U : jnp.ndarray
        square unitary matrix

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    nqubits = int(math.log2(U.shape[0]))
    assert len(wires) == nqubits, BUG.format(nqubits, wires)
    assert c.ndim == 3

    if len(wires) == 1:
        return op1(c, wires, U)

    if c.shape[1] == nqubits:
        return mat_opf(c, U)

    return opN(c, wires, lambda ci: mat_opf(jnp.expand_dims(ci, 0), U))


def ControlledQubitUnitary(c: jnp.ndarray, wires: Tuple[int],
                           U: jnp.ndarray) -> jnp.ndarray:
    """
    Controlled Unitary Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1 + log2(U.ndim)``
    U : jnp.ndarray
        square unitary matrix

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    nqubits = 1 + int(math.log2(U.shape[0]))
    assert len(wires) == nqubits, BUG.format(nqubits, wires)

    CU = jnp.identity(U.shape[0] * 2,
                      dtype=c.dtype).at[
                          U.shape[0]:,
                          U.shape[1]:
                      ].set(U)

    return QubitUnitary(c, wires, CU)


def RXX(c: jnp.ndarray, wires: Tuple[int], theta: float) -> jnp.ndarray:
    """
    Rotate XX (exp(-iXX * thta))

    Parameters
    ----------
    c : jnp.ndarray
        qubit state
    wires : tuple of ints
        wires to apply. ``len(wires)`` must be ``2``

    Returns
    -------
    jnp.ndarray
        applied qubit state
    """
    return opN(c, wires, entangle_op2(_op.RXX(c.dtype, theta)))


def RYY(c: jnp.ndarray, wires: Tuple[int], theta: float) -> jnp.ndarray:
    """
    Rotate YY (exp(-iYY * thta))

    Parameters
    ----------
    c : jnp.ndarray
        qubit state
    wires : tuple of ints
        wires to apply. ``len(wires)`` must be ``2``

    Returns
    -------
    jnp.ndarray
        applied qubit state
    """
    return opN(c, wires, entangle_op2(_op.RYY(c.dtype, theta)))


def RZZ(c: jnp.ndarray, wires: Tuple[int], theta: float) -> jnp.ndarray:
    """
    Rotate ZZ (exp(-iZZ * thta))

    Parameters
    ----------
    c : jnp.ndarray
        qubit state
    wires : tuple of ints
        wires to apply. ``len(wires)`` must be ``2``

    Returns
    -------
    jnp.ndarray
        applied qubit state
    """
    return opN(c, wires, entangle_op2(_op.RZZ(c.dtype, theta)))
