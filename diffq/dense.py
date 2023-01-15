import math
from typing import Tuple

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

    # General Matrix Operation
    "QubitUnitary",
    "ControlledQubitUnitary",
]

# StateVec Shape: [qubits...]
def zeros(nqubits: int, dtype: jnp.dtype) -> jnp.ndarray:
    """
    Create |00...0> state

    Parameters
    ----------
    nqubits : int
        Number of qubits
    dtype : jnp.dtype
        dtype.

    Returns
    -------
    jnp.ndarray
        Zero state aka. |00...0>
    """
    size = 2 ** nqubits
    shape = (2,) * nqubits
    return jnp.reshape(jnp.zeros(size, dtype=dtype).at[0].set(1), shape)

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
        State vector, e.g. [|000>, |001>, |010>, ..., |111>]
    """
    return jnp.reshape(x, (-1,))

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
    assert len(wires) == 1, BUG.format(1, len(wires))
    return jnp.moveaxis(jnp.tensordot(op, c, ((1,), wires)), (0,), wires)

def op2(c, wires, op):
    assert len(wires) == 2, BUG.format(2, len(wires))
    op2x2 = jnp.reshape(op, (2,2,2,2))
    return jnp.moveaxis(jnp.tensordot(op2x2, c, axes=((2,3), wires)), (0,1), wires)

def op3(c, wires, op):
    assert len(wires) == 3, BUG.format(3, len(wires))
    op2x3 = jnp.reshape(op, (2,2,2,2,2,2))
    return jnp.moveaxis(jnp.tensordot(op2x3, c, axes=((3,4,5), wires)),(0,1,2), wires)

def control_op2(op):
    return jnp.identity(4, dtype=op.dtype).at[2:,2:].set(op)

def control_op3(op2):
    return jnp.identity(8, dtype=op2.dtype).at[4:,4:].set(op2)

# Quantum Operators
def Hadamard(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply Hadamard Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``.

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
        wire to apply. ``len(wires)`` must be ``1``.

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
        wire to apply. ``len(wires)`` must be ``1``.

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
        wire to apply. ``len(wires)`` must be ``1``.

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
        wire to apply. ``len(wires)`` must be ``1``.

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
        wire to apply. ``len(wires)`` must be ``1``.

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
        wire to apply. ``len(wires)`` must be ``1``.

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
        wire to apply. ``len(wires)`` must be ``2``.

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, control_op2(_op.sigmaX(c.dtype)))

def CZ(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply controlled-Z Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, control_op2(_op.sigmaZ(c.dtype)))

def CY(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply controlled-Y Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, control_op2(_op.sigmaY(c.dtype)))

def SWAP(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply SWAP Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, _op.SWAP(c.dtype))

def ISWAP(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply i-SWAP Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, _op.ISWAP(c.dtype))

def ECR(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply Echoed RZX(pi/2) Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, _op.ECR(c.dtype))

def SISWAP(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply Square Root i-SWAP Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, _op.SISWAP(c.dtype))

SQISWAP = SISWAP

def CSWAP(c: jnp.ndarray, wires: Tuple[int]) -> jnp.ndarray:
    """
    Apply controlled SWAP Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``3``.

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op3(c, wires, control_op3(_op.SWAP(c.dtype)))

def Toffoli(c: jnp.ndarray, wires: Tuple[int]):
    """
    Apply Toffoli Gate (controlled controlled X Gate)

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``3``.

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op3(c, wires, control_op3(control_op2(_op.sigmaX(c.dtype))))

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
        wire to apply. ``len(wires)`` must be ``1``.
    phi, theta, omega : float
        rotation angles

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op1(c, wires, _op.Rot(c.dtype, phi, theta, omega))

def RX(c: jnp.ndarray, wires: Tuple[int], phi: float):
    """
    Apply Rotation X Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``.
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
        wire to apply. ``len(wires)`` must be ``1``.
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
        wire to apply. ``len(wires)`` must be ``1``.
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
        wire to apply. ``len(wires)`` must be ``1``.
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
        wire to apply. ``len(wires)`` must be ``2``.
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, control_op2(_op.phaseShift(c.dtype, phi)))

CPhase = ControlledPhaseShift

def CPhaseShift00(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply Phase Shift Gate for |00>

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, _op.phaseShift00(c.dtype, phi))

def CPhaseShift01(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply Phase Shift Gate for |01>

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, _op.phaseShift01(c.dtype, phi))

def CPhaseShift10(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply Phase Shift Gate for |10>

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, _op.phaseShift10(c.dtype, phi))

def CRX(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply controlled RX Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, control_op2(_op.RX(c.dtype, phi)))

def CRY(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply controlled RY Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, control_op2(_op.RY(c.dtype, phi)))

def CRZ(c: jnp.ndarray, wires: Tuple[int], phi: float) -> jnp.ndarray:
    """
    Apply controlled RZ Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, control_op2(_op.RZ(c.dtype, phi)))

def CRot(c: jnp.ndarray, wires: Tuple[int],
         phi: float, theta: float, omega: float) -> jnp.ndarray:
    """
    Apply controlled Rotation Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``2``.
    phi, theta, omega : float
        rotation angles

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, control_op2(_op.Rot(c.dtype, phi, theta, omega)))

U1 = PhaseShift

def U2(c: jnp.ndarray, wires: Tuple[int], phi: float, delta: float) -> jnp.ndarray:
    """
    Apply controlled U2 Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1``.
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
        wire to apply. ``len(wires)`` must be ``1``.
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
        wire to apply. ``len(wires)`` must be ``2``.
    phi : float
        rotation angle

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    return op2(c, wires, _op.PSWAP(c.dtype, phi))


def QubitUnitary(c: jnp.ndarray, wires: Tuple[int],
                 U: jnp.ndarray) -> jnp.ndarray:
    """
    Unitary Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``log2(U.shape[0])``.
    U : jnp.ndarray
        square unitary matrix

    Returns
    -------
    jnp.ndarray
        applied qubits state
    """
    nqubits = int(math.log2(U.shape[0]))
    assert len(wires) == nqubits, BUG.format(nqubits, wires)

    U = jnp.reshape(U, (2, 2) * len(wires))

    a = tuple(i for i in range(len(wires)))
    b = tuple(i + len(wires) for i in range(len(wires)))
    return jnp.moveaxis(jnp.tensordot(U, c, axes=(b, wires)), a, wires)


def ControlledQubitUnitary(c: jnp.ndarray, wires: Tuple[int],
                           U: jnp.ndarray) -> jnp.ndarray:
    """
    Controlled Unitary Gate

    Parameters
    ----------
    c : jnp.ndarray
        qubits state
    wires : tuple of ints
        wire to apply. ``len(wires)`` must be ``1 + log2(U.shape[0])``.
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
