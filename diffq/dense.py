import jax
import jax.numpy as jnp

from . import _operators as _op

__all__ = [
    # Util
    "zero",
    "to_state",

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
]

# StateVec Shape: [qubits...]
def zero(nqubits, dtype):
    size = 2 ** nqubits
    shape = (2,) * nqubits
    return jnp.reshape(jnp.zeros(size, dtype=dtype).at[0].set(1), shape)

def to_state(x):
    return jnp.reshape(x, (-1,))


# Internal Functions
BUG = "BUG: {} quantum operation is called with wrong wires: {}"

def op1(c, wires, op):
    assert wires.shape == (1,), BUG.format(1, wires.shape)
    i = wires.at[0].get()
    return jnp.moveaxis(jnp.tensordot(op, c, ((1,), (i,))), 0, i)

def op2(c, wires, op):
    assert wires.shape == (2,), BUG.format(2, wires.shape)
    i = (wires.at[0].get(), wires.at[1].get())
    op2x2 = jnp.reshape(op, (2,2,2,2))
    return jnp.moveaxis(jnp.tensordot(op2x2, c, axes=((2,3), i)), (0,1), i)

def op3(c, wires, op):
    assert wires.shape == (3,), BUG.format(3, wires.shape)
    i = (wires.at[0].get(), wires.at[1].get(), wires.at[2].get())
    op2x3 = jnp.reshape(op, (2,2,2,2,2))
    return jnp.moveaxis(jnp.tensordot(op2x3, c, axes=((2,3,4), i)), (0,1,2), i)

def control_op2(op):
    return jnp.identity(4, dtype=op.dtype).at[2:,2:].set(op)

def control_op3(op2):
    return jnp.identity(8, dtype=op.dtype).at[4:,4:].set(op2)

# Quantum Operators
def Hadamard(c, wires):
    return op1(c, wires, _op.H(c.dtype))

def PauliX(c, wires):
    return op1(c, wires, _op.sigmaX(c.dtype))

def PauliY(c, wires):
    return op1(c, wires, _op.sigmaY(c.dtype))

def PauliZ(c, wires):
    return op1(c, wires, _op.sigmaZ(c.dtype))

def S(c, wires):
    return op1(c, wires, _op.phaseS(c.dtype))

def T(c, wires):
    return op1(c, wires, _op.phaseT(c.dtype))

def SX(c, wires):
    return op1(c, wires, _op.sqrtX(c.dtype))

def CNOT(c, wires):
    return op2(c, wires, control_op2(_op.sigmaX(c.dtype)))

def CZ(c, wires):
    return op2(c, wires, control_op2(_op.sigmaZ(c.dtype)))

def CY(c, wires):
    return op2(c, wires, control_op2(_op.sigmaY(c.dtype)))

def SWAP(c, wires):
    return op2(c, wires, _op.SWAP(c.dtype))

def ISWAP(c, wires):
    return op2(c, wires, _op.ISWAP(c.dtype))

def ECR(c, wires):
    return op2(c, wires, _op.ECR(c.dtype))

def SISWAP(c, wires):
    return op2(c, wires, _op.SISWAP(c.dtype))

SQISWAP = SISWAP

def CSWAP(c, wires):
    return op3(c, wires, control_op3(_op.SWAP(c.dtype)))

def Toffoli(c, wires):
    return op3(c, wires, control_op3(control_op2(_op.sigmaX(c.dtype))))

def Rot(c, wires, phi, theta, omega):
    return op1(c, wires, _op.Rot(c.dtype, phi, theta, omega))

def RX(c, wires, phi):
    return op1(c, wires, _op.RX(c.dtype, phi))

def RY(c, wires, phi):
    return op1(c, wires, _op.RY(c.dtype, phi))

def RZ(c, wires, phi):
    return op1(c, wires, _op.RZ(c.dtype, phi))

def PhaseShift(c, wires, phi):
    return op1(c, wires, _op.phaseShift(c.dtype, phi))

def ControlledPhaseShift(c, wires, phi):
    return op2(c, wires, control_op2(_op.phaseShift(c.dtype, phi)))

CPhase = ControlledPhaseShift

def CPhaseShift00(c, wires, phi):
    return op2(c, wires, _op.phaseShift00(c.dtype, phi))

def CPhaseShift01(c, wires, phi):
    return op2(c, wires, _op.phaseShift01(c.dtype, phi))

def CPhaseShift10(c, wires, phi):
    return op2(c, wires, _op.phaseShift10(c.dtype, phi))

def CRX(c, wires, phi):
    return op2(c, wires, control_op2(_op.RX(c.dtype, phi)))

def CRY(c, wires, phi):
    return op2(c, wires, control_op2(_op.RY(c.dtype, phi)))

def CRZ(c, wires, phi):
    return op2(c, wires, control_op2(_op.RZ(c.dtype, phi)))

def CRot(c, wires, phi, theta, omega):
    return op2(c, wires, control_op2(_op.Rot(c.dtype, phi, theta, omega)))

U1 = PhaseShift

def U2(c, wires, phi, delta):
    return op1(c, wires, _op.U2(c.dtype, phi, delta))

def U3(c, wires, theta, phi, delta):
    return op1(c, wires, _op.U3(c.dtype, theta, phi, delta))

def PSWAP(c, wires, phi):
    return op2(c, wires, _op.PSWAP(c.dtype, phi))
