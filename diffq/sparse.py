import jax
import jax.numpy as jnp

from . import _operators as _op

__all__ = [
    # Util
    "zeros",
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

# StateVec Shape: [possibility, nqubit, qubit]
def zeros(nqubits, dtype):
    return jnp.zeros((1, nqubits, 2), dtype=dtype).at[:,:,0].set(1)

def to_state(x):
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


# Internal Functions
def op1(c, wires, op):
    i = wires.at[0].get()
    q = c.at[i, :]
    c = q.set(op @ q.get())
    return c

def opN(c, wires, opf):
    @jax.vmap
    def set(xi):
        return c.copy().at[wires, :].set(xi)
    return set(opf(c.at[wires, :].get()))

def control_op(op):
    def COP(q12):
        zero = q12.at[0, 0].get()
        one  = q12.at[0, 1].get()
        ret = jnp.zeros((2, 2, 2), dtype=q12.dtype)
        ret = ret.at[0, 0, 0].set(zero).at[1, 0, 1].set(one)
        ret = ret.at[0, 1, :].set(zero * q12.at[1, :].get())
        ret = ret.at[1, 1, :].set(one * op @ q12.at[1, :].get())
        return ret
    return COP

def entangle_op2(op):
    def EOP(q12):
        q = jnp.asarray([q12.at[0, 0].get() * q12.at[1, 0].get(), # |00>
                         q12.at[0, 0].get() * q12.at[1, 1].get(), # |01>
                         q12.at[0, 1].get() * q12.at[1, 0].get(), # |10>
                         q12.at[0, 1].get() * q12.at[1, 1].get()],# |11>
                        dtype=q12.dtype)
        coef = jnp.expand_dims(op @ q, (1, 2))
        return coef * jnp.asarray([[[1, 0], [1, 0]],
                                   [[1, 0], [0, 1]],
                                   [[0, 1], [1, 0]],
                                   [[0, 1], [0, 1]]],
                                  dtype=q12.dtype)
    return EOP


# Quantum Operators
@jax.vmap
def Hadamard(c, wires):
    return op1(c, wires, _op.H(c.dtype))

@jax.vmap
def PauliX(c, wires):
    return op1(c, wires, _op.sigmaX(c.dtype))

@jax.vmap
def PauliY(c, wires):
    return op1(c, wires, _op.sigmaY(c.dtype))

@jax.vmap
def PauliZ(c, wires):
    return op1(c, wires, _op.sigmaZ(c.dtype))

@jax.vmap
def S(c, wires):
    return op1(c, wires, _op.phaseS(c.dtype))

@jax.vmap
def T(c, wires):
    return op1(c, wires, _op.phaseT(c.dtype))

@jax.vmap
def SX(c, wires):
    return op1(c, wires, _op.sqrtX(c.dtype))

def CNOT(c, wires):
    return opN(c, wires, control_op(_op.sigmaX(c.dtype)))

def CZ(c, wires):
    return opN(c, wires, control_op(_op.sigmaZ(c.dtype)))

def CY(c, wires):
    return opN(c, wires, control_op(_op.sigmaY(c.dtype)))

def SWAP(c, wires):
    return opN(c, wires, lambda q12: jnp.expand_dims(q12.at[(1, 0), :].get(), 0))

def ISWAP(c, wires):
    return opN(c, wires, entangle_op2(_op.ISWAP(c.dtype)))

def ECR(c, wires):
    return opN(c, wires, entangle_op2(_op.ECR(c.dtype)))

def SISWAP(c, wires):
    return opN(c, wires, entangle_op2(_op.SISWAP(c.dtype)))

SQISWAP = SISWAP

def CSWAP(c, wires):
    def cswap(q123):
        s0 = q123.at[0, 0].get()
        s1 = q123.at[0, 1].get()
        return (jnp.zeros((2, 3, 2), dtype=c.dtype)
                .at[0, 0, 0].set(s0)
                .at[1, 0, 1].set(s1)
                .at[0, 1:3, :].set(s0 * q123.at[1:3, :].get())
                .at[1, 1:3, :].set(s1 * q123.at[(2,1), :].get()))
    return opN(c, wires, cswap)

def Toffoli(c, wires):
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


@jax.vmap
def Rot(c, wires, phi, theta, omega):
    return op1(c, wires, _op.Rot(c.dtype, phi, theta, omega))

@jax.vmap
def RX(c, wires, phi):
    return op1(c, wires, _op.RX(c.dtype, phi))

@jax.vmap
def RY(c, wires, phi):
    return op1(c, wires, _op.RY(c.dtype, phi))

@jax.vmap
def RZ(c, wires, phi):
    return op1(c, wires, _op.RZ(c.dtype, phi))

@jax.vmap
def PhaseShift(c, wires, phi):
    return op1(c, wires, _op.phaseShift(c.dtype, phi))

def ControlledPhaseShift(c, wires, phi):
    return opN(c, wires, control_op(_op.phaseShift(c.dtype, phi)))

CPhase = ControlledPhaseShift

def CPhaseShift00(c, wires, phi):
    return opN(c, wires, entangle_op2(_op.phaseShift00(c.dtype, phi)))

def CPhaseShift01(c, wires, phi):
    return opN(c, wires, entangle_op2(_op.phaseShift01(c.dtype, phi)))

def CPhaseShift10(c, wires, phi):
    return opN(c, wires, entangle_op2(_op.phaseShift10(c.dtype, phi)))

def CRX(c, wires, phi):
    return opN(c, wires, control_op(_op.RX(c.dtype, phi)))

def CRY(c, wires, phi):
    return opN(c, wires, control_op(_op.RY(c.dtype, phi)))

def CRZ(c, wires, phi):
    return opN(c, wires, control_op(_op.RZ(c.dtype, phi)))

def CRot(c, wires, phi, theta, omega):
    return opN(c, wires, control_op(_op.Rot(c.dtype, phi, theta, omega)))

U1 = PhaseShift

@jax.vmap
def U2(c, wires, phi, delta):
    return op1(c, wires, _op.U2(c.dtype, phi, delta))

@jax.vmap
def U3(c, wires, theta, phi, delta):
    return op1(c, wires, _op.U3(c.dtype, theta, phi, delta))

def PSWAP(c, wires, phi):
    return opN(c, wires, entangle_op2(_op.PSWAP(c.dtype, phi)))
