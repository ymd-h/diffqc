import jax
import jax.numpy as jnp

# Internal Functions

# Non-Parametric
def H(dtype):
    return jnp.asarray([[1,  1],
                        [1, -1]], dtype=dtype) * jnp.sqrt(0.5)

def sigmaX(dtype):
    return jnp.asarray([[0,  1],
                        [1,  0]], dtype=dtype)

def sigmaY(dtype):
    return jnp.asarray([[ 0, -1j],
                        [1j,   0]], dtype=dtype)

def sigmaZ(dtype):
    return jnp.asarray([[1,  0],
                        [0, -1]], dtype=dtype)

def phaseS(dtype):
    return jnp.asarray([[1,  0],
                        [0, 1j]], dtype=dtype)

def phaseT(dtype):
    return jnp.asarray([[1, 0],
                        [0, jnp.exp(0.25j * jnp.pi)]], dtype=dtype)

def sqrtX(dtype):
    return jnp.asarray([[1+1j, 1-1j],
                        [1-1j, 1+1j]], dtype=dtype) * 0.5

def SWAP(dtype):
    return jnp.asarray([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]], dtype=dtype)

def ISWAP(dtype):
    return jnp.asarray([[1,  0,  0, 0],
                        [0,  0, 1j, 0],
                        [0, 1j,  0, 0],
                        [0,  0,  0, 1]],
                       dtype=dtype)

def ECR(dtype):
    return jnp.asarray([[  0,   0,  1, 1j],
                        [  0,   0, 1j,  1],
                        [  1, -1j,  0,  0],
                        [-1j,   1,  0,  0]],
                       dtype=dtype) * jnp.sqrt(0.5)

def SISWAP(dtype):
    inv_sqrt2 = jnp.sqrt(0.5)
    return jnp.asarray([[1,              0,              0, 0],
                        [0,      inv_sqrt2, 1j * inv_sqrt2, 0],
                        [0, 1j * inv_sqrt2,      inv_sqrt2, 0],
                        [0,              0,              0, 1]],
                       dtype=dtype)

# Parametric
def phaseShift(dtype, phi):
    return jnp.asarray([[1,               0],
                        [0, jnp.exp(1j*phi)]], dtype=dtype)

def phaseShift00(dtype, phi):
    return jnp.identity(4, dtype=dtype).at[0, 0].set(jnp.exp(1j*phi))

def phaseShift01(dtype, phi):
    return jnp.identity(4, dtype=dtype).at[1, 1].set(jnp.exp(1j*phi))

def phaseShift10(dtype, phi):
    return jnp.identity(4, dtype=dtype).at[2, 2].set(jnp.exp(1j*phi))

def RX(dtype, phi):
    cos = jnp.cos(0.5*phi)
    sin = jnp.sin(0.5*phi)
    return jnp.asarray([[      cos, -1j * sin],
                        [-1j * sin,       cos]],
                       dtype=dtype)

def RY(dtype, phi):
    cos = jnp.cos(0.5*phi)
    sin = jnp.sin(0.5*phi)
    return jnp.asarray([[cos, -sin],
                        [sin,  cos]],
                       dtype=dtype)

def RZ(dtype, phi):
    exp = jnp.exp(0.5j*phi)
    return jnp.asarray([[1/exp,   0],
                        [    0, exp]],
                       dtype=dtype)

def Rot(dtype, phi, theta, omega):
    a = jnp.exp(0.5j * (phi+omega))
    s = jnp.exp(0.5j * (phi-omega))
    cos = jnp.cos(0.5*theta)
    sin = jnp.sin(0.5*theta)
    return jnp.asarray([[(1/a) * cos, -s * sin],
                        [(1/s) * sin,  a * cos]],
                       dtype=dtype)

def U2(dtype, phi, delta):
    p = jnp.exp(1j*phi)
    d = jnp.exp(1j*delta)
    return jnp.asarray([[1,  -d],
                        [p, p*d]],
                       dtype=dtype) * jnp.sqrt(0.5)

def U3(dtype, theta, phi, delta):
    cos = jnp.cos(0.5*theta)
    sin = jnp.sin(0.5*theta)
    p = jnp.exp(1j*phi)
    d = jnp.exp(1j*delta)
    return jnp.asarray([[cos,      -d*sin],
                        [p*sin, (p*d)*cos]],
                       dtype=dtype)

def PSWAP(dtype, phi):
    exp = jnp.exp(1j*phi)
    return (jnp.zeros((4, 4), dtype=dtype)
            .at[0, 0].set(1)
            .at[1, 2].set(exp)
            .at[2, 1].set(exp)
            .at[3, 3].set(1))


