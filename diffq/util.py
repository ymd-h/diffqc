import jax
import jax.numpy as jnp

__all__ = [
    "prob",
    "marginal_prob",
    "sample",
    "expval"
]


# jax.jit compatible `jnp.reshape(s, (2,) * int(jnp.log2(s.shape[0])))`
def _qubit_shape(s):
    if s.shape[0] <= 2:
        return s

    s = jnp.reshape(s, (2, -1))
    return jax.vmap(_qubit_shape)(s)


def prob(state, dtype=jnp.float32):
    return jnp.asarray(jnp.square(jnp.abs(state)), dtype=dtype)


def marginal_prob(probs, integrate_wires):
    p = _qubit_shape(probs)
    return jnp.reshape(jnp.sum(p, axis=integrate_wires), (-1,))


def sample(key, probs, shape):
    return jax.random.choice(key,
                             jnp.arange(probs.shape[0]),
                             shape=shape,
                             p=probs)


def expval(probs, wire):
    p = _qubit_shape(probs)
    return jnp.sum(jnp.take(p, jnp.ones((1,), dtype=jnp.int32), axis=wire))
