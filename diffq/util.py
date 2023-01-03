import jax
import jax.numpy as jnp

__all__ = [
    "prob",
    "marginal_prob",
    "sample",
]


def prob(state, dtype=jnp.float32):
    return jnp.asarray(jnp.square(jnp.abs(state)), dtype=dtype)


def marginal_prob(probs, integrage_wires):
    p = jnp.reshape(probs, (2,) * int(jnp.log2(probs.shape[0])))
    return jnp.reshape(jnp.sum(p, axis=integrage_wires), (-1,))


def sample(key, probs, shape):
    return jax.random.choice(key,
                             jnp.arange(probs.shape[0]),
                             shape=shape,
                             p=probs)
