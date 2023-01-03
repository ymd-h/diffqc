import jax
import jax.numpy as jnp

__all__ = [
    "prob",
    "marginal_prob",
    "sample",
]


def prob(state):
    return jnp.square(jnp.abs(state))


def marginal_prob(probs, integrage_wires):
    p = jnp.reshape(probs, (2,) * int(jnp.log2(probs.shape[0])))
    return jnp.sum(p, axis=integrage_wires)


def sample(key, probs, shape):
    return jax.random.choice(key,
                             jnp.arange(probs.shape[0]),
                             shape=shape,
                             p=probs)
