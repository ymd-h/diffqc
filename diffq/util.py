import jax
import jax.numpy as jnp

__all__ = [
    "prob",
    "marginal_prob",
]


def prob(state):
    return jnp.square(jnp.abs(state))


def marginal_prob(prob, integrage_wires):
    p = jnp.reshape(prob, (2,) * int(jnp.log2(prob.shape[0])))
    return jnp.sum(p, axis=integrage_wires)
