import unittest

import numpy as np
import jax
import jax.numpy as jnp

import diffqc


class TestProb(unittest.TestCase):
    def test_2qubit(self):
        s = jnp.asarray([0, 0, 1, 0], dtype=jnp.complex64)
        np.testing.assert_allclose(diffqc.prob(s), jnp.asarray([0, 0, 1, 0]))


class TestMarginalProb(unittest.TestCase):
    def test_3to2(self):
        p = jnp.asarray([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        np.testing.assert_allclose(diffqc.marginal_prob(p, 2),
                                   [0.4, 0.2, 0.2, 0.2])

    def test_jit(self):
        p = jnp.asarray([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        np.testing.assert_allclose(jax.jit(diffqc.marginal_prob,
                                           static_argnums=1)(p, 2),
                                   [0.4, 0.2, 0.2, 0.2])


class TestSample(unittest.TestCase):
    def test_sample(self):
        p = jnp.asarray([0, 1])
        key = jax.random.PRNGKey(0)
        np.testing.assert_allclose(diffqc.sample(key, p, (4,)),
                                   [1, 1, 1, 1])


class TestExpVal(unittest.TestCase):
    def test_expval(self):
        p = jnp.asarray([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        np.testing.assert_allclose(diffqc.expval(p, 0), 0.4)

    def test_jit(self):
        p = jnp.asarray([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        np.testing.assert_allclose(jax.jit(diffqc.expval, static_argnums=1)(p, 0), 0.4)


if __name__ == "__main__":
    unittest.main()
