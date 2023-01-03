import unittest

import numpy as np
import jax
import jax.numpy as jnp

import diffq


class TestProb(unittest.TestCase):
    def test_2qubit(self):
        s = jnp.asarray([0, 0, 1, 0], dtype=jnp.complex64)
        np.testing.assert_allclose(diffq.prob(s), jnp.asarray([0, 0, 1, 0]))


class TestMarginalProb(unittest.TestCase):
    def test_3to2(self):
        p = jnp.asarray([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        np.testing.assert_allclose(diffq.marginal_prob(p, 2),
                                   [0.4, 0.2, 0.2, 0.2])


class TestSample(unittest.TestCase):
    def test_sample(self):
        p = jnp.asarray([0, 1])
        key = jax.random.PRNGKey(0)
        np.testing.assert_allclose(diffq.sample(key, p, (4,)),
                                   [1, 1, 1, 1])

if __name__ == "__main__":
    unittest.main()
