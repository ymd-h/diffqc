import unittest

import numpy as np
import jax
import jax.numpy as jnp

from diffq import dense

class TestZero(unittest.TestCase):
    def test_zero(self):
        z = dense.zero(2, jnp.complex64)
        np.testing.assert_allclose(z.shape, (2, 2))
        self.assertEqual(z[0, 0], 1+0j)
        self.assertEqual(jnp.sum(z), 1+0j)

class TestToState(unittest.TestCase):
    def test_to_state(self):
        z = dense.zero(3, jnp.complex64)
        s = dense.to_state(z)
        np.testing.assert_allclose(s.shape, (8,))
        self.assertEqual(s[0], 1+0j)
        self.assertEqual(jnp.sum(s), 1+0j)

class TestHadamard(unittest.TestCase):
    def test_twice(self):
        w = jnp.arange(1)
        z = dense.zero(2, jnp.complex64)
        zz = dense.Hadamard(dense.Hadamard(z, w), w)
        np.testing.assert_allclose(z, zz)

class TestPauliX(unittest.TestCase):
    def test_pauliX(self):
        w = jnp.arange(1)
        z = dense.zero(2, jnp.complex64)
        x = dense.PauliX(z, w)
        np.testing.assert_allclose(x[:, 0], [0+0j, 1+0j])

if __name__ == "__main__":
    unittest.main()
