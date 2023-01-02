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
    def test_H(self):
        w = jnp.arange(1)
        z = dense.zero(1, jnp.complex64)
        h = dense.Hadamard(z, w)
        np.testing.assert_allclose(h, [1/jnp.sqrt(2), 1/jnp.sqrt(2)])

    def test_1(self):
        w = jnp.arange(1)
        o = dense.zero(1, jnp.complex64).at[:].set([0, 1])
        h = dense.Hadamard(o, w)
        np.testing.assert_allclose(h, [1/jnp.sqrt(2), -1/jnp.sqrt(2)])

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

    def test_twice(self):
        w = jnp.arange(1)
        z = dense.zero(1, jnp.complex64)
        xx = dense.PauliX(dense.PauliX(z, w), w)
        np.testing.assert_allclose(z, xx)

class TestPauliY(unittest.TestCase):
    def test_pauliY(self):
        w = jnp.arange(1)
        z = dense.zero(1, jnp.complex64)
        y = dense.PauliY(z, w)
        np.testing.assert_allclose(y, [0+0j, 0+1j])

    def test_1(self):
        w = jnp.arange(1)
        o = dense.zero(1, jnp.complex64).at[:].set([0, 1])
        y = dense.PauliY(o, w)
        np.testing.assert_allclose(y, [-1j, 0])

    def test_twice(self):
        w = jnp.arange(1)
        z = dense.zero(1, jnp.complex64)
        yy = dense.PauliY(dense.PauliY(z, w), w)
        np.testing.assert_allclose(yy, z)

class TestPauliZ(unittest.TestCase):
    def test_pauliZ(self):
        w = jnp.arange(1)
        z = dense.zero(1, jnp.complex64)
        x = dense.PauliZ(z, w)
        np.testing.assert_allclose(x, [1, 0])

    def test_1(self):
        w = jnp.arange(1)
        o = dense.zero(1, jnp.complex64).at[:].set([0, 1])
        x = dense.PauliZ(o, w)
        np.testing.assert_allclose(x, [0, -1])

    def test_twice(self):
        w = jnp.arange(1)
        z = dense.zero(1, jnp.complex64)
        zz = dense.PauliZ(dense.PauliZ(z, w), w)
        np.testing.assert_allclose(zz, z)

class TestS(unittest.TestCase):
    def test_S(self):
        w = jnp.arange(1)
        z = dense.zero(1, jnp.complex64)
        s = dense.S(z, w)
        np.testing.assert_allclose(s, [1, 0])

    def test_1(self):
        w = jnp.arange(1)
        o = dense.zero(1, jnp.complex64).at[:].set([0, 1])
        s = dense.S(o, w)
        np.testing.assert_allclose(s, [0, 1j])

class TestT(unittest.TestCase):
    def test_T(self):
        w = jnp.arange(1)
        z = dense.zero(1, jnp.complex64)
        t = dense.T(z, w)
        np.testing.assert_allclose(t, [1, 0])

    def test_1(self):
        w = jnp.arange(1)
        o = dense.zero(1, jnp.complex64).at[:].set([0, 1])
        t = dense.T(o, w)
        np.testing.assert_allclose(t, [0, jnp.exp(0.25j * jnp.pi)])

class TestSX(unittest.TestCase):
    def test_SX(self):
        w = jnp.arange(1)
        z = dense.zero(1, jnp.complex64)
        sx = dense.SX(z, w)
        np.testing.assert_allclose(sx, [0.5*(1+1j), 0.5*(1-1j)])

    def test_1(self):
        w = jnp.arange(1)
        o = dense.zero(1, jnp.complex64).at[:].set([0, 1])
        sx = dense.SX(o, w)
        np.testing.assert_allclose(sx, [0.5*(1-1j), 0.5*(1+1j)])

if __name__ == "__main__":
    unittest.main()
