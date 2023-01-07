import unittest

import numpy as np
import jax
import jax.numpy as jnp

from diffq import sparse

class TestZeros(unittest.TestCase):
    def test_zeros(self):
        z = sparse.zeros(2, jnp.complex64)
        np.testing.assert_allclose(z.shape, (1, 2, 2))
        np.testing.assert_allclose(z[0, 0], [1, 0])
        np.testing.assert_allclose(z[0, 1], [1, 0])

class TestToState(unittest.TestCase):
    def test_to_state(self):
        z = sparse.zeros(3, jnp.complex64)
        s = sparse.to_state(z)
        np.testing.assert_allclose(s.shape, (8,))
        self.assertEqual(s[0], 1+0j)
        self.assertEqual(jnp.sum(s), 1+0j)

    def test_entangled(self):
        s000 = sparse.zeros(3, jnp.complex64)
        s111 = s000.copy().at[:,:].set([0, 1])
        s = sparse.to_state(jnp.concatenate((s000.at[:,0].divide(jnp.sqrt(2)),
                                             s111.at[:,0].divide(jnp.sqrt(2))),
                                            axis=0))
        np.testing.assert_allclose(s, ([1/jnp.sqrt(2), 0, 0, 0,
                                        0, 0, 0, 1/jnp.sqrt(2)]))

class TestHadamard(unittest.TestCase):
    def test_0(self):
        w = (0,)
        z = sparse.zeros(1, jnp.complex64)
        h = sparse.Hadamard(z, w)
        np.testing.assert_allclose(h[0, 0], [1/jnp.sqrt(2), 1/jnp.sqrt(2)])

    def test_1(self):
        w = (0,)
        z = sparse.zeros(1, jnp.complex64).at[:,:].set([0, 1])
        h = sparse.Hadamard(z, w)
        np.testing.assert_allclose(h[0, 0], [1/jnp.sqrt(2), -1/jnp.sqrt(2)])

    def test_twice(self):
        w = (0,)
        z = sparse.zeros(1, jnp.complex64)
        zz = sparse.Hadamard(sparse.Hadamard(z, w), w)
        np.testing.assert_allclose(zz, z)

class TestPauliX(unittest.TestCase):
    def test_pauliX(self):
        w = (0,)
        s0 = sparse.zeros(2, jnp.complex64)
        x = sparse.PauliX(s0, w)
        np.testing.assert_allclose(sparse.to_state(x), [0,0,1,0])

    def test_twice(self):
        w = (0,)
        s0 = sparse.zeros(2, jnp.complex64)
        xx = sparse.PauliX(sparse.PauliX(s0, w), w)
        np.testing.assert_allclose(xx, s0)

class TestPauliY(unittest.TestCase):
    def test_pauliY(self):
        w = (0,)
        s0 = sparse.zeros(1, jnp.complex64)
        y = sparse.PauliY(s0, w)
        np.testing.assert_allclose(sparse.to_state(y), [0+0j, 0+1j])

    def test_1(self):
        w = (0,)
        s0 = sparse.zeros(1, jnp.complex64)
        s1 = sparse.PauliX(s0, w)
        y = sparse.PauliY(s1, w)
        np.testing.assert_allclose(sparse.to_state(y), [-1j, 0])

    def test_twice(self):
        w = (0,)
        s0 = sparse.zeros(1, jnp.complex64)
        yy = sparse.PauliY(sparse.PauliY(s0, w), w)
        np.testing.assert_allclose(sparse.to_state(yy), sparse.to_state(s0))


if __name__ == "__main__":
    unittest.main()
