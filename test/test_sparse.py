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

class TestS(unittest.TestCase):
    def test_S(self):
        w = (0,)
        s0 = sparse.zeros(1, jnp.complex64)
        s = sparse.S(s0, w)
        np.testing.assert_allclose(s, [[[1, 0]]])

    def test_1(self):
        w = (0,)
        s1 = sparse.PauliX(sparse.zeros(1, jnp.complex64), w)
        s = sparse.S(s1, w)
        np.testing.assert_allclose(s, [[[0, 1j]]])

class TestT(unittest.TestCase):
    def test_T(self):
        w = (0,)
        s0 = sparse.zeros(1, jnp.complex64)
        t = sparse.T(s0, w)
        np.testing.assert_allclose(t, [[[1, 0]]])

    def test_1(self):
        w = (0,)
        s1 = sparse.PauliX(sparse.zeros(1, jnp.complex64), w)
        t = sparse.T(s1, w)
        np.testing.assert_allclose(t, [[[0, jnp.exp(0.25j * jnp.pi)]]])

class TestSX(unittest.TestCase):
    def test_SX(self):
        w = (0,)
        s0 = sparse.zeros(1, jnp.complex64)
        sx = sparse.SX(s0, w)
        np.testing.assert_allclose(sx, [[[0.5*(1+1j), 0.5*(1-1j)]]])

    def test_1(self):
        w = (0,)
        s1 = sparse.PauliX(sparse.zeros(1, jnp.complex64), w)
        sx = sparse.SX(s1, w)
        np.testing.assert_allclose(sx, [[[0.5*(1-1j), 0.5*(1+1j)]]])

class TestCNOT(unittest.TestCase):
    def test_0(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        cn = sparse.CNOT(s00, w)
        np.testing.assert_allclose(cn, [[[1,0],[1,0]],[[0,0],[0,1]]])
        np.testing.assert_allclose(sparse.to_state(cn), sparse.to_state(s00))

    def test_1(self):
        w = (0, 1)
        s10 = sparse.PauliX(sparse.zeros(2, jnp.complex64), (0,))
        cn = sparse.CNOT(s10, w)
        np.testing.assert_allclose(cn, [[[0,0],[1,0]],[[0,1],[0,1]]])
        np.testing.assert_allclose(sparse.to_state(cn),
                                   sparse.to_state(sparse.PauliX(s10, (1,))))


class TestCZ(unittest.TestCase):
    def test_0(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        cz = sparse.CZ(s00, w)
        np.testing.assert_allclose(sparse.to_state(cz), sparse.to_state(s00))

    def test_1(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        s10 = sparse.PauliZ(s00, (0,))
        s11 = sparse.PauliZ(s10, (1,))
        cz = sparse.CZ(s10, w)
        np.testing.assert_allclose(sparse.to_state(cz), sparse.to_state(s11))

class TestCY(unittest.TestCase):
    def test_0(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        cy = sparse.CY(s00, w)
        np.testing.assert_allclose(sparse.to_state(cy), sparse.to_state(s00))

    def test_1(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        s10 = sparse.PauliX(s00, (0,))
        s1i = sparse.PauliY(s10, (1,))
        cy = sparse.CY(s10, w)
        np.testing.assert_allclose(sparse.to_state(cy), sparse.to_state(s1i))


class TestSWAP(unittest.TestCase):
    def test_0(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        s = sparse.SWAP(s00, w)
        np.testing.assert_allclose(s, s00)

    def test_1(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        s10 = sparse.PauliX(s00, (0,))
        s01 = sparse.PauliX(s00, (1,))
        s = sparse.SWAP(s10, w)
        np.testing.assert_allclose(sparse.to_state(s), sparse.to_state(s01))

    def test_twice(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        s10 = sparse.PauliX(s00, (0,))
        ss = sparse.SWAP(sparse.SWAP(s10, w), w)
        np.testing.assert_allclose(ss, s10)

class TestISWAP(unittest.TestCase):
    def test_00(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        s = sparse.ISWAP(s00, w)
        np.testing.assert_allclose(sparse.to_state(s), sparse.to_state(s00))

    def test_01(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        s01 = sparse.PauliX(s00, (1,))
        s10 = sparse.PauliX(s00, (0,))
        s = sparse.ISWAP(s01, w)
        np.testing.assert_allclose(sparse.to_state(s), sparse.to_state(1j * s10))

    def test_10(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        s01 = sparse.PauliX(s00, (1,))
        s10 = sparse.PauliX(s00, (0,))
        s = sparse.ISWAP(s10, w)
        np.testing.assert_allclose(sparse.to_state(s), sparse.to_state(1j * s01))

    def test_11(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        s11 = sparse.PauliX(sparse.PauliX(s00, (1,)), (0,))
        s = sparse.ISWAP(s11, w)
        np.testing.assert_allclose(sparse.to_state(s), sparse.to_state(s11))

class TestECR(unittest.TestCase):
    def test_ECR(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        s01 = sparse.PauliX(s00, (1,))
        s10 = sparse.PauliX(s00, (0,))
        s11 = sparse.PauliX(s01, (0,))

        s = jnp.stack((s00, s01, s10, s11))
        ans = jnp.asarray([
            [ 0, 0,  1,-1j],
            [ 0, 0,-1j,  1],
            [ 1,1j,  0,  0],
            [1j, 1,  0,  0],
        ]) / jnp.sqrt(2)

        @jax.vmap
        def ecr(si):
            return sparse.to_state(sparse.ECR(si, w))

        np.testing.assert_allclose(ecr(s), ans)

class TestSISWAP(unittest.TestCase):
    def test_SISWAP(self):
        w = (0, 1)
        s00 = sparse.zeros(2, jnp.complex64)
        s10 = sparse.PauliX(s00, (0,))
        s01 = sparse.PauliX(s00, (1,))
        s11 = sparse.PauliX(s01, (0,))

        s = jnp.stack((s00, s01, s10, s11))
        ans = jnp.asarray([
            [1,              0,              0, 0],
            [0,  1/jnp.sqrt(2), 1j/jnp.sqrt(2), 0],
            [0, 1j/jnp.sqrt(2),  1/jnp.sqrt(2), 0],
            [0,              0,              0, 1],
        ])

        @jax.vmap
        def siswap(si):
            return sparse.to_state(sparse.SISWAP(si, w))

        np.testing.assert_allclose(siswap(s), ans)

if __name__ == "__main__":
    unittest.main()
