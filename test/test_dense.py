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

class TestCNOT(unittest.TestCase):
    def test_CNOT(self):
        w = jnp.arange(2)
        z = dense.zero(2, jnp.complex64)
        cn = dense.CNOT(z, w)
        np.testing.assert_allclose(cn, z)

    def test_1(self):
        w = jnp.arange(2)
        o = dense.PauliX(dense.zero(2, jnp.complex64), jnp.arange(1))
        cn = dense.CNOT(o, w)
        np.testing.assert_allclose(cn, dense.PauliX(o, jnp.ones((1,), dtype=jnp.int32)))

class TestCZ(unittest.TestCase):
    def test_CZ(self):
        w = jnp.arange(2)
        z = dense.zero(2, jnp.complex64)
        cz = dense.CZ(z, w)
        np.testing.assert_allclose(cz, z)

    def test_1(self):
        w = jnp.arange(2)
        o = dense.PauliX(dense.zero(2, jnp.complex64), jnp.arange(1))
        cz = dense.CZ(o, w)
        np.testing.assert_allclose(cz, dense.PauliZ(o, jnp.ones((1,), dtype=jnp.int32)))

class TestCY(unittest.TestCase):
    def test_CY(self):
        w = jnp.arange(2)
        z = dense.zero(2, jnp.complex64)
        cy = dense.CY(z, w)
        np.testing.assert_allclose(cy, z)

    def test_1(self):
        w = jnp.arange(2)
        o = dense.PauliX(dense.zero(2, jnp.complex64), jnp.arange(1))
        cy = dense.CY(o, w)
        np.testing.assert_allclose(cy, dense.PauliY(o, jnp.ones((1,), dtype=jnp.int32)))

class TestSWAP(unittest.TestCase):
    def test_SWAP(self):
        w = jnp.arange(2)
        z = dense.zero(2, jnp.complex64)
        s = dense.SWAP(z, w)
        np.testing.assert_allclose(s, z)

    def test_1(self):
        w = jnp.arange(2)
        z = dense.zero(2, jnp.complex64)
        o = dense.PauliX(z, jnp.arange(1))
        s = dense.SWAP(o, w)
        np.testing.assert_allclose(s, dense.PauliX(z, jnp.ones((1,), dtype=jnp.int32)))

    def test_twice(self):
        w = jnp.arange(2)
        o = dense.PauliX(dense.zero(2, jnp.complex64), jnp.arange(1))
        ss = dense.SWAP(dense.SWAP(o, w), w)
        np.testing.assert_allclose(ss, o)

class TestISWAP(unittest.TestCase):
    def test_00(self):
        w = jnp.arange(2)
        s00 = dense.zero(2, jnp.complex64)
        s = dense.ISWAP(s00, w)
        np.testing.assert_allclose(s, s00)

    def test_01(self):
        w = jnp.arange(2)
        s00 = dense.zero(2, jnp.complex64)
        s01 = dense.PauliX(s00, jnp.ones((1,), dtype=jnp.int32))
        s10 = dense.PauliX(s00, jnp.arange(1))
        s = dense.ISWAP(s01, w)
        np.testing.assert_allclose(s, 1j * s10)

    def test_10(self):
        w = jnp.arange(2)
        s00 = dense.zero(2, jnp.complex64)
        s01 = dense.PauliX(s00, jnp.ones((1,), dtype=jnp.int32))
        s10 = dense.PauliX(s00, jnp.arange(1))
        s = dense.ISWAP(s10, w)
        np.testing.assert_allclose(s, 1j * s01)

    def test_11(self):
        w = jnp.arange(2)
        s00 = dense.zero(2, jnp.complex64)
        s11 = dense.PauliX(dense.PauliX(s00, jnp.arange(1)),
                           jnp.ones((1,), dtype=jnp.int32))
        s = dense.ISWAP(s11, w)
        np.testing.assert_allclose(s, s11)

class TestECR(unittest.TestCase):
    def test_ECR(self):
        w = jnp.arange(2)
        s00 = dense.zero(2, jnp.complex64)
        s01 = dense.PauliX(s00, jnp.ones((1,), dtype=jnp.int32))
        s10 = dense.PauliX(s00, jnp.arange(1))
        s11 = dense.PauliX(s01, jnp.arange(1))

        for s, ans in [[s00, [0, 0, 1, -1j]],
                       [s01, [0, 0, -1j, 1]],
                       [s10, [1, 1j, 0, 0]],
                       [s11, [1j, 1, 0, 0]]]:
            with self.subTest(state=s):
                np.testing.assert_allclose(dense.to_state(dense.ECR(s, w)),
                                           jnp.asarray(ans)/jnp.sqrt(2))

class TestSISWAP(unittest.TestCase):
    def test_SISWAP(self):
        w = jnp.arange(2)
        s00 = dense.zero(2, jnp.complex64)
        s01 = dense.PauliX(s00, jnp.ones((1,), dtype=jnp.int32))
        s10 = dense.PauliX(s00, jnp.arange(1))
        s11 = dense.PauliX(s01, jnp.arange(1))

        for name, s, ans in [["s00", s00, [1, 0, 0, 0]],
                             ["s01", s01, [0,  1/jnp.sqrt(2), 1j/jnp.sqrt(2), 0]],
                             ["s10", s10, [0, 1j/jnp.sqrt(2),  1/jnp.sqrt(2), 0]],
                             ["s11", s11, [0, 0, 0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.SISWAP(s, w)), ans)

class TestCSWAP(unittest.TestCase):
    def test_CSWAP(self):
        w = jnp.arange(3)
        s001 = dense.PauliX(dense.zero(3, jnp.complex64), jnp.arange(1)+2)
        np.testing.assert_allclose(dense.CSWAP(s001, w), s001)

        s101 = dense.PauliX(s001, jnp.arange(1))
        np.testing.assert_allclose(dense.CSWAP(s101, w),
                                   dense.SWAP(s101, jnp.arange(2)+1))

class TestToffoli(unittest.TestCase):
    def test_Toffoli(self):
        w = jnp.arange(3)
        s001 = dense.PauliX(dense.zero(3, jnp.complex64), jnp.arange(1)+2)
        np.testing.assert_allclose(dense.Toffoli(s001, w), s001)

        s011 = dense.PauliX(s001, jnp.arange(1)+1)
        np.testing.assert_allclose(dense.Toffoli(s011, w), s011)

        s110 = dense.SWAP(s011, jnp.asarray([0,2], dtype=jnp.int32))
        np.testing.assert_allclose(dense.Toffoli(s110, w),
                                   dense.PauliX(s110, jnp.arange(1)+2))

class TestRot(unittest.TestCase):
    def test_000(self):
        w = jnp.arange(1)
        s0 = dense.zero(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        for name, s, ans in [["s0", s0, [1, 0]],
                             ["s1", s1, [0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.Rot(s, w, 0, 0, 0)),
                                           ans)

    def test_0y0(self):
        w = jnp.arange(1)
        s0 = dense.zero(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        y = 0.25 * jnp.pi
        for name, s, ans in [["s0", s0, [ jnp.cos(0.5*y), jnp.sin(0.5*y)]],
                             ["s1", s1, [-jnp.sin(0.5*y), jnp.cos(0.5*y)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.Rot(s, w, 0, y, 0)),
                                           ans)

    def test_z00(self):
        w = jnp.arange(1)
        s0 = dense.zero(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        z = jnp.pi / 8
        for name, s, ans in [["s0", s0, [jnp.exp(-0.5j*z), 0]],
                             ["s1", s1, [0, jnp.exp(0.5j*z)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.Rot(s, w, z, 0, 0)),
                                           ans, atol=1e-7)

    def test_00z(self):
        w = jnp.arange(1)
        s0 = dense.zero(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        z = jnp.pi * 0.75
        for name, s, ans in [["s0", s0, [jnp.exp(-0.5j*z), 0]],
                             ["s1", s1, [0, jnp.exp(0.5j*z)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.Rot(s, w, 0, 0, z)),
                                           ans, atol=1e-7)

    def test_zyz(self):
        w = jnp.arange(1)
        s0 = dense.zero(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        z1 = jnp.pi * 1.5
        y = jnp.pi
        z2 = jnp.pi / 8

        for name, s, ans in [["s0", s0, [0, jnp.exp(-0.5j*(z1-z2))]],
                             ["s1", s1, [-jnp.exp(0.5j*(z1-z2)), 0]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.Rot(s, w, z1, y, z2)),
                                           ans, atol=1e-7)

class TestRX(unittest.TestCase):
    def test_0(self):
        w = jnp.arange(1)
        s0 = dense.zero(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        for name, s, ans in [["s0", s0, [1, 0]],
                             ["s1", s1, [0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RX(s, w, 0)), ans)

    def test_x(self):
        w = jnp.arange(1)
        s0 = dense.zero(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        x = jnp.pi
        for name, s, ans in [["s0", s0, [ 0, -1j]],
                             ["s1", s1, [-1j,   0]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RX(s, w, x)), ans,
                                           atol=1e-7)

class TestRY(unittest.TestCase):
    def test_0(self):
        w = jnp.arange(1)
        s0 = dense.zero(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        for name, s, ans in [["s0", s0, [1, 0]],
                             ["s1", s1, [0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RY(s, w, 0)), ans,
                                           atol=1e-7)

    def test_y(self):
        w = jnp.arange(1)
        s0 = dense.zero(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        y = jnp.pi
        for name, s, ans in [["s0", s0, [ 0, 1]],
                             ["s1", s1, [-1, 0]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RY(s, w, y)), ans,
                                           atol=1e-7)

class TestRZ(unittest.TestCase):
    def test_0(self):
        w = jnp.arange(1)
        s0 = dense.zero(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        for name, s, ans in [["s0", s0, [1, 0]],
                             ["s1", s1, [0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RZ(s, w, 0)), ans,
                                           atol=1e-7)

    def test_z(self):
        w = jnp.arange(1)
        s0 = dense.zero(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        z = jnp.pi
        for name, s, ans in [["s0", s0, [jnp.exp(-0.5j*z), 0]],
                             ["s1", s1, [0,  jnp.exp(0.5j*z)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RZ(s, w, z)), ans,
                                           atol=1e-7)

if __name__ == "__main__":
    unittest.main()
