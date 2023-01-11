import unittest

import numpy as np
import jax
import jax.numpy as jnp

from diffq import dense, util

class TestZeros(unittest.TestCase):
    def test_zeros(self):
        z = dense.zeros(2, jnp.complex64)
        np.testing.assert_allclose(z.shape, (2, 2))
        self.assertEqual(z[0, 0], 1+0j)
        self.assertEqual(jnp.sum(z), 1+0j)

class TestToState(unittest.TestCase):
    def test_to_state(self):
        z = dense.zeros(3, jnp.complex64)
        s = dense.to_state(z)
        np.testing.assert_allclose(s.shape, (8,))
        self.assertEqual(s[0], 1+0j)
        self.assertEqual(jnp.sum(s), 1+0j)

class TestExpect(unittest.TestCase):
    def test_X(self):
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, (0,))
        sH = dense.Hadamard(s0, (0,))
        sY = dense.S(sH, (0,))
        s = jnp.stack((s0, s1, sH, sY))

        @jax.vmap
        def expect(si):
            return dense.expectX(si, (0,))

        np.testing.assert_allclose(expect(s), [0, 0, 1, 0])

    def test_Y(self):
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, (0,))
        sH = dense.Hadamard(s0, (0,))
        sY = dense.S(sH, (0,))
        s = jnp.stack((s0, s1, sH, sY))

        @jax.vmap
        def expect(si):
            return dense.expectY(si, (0,))

        np.testing.assert_allclose(expect(s), [0, 0, 0, 1])

    def test_Z(self):
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, (0,))
        sH = dense.Hadamard(s0, (0,))
        sY = dense.S(sH, (0,))
        s = jnp.stack((s0, s1, sH, sY))

        @jax.vmap
        def expect(si):
            return dense.expectZ(si, (0,))

        np.testing.assert_allclose(expect(s), [1, -1, 0, 0])

    def test_U(self):
        s0 = dense.zeros(2, jnp.complex64)
        s1 = dense.PauliX(s0, (0,))
        sH = dense.Hadamard(s0, (0,))
        sY = dense.S(sH, (0,))
        s = jnp.stack((s0, s1, sH, sY))

        U = jnp.asarray([
            [[1,0,0, 0],
             [0,1,0, 0],
             [0,0,1, 0],
             [0,0,0, 1]],
            [[0,1,0, 0],
             [1,0,0, 0],
             [0,0,1, 0],
             [0,0,0,-1]],
        ])

        # |00> -> |01>
        # |10> -> |10>
        # |+0> -> (|01> + |10>)/sqrt(2)
        # |i0> -> (|01> +i|10>)/sqrt(2)

        ans = jnp.asarray([
            [1,1,1,1],
            [0,1,0.5,0.5],
        ])

        @jax.vmap
        def expectAll(u):
            @jax.vmap
            def expect(si):
                return dense.expectUnitary(si, (0, 1), u)
            return expect(s)

        np.testing.assert_allclose(expectAll(U), ans)


class TestHadamard(unittest.TestCase):
    def test_H(self):
        w = (0,)
        z = dense.zeros(1, jnp.complex64)
        h = dense.Hadamard(z, w)
        np.testing.assert_allclose(h, [1/jnp.sqrt(2), 1/jnp.sqrt(2)])

    def test_1(self):
        w = (0,)
        o = dense.zeros(1, jnp.complex64).at[:].set([0, 1])
        h = dense.Hadamard(o, w)
        np.testing.assert_allclose(h, [1/jnp.sqrt(2), -1/jnp.sqrt(2)])

    def test_twice(self):
        w = (0,)
        z = dense.zeros(2, jnp.complex64)
        zz = dense.Hadamard(dense.Hadamard(z, w), w)
        np.testing.assert_allclose(z, zz)

class TestPauliX(unittest.TestCase):
    def test_pauliX(self):
        w = (0,)
        z = dense.zeros(2, jnp.complex64)
        x = dense.PauliX(z, w)
        np.testing.assert_allclose(x[:, 0], [0+0j, 1+0j])

    def test_twice(self):
        w = (0,)
        z = dense.zeros(1, jnp.complex64)
        xx = dense.PauliX(dense.PauliX(z, w), w)
        np.testing.assert_allclose(z, xx)

class TestPauliY(unittest.TestCase):
    def test_pauliY(self):
        w = (0,)
        z = dense.zeros(1, jnp.complex64)
        y = dense.PauliY(z, w)
        np.testing.assert_allclose(y, [0+0j, 0+1j])

    def test_1(self):
        w = (0,)
        o = dense.zeros(1, jnp.complex64).at[:].set([0, 1])
        y = dense.PauliY(o, w)
        np.testing.assert_allclose(y, [-1j, 0])

    def test_twice(self):
        w = (0,)
        z = dense.zeros(1, jnp.complex64)
        yy = dense.PauliY(dense.PauliY(z, w), w)
        np.testing.assert_allclose(yy, z)

class TestPauliZ(unittest.TestCase):
    def test_pauliZ(self):
        w = (0,)
        z = dense.zeros(1, jnp.complex64)
        x = dense.PauliZ(z, w)
        np.testing.assert_allclose(x, [1, 0])

    def test_1(self):
        w = (0,)
        o = dense.zeros(1, jnp.complex64).at[:].set([0, 1])
        x = dense.PauliZ(o, w)
        np.testing.assert_allclose(x, [0, -1])

    def test_twice(self):
        w = (0,)
        z = dense.zeros(1, jnp.complex64)
        zz = dense.PauliZ(dense.PauliZ(z, w), w)
        np.testing.assert_allclose(zz, z)

class TestS(unittest.TestCase):
    def test_S(self):
        w = (0,)
        z = dense.zeros(1, jnp.complex64)
        s = dense.S(z, w)
        np.testing.assert_allclose(s, [1, 0])

    def test_1(self):
        w = (0,)
        o = dense.zeros(1, jnp.complex64).at[:].set([0, 1])
        s = dense.S(o, w)
        np.testing.assert_allclose(s, [0, 1j])

class TestT(unittest.TestCase):
    def test_T(self):
        w = (0,)
        z = dense.zeros(1, jnp.complex64)
        t = dense.T(z, w)
        np.testing.assert_allclose(t, [1, 0])

    def test_1(self):
        w = (0,)
        o = dense.zeros(1, jnp.complex64).at[:].set([0, 1])
        t = dense.T(o, w)
        np.testing.assert_allclose(t, [0, jnp.exp(0.25j * jnp.pi)])

class TestSX(unittest.TestCase):
    def test_SX(self):
        w = (0,)
        z = dense.zeros(1, jnp.complex64)
        sx = dense.SX(z, w)
        np.testing.assert_allclose(sx, [0.5*(1+1j), 0.5*(1-1j)])

    def test_1(self):
        w = (0,)
        o = dense.zeros(1, jnp.complex64).at[:].set([0, 1])
        sx = dense.SX(o, w)
        np.testing.assert_allclose(sx, [0.5*(1-1j), 0.5*(1+1j)])

class TestCNOT(unittest.TestCase):
    def test_CNOT(self):
        w = (0, 1)
        z = dense.zeros(2, jnp.complex64)
        cn = dense.CNOT(z, w)
        np.testing.assert_allclose(cn, z)

    def test_1(self):
        w = (0, 1)
        o = dense.PauliX(dense.zeros(2, jnp.complex64), (0,))
        cn = dense.CNOT(o, w)
        np.testing.assert_allclose(cn, dense.PauliX(o, (1,)))

class TestCZ(unittest.TestCase):
    def test_CZ(self):
        w = (0, 1)
        z = dense.zeros(2, jnp.complex64)
        cz = dense.CZ(z, w)
        np.testing.assert_allclose(cz, z)

    def test_1(self):
        w = (0, 1)
        o = dense.PauliX(dense.zeros(2, jnp.complex64), (0,))
        cz = dense.CZ(o, w)
        np.testing.assert_allclose(cz, dense.PauliZ(o, (1,)))

class TestCY(unittest.TestCase):
    def test_CY(self):
        w = (0, 1)
        z = dense.zeros(2, jnp.complex64)
        cy = dense.CY(z, w)
        np.testing.assert_allclose(cy, z)

    def test_1(self):
        w = (0, 1)
        o = dense.PauliX(dense.zeros(2, jnp.complex64), (0,))
        cy = dense.CY(o, w)
        np.testing.assert_allclose(cy, dense.PauliY(o, (1,)))

class TestSWAP(unittest.TestCase):
    def test_SWAP(self):
        w = (0, 1)
        z = dense.zeros(2, jnp.complex64)
        s = dense.SWAP(z, w)
        np.testing.assert_allclose(s, z)

    def test_1(self):
        w = (0, 1)
        z = dense.zeros(2, jnp.complex64)
        o = dense.PauliX(z, (0,))
        s = dense.SWAP(o, w)
        np.testing.assert_allclose(s, dense.PauliX(z, (1,)))

    def test_twice(self):
        w = (0, 1)
        o = dense.PauliX(dense.zeros(2, jnp.complex64), (0,))
        ss = dense.SWAP(dense.SWAP(o, w), w)
        np.testing.assert_allclose(ss, o)

class TestISWAP(unittest.TestCase):
    def test_00(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s = dense.ISWAP(s00, w)
        np.testing.assert_allclose(s, s00)

    def test_01(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s = dense.ISWAP(s01, w)
        np.testing.assert_allclose(s, 1j * s10)

    def test_10(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s = dense.ISWAP(s10, w)
        np.testing.assert_allclose(s, 1j * s01)

    def test_11(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s11 = dense.PauliX(dense.PauliX(s00, (0,)),
                           (1,))
        s = dense.ISWAP(s11, w)
        np.testing.assert_allclose(s, s11)

class TestECR(unittest.TestCase):
    def test_ECR(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s11 = dense.PauliX(s01, (0,))

        for s, ans in [[s00, [0, 0, 1, -1j]],
                       [s01, [0, 0, -1j, 1]],
                       [s10, [1, 1j, 0, 0]],
                       [s11, [1j, 1, 0, 0]]]:
            with self.subTest(state=s):
                np.testing.assert_allclose(dense.to_state(dense.ECR(s, w)),
                                           jnp.asarray(ans)/jnp.sqrt(2))

class TestSISWAP(unittest.TestCase):
    def test_SISWAP(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s11 = dense.PauliX(s01, (0,))

        for name, s, ans in [["s00", s00, [1, 0, 0, 0]],
                             ["s01", s01, [0,  1/jnp.sqrt(2), 1j/jnp.sqrt(2), 0]],
                             ["s10", s10, [0, 1j/jnp.sqrt(2),  1/jnp.sqrt(2), 0]],
                             ["s11", s11, [0, 0, 0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.SISWAP(s, w)), ans)

class TestCSWAP(unittest.TestCase):
    def test_CSWAP(self):
        w = (0, 1, 2)
        s001 = dense.PauliX(dense.zeros(3, jnp.complex64), (2,))
        np.testing.assert_allclose(dense.CSWAP(s001, w), s001)

        s101 = dense.PauliX(s001, (0,))
        np.testing.assert_allclose(dense.CSWAP(s101, w),
                                   dense.SWAP(s101, (1, 2)))

class TestToffoli(unittest.TestCase):
    def test_Toffoli(self):
        w = (0, 1, 2)
        s001 = dense.PauliX(dense.zeros(3, jnp.complex64), (2,))
        np.testing.assert_allclose(dense.Toffoli(s001, w), s001)

        s011 = dense.PauliX(s001, (1,))
        np.testing.assert_allclose(dense.Toffoli(s011, w), s011)

        s110 = dense.SWAP(s011, (0, 2))
        np.testing.assert_allclose(dense.Toffoli(s110, w),
                                   dense.PauliX(s110, (2,)))

class TestRot(unittest.TestCase):
    def test_000(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        for name, s, ans in [["s0", s0, [1, 0]],
                             ["s1", s1, [0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.Rot(s, w, 0, 0, 0)),
                                           ans)

    def test_0y0(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        y = 0.25 * jnp.pi
        for name, s, ans in [["s0", s0, [ jnp.cos(0.5*y), jnp.sin(0.5*y)]],
                             ["s1", s1, [-jnp.sin(0.5*y), jnp.cos(0.5*y)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.Rot(s, w, 0, y, 0)),
                                           ans)

    def test_z00(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        z = jnp.pi / 8
        for name, s, ans in [["s0", s0, [jnp.exp(-0.5j*z), 0]],
                             ["s1", s1, [0, jnp.exp(0.5j*z)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.Rot(s, w, z, 0, 0)),
                                           ans, atol=1e-7)

    def test_00z(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        z = jnp.pi * 0.75
        for name, s, ans in [["s0", s0, [jnp.exp(-0.5j*z), 0]],
                             ["s1", s1, [0, jnp.exp(0.5j*z)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.Rot(s, w, 0, 0, z)),
                                           ans, atol=1e-7)

    def test_zyz(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
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
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        for name, s, ans in [["s0", s0, [1, 0]],
                             ["s1", s1, [0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RX(s, w, 0)), ans)

    def test_x(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        x = jnp.pi
        for name, s, ans in [["s0", s0, [ 0, -1j]],
                             ["s1", s1, [-1j,   0]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RX(s, w, x)), ans,
                                           atol=1e-7)

class TestRY(unittest.TestCase):
    def test_0(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        for name, s, ans in [["s0", s0, [1, 0]],
                             ["s1", s1, [0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RY(s, w, 0)), ans,
                                           atol=1e-7)

    def test_y(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        y = jnp.pi
        for name, s, ans in [["s0", s0, [ 0, 1]],
                             ["s1", s1, [-1, 0]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RY(s, w, y)), ans,
                                           atol=1e-7)

class TestRZ(unittest.TestCase):
    def test_0(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        for name, s, ans in [["s0", s0, [1, 0]],
                             ["s1", s1, [0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RZ(s, w, 0)), ans,
                                           atol=1e-7)

    def test_z(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        z = jnp.pi
        for name, s, ans in [["s0", s0, [jnp.exp(-0.5j*z), 0]],
                             ["s1", s1, [0,  jnp.exp(0.5j*z)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.RZ(s, w, z)), ans,
                                           atol=1e-7)

class TestPhaseShift(unittest.TestCase):
    def test_0(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        for name, s, ans in [["s0", s0, [1, 0]],
                             ["s1", s1, [0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.PhaseShift(s, w, 0)),
                                           ans)
    def test_p(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        p = jnp.pi
        for name, s, ans in [["s0", s0, [1, 0]],
                             ["s1", s1, [0, jnp.exp(1j*p)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.to_state(dense.PhaseShift(s, w, p)),
                                           ans)

class TestControlledPhaseShift(unittest.TestCase):
    def test_CPS(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s11 = dense.PauliX(s01, (0,))

        p = jnp.pi
        for name, s, ans in [["s00", s00, s00],
                             ["s01", s01, s01],
                             ["s10", s10, dense.PhaseShift(s10, (1,), p)],
                             ["s11", s11, dense.PhaseShift(s11, (1,), p)]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.ControlledPhaseShift(s, w, p), ans)

class TestCPhaseShift(unittest.TestCase):
    def test_cps(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s11 = dense.PauliX(s01, (0,))

        p = jnp.pi
        for name, s, ans in [["s00", s00, s00 * jnp.exp(1j*p)],
                             ["s01", s01, s01],
                             ["s10", s10, s10],
                             ["s11", s11, s11]]:
            with self.subTest(mode = "00", state=name):
                np.testing.assert_allclose(dense.CPhaseShift00(s, w, p), ans)

        for name, s, ans in [["s00", s00, s00],
                             ["s01", s01, s01 * jnp.exp(1j*p)],
                             ["s10", s10, s10],
                             ["s11", s11, s11]]:
            with self.subTest(mode = "01", state=name):
                np.testing.assert_allclose(dense.CPhaseShift01(s, w, p), ans)

        for name, s, ans in [["s00", s00, s00],
                             ["s01", s01, s01],
                             ["s10", s10, s10 * jnp.exp(1j*p)],
                             ["s11", s11, s11]]:
            with self.subTest(mode = "10", state=name):
                np.testing.assert_allclose(dense.CPhaseShift10(s, w, p), ans)

class TestCRX(unittest.TestCase):
    def test_CRX(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s11 = dense.PauliX(s01, (0,))

        p = jnp.pi
        for name, s, ans in [["s00", s00, s00],
                             ["s01", s01, s01],
                             ["s10", s10, dense.RX(s10, (1,), p)],
                             ["s11", s11, dense.RX(s11, (1,), p)]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.CRX(s, w, p), ans)

class TestCRY(unittest.TestCase):
    def test_CRY(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s11 = dense.PauliX(s01, (0,))

        p = jnp.pi
        for name, s, ans in [["s00", s00, s00],
                             ["s01", s01, s01],
                             ["s10", s10, dense.RY(s10, (1,), p)],
                             ["s11", s11, dense.RY(s11, (1,), p)]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.CRY(s, w, p), ans)

class TestCRZ(unittest.TestCase):
    def test_CRZ(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s11 = dense.PauliX(s01, (0,))

        p = jnp.pi
        for name, s, ans in [["s00", s00, s00],
                             ["s01", s01, s01],
                             ["s10", s10, dense.RZ(s10, (1,), p)],
                             ["s11", s11, dense.RZ(s11, (1,), p)]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.CRZ(s, w, p), ans)

class TestCRot(unittest.TestCase):
    def test_CRot(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s11 = dense.PauliX(s01, (0,))

        p = jnp.pi * 1.5
        t = jnp.pi / 8
        o = jnp.pi / 6
        for name, s, ans in [["s00", s00, s00],
                             ["s01", s01, s01],
                             ["s10", s10, dense.Rot(s10, (1,), p, t, o)],
                             ["s11", s11, dense.Rot(s11, (1,), p, t, o)]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.CRot(s, w, p, t, o), ans)

class TestU2(unittest.TestCase):
    def test_00(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        for name, s, ans in [["s0", s0, [ 1, 1]],
                             ["s1", s1, [-1, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.U2(s, w, 0, 0),
                                           jnp.asarray(ans)/jnp.sqrt(2))

    def test_p0(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        p = jnp.pi * 1.5
        for name, s, ans in [["s0", s0, [ 1, jnp.exp(1j*p)]],
                             ["s1", s1, [-1, jnp.exp(1j*p)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.U2(s, w, p, 0),
                                           jnp.asarray(ans)/jnp.sqrt(2))

    def test_0d(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        d = jnp.pi * 1.5
        for name, s, ans in [["s0", s0, [ 1, 1]],
                             ["s1", s1, [-jnp.exp(1j*d), jnp.exp(1j*d)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.U2(s, w, 0, d),
                                           jnp.asarray(ans)/jnp.sqrt(2))

    def test_pd(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        p = jnp.pi * 1.5
        d = jnp.pi / 4
        for name, s, ans in [["s0", s0, [ 1, jnp.exp(1j*p)]],
                             ["s1", s1, [-jnp.exp(1j*d), jnp.exp(1j*(p+d))]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.U2(s, w, p, d),
                                           jnp.asarray(ans)/jnp.sqrt(2), atol=1e-7)

class TestU3(unittest.TestCase):
    def test_000(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        for name, s, ans in [["s0", s0, [1, 0]],
                             ["s1", s1, [0, 1]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.U3(s, w, 0, 0, 0), ans, atol=1e-7)

    def test_t00(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        t = jnp.pi / 6
        for name, s, ans in [["s0", s0, [jnp.cos(0.5*t), jnp.sin(0.5*t)]],
                             ["s1", s1, [-jnp.sin(0.5*t), jnp.cos(0.5*t)]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.U3(s, w, t, 0, 0), ans, atol=1e-7)

    def test_tp0(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        t = jnp.pi
        p = jnp.pi * 1.5
        for name, s, ans in [["s0", s0, [0, jnp.exp(1j*p)]],
                             ["s1", s1, [-1, 0]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.U3(s, w, t, p, 0), ans, atol=1e-7)

    def test_t0d(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        t = jnp.pi
        d = jnp.pi * 1.5
        for name, s, ans in [["s0", s0, [0, 1]],
                             ["s1", s1, [-jnp.exp(1j*d), 0]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.U3(s, w, t, 0, d), ans, atol=1e-7)

    def test_tpd(self):
        w = (0,)
        s0 = dense.zeros(1, jnp.complex64)
        s1 = dense.PauliX(s0, w)

        t = jnp.pi
        p = jnp.pi / 4
        d = jnp.pi * 1.5
        for name, s, ans in [["s0", s0, [0, jnp.exp(1j*p)]],
                             ["s1", s1, [-jnp.exp(1j*d), 0]]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.U3(s, w, t, p, d), ans, atol=1e-7)

class TestPSWAP(unittest.TestCase):
    def test_0(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s11 = dense.PauliX(s01, (0,))

        for name, s, ans in [["s00", s00, dense.SWAP(s00, w)],
                             ["s01", s01, dense.SWAP(s01, w)],
                             ["s10", s10, dense.SWAP(s10, w)],
                             ["s11", s11, dense.SWAP(s11, w)]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.PSWAP(s, w, 0), ans)

    def test_p(self):
        w = (0, 1)
        s00 = dense.zeros(2, jnp.complex64)
        s01 = dense.PauliX(s00, (1,))
        s10 = dense.PauliX(s00, (0,))
        s11 = dense.PauliX(s01, (0,))

        p = jnp.pi * 1.5
        for name, s, ans in [["s00", s00, dense.SWAP(s00, w)],
                             ["s01", s01, dense.SWAP(s01, w) * jnp.exp(1j*p)],
                             ["s10", s10, dense.SWAP(s10, w) * jnp.exp(1j*p)],
                             ["s11", s11, dense.SWAP(s11, w)]]:
            with self.subTest(state=name):
                np.testing.assert_allclose(dense.PSWAP(s, w, p), ans)

class TestQubitUnitary(unittest.TestCase):
    def test_X(self):
        from diffq import _operators as _op
        w = (0,)
        q0 = dense.zeros(1, jnp.complex64)
        np.testing.assert_allclose(dense.QubitUnitary(q0, w, _op.sigmaX(q0.dtype)),
                                   dense.PauliX(q0, w))

    def test_H(self):
        from diffq import _operators as _op
        w = (0,)
        q0 = dense.zeros(1, jnp.complex64)
        np.testing.assert_allclose(dense.QubitUnitary(q0, w, _op.H(q0.dtype)),
                                   dense.Hadamard(q0, w))

    def test_2qubit(self):
        w = (0, 1)
        q00 = dense.zeros(2, jnp.complex64)

        def f(c):
            c = dense.PauliX(c, (0,))
            c = dense.PauliZ(c, (1,))
            return c
        U = util.CreateMatrix(dense, 2, jnp.complex64, f)
        np.testing.assert_allclose(dense.QubitUnitary(q00, w, U),
                                   dense.PauliZ(dense.PauliX(q00, (0,)), (1,)))

class TestControlledQubitUnitary(unittest.TestCase):
    def test_CX(self):
        from diffq import _operators as _op
        w = (0,1)
        q00 = dense.zeros(2, jnp.complex64)
        np.testing.assert_allclose(
            dense.ControlledQubitUnitary(q00, w, _op.sigmaX(q00.dtype)),
            dense.CNOT(q00, w)
        )

        q10 = dense.PauliX(q00, (0,))
        np.testing.assert_allclose(
            dense.ControlledQubitUnitary(q10, w, _op.sigmaX(q10.dtype)),
            dense.CNOT(q10, w)
        )


if __name__ == "__main__":
    unittest.main()
