import unittest

import jax
import jax.numpy as jnp

import numpy as np

import diffq
from diffq import dense, sparse
from diffq import lib


class TestGHZ(unittest.TestCase):
    def _check(self, s):
        p = diffq.prob(s)
        np.testing.assert_allclose(p,
                                   jnp.zeros_like(p)
                                   .at[0]
                                   .set(0.5)
                                   .at[p.shape[0]-1]
                                   .set(0.5))

    def test_dense(self):
        q = dense.zeros(3, jnp.complex64)
        q = lib.GHZ(dense, q, (0,1,2))
        self._check(dense.to_state(q))

    def test_sparse(self):
        q = sparse.zeros(3, jnp.complex64)
        q = lib.GHZ(sparse, q, (0,1,2))
        self._check(sparse.to_state(q))

    def test_dense_jit(self):
        @jax.jit
        def f():
            q = dense.zeros(3, jnp.complex64)
            q = lib.GHZ(dense, q, (0,1,2))
            return dense.to_state(q)
        self._check(f())

    def test_sparse_jit(self):
        @jax.jit
        def f():
            q = sparse.zeros(3, jnp.complex64)
            q = lib.GHZ(sparse, q, (0,1,2))
            return sparse.to_state(q)
        self._check(f())


class TestQFT(unittest.TestCase):
    def _f(self, op):
        def f():
            q = op.zeros(3, jnp.complex64)
            for i in range(3):
                q = op.Hadamard(q, (i,))
            q = lib.QFT(op, q, (0,1,2))
            return op.to_state(q)
        return f

    def _check(self, s):
        # QFT: |+++>  -->  |000>
        p = diffq.prob(s)
        np.testing.assert_allclose(p, jnp.zeros_like(p).at[0].set(1),
                                   atol=1e-7, rtol=1e-6)

    def test_dense(self):
        f = self._f(dense)
        self._check(f())

    def test_sparse(self):
        f = self._f(sparse)
        self._check(f())

    def test_dense_jit(self):
        f = jax.jit(self._f(dense))
        self._check(f())

    def test_sparse_jit(self):
        f = jax.jit(self._f(sparse))
        self._check(f())

class TestQPE(unittest.TestCase):
    def _f(self, op):
        def f():
            s00 = op.zeros(5, jnp.complex64)
            s01 = op.PauliZ(s00, (1,))
            s10 = op.PauliZ(s00, (0,))
            s11 = op.PauliZ(s10, (1,))
            s = jnp.stack((s00, s01, s10, s11))

            s = jax.vmap(lambda q: op.to_state(lib.QPE(op, q, (0, 1), (2, 3, 4))))(s)
            return s
        return f

    def _check(self, s):
        p = jax.vmap(lambda si: diffq.marginal_prob(diffq.prob(si), (0,1)))(s)
        np.testing.assert_allclose(p,
                                   [
                                       [1,0,0,0,0,0,0,0], # |000>
                                       [0,1,0,0,0,0,0,0], # |001>
                                       [0,0,1,0,0,0,0,0], # |010>
                                       [0,0,0,1,0,0,0,0], # |011>
                                   ])

    def test_dense(self):
        f = self._f(dense)
        self._check(f())

    def test_sparse(self):
        f = self._f(sparse)
        self._check(f())

    def test_dense_jit(self):
        f = jax.jit(self._f(dense))
        self._check(f())

    def test_sparse_jit(self):
        f = jax.jit(self._f(sparse))
        self._check(f())


if __name__ == "__main__":
    unittest.main()
