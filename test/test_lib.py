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
                                   .set(1/jnp.sqrt(2))
                                   .at[p.shape[0]-1]
                                   .set(1/jnp.sqrt(2)))

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


if __name__ == "__main__":
    unittest.main()
