import unittest

import jax
import jax.numpy as jnp
import numpy as np

import diffq
from diffq import dense, sparse, util


class TestCreatePossibleState(unittest.TestCase):
    def test_dense(self):
        q = util.CreatePossibleState(dense, 2, jnp.complex64)
        np.testing.assert_allclose(jax.vmap(dense.to_state)(q),
                                   jnp.asarray([
                                       [1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1],
                                   ]))

    def test_sparse(self):
        q = util.CreatePossibleState(sparse, 2, jnp.complex64)
        np.testing.assert_allclose(jax.vmap(sparse.to_state)(q),
                                   jnp.asarray([
                                       [1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1],
                                   ]))

if __name__ == "__main__":
    unittest.main()
