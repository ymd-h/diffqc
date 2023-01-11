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

class TestCreateMatrix(unittest.TestCase):
    def test_dense(self):
        def f(c):
            c = dense.Hadamard(c, (0,))
            c = dense.PauliY(c, (1,))
            c = dense.CNOT(c, (0, 1))
            return c

        # |00>
        #   ->  (|0> + |1>)|0>/sqrt(2)
        #   -> i(|0> + |1>)|1>/sqrt(2)
        #   -> i(|01> + |10>)/sqrt(2)
        #
        # |01>
        #   ->   (|0> + |1>)|1>/sqrt(2)
        #   -> -i(|0> + |1>)|0>/sqrt(2)
        #   -> -i(|00> + |11>)/sqrt(2)
        #
        # |10>
        #   ->  (|0> - |1>)|0>/sqrt(2)
        #   -> i(|0> - |1>)|1>/sqrt(2)
        #   -> i(|01> - |10>)/sqrt(2)
        #
        # |11>
        #   ->   (|0> - |1>)|1>/sqrt(2)
        #   -> -i(|0> - |1>)|0>/sqrt(2)
        #   -> -i(|00> - |11>)/sqrt(2)

        mat = util.CreateMatrix(dense, 2, jnp.complex64, f)
        np.testing.assert_allclose(mat,
                                   jnp.asarray([
                                       [0 ,-1j, 0 ,-1j],
                                       [1j, 0 , 1j, 0 ],
                                       [1j, 0 ,-1j, 0 ],
                                       [0 ,-1j, 0 , 1j],
                                   ]) / jnp.sqrt(2))

if __name__ == "__main__":
    unittest.main()
