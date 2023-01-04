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


if __name__ == "__main__":
    unittest.main()
