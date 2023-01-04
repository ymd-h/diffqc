import jax
import jax.numpy as jnp
import numpy as np

import diffq
from diffq import dense as op

def main():
    nqubits = 5

    @jax.jit
    def circuit(params):
        x = op.zero(nqubits, jnp.complex64)

        for i in range(nqubits):
            x = op.Hadamard(x, (i,))

        for i in range(nqubits-1):
            x = op.CRZ(x, (i, i+1), params["CRZ"][i])

        x = op.Rot(x, (4, ),
                   params["Rot"]["phi"], params["Rot"]["theta"], params["Rot"]["omega"])

        return op.to_state(x)

    @jax.jit
    def expval(params):
        s = circuit(params)
        p = diffq.prob(s)
        return diffq.expval(p, 4)


    p = {
        "CRZ": jnp.ones((4,)),
        "Rot": {
            "phi": jnp.ones((1,)),
            "theta": jnp.ones((1,)),
            "omega": jnp.ones((1,)),
        }
    }

    print(f"expval: {expval(p)}")
    print(f"grad: {jax.grad(expval)(p)}")

if __name__ == "__main__":
    main()
