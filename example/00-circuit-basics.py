"""
Example 00: Circuit Basics
"""
import jax
import jax.numpy as jnp

import diffqc
from diffqc import dense as op

def main():
    nqubits = 5

    @jax.jit
    def circuit(params):
        # Initialize |00..0> state
        x = op.zeros(nqubits, jnp.complex64)

        # Apply quantum operation
        # At quantum operator,
        # - the first argument is qubit state,
        # - the second argument is wire(s) and its type is `tuple`,
        # - the rest arguments are parameters, and
        # - the return value is the new qubit state.

        for i in range(nqubits):
            # 1 qubit operation H on i-th wire
            x = op.Hadamard(x, (i,))

        for i in range(nqubits-1):
            # 2 qubit operation CRZ on i-th and (i+1)-th wires
            # The first wire is the control qubit.
            x = op.CRZ(x, (i, i+1), params["CRZ"][i])

        x = op.Rot(x, (4, ),
                   params["Rot"]["phi"], params["Rot"]["theta"], params["Rot"]["omega"])

        # Convert internal representation to state-vector
        # aka. [|00000>, |00001>, |00010>, ..., |11111>]
        return op.to_state(x)

    @jax.jit
    def expval(params):
        s = circuit(params)

        # Caluculate probabilities of each state
        p = diffqc.prob(s)

        # Expectation of |1> at certain wire.
        return diffqc.expval(p, 4)


    # Define parameters as compatible with JAX pytree.
    # Ref: https://jax.readthedocs.io/en/latest/pytrees.html
    p = {
        "CRZ": jnp.ones((4,)),
        "Rot": {
            "phi": jnp.ones((1,)),
            "theta": jnp.ones((1,)),
            "omega": jnp.ones((1,)),
        }
    }

    # Now you can call circuit and take gradient.
    print(f"expval: {expval(p)}")
    print(f"grad: {jax.grad(expval)(p)}")

if __name__ == "__main__":
    main()
