"""
Example 03: PennyLane plugin

Ref: https://pennylane.ai/qml/demos/tutorial_jax_transformations.html
"""
import pennylane as qml

import jax
import jax.numpy as jnp


def main():
    dev = qml.device("diffqc.qubit", wires=2)

    @qml.qnode(dev, interface="jax")
    def circuit(param):
        qml.RX(param, wires=0)
        qml.CNOT(wires=[0,1])

        return qml.expval(qml.PauliZ(0))

    print(circuit(0.123))


if __name__ == "__main__":
    main()
