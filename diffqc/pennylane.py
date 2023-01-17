from typing import Any, Dict, Iterable, List, Literal, Union

import numpy as np
import jax
import jax.numpy as jnp
import pennylane as qml

import diffqc


class diffqcQubitDevice(qml.QubitDevice):
    name = "PennyLane plugin for diffqc"
    short_name = "diffqc.qubit"
    pennylane_requires = ">=0.20.0"
    version = "0.0.0"
    author = "ymd-h"

    operations = {
        # Non-Parameterized Gate
        "Identity",
        "Hadamard",
        "PauliX",
        "PauliY",
        "PauliZ",
        "PauliZ",
        "S",
        "T",
        "SX",
        "CNOT",
        "CZ",
        "CY",
        "SWAP",
        "ISWAP",
        "ECR",
        "SISWAP",
        "SQISWAP",
        "CSWAP",
        "Toffoli",
        #"MultiControlledX",
        #"Barrier",
        #"WireCut",

        # Parameterized Gate
        "Rot",
        "RX",
        "RY",
        "RZ",
        #"MultiRZ",
        #"PauliRot",
        "PhaseShift",
        "ControlledPhaseShift",
        "CPhase",
        "CPhaseShift",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "U1",
        "U2",
        "U3",
        "IsingXX",
        #"IsingXY",
        "IsingYY",
        "IsingZZ",
        "PSWAP",
    }
    observables = {
        #"Hadamard",
        #"Hermitian",
        #"Identity",
        "PauliX",
        "PauliY",
        "PauliZ",
        #"Projector",
        #"Hamiltonian",
        #"SparseHamiltonian",
    }

    def __init__(self,
                 wires: Union[int, Iterable[Union[int, str]]],
                 shots: Union[None, int, List[int]] = None,
                 mode: Literal["dense", "sparse"] = "dense"):
        super().__init__(wires=wires, shots=shots)

        try:
            self.op = {
                "dense": diffqc.dense,
                "sparse": diffqc.sparse,
            }[mode]
        except KeyError:
            raise ValueError(f"Unknown mode: {mode}")

    def apply(self, operations: List[qml.operation.Operation], **kwargs):
        f = lambda: self.op.zeros(len(self.wires), jnp.complex64)

        for op in operations:
            if op.name == "Identity":
                continue

            if op.name in ["IsingXX", "IsingYY", "IsingZZ"]:
                opf = {"IsingXX": op.RXX, "IsingYY": op.RYY, "IsingZZ": op.RZZ}
            else:
                opf = getattr(self.op, op.name)

            f = lambda: opf(f(),
                            jnp.asarray(op.wires),
                            *tuple(jnp.asarray(p) for p in op.parameters))

        f = lambda: self.op.to_state(f())

        self._state = jax.jit(f)()
        return np.array(self._state, copy=True)

    def analytic_probability(self, wires: Union[None,
                                                Iterable[Union[int, str]],
                                                int,
                                                str,
                                                qml.wires.Wires] = None):
        if self._state is None:
            return None

        prob = self.marginal_prob(np.asarray(jnp.abs(self._state) ** 2), wires)
        return prob

    @classmethod
    def capability(cls) -> Dict[str, Any]:
        return {
            "model": "qubit",
            "return_state": True,
            "supports_tracker": True,
        }
