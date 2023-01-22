# Design Parameterized Quantum Circuit (PQC)


When we use Variational Quantum Algorithm (VQA), especially Quantum
Circuit Learning (QCL)[^1], we need to design quantum circuit.

As far as we know, designing good circuit is still active research area.

## Expressivity and Entangle Capability

S. Sim _et al_.[^2] proposed 2 metrics of PQC, aka. "expressivity"
and "entangle capability".

The expressivity is defined as a circuit's ability to generate pure
states that are well representative of the Hilbert space.
Let's say, if output quantum states are well distributed over all
bloch spheres when moving parameters, the circuit has good
expressivity.
For example, arbitary unitary rotation gate has better expressivity
than RX rotation gate.

The entangle capability is defined by Mayer-Wallach entanglement
mesearment[^3]. Emprically we know highly entangled circuits can
capture good representation from data structure even though shallow
depth.

The authors also calculated (numerically simulated) these metrics for
19 circuits which have been proposed before and they indicated good
candidates;

|               | Josephson Sampler [^4] | Circuit Centric [^5]  |
|---------------|:----------------------:|:---------------------:|
| No.           | 12                     | 19                    |
| Parameters    | {math}`(4n-4)L`        | {math}`3nL`           |
| 2 qubit gates | {math}`(n-1)L`         | {math}`nL`            |
| Circuit depth | {math}`6L`             | {math}`(n+2)L`        |
| Saturate      | {math}`L \sim 3`       | {math}`L \sim 3`      |
| `diffqc.nn`   | `JosephsonSampler`     | `CircuitCentricBlock` |



where {math}`n` is number of qubits, {math}`L` is number of layer repetition.

Another research conducted by T Hubregtsen _et al_.[^6] showed these
circuits worked well for classification of toy dataset.


## Barren Plateaus
J. R. McClean _et al_.[^7] pointed out there is "barren plateaus" at
training of PQC.



[^1]: K. Mitarai _et al_., "Quantum Circuit Learning", Phys. Rev. A 98,
    032309 (2018)  
    DOI: <https://doi.org/10.1103/PhysRevA.98.032309>  
    arXiv: <https://arxiv.org/abs/1803.00745>

[^2]: S. Sim _et al_., "Expressibility and entangling capability of
    parameterized quantum circuits for hybrid quantum-classical algorithms",
    Adv. Quantum Technol. 2 (2019) 1900070  
    DOI: <https://doi.org/10.1002/qute.201900070>  
    arXiv: <https://arxiv.org/abs/1905.10876>

[^3]: D. A. Meyer and N. R. Wallach, "Global entanglement in multiparticle systems",
    J. Math. Phys. 43, 4273 (2002)  
    DOI: <https://doi.org/10.1063/1.1497700>  
    arXiv: <https://arxiv.org/abs/quant-ph/0108104>

[^4]: M. Schuld _et al_., "Circuit-centric quantum classifiers",
    Phys. Rev. A 101, 032308 (2020)  
    DOI: <https://doi.org/10.1103/PhysRevA.101.032308>  
    arXiv: <https://arxiv.org/abs/1804.00633>

[^5]: M. R. Geller, "Sampling and scrambling on a chain of superconducting qubits",
    Phys. Rev. Applied 10, 024052 (2018)  
    DOI: <https://doi.org/10.1103/PhysRevApplied.10.024052>  
    arXiv: <https://arxiv.org/abs/1711.11026>

[^6]: T. Hubregtsen _et al_., "Evaluation of Parameterized Quantum
    Circuits: on the relation between classification accuracy,
    expressibility and entangling capability", Quantum Machine
    Intelligence volume 3, Article number: 9 (2021)  
    DOI: <https://doi.org/10.1007/s42484-021-00038-w>  
    arXiv: <https://arxiv.org/abs/2003.09887>

[^7]: J. R. McClean _et al_., "Barren plateaus in quantum neural network
    training landscapes", Nat Commun 9, 4812 (2018)  
    DOI: <https://doi.org/10.1038/s41467-018-07090-4>  
    arXiv: <https://arxiv.org/abs/1803.11173>
