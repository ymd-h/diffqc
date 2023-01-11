"""
Exaple 01: QCL classification with Flax

K. Mitarai et al., "Quantum Circuit Learning", Phys. Rev. A 98, 032309 (2018)
https://doi.org/10.1103/PhysRevA.98.032309
https://arxiv.org/abs/1803.00745

This example additionally requires followings;
* Flax: https://flax.readthedocs.io/en/latest/index.html
* Optax: https://optax.readthedocs.io/en/latest/
* scikit-learn: https://scikit-learn.org/stable/
"""
import functools
import time
from typing import Callable

import diffq
from diffq import dense as op

from flax import linen as nn
from flax.training import train_state

import jax
import jax.numpy as jnp

import optax

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def circuit(n_qubits, depth, features, weights):
    msg = "BUG: n_qubits ({}) must be greater than feature size ({})."
    assert n_qubits >= features.shape[0], msg.format(n_qubits, features.shape[0])

    q = op.zeros(n_qubits, jnp.complex64)

    for idx in range(n_qubits):
        i = idx % features.shape[0]
        q = op.RY(q, (idx,), jnp.arcsin(features.at[i].get()))
        q = op.RZ(q, (idx,), jnp.arccos(features.at[i].get() ** 2))

    # Note: We use much simpler circuit than that of the original paper,
    #       however, it seems fine for this easy task.
    for k in range(depth):
        for idx in range(0, n_qubits-1):
            q = op.CNOT(q, (idx, idx+1))
        for idx in range(n_qubits):
            q = op.RY(q, (idx,), weights.at[k, idx].get())

    return jnp.stack(tuple(op.expectZ(q, (idx,)) for idx in range(n_qubits)))


class QCL(nn.Module):
    n_qubits: int
    depth: int
    output_dim: int
    circuit_init: Callable = nn.initializers.uniform(2 * jnp.pi)

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        w = self.param("circuit", self.circuit_init, (self.depth, self.n_qubits))

        @jax.vmap
        def batch_circuit(e):
            return circuit(self.n_qubits, self.depth, e, w)

        x = batch_circuit(x)
        x = nn.Dense(self.output_dim)(x)
        return x


def load_data():
    features, labels = load_iris(return_X_y=True)
    features = features[:,2:] # only sepal length/width
    scalar = MinMaxScaler(feature_range=(-1, 1))
    features_std = scalar.fit_transform(features)
    x_train, x_test, y_train, y_test = train_test_split(features_std, labels,
                                                        random_state=42, shuffle=True)
    return (jnp.asarray(x_train), jnp.asarray(y_train),
            jnp.asarray(x_test) , jnp.asarray(y_test))


def cross_entropy_loss(logits, labels, num_classes):
    y = jax.nn.one_hot(labels, num_classes=num_classes)
    return optax.softmax_cross_entropy(logits=logits, labels=y).mean()


def compute_metrics(logits, labels, num_classes):
    return {
        "loss": cross_entropy_loss(logits, labels, num_classes),
        "accuracy": jnp.mean(jnp.argmax(logits, -1) == labels),
    }


def create_train_state(rng, n_qubits, depth, lr, feature_shape, output_dim):
    qcl = QCL(n_qubits=n_qubits, depth=depth, output_dim=output_dim)
    params = qcl.init(rng, jnp.ones((1, *feature_shape)))["params"]

    tx = optax.adam(learning_rate=lr)
    return train_state.TrainState.create(apply_fn=qcl.apply, params=params, tx=tx)


@functools.partial(jax.jit, static_argnums=(3,4,5))
def train_step(state, x, y, n_qubits, depth, output_dim):
    def loss_fn(params):
        logits = QCL(n_qubits=n_qubits,
                     depth=depth,
                     output_dim=output_dim).apply({"params": params}, x)
        loss = cross_entropy_loss(logits, y, output_dim)
        return loss, logits

    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, logits = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=y, num_classes=output_dim)
    return state, metrics


@functools.partial(jax.jit, static_argnums=(3,4,5))
def eval_step(params, x, y, n_qubits, depth, output_dim):
    logits = QCL(n_qubits=n_qubits,
                 depth=depth,
                 output_dim=output_dim).apply({"params": params}, x)
    return compute_metrics(logits=logits, labels=y, num_classes=output_dim)


def main():
    # Circuit
    n_qubits = 4
    depth = 2

    # Learning
    lr = 0.05
    epochs = 100

    # Data
    num_classes = 3
    x_train, y_train, x_test, y_test = load_data()

    rng = jax.random.PRNGKey(0)
    rng, rng_apply = jax.random.split(rng)

    state = create_train_state(rng_apply,
                               n_qubits=n_qubits, depth=depth, lr=lr,
                               feature_shape=x_train.shape[1:],
                               output_dim=num_classes)


    for e in range(epochs):
        # Notes: Data size is enough small, we don't divide data to mini-batch
        t = time.perf_counter()

        state, metrics = train_step(state, x_train, y_train,
                                    n_qubits=n_qubits, depth=depth,
                                    output_dim=num_classes)
        print(f"Epoch: {e:2d} [Train] Loss: {metrics['loss']:.4f}, " +
              f"Accuracy: {metrics['accuracy'] * 100:.2f}%", end=" ")

        metrics = eval_step(state.params, x_test, y_test,
                            n_qubits=n_qubits, depth=depth,
                            output_dim=num_classes)
        print(f"[Eval] Loss: {metrics['loss']:.4f}, " +
              f"Accuracy: {metrics['accuracy'] * 100:.2f}%", end=" ")

        print(f"[Time] Elapsed: {time.perf_counter() - t:.4f}s")


if __name__ == "__main__":
    main()
