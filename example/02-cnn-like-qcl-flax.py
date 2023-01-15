"""
Example 02: CNN-like QCL classification with Flax

This example additionally requires followings;
* Flax: https://flax.readthedocs.io/en/latest/index.html
* Optax: https://optax.readthedocs.io/en/latest/
* scikit-learn: https://scikit-learn.org/stable/

Warnings
--------
This implementation is different from QCNN[1],
because the intermediate measurement and reaction are not easy for diffq simulation,
and because this example implementation needs smaller qubits and shallow circuit.

[1] I. Cong et al., "Quantum Convolutional Neural Networks",
    Nature Phys. 15 1273-1278 (2019)
    https://doi.org/10.1038/s41567-019-0648-8
    https://arxiv.org/abs/1810.03787
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

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from tqdm import tqdm


def conv3x3cell(x, w):
    x = jnp.reshape(x, (9,))

    q = op.zeros(9, jnp.complex64)
    for i in range(9):
        q = op.RY(q, (i,), x.at[i].get())

    for k in range(w.shape[0]):
        for i in range(8):
            q = op.CNOT(q, (i, i+1))
        for i in range(9):
            q = op.RY(q, (i,), w.at[k, i].get())

    return op.expectZ(q, (0,))


def ConvLayer(x, w):
    # [0, 1) -> [-pi/2, pi/2)
    x = jnp.arcsin(2 *(x - 0.5))

    # convolution
    F = diffq.nn.Convolution(op, conv3x3cell,
                             kernel_shape = (3, 3),
                             slide = (1, 1),
                             padding = (1, 1))
    x = F(x, w)

    # pooling
    x = diffq.nn.MaxPooling(x, (2, 2))

    return x

def DenseLayer(x, w):
    x = jnp.reshape(x, (-1,))

    q = op.zeros(x.shape[0], jnp.complex64)
    for i in range(x.shape[0]):
        q = op.RY(q, (i,), jnp.arcsin(x.at[i].get()))
        q = op.RZ(q, (i,), jnp.arccos(x.at[i].get() ** 2))

    for k in range(w.shape[0]):
        for i in range(x.shape[0]-1):
            q = op.CNOT(q, (i, i+1))
        for i in range(x.shape[0]):
            q = op.RY(q, (i,), w.at[k,i].get())

    for i in range(x.shape[0]):
        q = op.PauliZ(q, (i,))

    p = diffq.prob(op.to_state(q))
    return jnp.stack(tuple(diffq.expval(p, i) for i in range(x.shape[0])))


class ConvQCL(nn.Module):
    cdepth: int
    ddepth: int
    output_dim: int
    circuit_init: Callable = nn.initializers.uniform(2 * jnp.pi)

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        wc = self.param("conv", self.circuit_init, (self.cdepth, 9))
        wd = self.param("dense", self.circuit_init, (self.ddepth, 16))

        @jax.vmap
        def batch(xi):
            # 8 x 8 -> 4 x 4
            xi = ConvLayer(xi, wc)

            xi = DenseLayer(xi, wd)
            return xi

        x = batch(x)
        x = nn.Dense(self.output_dim)(x)
        return x

def load_data():
    features, labels = load_digits(return_X_y=True)
    features_std = features.reshape((-1, 8, 8)) / 256.0
    x_train, x_test, y_train, y_test = train_test_split(features_std, labels,
                                                        random_state=42, shuffle=True)
    return (jnp.asarray(x_train), jnp.asarray(y_train),
            jnp.asarray(x_test) , jnp.asarray(y_test))


def cross_entropy_loss(logits, labels, num_classes):
    y = jax.nn.one_hot(labels, num_classes)
    return optax.softmax_cross_entropy(logits=logits, labels=y).mean()


def compute_metrics(logits, labels, num_classes):
    return {
        "loss": cross_entropy_loss(logits, labels, num_classes),
        "accuracy": jnp.mean(jnp.argmax(logits, -1) == labels),
    }


def create_train_state(rng, cdepth, ddepth, lr, input_shape, output_dim):
    qcl = ConvQCL(cdepth=cdepth, ddepth=ddepth, output_dim=output_dim)
    params = qcl.init(rng, jnp.ones(input_shape))["params"]

    tx = optax.adam(learning_rate=lr)
    return train_state.TrainState.create(apply_fn=qcl.apply, params=params, tx=tx)


@functools.partial(jax.jit, static_argnums=(3,4,5))
def train_step(state, x, y, cdepth, ddepth, output_dim):
    def loss_fn(params):
        logits = ConvQCL(cdepth=cdepth,
                         ddepth=ddepth,
                         output_dim=output_dim).apply({"params": params}, x)
        loss = cross_entropy_loss(logits, y, output_dim)
        return loss, logits

    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, logits = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=y, num_classes=output_dim)
    return state, metrics


@functools.partial(jax.jit, static_argnums=(3,4,5))
def eval_step(params, x, y, cdepth, ddepth, output_dim):
    logits = ConvQCL(cdepth=cdepth,
                     ddepth=ddepth,
                     output_dim=output_dim).apply({"params": params}, x)
    return compute_metrics(logits=logits, labels=y, num_classes=output_dim)


def main():
    # Circuit
    cdepth = 2
    ddepth = 2

    # Learning
    lr = 0.05
    epochs = 50
    batch_size = 16

    # Data
    num_classes = 10
    x_train, y_train, x_test, y_test = load_data()

    rng = jax.random.PRNGKey(0)
    rng, rng_apply = jax.random.split(rng)

    state = create_train_state(rng_apply,
                               cdepth=cdepth, ddepth=ddepth, lr=lr,
                               input_shape=(batch_size, *x_train.shape[1:]),
                               output_dim=num_classes)


    train_ds_size = len(x_train)
    steps_per_epoch = train_ds_size // batch_size
    for e in range(epochs):
        t = time.perf_counter()

        rng, rng_apply = jax.random.split(rng)
        perms = jax.random.permutation(rng_apply, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))

        batch_metrics = []
        for perm_idx in tqdm(perms, ascii=True):
            state, metrics = train_step(state,
                                        x_train.at[perm_idx].get(),
                                        y_train.at[perm_idx].get(),
                                        cdepth=cdepth, ddepth=ddepth,
                                        output_dim=num_classes)
            batch_metrics.append(metrics)

        loss = jnp.mean(jnp.asarray(tuple(m["loss"] for m in batch_metrics)))
        acc = jnp.mean(jnp.asarray(tuple(m["accuracy"] for m in batch_metrics)))
        print(f"Epoch: {e:2d} [Train] Loss: {loss:.4f}, " +
              f"Accuracy: {acc * 100:.2f}%", end=" ")

        metrics = eval_step(state.params, x_test, y_test,
                            cdepth=cdepth, ddepth=ddepth,
                            output_dim=num_classes)
        print(f"[Eval] Loss: {metrics['loss']:.4f}, " +
              f"Accuracy: {metrics['accuracy'] * 100:.2f}%", end=" ")

        print(f"[Time] Elapsed: {time.perf_counter() - t:.4f}s")


if __name__ == "__main__":
    main()
