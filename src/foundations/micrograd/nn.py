"""Scaffolding for neural networks built on micrograd Values."""

import random

from .engine import Value


class Neuron:
    """A single neuron: a weighted sum of inputs plus bias, passed through tanh.

    Args:
        nin: Number of inputs the neuron accepts.
        step_size: Learning rate used by `update`.

    Attributes:
        w: One weight per input, initialized uniformly in [-1, 1].
        b: The bias, initialized uniformly in [-1, 1].
    """

    def __init__(self, nin, step_size):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self._step_size = step_size

    def __call__(self, x):
        """Compute the neuron's activation, tanh(w · x + b).

        Args:
            x: Sequence of `nin` inputs (Values or numbers).

        Returns:
            The activation as a Value.
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()

        return out

    def _parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        """Reset the gradients of the weights and bias to zero."""
        params = self._parameters()

        for p in params:
            p.zero_grad()

    def update(self):
        """Take one gradient descent step on the weights and bias."""
        params = self._parameters()

        for p in params:
            p.data += -1 * self._step_size * p.grad


class Layer:
    """A fully connected layer of independent neurons sharing the same inputs.

    Args:
        nin: Number of inputs to each neuron.
        nout: Number of neurons, i.e. the layer's output size.
        step_size: Learning rate passed to each neuron.

    Attributes:
        neurons: The layer's `nout` neurons.
    """

    def __init__(self, nin, nout, step_size):
        self.neurons = [Neuron(nin, step_size) for _ in range(nout)]

    def __call__(self, x):
        """Run every neuron on the same input.

        Args:
            x: Sequence of `nin` inputs (Values or numbers).

        Returns:
            A list of `nout` Values, one per neuron, or a single Value if
            `nout` is 1.
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def zero_grad(self):
        """Reset the gradients of every neuron in the layer."""
        for neuron in self.neurons:
            neuron.zero_grad()

    def update(self):
        """Take one gradient descent step on every neuron in the layer."""
        for neuron in self.neurons:
            neuron.update()


class MLP:
    """A multi-layer perceptron: layers applied in sequence.

    Args:
        nin: Number of inputs to the network.
        nouts: Output size of each layer, in order. The last entry is the
            network's output size, e.g. `[4, 4, 1]` for two hidden layers of 4.
        step_size: Learning rate passed to every layer.

    Attributes:
        layers: The network's layers, one per entry in `nouts`.
    """

    def __init__(self, nin, nouts, step_size):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], step_size) for i in range(len(nouts))]

    def __call__(self, x):
        """Run a forward pass through every layer.

        Args:
            x: Sequence of `nin` inputs (Values or numbers).

        Returns:
            A list of Values with length `nouts[-1]`, or a single Value if
            `nouts[-1]` is 1.
        """
        out = x
        for layer in self.layers:
            out = layer(out)

        return out

    def zero_grad(self):
        """Reset the gradients of every parameter in the network."""
        for layer in self.layers:
            layer.zero_grad()

    def update(self):
        """Take one gradient descent step on every parameter in the network."""
        for layer in self.layers:
            layer.update()
