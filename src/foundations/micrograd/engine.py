"""Micrograd - Scalar autograd engine."""

import math


class Value:
    """An autograd wrapper for scalar values.

    Value wraps a scalar, records how it was computed and calculates the gradient based
    on the record. Operations on Values build a computation graph; calling `backward()`
    on the final Value fills in `grad` for every Value in the graph.

    Args:
        data: The scalar value.
        _children: The Values this one was computed from. Set internally by
            operations; leave empty when creating a Value directly.

    Attributes:
        data: The scalar value.
        grad: Derivative of the final Value with respect to the current value.
            Starts at 0.0 and accumulates across `backward()` calls, so reset it
            before each new pass.
    """

    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def _accumulate_grad(self, value):
        """Add `value` to `grad`, summing contributions from every use of this Value."""
        self.grad += value

    def __add__(self, other):
        """Add two Values, recording the operation for backpropagation.

        Args:
            other: The Value or number to add. Numbers are wrapped in a Value.

        Returns:
            A new Value holding `self.data + other.data`.
        """
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other))
        out._backward = lambda: (
            self._accumulate_grad(out.grad),
            other._accumulate_grad(out.grad),
        )
        return out

    def __radd__(self, other):
        """Handle `number + Value` by delegating to `__add__` (addition commutes)."""
        return self + other

    def __mul__(self, other):
        """Multiply two Values, recording the operation for backpropagation.

        Args:
            other: The Value or number to multiply by. Numbers are wrapped in a Value.

        Returns:
            A new Value holding `self.data * other.data`.
        """
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other))
        out._backward = lambda: (
            self._accumulate_grad(other.data * out.grad),
            other._accumulate_grad(self.data * out.grad),
        )
        return out

    def __rmul__(self, other):
        """Handle `number * Value` by delegating to `__mul__` (it commutes)."""
        return self * other

    def __pow__(self, other):
        """Raise this Value to a constant power, recording it for backpropagation.

        Args:
            other: The exponent. Must be an int or float; Value exponents are not
                supported.

        Returns:
            A new Value holding `self.data ** other`.

        Raises:
            AssertionError: If `other` is not an int or float.
        """
        assert isinstance(other, (int, float))

        out = Value(self.data**other, (self,))
        out._backward = lambda: self._accumulate_grad(
            other * (self.data ** (other - 1)) * out.grad
        )

        return out

    def __truediv__(self, other):
        """Divide by a Value or number, computed as `self * other**-1`."""
        return self * other**-1

    def __rtruediv__(self, other):
        """Handle `number / Value`, computed as `other * self**-1`."""
        return other * self**-1

    def __neg__(self):
        """Negate this Value, computed as `self * -1`."""
        return self * -1

    def __sub__(self, other):
        """Subtract a Value or number, computed as `self + (-other)`."""
        return self + (-other)

    def __rsub__(self, other):
        """Handle `number - Value`, computed as `other + (-self)`."""
        return other + (-self)

    def exp(self):
        """Compute e raised to this Value, recording it for backpropagation.

        Returns:
            A new Value holding `math.exp(self.data)`.
        """
        out = Value(math.exp(self.data), (self,))
        out._backward = lambda: self._accumulate_grad(out.grad * out.data)

        return out

    def tanh(self):
        """Compute the tanh of this Value, recording it for backpropagation.

        Returns:
            A new Value holding `tanh(self.data)`.
        """
        x = self.data
        th = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(th, (self,))
        out._backward = lambda: self._accumulate_grad(out.grad * (1 - th**2))

        return out

    def zero_grad(self):
        """Reset gradient of this Value.

        Utility step requried in NN training.
        """
        self.grad = 0.0

    def backward(self):
        """Backpropagate from this Value, filling in `grad` across its whole graph."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
