# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: ml-infra-foundations
#     language: python
#     name: ml-infra-foundations
# ---

# %% [markdown]
# # Micrograd from scratch
#
# Learning project on backpropagation by implementing it from scratch. This is a follow-along implementation of Andrej Karpathy's lecture on neural networks ([Link](https://www.youtube.com/watch?v=VMj-3S1tku0))
#
# ## What's being implemented?
# - A tiny scalar auto gradient engine
# - Small neural network trained using it
# %%
import math
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# %% [markdown]
# ## Functions and derivatives
#
# %%
def f(x):
    return 3*x**2 - 4*x + 5
# %%
f(3.0)
# %%
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
# %%
h = 0.00000001
x = -2.0
(f(x + h) - f(x))/h
# %% [markdown]
# ### Derivatives across multiple variables
#

# %%
a = 2.0
b = -3.0
c = 10.0

# %%
h = 0.0001

d1 = a*b + c

c += h
d2 = a*b + c

print("d1", d1)
print("d2", d2)
print("slope", (d2 - d1)/h)


# %% [markdown]
# ## Components of micrograd
#

# %%
class Value:

    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data * other.data, (self,other))

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,))

        def _backward():
            self.grad += out.grad * other * (self.data**(other - 1))
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
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
        


# %%
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

# %%
d = a*b + c

# %%
# Skipping part of graph viz and deriving the logic/intuition for back propagation algorithm
# Skipping part on tanh as activation and computing the gradients of an expression using tanh
# Skipping section on pytorch

# %%
d.backward()

# %%
print('grad(d)', d.grad)
print('grad(c)', c.grad)
print('grad(b)', b.grad)
print('grad(a)', a.grad)

# %%
a = Value(2.0)
b = Value(5.0)
e = a / b  
e.backward()

print('e', e.data)
print('grad(e)', e.grad)
print('grad(b)', b.grad)
print('grad(a)', a.grad)


# %% [markdown]
# ## Building Neural Nets
#

# %%
class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        out - act.

# %%
