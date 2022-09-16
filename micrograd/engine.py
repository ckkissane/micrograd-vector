from typing import List
import math


class Vector:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = [0.0 for _ in range(len(self.data))]
        self._backward = lambda: None
        self._prev = set(_children)

    def __repr__(self):
        return f"Vector(data={self.data})"

    def __add__(self, other):
        other = (
            other
            if isinstance(other, Vector)
            else Vector([other for _ in range(len(self.data))])
        )
        out = Vector(
            [si + oi for si, oi in zip(self.data, other.data)], (self, other), "+"
        )

        def _backward():
            for i in range(len(self.grad)):
                self.grad[i] += out.grad[i]
                other.grad[i] += out.grad[i]

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = (
            other
            if isinstance(other, Vector)
            else Vector([other for _ in range(len(self.data))])
        )
        out = Vector(
            [si * oi for si, oi in zip(self.data, other.data)], (self, other), "*"
        )

        def _backward():
            for i in range(len(self.grad)):
                self.grad[i] += other.data[i] * out.grad[i]
                other.grad[i] += self.data[i] * out.grad[i]

        out._backward = _backward

        return out

    def sigmoid(self):
        def sig_helper(x: float):
            return 1.0 / (1.0 + math.exp(-x))

        out = Vector([sig_helper(x) for x in self.data], (self,), "sigmoid")

        def _backward():
            for i in range(len(self.grad)):
                self.grad[i] += out.data[i] * (1 - out.data[i]) * out.grad[i]

        out._backward = _backward

        return out

    def matmul(self, other):
        def mm(A: List[List[float]], B: List[List[float]]):
            assert len(A[0]) == len(B)
            out = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
            for i in range(len(out)):
                for j in range(len(out[0])):
                    for k in range(len(B)):
                        out[i][j] += A[i][k] * B[k][j]
            return out

        def transpose(A: List[List[float]]):
            out = [[0.0 for _ in range(len(A))] for _ in range(len(A[0]))]
            for i in range(len(out)):
                for j in range(len(out[0])):
                    out[i][j] = A[j][i]
            return out

        # other must be List[Vector]
        other_data = [p.data for p in other]
        out = Vector(
            mm([self.data], other_data)[0], (self, *[p for p in other]), "matmul"
        )

        def _backward():
            self_grad = mm([out.grad], transpose(other_data))
            other_grad = mm(transpose([self.data]), [out.grad])
            for i in range(len(self.grad)):
                self.grad[i] += self_grad[0][i]
            for i in range(len(other)):
                for j in range(len(other[i].grad)):
                    other[i].grad[j] += other_grad[i][j]

        out._backward = _backward

        return out

    def log(self):
        eps = 1e-6
        out = Vector([math.log(di + eps) for di in self.data], (self,), "log")

        def _backward():
            for i in range(len(self.grad)):
                self.grad[i] += (1.0 / self.data[i]) * out.grad[i]

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

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

        self.grad = [1.0 for _ in range(len(self.data))]
        for node in reversed(topo):
            node._backward()
