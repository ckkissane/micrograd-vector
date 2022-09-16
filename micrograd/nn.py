from engine import Vector
import random


class Module:
    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, nin: int, nout: int):
        self.nin, self.nout = nin, nout
        self.weight = [
            Vector([random.uniform(-1, 1) for _ in range(nout)]) for _ in range(nin)
        ]
        self.bias = Vector([0.0 for _ in range(nout)])

    def __call__(self, x: Vector):
        xW = x.matmul(self.weight)
        y = xW + self.bias
        return y

    def __repr__(self):
        return f"Linear(nin={self.nin}, nout={self.nout})"

    def parameters(self):
        return self.weight + [self.bias]


class Sigmoid(Module):
    def __call__(self, x: Vector):
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"
