from engine import Vector
import math


def binary_cross_entropy(pred: Vector, y: float):
    return -(y * pred.log() + (1.0 - y) * (1.0 - pred).log())


def cross_entropy(input: Vector, target: int):
    eps = 1e-6
    softmax_num = [math.exp(d_i + eps) for d_i in input.data]
    softmax_denom = sum(softmax_num)
    probs = [num / softmax_denom for num in softmax_num]
    p = probs[target]

    out = Vector([-math.log(p + eps)], (input,), "cross_entropy")

    def _backward():
        for i in range(len(input.grad)):
            input.grad[i] += (probs[i] - (1 if i == target else 0)) * out.grad[0]

    out._backward = _backward

    return out
