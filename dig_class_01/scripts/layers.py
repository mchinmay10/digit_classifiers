import time
from vector import vector_length, dot


class Neuron:

    def __init__(self, weights: list[float], bias: float):
        self.weights = weights
        self.bias = bias

    def forward(self, x):
        dot_product = dot(self.weights, x)
        if dot_product:
            return dot_product + self.bias
        else:
            return f"Invalid input {x}"


def forward_test():
    print(f"Executing test cases for forward function...")
    time.sleep(3)
    n1 = Neuron([2, 1], 3)
    print(f"forward([5, 4]) = {n1.forward([5, 4])}")


if __name__ == "__main__":
    print(f"----Running test cases for the layers.py file----")
    forward_test()
