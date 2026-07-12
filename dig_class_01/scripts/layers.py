import time
import random
from vector import dot
from activations import step, sigmoid, relu


class Neuron:

    def __init__(
        self,
        weights: list[float],
        bias: float,
        activation=sigmoid,
    ):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, x):
        dot_product = dot(self.weights, x)
        if dot_product:
            return self.activation(dot_product + self.bias)
        else:
            return f"Invalid input {x}"


class DenseLayer:

    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    def layer_forward(self, x):
        output = []
        for neuron in self.neurons:
            output.append(neuron.forward(x))

        return output


def neuron_forward_test():
    print(f"Executing test cases for forward function of neuron class...")
    time.sleep(3)
    n1 = Neuron([2, 1], 3)
    print(f"forward([5, 4]) = {n1.forward([5, 4])}")


def ten_neuron_fwd_test():
    print("Executing test cases for ten neurons accepting the same input...")
    time.sleep(3)
    input = [0.32, 0.55]
    print(f"Input: {input}")
    for i in range(10):
        w_i = [round(random.random(), 2) for _ in range(2)]
        b_i = round(random.random(), 2)
        n_i = Neuron(w_i, b_i)
        f_i = n_i.forward(input)
        print(f"----Printing details of Neuron {i + 1}----")
        time.sleep(2)
        print(f"Weights: {w_i}")
        print(f"Bias: {b_i}")
        print(f"Forward pass: {f_i:.2f}")


def dense_layer_forward_test():
    print(f"Executing test cases for forward function of dense layer class...")
    time.sleep(3)
    input = [0.17, 0.62]
    b = round(random.random(), 2)
    n_list = []
    print(f"Input: {input}")
    for i in range(3):
        w_i = [round(random.random(), 2) for _ in range(2)]
        n_list.append(Neuron(w_i, b))
    l = DenseLayer(n_list)
    output = l.layer_forward(input)
    print("----Printing output of Dense Layer 'l'----")
    time.sleep(2)
    print(f"Layer output: {[round(x, 2) for x in output]}")


if __name__ == "__main__":
    print(f"----Running test cases for the layers.py file----")
    neuron_forward_test()
    ten_neuron_fwd_test()
    dense_layer_forward_test()
