import time
import random
from vector import dot
from activations import step, sigmoid, relu
from losses import squared_error


# A Neuron is the basic computation unit of a Neural Network
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


# Denselayer is a fully connected layer of neurons
class DenseLayer_v1:

    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    def layer_forward(self, x):
        output = []
        for neuron in self.neurons:
            output.append(neuron.forward(x))

        return output


# A more robust version of the Denselayer class
class DenseLayer_v2:

    def __init__(
        self,
        num_neurons: int,
        weight_matrix: list[list[float]],
        bias: float,
    ):
        self.num_neurons = num_neurons
        self.weight_matrix = weight_matrix
        self.bias = bias

        self.neurons: list[Neuron] = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(weight_matrix[i], bias))

    def layer_forward(self, x):
        output = []
        for neuron in self.neurons:
            output.append(neuron.forward(x))

        return output

    def layer_forward_debug(self, x):
        print(f"Input vector: {x}")
        for i in range(self.num_neurons):
            print(f"----For Neuron {i+1}----")
            print(f"Weights: {self.weight_matrix[i]}")
            print(f"Weighted Sum: {dot(x, self.weight_matrix[i])}")
        print(f"Value of Bias: {self.bias}")
        activ_output = self.layer_forward(x)
        print(f"Activation output: {activ_output}")

        return activ_output


# function to predict the output of the neural network.
# we use the logic that index of neuron with the highest probability represents the predicted digit
def predict_digit(output):
    out_index = -1
    max_prob = 0
    for index, prob in enumerate(output):
        if prob > max_prob:
            max_prob = prob
            out_index = index

    return out_index


# function to perform a complete forward pass
def simulate_fwd_pass():
    print("Executing complete forward pass...")
    time.sleep(3)
    input = [round(random.random(), 2) for _ in range(2)]
    target = 1
    num_neurons = 1
    weights_per_neurons = len(input)
    weight_matrix = [
        [0.5 for _ in range(weights_per_neurons)] for _ in range(num_neurons)
    ]
    bias = 0
    l = DenseLayer_v2(num_neurons, weight_matrix, bias)
    print("----Printing verbose output of Dense Layer v2 forward pass----")
    time.sleep(2)
    activ_output = l.layer_forward_debug(input)
    print(f"----Printing loss for given taget: {target}----")
    time.sleep(2)
    print(f"Loss: {round(squared_error(target, activ_output[0]), 3)}")


# Test cases:
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


def dense_layer_v1_forward_test():
    print(f"Executing test cases for forward function of dense layer v1 class...")
    time.sleep(3)
    input = [0.17, 0.62]
    b = round(random.random(), 2)
    n_list = []
    print(f"Input: {input}")
    for i in range(3):
        w_i = [round(random.random(), 2) for _ in range(2)]
        n_list.append(Neuron(w_i, b))
    l = DenseLayer_v1(n_list)
    output = l.layer_forward(input)
    print("----Printing output of Dense Layer 'l'----")
    time.sleep(2)
    print(f"Layer output: {[round(x, 2) for x in output]}")


def dense_layer_v2_forward_test():
    print(
        f"Executing test cases for forward function of dense layer version 2 class..."
    )
    time.sleep(3)
    input = [0.17, 0.62]
    b = round(random.random(), 2)
    print(f"Input: {input}")
    num_neurons = 3
    weights_per_neuron = len(input)
    weight_matrix = [
        [round(random.random(), 2) for _ in range(weights_per_neuron)]
        for _ in range(num_neurons)
    ]
    l = DenseLayer_v2(num_neurons, weight_matrix, b)
    output = l.layer_forward(input)
    print("----Printing output of Dense Layer v2 l----")
    time.sleep(2)
    print(f"Layer output: {[round(x, 2) for x in output]}")


def predict_digit_test():
    print(f"Executing test cases for predict function...")
    time.sleep(3)
    output = [round(random.random(), 2) for _ in range(10)]
    print(f"Predicted digit = {predict_digit(output)}")


if __name__ == "__main__":
    print(f"----Running test cases for the layers.py file----")
    # neuron_forward_test()
    # ten_neuron_fwd_test()
    # dense_layer_v1_forward_test()
    # dense_layer_v2_forward_test()
    # predict_digit_test()
    simulate_fwd_pass()
