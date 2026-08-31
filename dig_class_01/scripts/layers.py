import time
import random
from vector import dot
from activations import identity_single, sigmoid
from visual import function_header, load_print, border_print_v1


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

    def forward(self, x) -> float | str:
        dot_product = dot(self.weights, x)
        if dot_product:
            return self.activation(dot_product + self.bias)
        else:
            return f"Invalid input {x}"


# Designing the second version of a Neuron
class Neuron_v2:

    def __init__(
        self,
        weights: list[float],
        bias: float,
    ):
        self.weights = weights
        self.bias = bias
        self.activation = identity_single

        self.fwd = 0
        self.bwd = 0

    def forward(
        self,
        x: list[float],
    ) -> None | str:

        self.fwd = 1
        self.x = x
        self.z = dot(self.weights, x)
        if self.z:
            self.a = self.activation(self.z + self.bias)
            self.loss = self.a**2
        else:
            return f"Invalid input {x}"

    def backprop(self):
        if self.fwd == 0:
            print("Please peform forward pass first!")
        else:
            self.bwd = 1
            self.dloss_da = 2 * self.a
            self.da_dz = 1
            self.da_db = 1
            self.dz_dw = self.x.copy()
            self.dloss_db = self.dloss_da * self.da_db
            self.dloss_dw = []
            for der in self.dz_dw:
                self.dloss_dw.append(self.dloss_da * self.da_dz * der)

    def view_props(self):
        if self.fwd == 0:
            print("Please perform forward pass first!")
        else:
            border_print_v1("Forward propagation:")
            load_print(f"x:       {self.x}")
            load_print(f"weights: {self.weights}")
            load_print(f"bias:    {self.bias}")
            load_print(f"z:       {self.z}")
            load_print(f"a:       {self.a}")
            load_print(f"Loss:    {self.loss}")
            if self.bwd == 0:
                print("Please perform backward pass first!")
            else:
                border_print_v1("Backward propagation:")
                load_print(f"Δloss / Δa = {self.dloss_da}")
                load_print(f"Δa / Δz =    {self.da_dz}")
                load_print(f"Δz / Δw =    {self.dz_dw}")
                load_print("Δloss / Δw =  (Δz / Δw) * (Δa / Δz) * (Δloss / Δa)")
                load_print(f"Δloss / Δw = {self.dloss_dw}")
                load_print(f"Δloss / Δb = {self.dloss_db}")


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
        activ_output = []
        for neuron in self.neurons:
            activ_output.append(neuron.forward(x))

        return activ_output

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


# Third version of Denselayer class including back propagation
class DenseLayer_v3:

    def __init__(
        self,
        size: int,
        weights: list[list[float]],
        bias: float,
    ):
        self.size = size
        self.weights = weights
        self.bias = bias

        self.neurons: list[Neuron_v2] = []
        for i in range(size):
            self.neurons.append(Neuron_v2(weights[i], bias))

    def forward(self, x: list[float]):
        for neuron in self.neurons:
            neuron.forward(x)

    def layer_output(self):
        border_print_v1("Layer Output:")
        for neuron in self.neurons:
            load_print(f"{neuron.a}")

    def backprop(self):
        for neuron in self.neurons:
            neuron.backprop()


def compare_with_numerical_gradient():
    function_header(
        "Executing comparision for comparing analytical and numerical gradients"
    )
    border_print_v1("Analytical Gradients:")
    n = Neuron_v2([3], 1)
    n.forward([2])
    n.backprop()
    n.view_props()
    from backprop import chain_rule_discovery

    border_print_v1("Numerical Gradients")
    chain_rule_discovery()


# Test cases:
def neuron_forward_test():
    print(f"Executing test cases for forward function of neuron class...")
    time.sleep(3)
    n1 = Neuron([2, 1], 3)
    print(f"forward([5, 4]) = {n1.forward([5, 4])}")


def neuron_v2_forward_test():
    function_header("Executing test cases for forward function of neuron v2 class")
    n1 = Neuron_v2([1, 2, 3], 5)
    n1.forward([2, 3, 4])
    load_print(f"calculating forward([2, 3, 4])...")
    n1.backprop()
    load_print(f"performing back prop...")
    n1.view_props()


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


def dense_layer_v3_forward_test():
    function_header(
        "Executing test cases for the forward function of dense layer version 3 class..."
    )
    weights = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    bias = 0
    x = [7.0, 8.0, 9.0]
    l = DenseLayer_v3(2, weights, bias)
    l.forward(x)
    l.layer_output()


if __name__ == "__main__":
    print(f"----Running test cases for the layers.py file----")
    # neuron_forward_test()
    # ten_neuron_fwd_test()
    # dense_layer_v1_forward_test()
    # dense_layer_v2_forward_test()
    # neuron_v2_forward_test()
    # compare_with_numerical_gradient()
    dense_layer_v3_forward_test()
