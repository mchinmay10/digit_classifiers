import time
import random
from vector import dot
from activations import step, sigmoid, relu
from losses import mean_squared_error
from visual import clear_screen


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
    nw_input = [1, 1]
    target = [1]
    num_neurons = 1
    weights_per_neurons = len(nw_input)
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
    print(f"Loss: {round(mean_squared_error(target, activ_output), 3)}")


# outputs the current state of the system and return weights and biases
def current_state_of_nw(nw_input, num_neurons, weight_matrix, bias, target):
    l = DenseLayer_v2(num_neurons, weight_matrix, bias)
    time.sleep(2)
    active_output = l.layer_forward_debug(nw_input)
    loss = round(mean_squared_error(target, active_output), 3)
    print(f"Loss: {loss}")
    return loss


# function to manual optimize the weights and biases of a layer to minimise loss value
def manual_optimisation():
    print("----Initialising Manual Optimisation---")
    time.sleep(1)
    num_neurons = int(input("Please enter number of neurons: "))
    CONTINUE = 1
    nw_input = [1, 1]
    input_len = len(nw_input)
    weight_matrix = [[0.5 for _ in range(input_len)] for _ in range(num_neurons)]
    bias = 0
    target = [1]
    while CONTINUE == 1:
        clear_screen()
        loss = current_state_of_nw(nw_input, num_neurons, weight_matrix, bias, target)
        CONTINUE = int(input("Continue manual optimisation? [0/1]: "))
        if CONTINUE == 1:
            print("Update weight matrix...")
            time.sleep(1)
            for i in range(num_neurons):
                for j in range(input_len):
                    weight_matrix[i][j] = float(
                        input(f"Value of w at: row {i} and column {j}: ")
                    )
            print("Update bias...")
            time.sleep(1)
            bias = get_bias_from_user()
        else:
            print(f"Quitting manual optimisation...")
            time.sleep(1)


# helper function that temperorily changes one weight, computes loss and resores the original weight
def temp_weight_change(
    nw_input: list, layer: DenseLayer_v2, index: tuple, delta: float, target: list
):
    i, j = index
    loss_list = []
    activ_output = layer.layer_forward(nw_input)
    loss = mean_squared_error(target, activ_output)
    loss_list.append(loss)
    print(f"Inital / Previous Loss: {loss}")
    time.sleep(2)
    print(f"Temporarily changing weight at index: {index} by {delta}")
    save_weight = layer.weight_matrix[i][j]
    new_weight = save_weight + delta
    layer.weight_matrix[i][j] = new_weight
    time.sleep(2)
    print(f"Value of old weight: {save_weight}")
    print(f"Value of new weight: {new_weight}")
    activ_output = layer.layer_forward(nw_input)
    loss = mean_squared_error(target, activ_output)
    loss_list.append(loss)
    time.sleep(2)
    print(f"Loss: {loss}")
    time.sleep(2)
    if loss_list[0] > loss_list[1]:
        print(f"Loss decreased from {loss_list[0]} -> {loss_list[1]}")
    elif loss_list[0] < loss_list[1]:
        print(f"Loss increased from {loss_list[0]} -> {loss_list[1]}")
    else:
        print("Loss unchanged")
    time.sleep(2)
    print("Restoring weight...")
    time.sleep(2)
    layer.weight_matrix[i][j] = save_weight


# function to recieve weights from user (mainly for manual optimisation and initialisation purpose)
def get_weights_from_user(num_neurons, input_len):
    weight_matrix = [[0.0 for _ in range(input_len)] for _ in range(num_neurons)]
    print("Enter weight matrix")
    time.sleep(3)
    for i in range(num_neurons):
        for j in range(input_len):
            weight_matrix[i][j] = float(
                input(f"Value of w at: row {i} and column {j}: ")
            )

    return weight_matrix


# function to recieve bias term from user
def get_bias_from_user():
    bias = float(input("Enter bias term: "))
    return bias


# function that gets dummy target from user for purpose of manual simulation of learning process
def get_dummy_targets_from_user(n):
    targets = []
    for i in range(n):
        target = float(input(f"Enter target {i + 1}: "))
        targets.append(target)

    return targets


# function to get index of the weight to be changed temporarily
# to add bounding check on the row and column that we accept from user
def get_weight_index_from_user():
    weight_index = []
    for i in range(2):
        if i == 0:
            row = int(input("Enter row index: "))
            weight_index.append(row)
        else:
            col = int(input("Enter column index: "))
            weight_index.append(col)

    return tuple(weight_index)


# get input to network from user
def get_nw_input_from_user():
    nw_input = []
    n = int(input("Enter number of inputs: "))
    for i in range(n):
        single_input = float(input(f"Enter input {i + 1}: "))
        nw_input.append(single_input)

    return nw_input


# function that gracefully runs the temporary weight change simulation
def simulate_temp_weight_change():
    nw_input = get_nw_input_from_user()
    time.sleep(2)
    num_neurons = int(input("Enter number of neurons in the layer: "))
    time.sleep(2)
    weigth_matrix = get_weights_from_user(num_neurons, len(nw_input))
    time.sleep(2)
    bias = get_bias_from_user()
    time.sleep(2)
    l = DenseLayer_v2(
        num_neurons,
        weigth_matrix,
        bias,
    )
    target = get_dummy_targets_from_user(num_neurons)
    time.sleep(2)
    CONTINUE = 1
    while CONTINUE == 1:
        CONTINUE = int(input("Continue manual temperory weight change? [0/1]: "))
        if CONTINUE == 1:
            index = get_weight_index_from_user()
            time.sleep(2)
            delta = float(input("Enter delta for weight change: "))
            time.sleep(2)
            temp_weight_change(nw_input, l, index, delta, target)
            clear_screen()
        else:
            print("Quitting weight temperory change simulation...")
            time.sleep(2)


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
    # simulate_fwd_pass()
    # manual_optimisation()
    simulate_temp_weight_change()
