"""
Contains the following functions:
1. get_weights_from_user
2. get_bias_from_user
3. get_dummy_targts_from_user
4. get_weight_index_from_user
5. get_nw_input_from_user
6. generate_dummy_image
7. dummy_target_dig_vector
"""

import random
from visual import load_print


# function to recieve nueron weights from user
def get_neuron_weights_from_user(input_len):
    weights_array = [0.0 for _ in range(input_len)]
    load_print("Enter weights array")
    for i in range(input_len):
        weights_array[i] = float(input(f"Enter weight no. {i + 1}: "))

    return weights_array


# function to recieve layer weights from user (mainly for manual optimisation and initialisation purpose)
def get_weights_from_user(num_neurons, input_len):
    weight_matrix = [[0.0 for _ in range(input_len)] for _ in range(num_neurons)]
    load_print("Enter weights matrix")
    for i in range(num_neurons):
        for j in range(input_len):
            weight_matrix[i][j] = float(
                input(f"Value of w at: row {i} and column {j}: ")
            )

    return weight_matrix


# function to recieve bias term from user
def get_bias_from_user() -> float:
    bias = float(input("Enter bias term: "))
    return bias


# function that gets dummy target from user for purpose of manual simulation of learning process
# where 'n' is naturally the number of nuerons in the pre-output layer
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


# function that generates a dummy 28 * 28 image
def generate_dummy_image():
    return [random.randint(0, 255) for _ in range(784)]


# function that initialises weights for a neuron
def init_weights(n):
    return [round(random.random(), 2) for _ in range(n)]


def generate_num_not_zero():
    num = 1 - random.random()
    if random.choice([True, False]):
        num = -num
    return num


# function that generates random step sizes
def gen_step_sizes(n):
    return [round(generate_num_not_zero(), 3) for _ in range(n)]


# fuunction that generates a dummy target output as a digit in the form of a one-hot vector
def dummy_target_dig_vector():
    dummy_index = random.randint(0, 9)
    one_hot_digit = []
    for i in range(10):
        if i == dummy_index:
            one_hot_digit.append(1)
        else:
            one_hot_digit.append(0)

    return one_hot_digit
