import time
import random
from layers import DenseLayer_v2
from losses import mean_squared_error
from layer_helper import get_bias_from_user
from visual import clear_screen


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


# Test cases:
def predict_digit_test():
    print(f"Executing test cases for predict function...")
    time.sleep(3)
    output = [round(random.random(), 2) for _ in range(10)]
    print(f"Predicted digit = {predict_digit(output)}")


if __name__ == "__main__":
    print("----Running test cases for the layer_ops.py file----")
