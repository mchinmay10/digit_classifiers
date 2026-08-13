import time
import numpy as np
import matplotlib.pyplot as plt
from layers import Neuron, DenseLayer_v2
from losses import squared_error, mean_squared_error
from visual import clear_screen, function_header, load_print
from layer_helper import (
    get_nw_input_from_user,
    get_weights_from_user,
    get_neuron_weights_from_user,
    get_bias_from_user,
    get_weight_index_from_user,
    get_dummy_targets_from_user,
    generate_dummy_image,
    init_weights,
    gen_step_sizes,
)
from layer_ops import current_state_of_nw


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


# function to manually optimise the weights and then print the corresponding loss in a tabular format
def neighbourhood_optimisation_tabular() -> tuple | None:
    function_header("Initialising manual optimisation in tabular format")
    time.sleep(1)
    print("Enter following specs only!")
    print("num_neurons = 1")
    print("num of inputs = 1")
    time.sleep(2)
    num_neurons = int(input("Enter number of neurons: "))
    time.sleep(2)
    nw_inputs = get_nw_input_from_user()
    time.sleep(2)
    weights = get_weights_from_user(num_neurons, len(nw_inputs))
    time.sleep(2)
    weights_array = [weights[0][0] + float(dw) for dw in np.linspace(-0.5, 0.5, 9)]
    iter_items = len(weights_array)
    bias = get_bias_from_user()
    layer = DenseLayer_v2(num_neurons, weights, bias)
    targets = get_dummy_targets_from_user(num_neurons)
    prediction_array = []
    loss_array = []

    # loop for calculating loss array
    for i in range(iter_items):
        layer.weight_matrix[0][0] = weights_array[i]
        prediction = layer.layer_forward(nw_inputs)
        simple_loss = mean_squared_error(targets, prediction)
        prediction_array.append(prediction)
        loss_array.append(simple_loss)

    # loop for printing the table
    print("Weight\t Prediction\t\t Loss\t")
    for i in range(iter_items):
        print(f"{weights_array[i]}\t {prediction_array[i][0]}\t {loss_array[i]}")

    return (weights_array, loss_array)


# function to represent the manual optimisation in a graph format
def neighbourhood_optimisation_graph():
    axes = neighbourhood_optimisation_tabular()
    weights = []
    losses = []
    if axes:
        weights, losses = axes

    function_header("Representing optimisation in graphical format")

    plt.figure(figsize=(100, 100))

    plt.plot(weights, losses, color="blue", linestyle="-")

    plt.title("Loss Plot", fontsize=14, fontweight="bold")
    plt.xlabel("Weight", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.6)

    plt.show()


# function that returns loss in order to find better direction for a single neuron
def try_weight_directions(
    nw_input: list, neuron: Neuron, target: list, step_size: float
):
    activ_output = neuron.forward(nw_input)
    loss = squared_error(target[0], activ_output)
    print(f"Initial Loss: {loss}")
    store_weights = neuron.weights.copy()
    num_weights = len(nw_input)
    load_print(f"Incresing weights by {step_size}...")
    for i in range(num_weights):
        neuron.weights[i] += step_size
    load_print(f"Increased weights: {neuron.weights}")
    activ_output = neuron.forward(nw_input)
    weight_increase_loss = squared_error(target[0], activ_output)
    print(f"Loss if weight increased: {weight_increase_loss}")
    load_print("Restoring original weights...")
    neuron.weights = store_weights
    load_print(f"Decreasing weights by {step_size}...")
    for i in range(num_weights):
        neuron.weights[i] -= step_size
    load_print(f"Decreased weights: {neuron.weights}")
    activ_output = neuron.forward(nw_input)
    weight_decrease_loss = squared_error(target[0], activ_output)
    print(f"Loss if weight decreased: {weight_decrease_loss}")

    return (loss, weight_decrease_loss, weight_increase_loss)


# determine which direction is better for a single weight change (for a single neuron)
def find_better_direction():
    function_header("Determine which direction is better based on loss values")
    nw_input = get_nw_input_from_user()
    input_len = len(nw_input)
    weights = get_neuron_weights_from_user(input_len)
    bias = get_bias_from_user()
    load_print("Initialising neuron...")
    neuron = Neuron(weights, bias)
    target = get_dummy_targets_from_user(1)
    step_size = float(input("Enter step size: "))
    loss, weight_decrease_loss, weight_increase_loss = try_weight_directions(
        nw_input, neuron, target, step_size
    )
    if loss > weight_decrease_loss:
        load_print("Recommended action: decrease weight")
    if loss > weight_increase_loss:
        load_print("Recommended action: increase weight")
    if loss < weight_decrease_loss and loss < weight_increase_loss:
        load_print(
            "Recommended action: change step size to find ideal weight value for minimum loss"
        )


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
            print("Quitting temperory weight change simulation...")
            time.sleep(2)


# mini optimizer algorithm to automate temperory weight changes
def improve_once():
    # calculate which weight changes increased loss and which decreased loss
    # 23 weight changes decreased loss
    # 45 weight changes increased loss
    # final loss
    # by how much loss decreased / increased from the previous iteration
    # and this should be an iterable function inside the simulate_improve_once()
    # for every weight keep tweaking untill loss decreases
    pass


def simulate_improve_once():
    # change every weight by a random delta (in next iteration as in, in the simulation function do no again improve by a random delta, just improve by -delta to see which direction increased loss / decreased loss)
    pass


# function that calculates rate of change of loss with respect to that of the change of weight
def primitive_derivative(
    step_sizes: list[float], losses: list[float], init_loss: float
):
    loss_diff = []
    pri_der = []
    iters = len(losses)
    for i in range(iters):
        pri_der.append((losses[i] - init_loss) / step_sizes[i])

    return pri_der


# to experiment with a neuron of 784 weights
def primitive_derivative_simulation():
    function_header(
        "Derivative in its most primitive form (couldn't think of a better header)!"
    )
    load_print("Generating dummy image...")
    nw_input = generate_dummy_image()
    input_len = len(nw_input)
    load_print("Initialising random weights...")
    weights = init_weights(input_len)
    bias = get_bias_from_user()
    load_print("Initialising neuron...")
    neuron = Neuron(weights, bias)
    target = get_dummy_targets_from_user(1)
    activ_output = neuron.forward(nw_input)
    load_print("Calculating initial loss...")
    init_loss = round(squared_error(target[0], activ_output), 2)
    load_print(f"Initial loss: {init_loss}")
    load_print(
        "Generating random step sizes for primitive gradient descent initialisation..."
    )
    step_sizes = gen_step_sizes(input_len)


if __name__ == "__main__":
    # simulate_temp_weight_change()
    # neighbourhood_optimisation_tabular()
    # neighbourhood_optimisation_graph()
    # find_better_direction()
    primitive_derivative_simulation()
