import time
from layers import DenseLayer_v2
from losses import mean_squared_error
from visual import clear_screen
from layer_helper import (
    get_nw_input_from_user,
    get_weights_from_user,
    get_bias_from_user,
    get_weight_index_from_user,
    get_dummy_targets_from_user,
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
            print("Quitting weight temperory change simulation...")
            time.sleep(2)


# mini optimizer algorithm to automate temperory weight changes
def improve_once():
    # change every weight by a random delta (in next iteration as in, in the simulation function do no again improve by a random delta, just improve by -delta to see which direction increased loss / decreased loss)
    # calculate which weight changes increased loss and which decreased loss
    # 23 weight changes decreased loss
    # 45 weight changes increased loss
    # final loss
    # by how much loss decreased / increased from the previous iteration
    # and this should be an iterable function inside the simulate_improve_once()
    pass
    # for every weight keep tweaking untill loss decreases


def simulate_improve_once():
    pass
