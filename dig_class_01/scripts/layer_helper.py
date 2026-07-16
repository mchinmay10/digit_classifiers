import time


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
