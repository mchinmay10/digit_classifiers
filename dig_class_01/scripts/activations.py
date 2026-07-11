import time
import math
import matplotlib.pyplot as plt
import numpy as np


# implementing the step function for single input
def step_single(x):
    if x > 0:
        return 1
    else:
        return 0


# implementing step function for an array of nums as input
def step_arr(nums):
    step_array = []
    for x in nums:
        step_array.append(step_single(x))

    return step_array


# the final step function with added flexibility
def step(input):
    if type(input) != list:
        return step_single(input)
    else:
        return step_arr(input)


# implementing the sigmoid function for single input
def sigmoid_single(x):
    return 1 / (1 + math.exp(-x))


# sigmoid function for an array of nums as input
def sigmoid_arr(nums):
    sig_array = []
    for x in nums:
        sig_array.append(sigmoid_single(x))

    return sig_array


# the final sigmoid function with added flexibility
def sigmoid(input):
    if type(input) != list:
        return sigmoid_single(input)
    else:
        return sigmoid_arr(input)


# generate a graph of sigmoid function over a range of inputs
def generate_sigmoid_plot(nums):
    x = np.array(nums)
    y = sigmoid(nums)

    plt.figure(figsize=(15, 15))

    plt.plot(x, y, color="blue", linestyle="-")

    plt.title("Sigmoid Function", fontsize=14, fontweight="bold")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("sigmoid(x)", fontsize=12)
    plt.grid(True, alpha=0.6)

    plt.show()


# implementing the rectified linear unit function from scratch
def relu_single(x):
    if x > 0:
        return x
    else:
        return 0


# rectified linear unit for an array of nums as input
def relu_arr(nums):
    relu_array = []
    for x in nums:
        relu_array.append(relu_single(x))

    return relu_array


# the final relu function with added flexibility
def relu(input):
    if type(input) != list:
        return relu_single(input)
    else:
        return relu_arr(input)


# Test cases
def step_test():
    print(f"Executing test cases for step function...")
    time.sleep(3)
    print(f"step(0.1) = {step(0.1)}")
    print(f"step(-2) = {step(-2)}")


def generate_sigmoid_plot_test():
    print(f"Executing test cases for sigmoid plot function...")
    time.sleep(3)
    generate_sigmoid_plot([-10, -5, -2, -1, 0, 1, 2, 5, 10])
    generate_sigmoid_plot([x for x in range(-10, 10)])


if __name__ == "__main__":
    print("----Running test cases for activations.py file----")
    time.sleep(3)
    step_test()
    generate_sigmoid_plot_test()
