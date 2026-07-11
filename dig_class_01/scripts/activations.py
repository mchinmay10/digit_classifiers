import time
import math
import matplotlib.pyplot as plt
import numpy as np


# implementing the step function from scratch
def step(x):
    if x > 0:
        return 1
    else:
        return 0


# implementing the sigmoid function from scratch
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# sigmoid function for an array of nums as input
def sigmoid_arr(nums):
    sig_array = []
    for x in nums:
        sig_array.append(sigmoid(x))

    return sig_array


# generate a graph of sigmoid function over a range of inputs
def generate_sigmoid_plot(nums):
    x = np.array(nums)
    y = sigmoid_arr(nums)

    plt.figure(figsize=(15, 15))

    plt.plot(x, y, color="blue", linestyle="-")

    plt.title("Sigmoid Function", fontsize=14, fontweight="bold")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("sigmoid(x)", fontsize=12)
    plt.grid(True, alpha=0.6)

    plt.show()


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
