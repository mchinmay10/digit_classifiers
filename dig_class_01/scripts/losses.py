import time
from vector import ones, vector_length, dot, generate_random_vector


# calculate mean value of elements of list
def mean_val_v1(nums: list[float]) -> float:
    return sum(nums) / len(nums)


def mean_val_v2(nums: list[float]) -> float | str:
    nums_len = vector_length(nums)
    dot_prod = dot(nums, ones(nums_len))
    if dot_prod:
        return dot_prod / nums_len
    else:
        return "Please debug mean_val_v2(nums) function in losses.py"


# implementing absolute error function from scratch (for a single value)
def absolute_error(target, prediction):
    if target >= prediction:
        return target - prediction
    else:
        return prediction - target


# implementing the mean absolute error function froms scratch (for mulitple values)
def mean_absolute_error(targets, predictions):
    abs_error_arr = []
    for target, prediction in zip(targets, predictions):
        abs_error_arr.append(absolute_error(target, prediction))

    return mean_val_v1(abs_error_arr)


# implementing squared error function from scratch (for a single value)
def squared_error(target, prediction):
    return (target - prediction) ** 2


# implementing mean squared error function from scratch (for mulitple values)
def mean_squared_error(targets, predictions):
    squared_error_arr = []
    for target, prediction in zip(targets, predictions):
        squared_error_arr.append(squared_error(target, prediction))

    return mean_val_v1(squared_error_arr)


# Test cases
def abs_error_test():
    print("Executing test cases for the absolute error function...")
    time.sleep(3)
    print(f"abs_error(1, 0.9): {absolute_error(1, 0.9)}")
    print(f"abs_error(1, 0.5): {absolute_error(1, 0.5)}")
    print(f"abs_error(1, 0.1): {absolute_error(1, 0.1)}")


def squared_error_test():
    print("Executing test cases for the squared error function...")
    time.sleep(3)
    print(f"squared_error(1, 0.9): {squared_error(1, 0.9)}")
    print(f"squared_error(1, 0.5): {squared_error(1, 0.5)}")
    print(f"squared_error(1, 0.1): {squared_error(1, 0.1)}")


def mean_val_functions_test():
    print("Executing test cases for versions of the mean value functions...")
    time.sleep(3)
    if mean_val_v1([20, 50, 30]) == mean_val_v2([20, 50, 30]):
        print("Test 1 passed")
    else:
        print("Test 1 failed")
    time.sleep(1)
    if mean_val_v1([-60, -73, -39]) == mean_val_v2([-60, -73, -39]):
        print("Test 2 passed")
    else:
        print("Test 2 failed")
    time.sleep(1)
    if mean_val_v1([4678.5, 8085.9, 3434.79]) == mean_val_v2([4678.5, 8085.9, 3434.79]):
        print("Test 3 passed")
    else:
        print("Test 3 failed")


def mean_abs_err_test():
    print("Executing test cases for the mean absolute error function...")
    time.sleep(3)
    print("---Mean Absolute Error between random vectors----")
    print(
        f"{mean_absolute_error(generate_random_vector(5), generate_random_vector(5))}"
    )
    print(
        f"{mean_absolute_error(generate_random_vector(1), generate_random_vector(1))}"
    )
    print(
        f"{mean_absolute_error(generate_random_vector(12345), generate_random_vector(12345))}"
    )
    # To be added later:
    # vectors of negative lengths
    # vectors of unequal lengths


def mean_squared_err_test():
    print("Executing test cases for the mean squared function...")
    time.sleep(3)
    print("----Mean Squared Error between random vectors----")
    print(f"{mean_squared_error(generate_random_vector(5), generate_random_vector(5))}")
    print(f"{mean_squared_error(generate_random_vector(1), generate_random_vector(1))}")
    print(
        f"{mean_squared_error(generate_random_vector(12345), generate_random_vector(12345))}"
    )
    # To be added later
    # vectors of negative lengths
    # vectors of unequal lengths


if __name__ == "__main__":
    print("----Performing tests for the losses.py file----")
    abs_error_test()
    squared_error_test()
    mean_val_functions_test()
    mean_abs_err_test()
    mean_squared_err_test()
