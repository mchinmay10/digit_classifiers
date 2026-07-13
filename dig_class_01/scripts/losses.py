import time


# implementing absolute error function from scratch
def absolute_error(target, prediction):
    if target >= prediction:
        return target - prediction
    else:
        return prediction - target


# implementing squared error function from scratch
def squared_error(target, prediction):
    return (target - prediction) ** 2


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


if __name__ == "__main__":
    print("----Performing tests for the losses.py file----")
    abs_error_test()
    squared_error_test()
