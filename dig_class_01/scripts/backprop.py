from visual import load_print, function_header


# function that manually calculates output and loss (ver 1)
def output_and_loss_v1(x: float, w: float):
    z = w * x
    loss = z**2
    load_print(f"z = w * x\nz = {z}")
    load_print(f"loss = z ** 2\nloss = {loss}")


def output_and_loss_v2(x: float, w: float, b: float):
    z = w * x
    a = z + b
    loss = a**2
    load_print(f"z = w * x\na = z + b\na = w * x + b\na = {w} * {x} + {b}\na = {a}")
    load_print(f"loss = a ** 2\nloss = {a} ** 2\nloss = {loss}")


# Test cases
def output_and_loss_v1_test():
    function_header("Executing test cases for output_and_loss version 1 function")
    output_and_loss_v1(2, 3)
    output_and_loss_v1(2, 3.001)


def output_and_loss_v2_test():
    function_header("Executing test cases for output_and_loss version 2 function")
    output_and_loss_v2(2, 3, 1)


if __name__ == "__main__":
    output_and_loss_v1_test()
    output_and_loss_v2_test()
