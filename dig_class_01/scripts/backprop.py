from visual import load_print, border_print_v1, function_header
from optimizer import single_weight_prim_der


# function that manually calculates output and loss (ver 1)
def output_and_loss_v1(
    x: float,
    w: float,
):
    z = w * x
    loss = z**2
    load_print(f"z = w * x\nz = {w} * {x}\nz = {z}")
    load_print(f"loss = z ** 2\nloss = {z} ** 2\nloss = {loss}")
    return loss


# function that manually calculates output and loss (ver 2)
def output_and_loss_v2(
    x: float,
    w: float,
    b: float,
):
    z = w * x
    a = z + b
    loss = a**2
    load_print(f"z = w * x\na = z + b\na = w * x + b\na = {w} * {x} + {b}\na = {a}")
    load_print(f"loss = a ** 2\nloss = {a} ** 2\nloss = {loss}")
    return z, a, loss


# function that calculates derivatives of intermediate variables with respect to their parent variable as part of the ̦process of discovering the chain rule
def corresponding_derivatives():
    function_header(
        "Running experiments for calculating intermediate derivates as part of chain rule discovery"
    )
    z1, a1, l1 = output_and_loss_v2(2, 3, 1)
    z2, a2, l2 = output_and_loss_v2(2, 3.001, 1)
    load_print("Chaining different rates of change...")
    load_print(
        f"Rate of change of 'z' w.r.t. 'w' = {single_weight_prim_der(0.001, z2, z1)}"
    )
    load_print(
        f"Rate of change of 'a' w.r.t. 'z' = {single_weight_prim_der(z2 - z1, a2, a1)}"
    )
    load_print(
        f"Rate of change of 'loss' w.r.t 'a' = {single_weight_prim_der(a2 - a1, l2, l1)}"
    )


def chain_rule_discovery():
    function_header("Running experiments for discovering chain rule from scratch")
    z1, a1, l1 = output_and_loss_v2(2, 3, 1)
    z2, a2, l2 = output_and_loss_v2(2, 3.001, 1)
    border_print_v1(f"Δloss / Δw = {single_weight_prim_der(0.001, l2, l1)}")
    load_print("Verifying above calculated quantity with the chain rule...")
    dz_dw = single_weight_prim_der(0.001, z2, z1)
    da_dz = single_weight_prim_der(z2 - z1, a2, a1)
    dl_da = single_weight_prim_der(a2 - a1, l2, l1)
    dl_dw = dz_dw * da_dz * dl_da
    load_print(f"Δz / Δw = {dz_dw}")
    load_print(f"Δa / Δz = {da_dz}")
    load_print(f"Δloss / Δa = {dl_da}")
    load_print("Δloss / Δw = (Δz / Δw) * (Δa / Δz) * (Δloss / Δa)")
    load_print(f"Δloss / Δw = {dz_dw} * {da_dz} * {dl_da}")
    border_print_v1(f"Δloss / Δw = {dl_dw}")


# Test cases
def output_and_loss_v1_test():
    function_header("Executing test cases for output_and_loss version 1 function")
    loss1 = output_and_loss_v1(2, 3)
    loss2 = output_and_loss_v1(2, 3.001)
    load_print("Printing primitive derivative for the above change in weight...")
    load_print(
        f"Rate of change of loss w.r.t. change in weight = {single_weight_prim_der(0.001, loss2, loss1)}"
    )


def output_and_loss_v2_test():
    function_header("Executing test cases for output_and_loss version 2 function")
    _, _, _ = output_and_loss_v2(2, 3, 1)


if __name__ == "__main__":
    # output_and_loss_v1_test()
    # output_and_loss_v2_test()
    # corresponding_derivatives()
    chain_rule_discovery()
