from visual import load_print, border_print_v1, function_header


class Node_v1:

    def __init__(
        self,
        x: float,
        w: float,
        b: float,
    ):
        self.x = x
        self.w = w
        self.b = b

    def z(self) -> float:
        return self.w * self.x

    def a(self) -> float:
        return self.z() + self.b

    def loss(self) -> float:
        return self.a() ** 2

    def forward_prop(self):
        load_print(f"x:    {self.x}")
        load_print(f"w:    {self.w}")
        load_print(f"b:    {self.b}")
        load_print(f"z:    {self.z()}")
        load_print(f"a:    {self.a()}")
        load_print(f"Loss: {self.loss()}")

    def dloss_da(self):
        return 2 * self.a()

    def da_dz(self):
        return 1

    def dz_dw(self):
        return self.x

    def dloss_dw(self):
        return self.dloss_da() * self.da_dz() * self.dz_dw()

    def back_prop(self):
        load_print(f"Δloss / Δa = {self.dloss_da()}")
        load_print(f"Δa / Δz =    {self.da_dz()}")
        load_print(f"Δz / Δw =    {self.dz_dw()}")
        load_print("Δloss / Δw =  (Δz / Δw) * (Δa / Δz) * (Δloss / Δa)")
        load_print(f"Δloss / Δw = {self.dz_dw()} * {self.da_dz()} * {self.dloss_da()}")
        load_print(f"Δloss / Δw = {self.dloss_dw()}")


class Node_v2:

    def __init__(
        self,
        x: float,
        w: float,
        b: float,
    ):
        self.x = x
        self.w = w
        self.b = b

        self.forward = 0
        self.backward = 0

    def forward_prop(self):
        self.forward = 1
        self.z = self.w * self.x
        self.a = self.z + self.b
        self.loss = self.a**2

    def back_prop(self):
        if self.forward == 0:
            print("Please perform forward pass first!")
        else:
            self.backward = 1
            self.dloss_da = 2 * self.a
            self.da_dz = 1
            self.dz_dw = self.x
            self.dloss_dw = self.dloss_da * self.da_dz * self.dz_dw

    def view_props(self):
        if self.forward == 0:
            print("Please perform forward pass first!")
        else:
            border_print_v1("Forward propagation:")
            load_print(f"x:    {self.x}")
            load_print(f"w:    {self.w}")
            load_print(f"b:    {self.b}")
            load_print(f"z:    {self.z}")
            load_print(f"a:    {self.a}")
            load_print(f"Loss: {self.loss}")
            if self.backward == 0:
                print("Please perform backward pass first!")
            else:
                border_print_v1("Backward propagation")
                load_print(f"Δloss / Δa = {self.dloss_da}")
                load_print(f"Δa / Δz =    {self.da_dz}")
                load_print(f"Δz / Δw =    {self.dz_dw}")
                load_print("Δloss / Δw =  (Δz / Δw) * (Δa / Δz) * (Δloss / Δa)")
                load_print(
                    f"Δloss / Δw = {self.dz_dw} * {self.da_dz} * {self.dloss_da}"
                )
                load_print(f"Δloss / Δw = {self.dloss_dw}")


# Test cases
def node_v1_tests():
    function_header("Executing test cases for version 1 of Node class")
    node = Node_v1(2, 3, 1)
    node.forward_prop()
    node.back_prop()


def node_v2_tests():
    function_header("Executing test cases for version 2 of Node class")
    node = Node_v2(2, 3, 1)
    node.forward_prop()
    # node.back_prop()
    node.view_props()


if __name__ == "__main__":
    # node_v1_tests()
    node_v2_tests()
