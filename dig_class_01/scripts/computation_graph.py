from visual import load_print, function_header


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

    def view_props(self):
        load_print(f"x:    {self.x}")
        load_print(f"w:    {self.w}")
        load_print(f"b:    {self.b}")
        load_print(f"z:    {self.z()}")
        load_print(f"a:    {self.a()}")
        load_print(f"Loss: {self.loss()}")


# Test cases
def node_v1_tests():
    function_header("Executing test cases for version 1 of Node class")
    node = Node_v1(2, 3, 1)
    node.view_props()


if __name__ == "__main__":
    node_v1_tests()
