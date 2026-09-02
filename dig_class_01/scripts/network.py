from visual import function_header, load_print
from layers import DenseLayer_v3


# First iteration of Network class representing a simple neural network
class Network_v1:
    def __init__(
        self,
        num_layers: int,
        layer_dims: list[int],
        weights: list[list[list[float]]],
        biases: list[float],
    ):
        self.num_layers = num_layers
        self.layer_dims = layer_dims
        self.weights = weights
        self.biases = biases

        self.layers: list[DenseLayer_v3] = []
        for i in range(num_layers):
            self.layers.append(
                DenseLayer_v3(
                    layer_dims[i],
                    weights[i],
                    biases[i],
                )
            )

    def forward(self, x: list[float]):
        fwd_out = x.copy()
        for layer in self.layers:
            fwd_out = layer.forward_out(fwd_out)

        return fwd_out


# Test cases
def network_v1_forward_test():
    function_header("Executing test cases for network v1 forward pass")
    weights = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    biases = [0.0]
    x = [7.0, 8.0, 9.0]
    n = Network_v1(1, [2], weights, biases)
    load_print(f"Network output: {n.forward(x)}")


if __name__ == "__main__":
    network_v1_forward_test()
