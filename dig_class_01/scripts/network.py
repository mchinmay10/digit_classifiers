from visual import function_header, load_print
from layers import DenseLayer_v3
from losses import mean_squared_error


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

        self.intermediate_outs = []

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
        int_out = x.copy()
        for layer in self.layers:
            int_out = layer.forward_out(int_out)
            # To check where this intermediate_outs array is getting utilised
            self.intermediate_outs.append(int_out)

        return int_out

    def backward(self, target: list[float]):
        if len(target) != self.layer_dims[self.num_layers - 1]:
            return "Number of targets are not equal to number of neurons in the last / output layer!"
        else:
            for i in range(self.num_layers - 1, 0, -1):
                # Does the target acts as input to all neurons during a backprop pass ?
                # This is incorrect.
                self.layers[i].backprop(target)


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
