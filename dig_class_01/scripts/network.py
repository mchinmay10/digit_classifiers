from visual import function_header, load_print, border_print_v1
from layers import DenseLayer_v3, DenseLayer_v4
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

    def forward(
        self,
        x: list[float],
    ):
        int_out = x.copy()
        for layer in self.layers:
            int_out = layer.forward_out(int_out)
            # To check where this intermediate_outs array is getting utilised
            self.intermediate_outs.append(int_out)

        # Eventually int_out would be output of the last layer; hence output of the network
        return int_out

    def backward(self, target: list[float]):
        if len(target) != self.layer_dims[self.num_layers - 1]:
            return "Number of targets are not equal to number of neurons in the last / output layer!"
        else:
            for i in range(self.num_layers - 1, 0, -1):
                # Does the target acts as input to all neurons during a backprop pass ?
                # This is incorrect.
                self.layers[i].backprop(target)


# Independent iteration of Network version 2 corresponding to DenseLayer version 4
class Network_v2:

    def __init__(
        self,
        num_layers: int,
        neurons_in_each_layer: list[int],
        weights: list[list[list[float]]],
        biases_for_each_layer: list[list[float]],
    ):

        self.num_layers = num_layers
        self.neurons_in_each_layer = neurons_in_each_layer
        self.weights = weights
        self.biases_for_each_layer = biases_for_each_layer

        self.layers: list[DenseLayer_v4] = []
        for i in range(num_layers):
            self.layers.append(
                DenseLayer_v4(
                    neurons_in_each_layer[i],
                    weights[i],
                    biases_for_each_layer[i],
                )
            )

    def forward(
        self,
        x: list[float],
    ):
        intermediate_output = x.copy()
        for layer in self.layers:
            layer_out = layer.layer_forward(intermediate_output)
            intermediate_output = layer_out

        self.network_output = intermediate_output
        return self.network_output

    def loss_calc(
        self,
        target: list[float],
    ):
        self.loss = mean_squared_error(
            target,
            self.network_output,
        )

    def dloss_dnetwork_output(
        self,
        target: list[float],
    ):
        self.dloss_dn_outs: list[float] = []
        output_len = len(self.network_output)
        for i in range(output_len):
            # This depends on the error function used (derivative of error function / loss function)
            diff = self.network_output[i] - target[i]
            self.dloss_dn_outs.append(2 * diff / output_len)

    def backprop(
        self,
        target: list[float],
    ):
        self.loss_calc(target)
        self.dloss_dnetwork_output(target)
        intermediate_ders = []
        for i in range(self.num_layers - 1, -1, -1):
            # print(f"in netowrk, layer no: {i}")
            # Output layer
            if i == self.num_layers - 1:
                intermediate_ders = self.layers[i].layer_backprop(self.dloss_dn_outs)
                continue
            # Other layers other than outer layer
            layer_out_rev = self.layers[i].layer_backprop(intermediate_ders)
            intermediate_ders = layer_out_rev

    def show_grads_loss_wrt_w_and_b(self):
        function_header("Gradient of Loss w.r.t. weight for each neuron (Analytical): ")
        for i in range(self.num_layers):
            neurons_i_layer = self.layers[i].neurons
            border_print_v1(f"Layer {i + 1}:")
            for j in range(len(neurons_i_layer)):
                border_print_v1(f"Neuron {j + 1}:")
                weights_j_neuron = neurons_i_layer[j].weights
                for k in range(len(weights_j_neuron)):
                    load_print(f"Weight {k + 1}: {neurons_i_layer[j].dloss_dw[k]}")
                load_print(f"Bias {j + 1}: {neurons_i_layer[j].dloss_db}")

    def calc_gradients_numerically(
        self,
        n_input: list[float],
        epsilon: float,
        target: list[float],
    ):
        function_header("Gradient of Loss w.r.t. weight for each neuron (Numerical): ")
        for i in range(self.num_layers):
            neurons_i_layer = self.layers[i].neurons
            border_print_v1(f"Layer {i + 1}:")
            for j in range(len(neurons_i_layer)):
                border_print_v1(f"Neuron {j + 1}:")
                weights_j_neuron = neurons_i_layer[j].weights
                for k in range(len(weights_j_neuron)):
                    neurons_i_layer[j].weights[k] += epsilon
                    self.forward(n_input)
                    self.loss_calc(target)
                    arg_1 = self.loss
                    neurons_i_layer[j].weights[k] -= 2 * epsilon
                    self.forward(n_input)
                    self.loss_calc(target)
                    arg_2 = self.loss
                    dloss_dw_k = (arg_1 - arg_2) / (2 * epsilon)
                    load_print(f"Weight {k + 1}: {dloss_dw_k}")
                    neurons_i_layer[j].weights[k] += epsilon

    def gradient_descent_step(self, learning_rate):
        for layer in self.layers:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights)):
                    neuron.weights[i] -= learning_rate * neuron.dloss_dw[i]
                neuron.bias -= learning_rate * neuron.dloss_db


def compare_analytical_numerical_grads():
    num_layers = 2
    neurons_in_each_layer = [2, 1]
    weights = [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0]]]
    biases_for_each_layer = [[0.5, 0.5], [0.2]]
    network = Network_v2(
        num_layers,
        neurons_in_each_layer,
        weights,
        biases_for_each_layer,
    )
    n_input = [1.0, 1.0]
    target = [0.5, 0.5]
    epsilon = 0.001
    load_print("Forward pass started...")
    network.forward(n_input)
    load_print("Forward pass finished. Proceeding with backprop...")
    network.backprop(target)
    load_print("Backprop finished...")
    border_print_v1("Analytical Gradients:")
    network.show_grads_loss_wrt_w_and_b()
    border_print_v1("Numerical Gradients:")
    network.calc_gradients_numerically(n_input, epsilon, target)


# Test cases
def network_v1_forward_test():
    function_header("Executing test cases for network v1 forward pass")
    weights = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    biases = [0.0]
    x = [7.0, 8.0, 9.0]
    n = Network_v1(1, [2], weights, biases)
    load_print(f"Network output: {n.forward(x)}")


if __name__ == "__main__":
    # network_v1_forward_test()
    compare_analytical_numerical_grads()
