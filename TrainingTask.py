from utils.network import Network


class TrainingTask:

    def __init__(self, network: Network, input_data, z) -> None:
        self.network = network
        self.input_data = input_data
        self.z = z

    def initialize(self, weight_start = 0):

        self.network.generate_network_specs()
        w_list = self.network.initialize_network(weight_start)
        self.initial_weights = self.network.unpack_weights(w_list)

    def start_training(self, algorithm_function, e_t=0.001, alpha=0.05, max_iterations=100, verbose=False):

        print("\n\n")
        print(f"------------ start training -------------------- ")
        print(f"start training with algorithm function: {algorithm_function}")
        training_output = algorithm_function(self.initial_weights, self.input_data,
                           self.z, self.network, e_t, alpha, max_iterations,verbose)
        print(f"------------ end training ---------------------- ")
        return training_output
