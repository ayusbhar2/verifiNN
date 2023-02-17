from utils.model import Model


class TrainingTask:

    def __init__(self, model: Model, input_data, z) -> None:
        self.model = model
        self.input_data = input_data
        self.z = z

    def initialize(self, weight_start = 0):

        self.initial_weights = self.model.initialize(weight_start)

    def start_training(self, algorithm_function, e_t=0.001, alpha=0.05, max_iterations=100, verbose=False):

        print("\n\n")
        print(f"------------ start training -------------------- ")
        print(f"start training with algorithm function: {algorithm_function}")
        training_output = algorithm_function(self.initial_weights, self.input_data,
                           self.z, self.model, e_t, alpha, max_iterations,verbose)
        print(f"------------ end training ---------------------- ")
        return training_output
