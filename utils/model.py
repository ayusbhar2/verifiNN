from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def initialize(self, start = 0):
        pass

    @abstractmethod
    def is_trained(self):
        pass

    @abstractmethod
    def get_output(self,input_data, weight_list):
        pass

    @abstractmethod
    def get_trained_output(self,input_data):
        pass

    @abstractmethod
    def compute_loss(self,weights_vector, input_data, z):
        pass
      
