from numpy import random, array, dot


class Neuron:
    def __init__(self):
        self.__value = 0
        self.bias = 0
    
    @staticmethod
    def ReLU(x: float) -> float:
        return max(0, x)
    
    @property
    def value(self) -> float:
        return self.__value
    
    @value.setter
    def value(self, value: float):
        self.__value = self.ReLU(value + self.bias)


class Layer:
    def __init__(self, neurons: int):
        self.__neurons = [Neuron() for _ in range(neurons)]

    def __len__(self) -> int:
        return len(self.__neurons)
    
    def get_values(self) -> list:
        return [neuron.value for neuron in self.__neurons]
    
    def get_biases(self) -> list:
        return [neuron.bias for neuron in self.__neurons]
    
    def set_biases(self, biases: list):
        for idx, neuron in enumerate(self.__neurons):
            neuron.bias = biases[idx]
    
    def set_values(self, values: list, weights: list):
        for idx, value in enumerate(dot(array(values), weights)):
            self.__neurons[idx].value = value


class Network:
    def __init__(self, layers: list, weights: list = [], biases: list = []):
        self.__inputs, *self.__hiddens, self.__exits = layers
        self.__layers = []
        for n in range(1, len(layers)):
            self.__layers.append(Layer(layers[n]))
        self.set_biases(biases)
        self.set_weights(weights)
    
    def set_biases(self, biases: list):
        if biases:
            for idx, layer in enumerate(self.__layers):
                layer.set_biases(biases[idx])
    
    def set_weights(self, weights: list):
        self.__weights = weights
        if not weights:
            self.__weights = []
            self.__weights.append(random.randn(self.__inputs, len(self.__layers[0])) * 0.01)
            for idx in range(1, len(self.__layers)):
                self.__weights.append(random.randn(len(self.__layers[idx - 1]), len(self.__layers[idx])) * 0.01)
    
    def get_biases(self) -> list:
        return [layer.get_biases() for layer in self.__layers]
    
    def get_weights(self) -> list:
        return self.__weights
    
    def feedforward(self, parameters: list) -> list:
        self.__layers[0].set_values(parameters, self.__weights[0])
        for idx in range(1, len(self.__layers)):
            self.__layers[idx].set_values(self.__layers[idx - 1].get_values(), self.__weights[idx])
        return self.__layers[-1].get_values()
