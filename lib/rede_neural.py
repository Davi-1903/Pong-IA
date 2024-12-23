from numpy import exp, log, tanh, dot, array, random, sqrt
from typing import Literal, Callable


class Neuron:
    def __init__(self, activation_function: Literal['none', 'sigmoid', 'swish', 'tanh', 'relu', 'leaky_relu', 'softplus']):
        self.__value = 0
        self.bias = 0
        self.activation_function = self.get_activation_function(activation_function)

    def get_activation_function(self, name: str) -> Callable:
        functions = {
            'none': lambda x: x - self.bias,
            'sigmoid': lambda x: 1 / (1 + exp(-x)),
            'swish': lambda x: x / (1 + exp(-x)),
            'tanh': tanh,
            'relu': lambda x: max(0, x),
            'leaky_relu': lambda x: x if x > 0 else x * 0.01,
            'softplus': lambda x: log(1 + exp(x))
        }
        if name not in functions:
            raise ValueError(f'Invalid name "{name}". Try: {", ".join(functions)}')
        return functions[name]

    @property
    def value(self) -> float:
        return self.activation_function(self.__value + self.bias)

    @value.setter
    def value(self, raw_value: float):
        self.__value = raw_value

    def get_raw(self) -> float:
        return self.__value


class Layer:
    def __init__(self, neurons: list[str]):
        self.__neurons = [Neuron(function) for function in neurons]

    def __len__(self) -> int:
        return len(self.__neurons)

    def get_biases(self) -> list:
        return [neuron.bias for neuron in self.__neurons]

    def get_values(self) -> list:
        return [neuron.value for neuron in self.__neurons]

    def set_biases(self, biases: list):
        for idx, bias in enumerate(biases):
            self.__neurons[idx].bias = bias

    def set_values(self, values: list, weigths: array):
        for idx, value in enumerate(dot(array(values), weigths)):
            self.__neurons[idx].value = value


class Network:
    def __init__(self, layers: list[list], weights: list = [], weights_inicialization: Literal['random', 'xavier', 'he', 'lecun'] = 'random', biases: list = []):
        self.__inputs, *self.__hiddens, self.__outputs = layers
        self.__layers = []
        for idx in range(1, len(layers)):
            self.__layers.append(Layer(layers[idx]))
        self.set_weights(weights, weights_inicialization)
        self.set_biases(biases)

    def set_weights(self, weights: list, weights_inicialization: Literal['random', 'xavier', 'he', 'lecun'] = 'random'):
        self.__weights = weights
        if not weights:
            self.__weights = []
            if weights_inicialization == 'random':
                self.__weights.append(random.uniform(-0.01, 0.01, size=(self.__inputs, len(self.__layers[0]))))
                for idx in range(1, len(self.__layers)):
                    self.__weights.append(random.uniform(-0.01, 0.01, size=(len(self.__layers[idx - 1]), len(self.__layers[idx]))))
            elif weights_inicialization == 'xavier':
                limit = sqrt(6 / (self.__inputs + len(self.__layers[0])))
                self.__weights.append(random.uniform(-limit, limit, size=(self.__inputs, len(self.__layers[0]))))
                for idx in range(1, len(self.__layers)):
                    limit = sqrt(6 / (len(self.__layers[idx - 1]) + len(self.__layers[idx])))
                    self.__weights.append(random.uniform(-limit, limit, size=(len(self.__layers[idx - 1]), len(self.__layers[idx]))))
            elif weights_inicialization == 'he':
                self.__weights.append(random.randn(self.__inputs, len(self.__layers[0])) * sqrt(2 / self.__inputs))
                for idx in range(1, len(self.__layers)):
                    self.__weights.append(random.randn(len(self.__layers[idx - 1]), len(self.__layers[idx])) * sqrt(2 / self.__layers[idx - 1]))
            elif weights_inicialization == 'lecun':
                self.__weights.append(random.randn(self.__inputs, len(self.__layers[0])) * sqrt(1 / self.__inputs))
                for idx in range(1, len(self.__layers)):
                    self.__weights.append(random.randn(len(self.__layers[idx - 1]), len(self.__layers[idx])) * sqrt(1 / self.__layers[idx - 1]))
            else:
                raise ValueError(f'Invalid weights inicialization "{weights_inicialization}". Choose from random, xavier, he, lecun')

    def set_biases(self, biases: list):
        for idx, biases in enumerate(biases):
            self.__layers[idx].set_biases(biases)

    def get_biases(self) -> list:
        return [layer.get_biases() for layer in self.__layers]

    def get_weights(self) -> list:
        return self.__weights

    def feed_forward(self, parameters: list) -> list:
        self.__layers[0].set_values(parameters, self.__weights[0])
        for idx in range(1, len(self.__layers)):
            self.__layers[idx].set_values(self.__layers[idx - 1].get_values(), self.__weights[idx])
        return self.__layers[-1].get_values()
