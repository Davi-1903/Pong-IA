from numpy import exp, log, tanh, dot, array, random, sqrt
from typing import Literal, Callable


class Neuron:
    '''Neurônio matemático.
    
    Atributos:
    - `value:` Propriedade que guarda o valor recebido (privado);
    - `bias:` Viés cognitivo;
    - `activation_function:` Função de ativação;
    '''
    def __init__(self, activation_function: Literal['none', 'sigmoid', 'swish', 'tanh', 'relu', 'leaky_relu', 'softplus']):
        '''Método construtor.
        
        Parâmetros:
        - `activation_function:` Função de ativação;
            - none
            - sigmoid
            - swish
            - tanh
            - relu
            - leaky relu
            - softplus
        '''
        self.__value = 0
        self.bias = 0
        self.activation_function = self.get_activation_function(activation_function)

    def get_activation_function(self, name: str) -> Callable[[float], float]:
        '''Método que retorna a função de ativação.'''
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
        '''Método que retorna o valor sem viés e sem função de ativação.'''
        return self.__value


class Layer:
    '''Camada de neurônios.
    
    Atributos:
    - `neurons:` Lista que guarda os neurônios (privado);
    '''
    def __init__(self, neurons: list[str]):
        '''Método construtor.
        
        Parâmetros:
        - `neurons:` Lista de strings em que cada string é uma função de ativação para o neurônio;
        '''
        self.__neurons = [Neuron(function) for function in neurons]

    def __len__(self) -> int:
        return len(self.__neurons)

    def get_biases(self) -> list:
        '''Método que retorna as bias dos neurônios.'''
        return [neuron.bias for neuron in self.__neurons]

    def get_values(self) -> list:
        '''Método que retorna os valores dos neurônios.'''
        return [neuron.value for neuron in self.__neurons]

    def set_biases(self, biases: list):
        '''Método para definir as biases dos neurônios.
        
        Parâmetros:
        - `biases:` Lista de valores (bias);
        '''
        for idx, bias in enumerate(biases):
            self.__neurons[idx].bias = bias

    def set_values(self, values: list, weigths: array):
        '''Método para atribuir valor aos neurônios.
        
        Parâmetros:
        - `values:` Lista de valores da camada anterior ou da camada de entrada;
        - `weigths:` Matriz de pesos entre as camadas;
        '''
        for idx, value in enumerate(dot(array(values), weigths)):
            self.__neurons[idx].value = value


class Network:
    '''Rede neural genérica do tipo Feed Forward.

    Atributos:
    - `inputs:` Quantidade de entrada (privado);
    - `hiddens:` Lista com a estrutura das camadas ocultas (privado);
    - `outputs:` Lista com a estrutura da camada de saída (privado);
    - `layers:` Lista de camadas (privado);
    - `weights:` Pesos da rede neural (privado);
    '''
    def __init__(self, structure: list[list], weights: list = [], weights_initialization: Literal['random', 'xavier', 'he', 'lecun'] = 'random', biases: list = []):
        '''Método construtor.
        
        Parâmetros:
        - `structure:` Estrutura da rede neural, lista de lista com as funções de ativação (Primeira camada apenas com a quantidade de entradas);
        - `weights:` Lista de pesos da rede neural (opcional);
        - `weights_initialization:` Método de inicialização dos pesos (opcional);
            - `random`;
            - `xavier`;
            - `he`;
            - `lecun`;
        - `biases:` Lista de bias da rede neural (opcional);
        '''
        self.__inputs, *self.__hiddens, self.__outputs = structure
        self.__layers = []
        for idx in range(1, len(structure)):
            self.__layers.append(Layer(structure[idx]))
        self.set_weights(weights, weights_initialization)
        self.set_biases(biases)

    def set_weights(self, weights: list, weights_initialization: Literal['random', 'xavier', 'he', 'lecun'] = 'random'):
        '''Método para a definição dos pesos ou, caso não haja pesos, inicialização.
        
        Parâmetros:
        - `weights:` Lista de pesos da rede neural;
        - `weights_initialization:` Método de inicialização dos pesos (opcional);
            - `random`;
            - `xavier`;
            - `he`;
            - `lecun`;
        '''
        self.__weights = weights
        if not weights:
            self.__weights = []
            if weights_initialization == 'random':
                self.__weights.append(random.uniform(-0.01, 0.01, size=(self.__inputs, len(self.__layers[0]))))
                for idx in range(1, len(self.__layers)):
                    self.__weights.append(random.uniform(-0.01, 0.01, size=(len(self.__layers[idx - 1]), len(self.__layers[idx]))))
            elif weights_initialization == 'xavier':
                limit = sqrt(6 / (self.__inputs + len(self.__layers[0])))
                self.__weights.append(random.uniform(-limit, limit, size=(self.__inputs, len(self.__layers[0]))))
                for idx in range(1, len(self.__layers)):
                    limit = sqrt(6 / (len(self.__layers[idx - 1]) + len(self.__layers[idx])))
                    self.__weights.append(random.uniform(-limit, limit, size=(len(self.__layers[idx - 1]), len(self.__layers[idx]))))
            elif weights_initialization == 'he':
                self.__weights.append(random.randn(self.__inputs, len(self.__layers[0])) * sqrt(2 / self.__inputs))
                for idx in range(1, len(self.__layers)):
                    self.__weights.append(random.randn(len(self.__layers[idx - 1]), len(self.__layers[idx])) * sqrt(2 / len(self.__layers[idx - 1])))
            elif weights_initialization == 'lecun':
                self.__weights.append(random.randn(self.__inputs, len(self.__layers[0])) * sqrt(1 / self.__inputs))
                for idx in range(1, len(self.__layers)):
                    self.__weights.append(random.randn(len(self.__layers[idx - 1]), len(self.__layers[idx])) * sqrt(1 / len(self.__layers[idx - 1])))
            else:
                raise ValueError(f'Invalid weights inicialization "{weights_initialization}". Choose from random, xavier, he, lecun')

    def set_biases(self, biases: list):
        '''Método para a definição dos bias da rede neural.

        Parâmetros:
        - `biases:` Lista de bias da rede neural;
        '''
        for idx, biases in enumerate(biases):
            self.__layers[idx].set_biases(biases)

    def get_biases(self) -> list:
        '''Método que retorna as biases da rede neural.'''
        return [layer.get_biases() for layer in self.__layers]

    def get_weights(self) -> list:
        '''Método que retorna os pesos da rede neural.'''
        return self.__weights

    def feed_forward(self, parameters: list) -> list:
        '''Método que realiza a feedforward da rede neural.

        Parâmetros:
        - `parameters:` Parâmetros de entrada da rede neural.
        '''
        self.__layers[0].set_values(parameters, self.__weights[0])
        for idx in range(1, len(self.__layers)):
            self.__layers[idx].set_values(self.__layers[idx - 1].get_values(), self.__weights[idx])
        return self.__layers[-1].get_values()
