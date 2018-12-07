import random
import os
import numpy as np
import json
import math

def sigmoid(x):
        return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
        return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork(object):

    def ent_derivative(a, y):
      return (a - y)

    def fan_in(x):
      return math.sqrt(3/x)

    def fan_in_out(x,y):
      return math.sqrt(6/(x+y))


    def __init__(self, sizes, weights=[[[fan_in_out(2,1), fan_in(3)], [1.58, -0.48], [0.98, 0.92]], [[0.38, -1.2, 0.045]]], biases=[[[0.47], [-0.18], [0.17]], [[0.42]]]):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        if weights==0:
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = weights;

        if biases==0:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        else:
            self.biases = biases;
        self.numberOfBiases = sum(self.sizes[1:])
        self.numberOfWeights = 0
        for i in range (1, len(self.sizes)):
            self.numberOfWeights += self.sizes[i] * self.sizes [i-1]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def getweights(self, layer, neuroninput, neuronoutput):
        return self.weights[layer][neuronoutput][neuroninput]

    def setweights(self, layer, neuroninput, neuronoutput, value):
        self.weights[layer][neuronoutput][neuroninput] = value

    def getbiases(self, layer, neuronoutput):
        return self.biases[layer][neuronoutput]

    def setbiases(self, layer, neuronoutput, value):
        self.biases[layer][neuronoutput] = value

    def myGene(self):
        genes = []
        for x in range(0, (self.num_layers - 1)):
            for y in range(0, (self.sizes[x + 1])):
                    genes.extend(self.weights[x][y])
        for x in range(len(self.biases)):
            for element in self.biases[x]:
                genes.extend(element)
        return genes

    def setGeneData(self,genes):
        weights1D = genes[:-self.numberOfBiases]
        biases1D = genes[self.numberOfWeights:]
        i = 0
        weights = []
        for x in range(0,(self.num_layers - 1)):
            layer = np.zeros((self.sizes[x + 1],self.sizes[x]),dtype=np.float64)
            for y in range(0, (self.sizes[x + 1] )):
                for z in range(0,self.sizes[x] ):
                    weight = weights1D[i]
                    layer[y,z] = weight
                    i += 1
            weights.append(layer)
        biases = []
        i = 0
        for x in range(1,(self.num_layers)):
            biase = np.zeros((self.sizes[x],1),dtype=np.float64)
            for y in range(0, (self.sizes[x])):
                biase[y,0] = biases1D[i]
                i += 1
            biases.append(biase)

        self.weights = weights
        self.biases = biases

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        f = open(filename, "wb")
        json.dump(data, f)
        f.close()

def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = NeuralNetwork(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
