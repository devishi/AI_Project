import random
from setUp import setup
from network import NeuralNetwork

class gene(object):
    layer_info = [2,3,1]
    fitness = 0

    def __init__(self,network=0):
        if network==0:
            self.network = NeuralNetwork(self.layer_info)
        else:
            self.network = network

        self.genes = self.network.myGene()

    def clone(self):
        return gene(self.network)

    def getNetwork(self):
        return self.network

    def getGene(self,i):
        return self.genes[i]

    def setGene(self,i,value):
        self.genes[i]=value

    def size(self):
        return len(self.genes)

    def mutate(self, mutationRate):
        newgenes = []
        weights = self.genes[:-self.network.numberOfBiases]
        biases = self.genes[self.network.numberOfWeights:]

        for weight in weights:
            if random.random() <= mutationRate:
                weight += weight * (random.random() - 0.5) * 2

        for bias in biases:
            if random.random() <= mutationRate:
                bias += bias * (random.random() - 0.5) * 2

        newgenes.extend(weights)
        newgenes.extend(biases)
        self.genes = newgenes
        return
