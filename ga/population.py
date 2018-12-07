from genome import gene
from network import NeuralNetwork
from setUp import setup
import copy
import random
import math

def randomGenome(genomes):
    i = random.randint(0,len(genomes) - 1)
    return genomes[i]

def randGene():
    net = NeuralNetwork(setup.layer_info)
    return gene(net)

def playTournament(population):
    tournament = Population()
    for i in xrange(setup.tournamentSize):
        tournament.addGenome(population.genomes[random.randint(0,setup.birdsInPlay - 1)])
    fittest = tournament.findFittest()
    return fittest

class Population(object):
    def __init__(self):
        self.genomes = []

    def findFittest(self,number=1):
        fittest = sorted(self.genomes, key=lambda x: x.fitness, reverse=True)
        while len(fittest) > number:
            fittest.pop()
        if number==1:
            fittest=fittest[0]
        return fittest

    def randPop(self):
        for i in xrange(setup.birdsInPlay):
            gene = randGene()
            self.genomes.append(gene)

    def evolvePopulation(self):
        genomes = []
        elitismoffset = 0
        if setup.elites!=0:
            elite = self.findFittest(setup.elites)
            elitismoffset = setup.elites

        for x in xrange (self.size() - elitismoffset - setup.randomBirds):
            father = self.findFittest()
            mother = playTournament(self)
            newIndiv = self.crossover(father, mother)
            newIndiv.mutate(setup.mutationRate)
            genomes.append(newIndiv)
        for x in xrange (setup.randomBirds):
            newIndiv = randGene()
            newIndiv.mutate(setup.mutationRate)
            genomes.append(newIndiv)
        if setup.elites !=1:
            genomes = elite + genomes
        else:
            genomes = genomes.append(elite)
        self.genomes = genomes

    def crossover(self,genome1, genome2):
        if type(genome2)==type(None):
            genome2=randomGenome(self.genomes)
        g = gene()
        for i in xrange(genome1.size()):
            if random.random() <= setup.uniformRate:
                g.setGene(i, genome1.getGene(i))
            else:
                g.setGene(i, genome2.getGene(i))
        return g

    def addGenome(self, gene):
        self.genomes.append(gene)

    def setGenomeFitness(self,i,fitness):
        self.genomes[i].fitness = fitness

    def getGenome(self,i):
        return self.genomes[i]

    def averageFitness(self):
        fitness = []
        for gene in self.genomes:
            fitness.append(gene.fitness)
        return sum(fitness) / float(len(fitness))

    def sumFitness(self):
        fitness = []
        for gene in self.genomes:
            fitness.append(gene.fitness)
        return sum(fitness)

    def size(self):
        return len(self.genomes)
