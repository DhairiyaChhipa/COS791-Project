import numpy as np
from enum import Enum
from Chromosome import Chromosome
from Kapur import Kapur


class Constants(Enum):
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.3
    POPULATION_SIZE = 25
    GENERATIONS = 100
    TOURNAMENT_SIZE = 5
    # SELECTION_SIZE = 5  # quarter of population size (20 / 2 -> [10] / 2 -> [5])
    ELITIST_SIZE = 5
    MUTATION_STRATEGY = 0  # 0 = +- 10, 1 = random int (1, 254)


class GeneticAlgorithm:
    def __init__(self, image, threshold_count: int, kapur: Kapur):
        self._generation = []
        self._image = image
        self._threshold_count = threshold_count
        self._kapur = kapur

    # def __del__(self):
    #     for _ in range(int(Constants.POPULATION_SIZE.value)):
    #         item = self._generation.pop()
    #         del item
    # 
    #     if len(self._generation) == 0:
    #         del self._generation

    def start(self):
        for x in range(int(Constants.POPULATION_SIZE.value)):  # initialise generation
            self._generation.append(Chromosome(self._kapur, self._threshold_count))
        for _ in range(int(Constants.GENERATIONS.value)):
            self.propagate()
            print("best fitness: ", self.getBest().fitness)

        return self.getBest()

    def propagate(self):  # handles selection and repopulation
        newGeneration = self.selectIndividuals()
        for _ in range(int((Constants.POPULATION_SIZE.value - Constants.ELITIST_SIZE.value) / 2)):

            parent1 = np.random.randint(0, int(Constants.POPULATION_SIZE.value - 1))
            parent2 = np.random.randint(0, int(Constants.POPULATION_SIZE.value - 1))
            child1Thresholds, child2Thresholds = self._generation[parent1].copyThresholds(), self._generation[parent2].copyThresholds()

            while not (parent2 == parent1):
                parent2 = np.random.randint(0, int(Constants.POPULATION_SIZE.value - 1))

            if (np.random.randint(0, 1)) < Constants.CROSSOVER_RATE.value:  # check if crossover can be done
                child1Thresholds, child2Thresholds = self.crossover(self._generation[parent1], self._generation[parent2])

            if (np.random.randint(0, 1)) < Constants.MUTATION_RATE.value:  # check if mutation can be done
                child1Thresholds = self.mutation(child1Thresholds)
                child2Thresholds = self.mutation(child2Thresholds)
            child1Thresholds.sort()
            child2Thresholds.sort()
            child1 = Chromosome(self._kapur, thresholds=child1Thresholds)
            child2 = Chromosome(self._kapur, thresholds=child2Thresholds)
            newGeneration.append(child1)
            newGeneration.append(child2)
        self.repopulate(newGeneration)

    def crossover(self, chromosome1: Chromosome, chromosome2: Chromosome):
        # Single point crossover
        childThresholds1 = chromosome1.copyThresholds()
        childThresholds2 = chromosome2.copyThresholds()
        crossoverPoint = np.random.randint(1, 2)

        for i in range(crossoverPoint, len(chromosome1.thresholds)):
            childThresholds1[i], childThresholds2[i] = chromosome2.thresholds[i], chromosome1.thresholds[i]

        return childThresholds1, childThresholds2

    def mutation(self, thresholds: list):
        # Random mutation
        childThresholds = thresholds
        # print(len(chromosome.thresholds))
        index = np.random.randint(0, len(thresholds))

        if Constants.MUTATION_STRATEGY.value == 0:
            childThresholds[index] = np.clip(
                thresholds[index] + np.random.randint(-10, 10), 1, 254)
        elif Constants.MUTATION_STRATEGY.value == 1:
            childThresholds[index] = np.random.randint(1, 254)

        return childThresholds

    def selectIndividuals(self):
        selectedPopulation = []
        self.reorderPopulation()

        for x in range(int(Constants.ELITIST_SIZE.value)):  # elitism, get the best chromosomes from population
            selectedPopulation.append(
                Chromosome(self._kapur, thresholds=self._generation[x].thresholds, fitness=self._generation[x].fitness))

        return selectedPopulation

    def repopulate(self, newGeneration):
        for x in range(int(Constants.POPULATION_SIZE.value)):
            self._generation[x] = newGeneration[x]

    def compareFitness(self, x: Chromosome, y: Chromosome):
        return x.fitness > y.fitness

    def reorderPopulation(self):
        for i in range(int(Constants.POPULATION_SIZE.value)):
            swapped = False

            for j in range(int(Constants.POPULATION_SIZE.value) - i - 1):
                if self.compareFitness(self._generation[j + 1], self._generation[j]):
                    self._generation[j], self._generation[j + 1] = self._generation[j + 1], self._generation[j]
                    swapped = True

            if not swapped:
                break

    def getBest(self):
        return self._generation[0]
