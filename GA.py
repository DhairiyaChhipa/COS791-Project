import numpy as np
from enum import Enum
from Chromosome import Chromosome
from Kapur import Kapur


class Constants(Enum):
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.3
    POPULATION_SIZE = 20
    GENERATIONS = 10
    TOURNAMENT_SIZE = 5
    SELECTION_SIZE = 5  # quarter of population size (20 / 2 -> [10] / 2 -> [5])
    ELITIST_SIZE = 10
    SEED = 1234
    MUTATION_STRATEGY = 1  # 0 = +- 10, 1 = random int (1, 254)
    
class GeneticAlgorithm:
    def __init__(self, image, threshold_count: int, kapur: Kapur):
        self._generation = []
        self._image = image
        self._threshold_count = threshold_count
        self._kapur = kapur
        np.random.seed(Constants.SEED.value)

    def __del__(self):
        for _ in range(int(Constants.POPULATION_SIZE.value)):
            item = self._generation.pop()
            del item

        if len(self._generation) == 0:
            del self._generation

    def start(self):
        for x in range(int(Constants.POPULATION_SIZE.value)):  # initialise generation
            self._generation.append(Chromosome(self._threshold_count, self._kapur))

        for _ in range(int(Constants.GENERATIONS.value)):
            self.propagate()

        return self.getBest()

    def propagate(self):  # handles selection and repopulation
        newGeneration = self.selectIndividuals()

        for _ in range(int(Constants.SELECTION_SIZE.value)):

            parent1 = np.random.randint(0, int(Constants.GENERATIONS.value - 1))
            parent2 = np.random.randint(0, int(Constants.GENERATIONS.value - 1))
            while not (parent2 == parent1):
                parent2 = np.random.randint(0, int(Constants.GENERATIONS.value - 1))

            if (np.random.randint(0, 1)) < Constants.CROSSOVER_RATE.value:  # check if crossover can be done
                child1, child2 = self.crossover(self._generation[parent1], self._generation[parent2])
                newGeneration.append(child1)
                newGeneration.append(child2)

            if (np.random.randint(0, 1)) < Constants.MUTATION_RATE.value:  # check if mutation can be done
                newGeneration.append(self.mutation(self._generation[parent1]))
                newGeneration.append(self.mutation(self._generation[parent2]))

        self.repopulate(newGeneration)

    def crossover(self, chromosome1: Chromosome, chromosome2: Chromosome):
        # Single point crossover
        child1 = Chromosome(thresholds=chromosome1.thresholds, fitness=chromosome1.fitness)
        child2 = Chromosome(thresholds=chromosome2.thresholds, fitness=chromosome2.fitness)
        crossoverPoint = np.random.randint(1, 2)

        for i in range(crossoverPoint, len(chromosome1.thresholds)):
            child1.thresholds[i], child2.thresholds[i] = chromosome2.thresholds[i], chromosome1.thresholds[i]

        return child1, child2

    def mutation(self, chromosome: Chromosome):
        # "Flip" bit mutation
        child = Chromosome(thresholds=chromosome.thresholds, fitness=chromosome.fitness)
        index = np.random.randint(0, len(chromosome.thresholds))
        if Constants.MUTATION_STRATEGY.value == 0:
            child.thresholds[index] = chromosome.thresholds[index] = np.clip(chromosome.thresholds[index] + np.random.randint(-10, 10), 1, 254)
        elif Constants.MUTATION_STRATEGY.value == 1:
            child.thresholds[index] = chromosome.thresholds[index] = np.random.randint(1, 254)
        return child

    def selectIndividuals(self):
        selectedPopulation = []
        self.reorderPopulation()

        for x in range(int(Constants.ELITIST_SIZE.value)):  # elitism, get the best chromosomes from population
            selectedPopulation.append(Chromosome(thresholds=self._generation[x].thresholds, fitness=self._generation[x].fitness))

        return selectedPopulation

    def repopulate(self, newGeneration):
        for x in range(int(Constants.POPULATION_SIZE.value)):
            self._generation[x] = newGeneration[x]

    def compareFitness(self, x: Chromosome, y: Chromosome):
        if x.fitness > y.fitness:
            return True
        else:
            return False

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
