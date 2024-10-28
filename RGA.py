import numpy as np
from enum import Enum
from Chromosome import Chromosome
from Kapur import Kapur


class Constants(Enum):
    CROSSOVER_RATE = 0.6
    MUTATION_RATE = 0.4
    GENERATIONS = 10
    POPULATION_SIZE = 20
    ELITIST_SIZE = 4
    TOURNAMENT_SIZE = 5
    SELECTION_SIZE = (POPULATION_SIZE / 2) + 1


class GeneticAlgorithm:
    def __init__(self, image, threshold_count: int, kapur: Kapur):
        self._generation = []
        self._image = image
        self._threshold_count = threshold_count
        self._kapur = kapur

    def start(self):
        for _ in range(int(Constants.POPULATION_SIZE.value)):  # initialise generation
            self._generation.append(Chromosome(self._kapur, self._threshold_count))

        bestIndividual = None
        for _ in range(int(Constants.GENERATIONS.value)):
            self.propagate()
            bestIndividual = self.getBest(self._generation)

        return bestIndividual

    def propagate(self):  # handles selection and repopulation
        tournamentGeneration = self.tournamentSelection()
        newGeneration = []

        for i in range(int(Constants.SELECTION_SIZE.value) - 1):
            parent1 = i
            parent2 = i + 1
            child1Thresholds, child2Thresholds = tournamentGeneration[parent1].copyThresholds(), tournamentGeneration[parent2].copyThresholds()

            if (np.random.randint(0, 1)) < Constants.CROSSOVER_RATE.value:  # check if crossover can be done
                child1Thresholds, child2Thresholds = self.crossover(tournamentGeneration[parent1], tournamentGeneration[parent2])

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
        childThresholds1.sort()
        childThresholds2.sort()
        return childThresholds1, childThresholds2

    def mutation(self, thresholds: list):
        # Random mutation
        childThresholds = thresholds
        index = np.random.randint(0, len(thresholds))
        childThresholds[index] = np.clip(thresholds[index] + np.random.randint(-10, 10), 1, 254)
        childThresholds.sort()
        return childThresholds

    def tournamentSelection(self):
        bestIndividuals = []
        selectionCounter = 0

        while selectionCounter < int(Constants.SELECTION_SIZE.value):
            tournament = []
            tournamentCounter = 0

            while tournamentCounter < int(Constants.TOURNAMENT_SIZE.value):
                randomIndividual = self._generation[np.random.randint(0, len(self._generation))]
                if randomIndividual not in tournament:
                    tournamentCounter += 1
                    tournament.append(randomIndividual)

            bestIndividual = self.getBest(tournament)

            if bestIndividual not in bestIndividuals:
                selectionCounter += 1
                bestIndividuals.append(Chromosome(self._kapur, thresholds=bestIndividual.thresholds, fitness=bestIndividual.fitness))

        return bestIndividuals

    def repopulate(self, newGeneration):
        for x in range(int(Constants.POPULATION_SIZE.value)):
            self._generation[x] = newGeneration[x]

    def compareFitness(self, x: Chromosome, y: Chromosome):
        return x.fitness > y.fitness

    # def reorderList(self, currentList, size):
    #     for i in range(size):
    #         swapped = False
    #         for j in range(size - i - 1):
    #             if self.compareFitness(currentList[j + 1], currentList[j]):
    #                 currentList[j], currentList[j + 1] = currentList[j + 1], currentList[j]
    #                 swapped = True
    # 
    #         if not swapped:
    #             break
    #     return currentList

    def getBest(self, selection: list):
        best = selection[0]
        for i in range(1, len(selection)):
            if self.compareFitness(selection[i], best):
                best = selection[i]
        return best
