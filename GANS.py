import heapq
import math
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
    LOCAL_ITERATIONS = 10
    LOCAL_SEARCH_RATE = 0.2
    COOLING_RATE = 0.7
    INITIAL_TEMP = 100
    STOP_TEMPERATURE = 1


class GeneticAlgorithmNeighbourSearch:
    def __init__(self, image, threshold_count: int, kapur: Kapur):
        self._generation = []
        self._image = image
        self._threshold_count = threshold_count
        self._kapur = kapur
        self._bestIndividual = None

    def start(self):
        bestGlobal = None
        for _ in range(int(Constants.POPULATION_SIZE.value)):  # initialise generation
            self._generation.append(Chromosome(self._kapur, self._threshold_count))

        for _ in range(int(Constants.GENERATIONS.value)):
            self.propagate()

            bestGlobal = self.getBestIndividual()
            print("Best fitness overall:", round(bestGlobal.fitness, 4))
            print("Generation:", _ + 1)
            # bestLocal = self.getBest(self._generation)
            # print("Best fitness this generation:", round(bestLocal.fitness, 4), "\n")

        return bestGlobal

    def propagate(self):  # handles selection, repopulation and local search
        tournamentGeneration = self.tournamentSelection()
        newGeneration = []

        for i in range(int(Constants.SELECTION_SIZE.value) - 1):
            parent1 = i
            parent2 = i + 1
            child1Thresholds, child2Thresholds = tournamentGeneration[parent1].copyThresholds(), tournamentGeneration[
                parent2].copyThresholds()

            if (np.random.randint(0, 1)) < Constants.CROSSOVER_RATE.value:  # check if crossover can be done
                child1Thresholds, child2Thresholds = self.crossover(tournamentGeneration[parent1],
                                                                    tournamentGeneration[parent2])

            if (np.random.randint(0, 1)) < Constants.MUTATION_RATE.value:  # check if mutation can be done
                child1Thresholds = self.mutation(child1Thresholds)
                child2Thresholds = self.mutation(child2Thresholds)

            child1Thresholds.sort()
            child2Thresholds.sort()

            child1 = Chromosome(self._kapur, thresholds=child1Thresholds)
            child2 = Chromosome(self._kapur, thresholds=child2Thresholds)

            if (np.random.randint(0, 1)) < Constants.LOCAL_SEARCH_RATE.value:
                if self.compareFitness(child1, tournamentGeneration[parent1]):
                    child1 = self.ILS(child1)
                if self.compareFitness(child2, tournamentGeneration[parent2]):
                    child2 = self.ILS(child2)

            newGeneration.append(child1)
            newGeneration.append(child2)

        # if (np.random.randint(0, 1)) < Constants.LOCAL_SEARCH_RATE.value:
        # newGeneration = self.ILS(newGeneration)

        newGeneration = self.eliteSelection(newGeneration)

        self.repopulate(newGeneration)

    def crossover(self, chromosome1: Chromosome, chromosome2: Chromosome):
        # Single point crossover
        childThresholds1 = chromosome1.copyThresholds()
        childThresholds2 = chromosome2.copyThresholds()
        crossoverPoint = np.random.randint(1, 2)

        for i in range(crossoverPoint, len(chromosome1.thresholds)):
            childThresholds1[i], childThresholds2[i] = chromosome2.thresholds[i], chromosome1.thresholds[i]

        return childThresholds1, childThresholds2

    def ILS(self, chromosome: Chromosome):
        # index = 0
        # for y in range(1, len(population)):  # pick the best individual from the population
        #     if self.compareFitness(population[y], population[index]):
        #         index = y

        # currSolution = Chromosome(self._kapur, thresholds=population[index].thresholds,
        #                           fitness=population[index].fitness)
        currSolution = Chromosome(self._kapur, thresholds=chromosome.thresholds, fitness=chromosome.fitness)

        for _ in range(int(Constants.LOCAL_ITERATIONS.value)):
            solution = self.perturbation(currSolution)  # perturbation
            newSolution = self.simulatedAnnealing(solution)  # local search

            if self.compareFitness(newSolution, currSolution):  # acceptance criteria
                currSolution = newSolution

        # if self.compareFitness(currSolution, population[index]):
        #     population[index] = currSolution
        if self.compareFitness(currSolution, chromosome):
            chromosome = currSolution

        return chromosome

    def simulatedAnnealing(self, chromosome: Chromosome):
        bestChromosome = chromosome
        temperature = int(Constants.INITIAL_TEMP.value)

        while temperature > Constants.STOP_TEMPERATURE.value:
            newChromosome = self.localSearch(bestChromosome)
            deltaCost = newChromosome.fitness - bestChromosome.fitness

            if self.compareFitness(newChromosome, bestChromosome):
                bestChromosome = newChromosome
            else:
                if self.accept(deltaCost, temperature):
                    bestChromosome = newChromosome

            temperature *= Constants.COOLING_RATE.value

        return bestChromosome

    def accept(self, delta, temperature):
        if delta < 0:
            return True
        else:
            randomValue = np.round(np.random.random(1)[0], 2)
            if randomValue < math.exp(-delta / temperature):
                return True
            else:
                return False

    def localSearch(self, chromosome: Chromosome):
        length = len(chromosome.thresholds)
        bestFitness = chromosome.fitness
        counter = -length
        rangeList = []

        while counter <= length:
            rangeList.append(counter)
            counter += 1

        index = np.random.randint(0, length)
        currentThreshold = chromosome.thresholds[index]
        bestThreshold = currentThreshold

        for thresholdRange in rangeList:
            if index == 0:
                newThreshold = min(max(1, chromosome.thresholds[index] + thresholdRange),
                                   chromosome.thresholds[index + 1] - 1)
            elif index == length - 1:
                newThreshold = max(min(254, chromosome.thresholds[index] + thresholdRange),
                                   chromosome.thresholds[index - 1])
            else:
                lower = chromosome.thresholds[index - 1] + 1
                upper = chromosome.thresholds[index + 1] - 1
                newThreshold = min(max(chromosome.thresholds[index] + thresholdRange, lower), upper)

            originalThreshold = chromosome.thresholds[index]
            chromosome.thresholds[index] = newThreshold
            chromosome.calculateFitness()

            if chromosome.fitness > bestFitness:
                bestFitness = chromosome.fitness
                bestThreshold = newThreshold
            chromosome.thresholds[index] = originalThreshold

        chromosome.thresholds[index] = bestThreshold

        chromosome.calculateFitness()
        return chromosome

    def perturbation(self, chromosome: Chromosome):
        chromosomeCopy = Chromosome(self._kapur, thresholds=chromosome.thresholds, fitness=chromosome.fitness)
        length = len(chromosomeCopy.thresholds) - 1
        index = np.random.randint(0, length + 1)
        lower = 0
        upper = 0

        if index == 0:
            lower = 1
            upper = chromosomeCopy.thresholds[index + 1]

        elif index == length:
            lower = chromosomeCopy.thresholds[index - 1] + 1
            upper = 255

        elif 0 < index < length:
            lower = chromosomeCopy.thresholds[index - 1] + 1
            upper = chromosomeCopy.thresholds[index + 1]
            if lower >= upper:
                lower = upper - 1

        newThreshold = np.random.randint(lower, upper)
        chromosomeCopy.thresholds[index] = newThreshold

        chromosomeCopy.calculateFitness()

        return chromosomeCopy

    def mutation(self, thresholds: list):
        # Random mutation
        childThresholds = thresholds
        index = np.random.randint(0, len(thresholds))
        childThresholds[index] = np.clip(thresholds[index] + np.random.randint(-30, 30), 1, 254)
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
                bestIndividuals.append(
                    Chromosome(self._kapur, thresholds=bestIndividual.thresholds, fitness=bestIndividual.fitness))

        return bestIndividuals

    def eliteSelection(self, population: list):
        bestIndividuals = heapq.nlargest(int(Constants.ELITIST_SIZE.value), population, key=lambda x: x.fitness)
        return bestIndividuals + population[Constants.ELITIST_SIZE.value:]

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

    # def copyGeneration(self):
    #     copy = []
    #     for i in range(len(self._generation)):
    #         temp = Chromosome(self._kapur, thresholds=self._generation[i].thresholds)
    #         copy.append(temp)
    #     return copy

    def getBest(self, selection: list):
        best = selection[0]
        for i in range(1, len(selection)):
            if self.compareFitness(selection[i], best):
                best = selection[i]
        return best

    def getBestIndividual(self):
        if self._bestIndividual is None:
            self._bestIndividual = self._generation[0]
        for i in range(0, int(Constants.POPULATION_SIZE.value)):
            if self.compareFitness(self._generation[i], self._bestIndividual):
                self._bestIndividual = self._generation[i]
        return self._bestIndividual
