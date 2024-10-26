import math
from colorama import Fore, Style
import numpy as np
from enum import Enum
from Chromosome import Chromosome
from Kapur import Kapur

class Constants(Enum):
    CROSSOVER_RATE = 0.6
    MUTATION_RATE = 0.4
    GENERATIONS = 100
    POPULATION_SIZE = 26
    ELITIST_SIZE = 4
    TOURNAMENT_SIZE = 5
    SELECTION_SIZE = (POPULATION_SIZE / 2) + 1
    MUTATION_STRATEGY = 0  # 0 = +- 10, 1 = random int (1, 254)
    LOCAL_ITERATIONS = 30
    LOCAL_SEARCH_RATE = 0.3

class GeneticAlgorithmNeighbourSearch:
    def __init__(self, image, threshold_count: int, kapur: Kapur):
        self._generation = []
        self._image = image
        self._threshold_count = threshold_count
        self._kapur = kapur
        self._bestIndividial = None

    # def __del__(self):
    #     # print("Constants.POPULATION_SIZE.value", Constants.POPULATION_SIZE.value)
    #     # print("len:", len(self._generation))

    #     for i in range(int(Constants.POPULATION_SIZE.value)):
    #         del self._generation[i]
    
    #     if len(self._generation) == 0:
    #         del self._generation

    def start(self):
        for _ in range(int(Constants.POPULATION_SIZE.value)):  # initialise generation
            self._generation.append(Chromosome(self._kapur, self._threshold_count))

        bestIndividual = None
        for _ in range(int(Constants.GENERATIONS.value)):
            self.propagate()

            bestGlobal = self.getBestIndividual()
            print("Best fitness overall:", round(bestGlobal.fitness, 4))

            # bestLocal = self.getBest(self._generation)
            # print("Best fitness this generation:", round(bestLocal.fitness, 4), "\n")

        return bestGlobal

    def propagate(self):  # handles selection, repopulation and local search
        tournamentGeneration = self.tournamentSelection()
        newGeneration = []

        for i in range(int(Constants.SELECTION_SIZE.value) - 1):
            parent1 = i
            parent2 = i + 1
            child1Thresholds, child2Thresholds = tournamentGeneration[parent1].copyThresholds(), tournamentGeneration[parent2].copyThresholds()

            if (round(np.random.random(1)[0], 2)) < Constants.CROSSOVER_RATE.value:  # check if crossover can be done
                child1Thresholds, child2Thresholds = self.crossover(tournamentGeneration[parent1], tournamentGeneration[parent2])

            if (round(np.random.random(1)[0], 2)) < Constants.MUTATION_RATE.value:  # check if mutation can be done
                child1Thresholds = self.mutation(child1Thresholds)
                child2Thresholds = self.mutation(child2Thresholds)

            child1Thresholds.sort()
            child2Thresholds.sort()

            child1 = Chromosome(self._kapur, thresholds=child1Thresholds)
            child2 = Chromosome(self._kapur, thresholds=child2Thresholds)

            newGeneration.append(child1)
            newGeneration.append(child2)

        if (round(np.random.random(1)[0], 2)) < Constants.LOCAL_SEARCH_RATE.value:
            newGeneration = self.ILS(newGeneration)

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

    def ILS(self, population: list):
        index = 0
        for y in range(1, len(population)): # pick best individual from the population
            if self.compareFitness(population[y], population[index]):
                index = y

        currSolution = Chromosome(self._kapur, thresholds=population[index].thresholds, fitness=population[index].fitness)

        for _ in range(int(Constants.LOCAL_ITERATIONS.value)):
            solution = self.perturbation(currSolution) # perturbation
            newSolution = self.localSearch(solution) # local search

            if self.compareFitness(newSolution, currSolution): # acceptance criteria
                currSolution = newSolution

        if self.compareFitness(currSolution, population[index]):
            population[index] = currSolution

        return population
    
    def localSearch(self, chromosone : Chromosome):
        length = len(chromosone.thresholds)
        bestFitness = chromosone.fitness
        counter = -length
        rangeList = []

        while (counter <= length):
            rangeList.append(counter)
            counter += 1

        index = np.random.randint(0, length)
        currentThreshold = chromosone.thresholds[index]
        bestThreshold = currentThreshold
        
        for thresholdRange in rangeList:
            newThreshold = 0

            if (index == 0):
                newThreshold = min(max(1, chromosone.thresholds[index] + thresholdRange), chromosone.thresholds[index + 1] - 1)
            elif (index == length - 1):
                newThreshold = max(min(254, chromosone.thresholds[index] + thresholdRange), chromosone.thresholds[index - 1])
            else:
                lower = chromosone.thresholds[index - 1] + 1
                upper = chromosone.thresholds[index + 1] - 1
                newThreshold = min(max(chromosone.thresholds[index] + thresholdRange, lower), upper)

            originalThreshold = chromosone.thresholds[index]
            chromosone.thresholds[index] = newThreshold
            chromosone.calculateFitness()

            if chromosone.fitness > bestFitness:
                bestFitness = chromosone.fitness
                bestThreshold = newThreshold
            chromosone.thresholds[index] = originalThreshold

        chromosone.thresholds[index] = bestThreshold

        chromosone.calculateFitness()
        return chromosone
    
    def perturbation(self, chromosone : Chromosome):
        chromosoneCopy = Chromosome(self._kapur, thresholds=chromosone.thresholds, fitness=chromosone.fitness)
        length = len(chromosoneCopy.thresholds) - 1
        index = np.random.randint(0, length + 1)
        newThreshold = None
        lower = 0
        upper = 0

        if (index == 0):
            lower = 1
            upper = chromosoneCopy.thresholds[index + 1]
            newThreshold = np.random.randint(lower, upper)

        elif (index == length):
            lower = chromosoneCopy.thresholds[index - 1] + 1
            upper = 255

        elif (index > 0 and index < length):
            lower = chromosoneCopy.thresholds[index - 1] + 1
            upper = chromosoneCopy.thresholds[index + 1]
            if (lower >= upper):
                lower = upper - 1

        newThreshold = np.random.randint(lower, upper)
        chromosoneCopy.thresholds[index] = newThreshold

        chromosoneCopy.calculateFitness()

        return chromosoneCopy

    def mutation(self, thresholds: list):
        # Random mutation
        childThresholds = thresholds
        index = np.random.randint(0, len(thresholds))

        if Constants.MUTATION_STRATEGY.value == 0:
            childThresholds[index] = np.clip(
                thresholds[index] + np.random.randint(-10, 10), 1, 255)
        elif Constants.MUTATION_STRATEGY.value == 1:
            childThresholds[index] = np.random.randint(1, 255)

        return childThresholds
    
    def tournamentSelection(self):
        bestIndividuals = []
        selectionCounter = 0

        while (selectionCounter < int(Constants.SELECTION_SIZE.value)):
            tournament = []
            tournamentCounter = 0

            while (tournamentCounter < int(Constants.TOURNAMENT_SIZE.value)):
                randomIndividual = self._generation[np.random.randint(0, len(self._generation))]
                if randomIndividual not in tournament:
                    tournamentCounter += 1
                    tournament.append(randomIndividual)
            
            bestIndividual = self.getBest(tournament)

            if bestIndividual not in bestIndividuals:
                selectionCounter += 1
                bestIndividuals.append(Chromosome(self._kapur, thresholds=bestIndividual.thresholds, fitness=bestIndividual.fitness))

        return bestIndividuals

    def eliteSelection(self, population: list):
        bestIndividuals = []
        counter = 0

        # Find best 4 individuals from original population to replace children
        sortedGeneration = self.reorderList(self.copyGeneration(), len(self._generation))
        for i in range(4):
            bestIndividuals.append(sortedGeneration[i])
        
        # temp = "("
        # for i in range(len(population)):
        #     temp += str(round(population[i].fitness, 4)) + ", "
        # temp += ")"
        # print("Length:", len(population))
        # print(temp)
        # print("--------------------------------------")

        # Find worst 4 children to replace (penultimate worst) - make sure not to overwrite elites just added in lmao
        indices = []
        for i in range(int(Constants.ELITIST_SIZE.value)):
            worst = 0
            for y in range(0, len(population)):
                if worst not in indices and self.compareFitness(population[worst], population[y]):
                    worst = y
            indices.append(worst)
            population[worst] = bestIndividuals[i]

        # temp = "("
        # for i in range(len(population)):
        #     if i in indices:
        #         temp += Fore.RED
        #     else: 
        #         temp += Style.RESET_ALL
        #     temp += str(round(population[i].fitness, 4)) + ", "
        # temp += Style.RESET_ALL + ")"
        # print("Length:", len(population))
        # print(temp, "\n")

        return population

    def repopulate(self, newGeneration):
        for x in range(int(Constants.POPULATION_SIZE.value)):
            self._generation[x] = newGeneration[x]

    def compareFitness(self, x: Chromosome, y: Chromosome):
        return x.fitness > y.fitness

    def reorderList(self, list, size):
        for i in range(size):
            swapped = False
            for j in range(size - i - 1):
                if self.compareFitness(list[j + 1], list[j]):
                    list[j], list[j + 1] = list[j + 1], list[j]
                    swapped = True

            if not swapped:
                break
        return list

    def copyGeneration(self):
        copy = []
        for i in range(len(self._generation)):
            temp = Chromosome(self._kapur, thresholds=self._generation[i].thresholds)
            copy.append(temp)
        return copy

    def getBest(self, selection : list):
        best = selection[0]
        for i in range(1, len(selection)):
            if self.compareFitness(selection[i], best):
                best = selection[i]
        return best

    def getBestIndividual(self):
        if self._bestIndividial is None:
            self._bestIndividial = self._generation[0]
        for i in range(0, int(Constants.POPULATION_SIZE.value)):
            if self.compareFitness(self._generation[i], self._bestIndividial):
                self._bestIndividial = self._generation[i]
        return self._bestIndividial