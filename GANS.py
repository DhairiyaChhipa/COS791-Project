import numpy as np
from enum import Enum
from Chromosome import Chromosome
from Kapur import Kapur

class Constants(Enum):
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.3
    # LOCAL_SEARCH_RATE = 0.4
    GENERATIONS = 100
    LOCAL_ITERATIONS = 30
    POPULATION_SIZE = 26
    ELITIST_SIZE = 4
    TOURNAMENT_SIZE = 5
    SELECTION_SIZE = (POPULATION_SIZE / 2) + 1
    MUTATION_STRATEGY = 0  # 0 = +- 10, 1 = random int (1, 254)

class GeneticAlgorithmNeighbourSearch:
    def __init__(self, image, threshold_count: int, kapur: Kapur):
        self._generation = []
        self._image = image
        self._threshold_count = threshold_count
        self._kapur = kapur

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
            bestIndividual = self.getBest(self._generation, int(Constants.POPULATION_SIZE.value))
            print("best fitness: ", bestIndividual.fitness)

        return bestIndividual

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

        self.ILS(newGeneration)
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
        randomIndex = np.random.randint(0, len(population))
        temp = Chromosome(self._kapur, thresholds=population[randomIndex].thresholds, fitness=population[randomIndex].fitness)
        currSolution = self.localSearch(temp) # local search
        # print(currSolution.thresholds)

        for _ in range(int(Constants.LOCAL_ITERATIONS.value)):
            # print("\n-----------------")

            solution = self.perturbation(currSolution) # perturbation
            # print(solution.thresholds)
            newSolution = self.localSearch(solution) # local search            
            # print(newSolution.thresholds)

            # print("-----------------\n")

            if self.compareFitness(newSolution, currSolution): # acceptance criteria
                currSolution = newSolution

        # CHECK IF FITNESS FUNCTION GETS PROPERLY UPDATED

        if self.compareFitness(currSolution, population[randomIndex]):
            print("============================== Improvement")
            population[randomIndex] = currSolution
    
    def localSearch(self, chromosone : Chromosome):
        length = len(chromosone.thresholds)
        index = np.random.randint(0, length)
        thresholdRange = np.random.choice([-1, 1])

        if (index == 0):
            chromosone.thresholds[index] = min(max(1, chromosone.thresholds[index] + thresholdRange), chromosone.thresholds[index + 1] - 1)
        elif (index == length - 1):
            chromosone.thresholds[index] = max(min(254, chromosone.thresholds[index] + thresholdRange), chromosone.thresholds[index - 1])
        else:
            lower = chromosone.thresholds[index - 1] + 1
            upper = chromosone.thresholds[index + 1] - 1
            chromosone.thresholds[index] = min(max(chromosone.thresholds[index] + thresholdRange, lower), upper)

        return chromosone
    
    def perturbation(self, chromosone : Chromosome):
        length = len(chromosone.thresholds) - 1
        index = np.random.randint(0, length + 1)
        newThreshold = None
        lower = 0
        upper = 0

        if (index == 0):
            lower = 1
            upper = chromosone.thresholds[index + 1]
            newThreshold = np.random.randint(lower, upper)

        elif (index == length):
            lower = chromosone.thresholds[index - 1] + 1
            upper = 255

        elif (index > 0 and index < length):
            lower = chromosone.thresholds[index - 1] + 1
            upper = chromosone.thresholds[index + 1]

        newThreshold = np.random.randint(lower, upper)
        chromosone.thresholds[index] = newThreshold

        return chromosone

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
            
            bestIndividual = self.getBest(tournament, len(tournament))

            if bestIndividual not in bestIndividuals:
                selectionCounter += 1
                bestIndividuals.append(Chromosome(self._kapur, thresholds=bestIndividual.thresholds, fitness=bestIndividual.fitness))

        return bestIndividuals

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

    def getBest(self, list, size):
        best = list[0]
        for i in range(1, size):
            if self.compareFitness(list[i], best):
                best = list[i]

        return best