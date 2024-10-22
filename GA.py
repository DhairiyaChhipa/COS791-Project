from enum import Enum
import random
from Chromosone import Chromosone

class Constants(Enum):
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.3
    POPULATION_SIZE = 20
    GENERATIONS = 10
    TOURNAMENT_SIZE = 5
    SELECTION_SIZE = 5 # half of half of population size (20 / 2 -> [10] / 2 -> [5])
    ELITIST_SIZE = 10
    SEED = 1234

class Optimiser:
    def __init__(self, image):
        self._generation = []
        self._image = image
        random.seed(Constants.SEED.value)

    def __del__(self):
        for _ in range(Constants.POPULATION_SIZE.value):
            item = self._generation.pop()
            del item

        if (len(self._generation) == 0):
            del self._generation
        
    def start(self):
        for x in range(Constants.POPULATION_SIZE.value): # initialise generation
            self._generation.append(Chromosone())
            self.calculateFitness(self._generation[x])

        for _ in range(Constants.GENERATIONS.value):
            self.propagate()

        return self.getBest()
        
    def propagate(self): # handles selection and repopulation
        newGeneration = self.selectIndividuals()

        for _ in range(Constants.SELECTION_SIZE.value):
            if ((random.randint(1, 100) / 100) < Constants.CROSSOVER_RATE.value): # check if crossover can be done
                parent1 = random.randint(0, Constants.GENERATIONS.value - 1)
                parent2 = random.randint(0, Constants.GENERATIONS.value - 1)
                child1, child2 = self.crossover(self._generation[parent1], self._generation[parent2])
 
                newGeneration.append(child1)
                newGeneration.append(child2)

            else: # else do mutation:
                parent1 = random.randint(0, Constants.GENERATIONS.value - 1)
                parent2 = random.randint(0, Constants.GENERATIONS.value - 1)

                while (not (parent2 == parent1)):
                    parent2 = random.randint(0, Constants.GENERATIONS.value - 1)

                newGeneration.append(self.mutation(self._generation[parent1]))
                newGeneration.append(self.mutation(self._generation[parent2]))

        self.repopulate(newGeneration)

    def crossover(self, chromosone1 : Chromosone, chromosone2 : Chromosone):
        # Single point crossover
        child1 = chromosone1._chromosone.copy()
        child2 = chromosone2._chromosone.copy()
        crossoverPoint = random.randint(1, 2)
    
        for i in range(crossoverPoint, 4):
            child1[i], child2[i] = child2[i], child1[i] 

        return child1, child2

    def mutation(self, chromosone : Chromosone):
        # "Flip" bit mutation
        child = chromosone._chromosone.copy()
        index = random.randint(0,3)
        child[index] = chromosone.getRandom(index)
        return child
    
    def selectIndividuals(self):
        selectedPopulation = []
        self.reorderPopulation()

        for x in range(Constants.ELITIST_SIZE.value): # elitism, get best chromosones from population
            selectedPopulation.append(self._generation[x]._chromosone.copy())

        return selectedPopulation        
    
    def repopulate(self, newGeneration):
        for x in range(Constants.POPULATION_SIZE.value):
            self._generation[x]._chromosone = newGeneration[x]
            self.calculateFitness(self._generation[x])

    def compareFitness(self, x, y):
        if (x._fitness > y._fitness):
          return True
        else:
          return False

    def calculateFitness(self, chromosone: Chromosone):
        chromosone._fitness = 0 # SWAP THIS OUT

    def reorderPopulation(self):
        for x in self._generation:
            self.calculateFitness(x)

        n = Constants.POPULATION_SIZE.value
        for i in range(n):
            swapped = False

            for j in range(n - i - 1):
                if (self.compareFitness(self._generation[j+1], self._generation[j])):
                    self._generation[j], self._generation[j+1] = self._generation[j+1], self._generation[j]
                    swapped = True

            if not swapped:
                break

    def getBest(self):
        return self._generation[0]