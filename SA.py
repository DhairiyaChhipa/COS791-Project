from enum import Enum
import math
import random
from Kapur import Kapur

class Constants(Enum):
    COOLING_RATE = 0.90
    INITIAL_TEMP = 100
    STOP_TEMPERATURE = 1
    SEED = 12345

class SA():
    def __init__(self, image):
        self._image = image
        self._segmentationMethod = Kapur(image)
        self._temperature = Constants.INITIAL_TEMP.value
        random.seed(Constants.SEED.value)

    def __del__(self):
        if (self._segmentationMethod is not None):
            del self._segmentationMethod
        self._image = None

    def run(self, thresholds):
        return super().run(thresholds)
    
    def reset(self):
        self._temperature = Constants.INITIAL_TEMP.value

    def start(self, level):
        self.reset()
        bestSolution = self.generateSolution(level)
        bestCost = self.calculateFitness(bestSolution)

        while (self._temperature > Constants.STOP_TEMPERATURE.value):
            newSolution = self.localSearch(bestSolution)
            newCost = self.calculateFitness(newSolution)
            deltaCost = newCost - bestCost

            if (self.compareFitness(newCost, bestCost)):
                bestSolution = newSolution
                bestCost = newCost
            else:
                if (self.accept(deltaCost)):
                    bestSolution = newSolution
                    bestCost = newCost

            self._temperature *= Constants.COOLING_RATE.value

        return bestSolution, bestCost
    
    def accept(self, delta):
        if delta < 0:
            return True
        else:
            randomValue = random.uniform(0, 0.99)
            if (randomValue < math.exp(-delta / self._temperature)):
                return True
            else:
                return False
            
    def localSearch(self, solution):
        newSolution = solution.copy()
        length = len(newSolution)
        index = random.randint(0, length - 1)
        thresholdRange = random.choice([-1, 1])
        
        if (index == 0):
            newSolution[index] = max(1, newSolution[index] + thresholdRange)
        elif (index == length - 1):
            newSolution[index] = min(254, newSolution[index] + thresholdRange)
        else:
            lower = newSolution[index - 1] + 1
            upper = newSolution[index + 1] - 1
            newSolution[index] = min(max(newSolution[index] + thresholdRange, lower), upper)
        
        return newSolution
    
    def buildImage(self, thresholds, type):
        if (type == "g"):
            return self._segmentationMethod.buildGrayImage(self._image, thresholds)
        elif (type == "c"):
            return self._segmentationMethod.buildColorImage(self._image, thresholds)
        
    def compareFitness(self, currFitness, bestFitness): # Maximising function
        return currFitness > bestFitness
    
    def setImage(self, image):
        self._image = image
        self._segmentationMethod.setImage(image)