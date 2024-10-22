import numpy as np
from Kapur import Kapur


class Chromosome:
    def __init__(self, kapur: Kapur, size: int = None, thresholds: list = None, fitness: float = 0):
        self.size = 0
        self.thresholds = []
        self.fitness = 0
        self.kapur = kapur
        if size is not None:  # Generate an entirely new chromosome
            self.size = size
            self.kapur = kapur
            self._generate()
        elif thresholds is not None:  # Copy an existing chromosome or create a new one with the given thresholds
            self.size = len(thresholds) + 1
            self.thresholds = thresholds
            self.fitness = fitness
            self.kapur = kapur
            if self.fitness == 0:
                self.fitness = self.kapur.run(self.thresholds)

    def _generate(self):
        for _ in range(self.size):
            self.thresholds.append(np.random.randint(1, 254))
            self.thresholds.sort()
            self.fitness = self.kapur.run(self.thresholds)

    def copy(self):
        threshold_copy = []
        for threshold in self.thresholds:
            threshold_copy.append(threshold)
        return Chromosome(kapur=self.kapur, size=self.size, thresholds=threshold_copy, fitness=self.fitness)

    def copyThresholds(self):
        threshold_copy = []
        for threshold in self.thresholds:
            threshold_copy.append(threshold)
        return threshold_copy
