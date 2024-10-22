import numpy as np
from Kapur import Kapur


class Chromosome:
    def __init__(self, size: int = None, kapur: Kapur = None, thresholds: list = None, fitness: float = 0):
        self.thresholds = []
        self.fitness = 0
        # Default Initialisation
        if size and kapur is not None:
            for _ in range(size):
                self.thresholds.append(np.random.randint(1, 254))
                self._fitness = kapur.run(self.thresholds)
            return
        # Copy Initialisation
        elif thresholds and fitness is not None:
            for i in range(size):
                self.thresholds.append(thresholds[i])
            self._fitness = fitness
            return
        # Empty Initialisation
        else:
            return
