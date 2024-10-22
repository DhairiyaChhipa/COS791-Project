import random


class Chromosome:
    def __init__(self):
        self._chromosome = []

        # Populate the pipeline with random components
        for x in range(4):
            self._chromosome.append(self.getRandom(x))

    def getRandom(self, index):
        if index == 2 or index == 3:
            return random.randint(0, 1)
        return random.randint(0, 2)
