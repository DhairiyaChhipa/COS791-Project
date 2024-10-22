import random

class Chromosone:
    def __init__(self):
        self._chromosone = []
        
        # Populate the pipeline with random components
        for x in range(4):
            self._chromosone.append(self.getRandom(x))
            
    def getRandom(self, index):
        if (index == 2 or index == 3):
            return random.randint(0,1)
        return random.randint(0,2)