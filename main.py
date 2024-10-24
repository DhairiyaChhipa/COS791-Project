import numpy as np
import cv2
import matplotlib.pyplot as plt
from GA import GeneticAlgorithm as GA
from GANS import GeneticAlgorithmNeighbourSearch as GANS
from Kapur import Kapur

seed = np.random.randint(0, 1000)
np.random.seed(seed)

def main():
    image = cv2.imread('dataset/042.png', cv2.IMREAD_GRAYSCALE)
    threshold_count = 5
    kapur = Kapur(image)
    ga = GANS(image, threshold_count, kapur)
    best = ga.start()
    print(best.thresholds)
    print(round(best.fitness, 4))
    colour_image = kapur.buildColorImage(image, best.thresholds)
    plt.imshow(colour_image)
    plt.show()

if __name__ == '__main__':
    main()
