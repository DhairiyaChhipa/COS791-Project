import numpy as np
import cv2
import matplotlib.pyplot as plt
from GA import GeneticAlgorithm as GA
from Kapur import Kapur

seed = np.random.randint(0, 1000)
np.random.seed(seed)


def main():
    image = cv2.imread('dataset/022.png', cv2.IMREAD_GRAYSCALE)
    threshold_count = 5
    kapur = Kapur(image)
    ga = GA(image, threshold_count, kapur)
    best = ga.start()
    print(best.thresholds)
    print(best.fitness)
    colour_image = kapur.buildColorImage(image, best.thresholds)
    plt.imshow(colour_image)
    plt.show()

if __name__ == '__main__':
    main()
