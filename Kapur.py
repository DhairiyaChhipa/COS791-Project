import math
from matplotlib import pyplot as plt
import numpy as np


class Kapur(object):
    def __init__(self, image):
        self._histogram = None
        self.histogram = None
        self._probabilities = None
        self.image = image

    def __del__(self):
        del self._histogram
        del self.histogram
        del self._probabilities
        del self.image

    def run(self, thresholds):
        self.generateHistogram(self.image)
        classProbabilities = self.calculateProbabilities(thresholds)
        entropy = self.calculateEntropy(thresholds, classProbabilities)
        return entropy

    def calculateProbabilities(self, thresholds):
        classProbabilities = []

        for x in range(len(thresholds) + 1):
            previousThreshold, nextThreshold = self.getThresholds(x, thresholds)

            classProbability = 0
            for i in range(previousThreshold, nextThreshold + 1):
                classProbability += self._probabilities[i]

            classProbabilities.append(classProbability)

        return classProbabilities

    def calculateEntropy(self, thresholds, classProbabilities):
        totalEntropy = 0

        for x in range(len(thresholds) + 1):
            previousThreshold, nextThreshold = self.getThresholds(x, thresholds)
            entropy = 0

            for i in range(previousThreshold, nextThreshold + 1):
                if self._probabilities[i] > 0:
                    ln = self._probabilities[i] / classProbabilities[x]
                    entropy += ln * math.log(ln)

            totalEntropy += -entropy

        return totalEntropy

    def generateHistogram(self, image):
        self.histogram = {}
        self._probabilities = {}
        totalPixels = image.shape[0] * image.shape[1]

        for i in range(0, 256):
            count = np.count_nonzero(image == i)
            self.histogram.update({i: count})

        for i in range(0, 256):
            self._probabilities.update({i: self.histogram[i] / totalPixels})

    def buildColorImage(self, image, thresholds):
        colourImage = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint32)
        k = len(thresholds)

        red = [255, 0, 0]
        orange = [255, 165, 0]
        yellow = [255, 255, 0]
        green = [0, 255, 0]
        blue = [0, 0, 255]
        purple = [160, 32, 420]

        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                currPixel = image[row][col]

                if currPixel >= 0 and currPixel <= thresholds[0]:
                    colourImage[row][col] = red
                elif currPixel > thresholds[k - 1] and currPixel <= 255:
                    colourImage[row][col] = purple

                if 1 < k and currPixel >= thresholds[0] and currPixel <= thresholds[1]:
                    colourImage[row][col] = orange

                elif 2 < k and currPixel >= thresholds[1] and currPixel <= thresholds[2]:
                    colourImage[row][col] = yellow

                elif 3 < k and currPixel >= thresholds[2] and currPixel <= thresholds[3]:
                    colourImage[row][col] = green

                elif 4 < k and currPixel >= thresholds[3] and currPixel <= thresholds[4]:
                    colourImage[row][col] = blue

        return colourImage

    def buildGrayImage(self, image, thresholds):
        grayImage = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        grayLevels = []
        k = len(thresholds)

        for x in range(len(thresholds) + 1):
            previousThreshold, nextThreshold = self.getThresholds(x, thresholds)
            classMean = 0
            totalPixels = 0

            for intensity, frequency in self._histogram.items():
                if intensity >= previousThreshold and intensity < nextThreshold:
                    classMean += intensity * frequency
                    totalPixels += frequency

            grayLevels.append(classMean // totalPixels)

        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                currPixel = image[row][col]
                if currPixel >= 0 and currPixel <= thresholds[0]:
                    grayImage[row][col] = grayLevels[0]
                elif currPixel > thresholds[k - 1] and currPixel <= 255:
                    grayImage[row][col] = grayLevels[len(grayLevels) - 1]

                if 1 < k and currPixel >= thresholds[0] and currPixel <= thresholds[1]:
                    grayImage[row][col] = grayLevels[1]

                elif 2 < k and currPixel >= thresholds[1] and currPixel <= thresholds[2]:
                    grayImage[row][col] = grayLevels[2]

                elif 3 < k and currPixel >= thresholds[2] and currPixel <= thresholds[3]:
                    grayImage[row][col] = grayLevels[3]

                elif 4 < k and currPixel >= thresholds[3] and currPixel <= thresholds[4]:
                    grayImage[row][col] = grayLevels[4]

        return grayImage

    def displayImages(self, image1, image2):
        fig, axs = plt.subplots(1, 2, figsize=(15, 9))
        axs[0].imshow(image1, cmap=plt.cm.gray)
        axs[1].imshow(image2, cmap=plt.cm.gray)

        axs[0].axis('off')
        axs[1].axis('off')

        axs[0].set_title('Original')
        axs[1].set_title('Segmented Image')
        plt.show()

    def displayHistogramImage(self, image):
        plt.hist(image.ravel(), 256, [0, 256])
        plt.show()

    def getThresholds(self, index, thresholds):
        previousThreshold = None
        nextThreshold = None

        if (index == 0):
            previousThreshold = 0
            nextThreshold = thresholds[index]

        elif (index == len(thresholds)):
            previousThreshold = thresholds[index - 1] + 1
            nextThreshold = 255

        else:
            previousThreshold = thresholds[index - 1] + 1
            nextThreshold = thresholds[index]

        return previousThreshold, nextThreshold
