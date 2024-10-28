from matplotlib.pylab import f
import numpy as np
import cv2
import matplotlib.pyplot as plt

from glob import glob

from RGA import GeneticAlgorithm as RGA
from HRGA import GeneticAlgorithmNeighbourSearch as HRGA
from Kapur import Kapur

# Helpers
def write_file(file_name, data):
    with open(file_name, 'a') as file:
        for line in data:
            file.write(f'{line}\n')

def write_image(path, image_name, image):
    cv2.imwrite(f'{path}{image_name}.png', image)

# Uniformity measure
def uniformity_measure(image, thresholds, n_thresholds):
    pixels = image.shape[0] * image.shape[1]
    # Histogram
    image = image.flatten()
    # Max grey level of pixels in the image
    max_grey_level = 255
    # Min grey level of pixels in the image
    min_grey_level = 0
    # for each segmented region
    grey_sum = 0
    for i in range(n_thresholds):
        region = None
        if i == 0:
            region = image[image <= thresholds[i]]
        elif i == n_thresholds - 1:
            region = image[image > thresholds[i - 1]]
        else:
            region = image[(image > thresholds[i - 1]) & (image <= thresholds[i])]
        # mean grey level of pixels in the region
        mean = np.mean(region)
        # for each pixel in the region
        for j in range(len(region)):
            grey_sum += (region[j] - mean) ** 2
    # Uniformity measure
    u = 1 - 2 * n_thresholds * (grey_sum / (pixels * (max_grey_level - min_grey_level) ** 2))
    return u

def run_trials(dataset, dataset_names, k, kapur, algorithm, file_name):
    print(f'Running {file_name}...')
    # Run GA
    k_levels = [2, 3, 4, 5]
    for i, image in enumerate(dataset):
        print(f'Image: {dataset_names[i]}')
        write_file(f'Results/{file_name}.txt', [f'===> Image: {dataset_names[i]} <==='])
        for k in k_levels:
            print(f'K Level: {k}')
            write_file(f'Results/{file_name}.txt', [f'==> K Level: {k}'])
            results = []
            results_text = []
            for j in range(10): # 100 trials
                kapur = Kapur(image.copy())
                algo = algorithm(image.copy(), k, kapur)
                best = algo.start()
                thr_string = ','.join([str(thr) for thr in best.thresholds])
                print(f'Trial: {j+1}, Thresholds: [{thr_string}], Fitness: {round(best.fitness, 4)}')
                results_text.append(f'Trial: {j+1}, Thresholds: [{thr_string}], Fitness: {round(best.fitness, 4)}')
                results.append(best)
            # Metrics
            best = max(results, key=lambda x: x.fitness)
            thr_string = ','.join([str(thr) for thr in best.thresholds])
            print(f'Best: [{thr_string}], Fitness: {round(best.fitness, 4)}')
            results_text.append(f'> Best: [{thr_string}], Best Fitness: {round(best.fitness, 4)}')
            fitness = [best.fitness for best in results]
            mean_fitness = np.mean(fitness)
            std_fitness = np.std(fitness)
            print(f'Mean Fitness: {round(mean_fitness, 4)}, Std Fitness: {round(std_fitness, 4)}')
            results_text.append(f'> Mean Fitness: {round(mean_fitness, 4)}, Std Fitness: {round(std_fitness, 4)}')
            # Uniformity measure
            u = uniformity_measure(image.copy(), best.thresholds, k)
            print(f'Uniformity Measure: {round(u, 4)}')
            results_text.append(f'> Uniformity Measure: {round(u, 4)}')
            # Save results
            print('Saving results...')
            write_file(f'Results/{file_name}.txt', results_text)
            # Save best image
            print('Saving best image...')
            colour_image = kapur.buildColorImage(image.copy(), best.thresholds)
            write_image(f'Results/{file_name}/{k}/', f'{dataset_names[i]}_K{k}_Best', colour_image)
    print(f'{file_name} Done!')

def main():
    print('Initializing...')
    # Seed - 
    seed = np.random.randint(0, 1000)
    # seed =
    np.random.seed(seed)
    print('Seed:', seed)
    write_file('Results/RGA.txt', [f'=== SEED: {seed} ==='])
    write_file('Results/HRGA.txt', [f'=== SEED: {seed} ==='])
    # Load images
    dataset = []
    dataset_names = []
    for image_path in glob('Dataset/*.png'):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        dataset_names.append(image_path.split('\\')[-1].split('.')[0])
        dataset.append(image)

    # Run trials
    # RGA
    run_trials(dataset.copy(), dataset_names, 2, Kapur, RGA, 'RGA')
    # HRGA
    run_trials(dataset.copy(), dataset_names, 2, Kapur, HRGA, 'HRGA')

    print('Done!')
    

if __name__ == '__main__':
    main()
