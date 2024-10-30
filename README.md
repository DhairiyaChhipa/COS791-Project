# COS791 Project

This is the code for our COS791 project. The project revolves around the Optimisation of a Genetic Algorithm for Multilevel Thresholding of Medical Images using a Neighbourhood Search. A Hybrid Real-coded Genetic Algorithm (HRGA) is implemented whereby Simulated Annealing (SA) was used as a neighborhood search component to optimise the RGA. This algorithm is compared to a classical Real-coded Genetic Algorithm (RGA) for the task of multilevel thresholding.

## Authors

This project was produced by: Tayla Orsmond, Ross Tordiffe and Dhairiya Chhipa.

## Getting Started

This project was developed using Python v3, which you will need to install here: 
https://www.python.org/downloads/

### Libraries:
This project makes use of the following libraries, please make sure these are installed before running.
This can be done by running the pip commands below.

  1. OpenCV v4.10:          pip install opencv-python
  2. numpy:                 pip install numpy
  3. matplotlib:            pip install matplotlib
  4. skimage:               pip install scikit-image
  5. glob:                  pip install glob2 

Alternatively, the following command can be run to install all dependencies that are listed in the requirements.txt file:

  pip install -r requirements.txt

### How to Run:
To run the project, open the terminal in the assignment root directory and run the following command:
  
  python main.py

This will run the main file which initialises the data, runs the RGA and HRGA and saves the results.
Please ensure the Dataset/ and Results/ folders are in the root directory (with the .py files). 

## Results:

All the resulting images and algorithm output are saved in the Results/ folder. This includes:
- The best thresholded image slices found by the RGA and HRGA algorithms for each k-level (in the RGA/ and HRGA/ folders respectively).
- The output of the RGA (RGA.txt).
- The output of the HRGA (HRGA.txt).