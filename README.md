PPREPIE: PREdictive maintenance PIPEline
=========================================

PPREPIE is an advanced predictive maintenance pipeline evaluated in the automotive case study. 
The code is given to support the paper:

> Danilo Giordano, Flavio Giobergia, Eliana Pastor, Antonio La Macchia,Tania Cerquitelli, Elena Baralis, Marco Mellia, Davide Tricarico, (2020), "Data-Driven Strategies for Predictive Maintenance: Lesson Learned from an Automotive Use Case", Computers in Industry.
please refer to it for the main concepts and for citation. 

![PREPIPE Framework](PREPIPE.png)

## Prerequisites

* The jupyter notebooks run on Linux, with: Python 3.7, sklearn 0.22 pandas 0.25.3 numpy 1.17.4 numpy 1.4.1.
* The feature extraction process runs on Linux, with: Python 3.7, vest-python, and tsfel.
* The grid search notebooks run on Spark version 2.4.0-cdh6.2.1

## Data samples

* data/ contains samples of the data. 

* Each cycle C0, C1, C2 is a CSV file where the first row is the header containing the signal names, while all the following rows store the samples of all signals reordered by Program A.

* cycle_order is a CSV file where the first row is the header with the cycle name and label, and all the following rows store the name of the cycle and the assigned label according to Program B. 
* Cycles in this file must be sorted by acquisition time. 

* All the code, except for the Unsupervised signal selection, run with tabular data. So either ad-hoc tabular data can be provided, or the 1c-DatasetCreation notebook must be used to transform cycle data into tabular data. 

* An example of tabular data is available in 1-SignalSelection/dataset/All.pkl.
* This pickle file is a pandas dataframe, where the header contains: ExpID (the name of the cycles), all the features, label. All the following rows contain the cycles' data.   

## Code
### In each notebook, the header reports the description of how to use it. 
### All the notebooks relying on cycle data assume that the data are stored in the data/ folder.


* 0-ValidationTestDivision: contains the jupyter notebook to compute the CAI Index.

* 1-SignalSelection: contains the jupyter notebooks to compute all the signal selection algorithms presented in the paper. To run, follow the alphabetic order. 

* 2-Windowing: contains the jupyter notebook to split the cycles into different time windows with different sizes.

* 3-FeatureSelection: contains the jupyter notebook to rank the features according to the FS algorithm.

* 4-Historicization: contains the jupyter notebook to create the dataset with historical features.

* 5-ModelTrainingTuning: contains the jupyter notebooks to run the grid search performing either the k-fold cross validation or the time series cross validation in D1 and the hold out validation in D2. 

* 6-DeepLearning: contains the scripts to create and validate the deep learning models.

* classes/parameters/ConfGenerator: the jupyter notebook creates the grid search space for each hyperparameter of the tree, forest, SVM classifier.

* classes/public/makerDatasetSpecialized: implements the code for the different feature extraction strategies. 

* Each step is used to create tabular datasets based on each step choice. 
* Since PREPIPE is based on a wrapping approach, to identify the best choice in each step (1,..,4), the created datasets must be tested with the 5-ModelTrainingTuning notebook. 
* For the identification of the best choice for each step, in 5-ModelTrainingTuning/gridresult, we report the notebooks to analyze the grid search results of each step. 

* As a grid search result, 5-ModelTrainingTuning/gridresult/1-SignalSelection/ reports two examples of grid search results for the 10-fold Cross Validation (CV) case and Time Series Cross Validation (TS) case. Please refer to the 5-ModelTrainingTuning/gridresult/README_Output.md for a complete overview of the output file.
