# Context-Aware Approximate Scientific Computing

This repository is a companion piece to the article entitle "Context-Aware Approximate Scientific Computing". It provides the code used for collecting the data (as well as applying the loop aggregation technique), in folder ```data_collection```, and the code used for the evaluation in the ```evaluation``` folder. 

## Requirements

### Data collection

- Python3 
- flopy==3.3.5 
- pandas==1.4.2 
- matplotlib==3.5.1 
- scipy==1.8.0 
- gdal==3.4.0 
- numpy==1.22.3 
- jupyter-notebook 
- libgfortran5

### Evaluation
- pandas
- numpy
- sklearn


## Usage

To directly go to the results of the evaluation process, we recommend to open the corresponding jupyter notebooks in folder ```evaluation/notebooks```:
- ```CostPrediction.ipynb``` for the evaluation of the cost predictive model (RQ 1.1)
- ```ValidityPrediction.ipynb``` for the evaluation of the validity predictive model (RQ 1.2)
- ```Approach.ipynb``` for the evaluation of the overall approach (RQs 2 & 3 & 4)


For the data collection, we recommend going to the provided example in ```data_collection/example```. The necessary information to run it is given in the associated ```README.md```.
