# Data collection

## Description
This is the repository associated to the submission of the paper entitled "Context-Aware Approximate Scientific Computing". It contains the code used for collecting the data about the simulation executions, as well as running the simulations by applying the loop aggregation technique. The code used for the experimentation execution is located in the folder `experimentation`. We provide a notebook with an example to launch and run a simulation with the Modflow model with an approximation rate in the folder `example` (See `README.md` in that folder).


## Structure

```
├── Experimentation_Results.csv
├── study_sites.txt
├── README.md
|
├── experimentation
│   ├── docker-simulation
│   ├── environment.yml
│   ├── run_H_ind.sh
│   └── run_simulations.sh
|
└── example
    ├── Dockerfile
    ├── Example.ipynb
    ├── environment.yml
    ├── instructions to run the Jupyter Notebook in Docker.md
    ├── modflow
    └── requirements.txt

```



### Experimental_Results.csv
The file contains for each simulation:
- the id number of the geographical site
- the upscaling factor used
- the value of the control metric expressed in meters
- the execution time (Time) expressed in seconds
- the number of simulation per hour (Nb Simu Per Hour)

### study_sites.txt
The files contains the id numbers, the names of the geographical sites, and the coordinates (expressed in the Lambert system unit).
