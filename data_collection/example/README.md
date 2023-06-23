# Description

## Requirements
The simulation model can only run on a Linux distribution as it is.

You will need to have the following libraries installed:
-Python3
-flopy==3.3.5
-pandas==1.4.2
-matplotlib==3.5.1
-scipy==1.8.0
-gdal==3.4.0
-numpy==1.22.3
-jupyter-notebook
-libgfortran5

## Running the example

In a terminal, inside folder `example`, run:
```
jupyter-notebook Example.ipynb
```

The notebook will open in your browser, and you will be able to use and run the different code cells.

## Docker
As the installation of ```gdal``` can be difficult and tricky, we provide a Dockerfile to build and run the jupyter notebook in an already set environment. See `instructions to run the Jupyter Notebook in Docker.md` for more details.