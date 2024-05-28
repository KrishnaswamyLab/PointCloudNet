# Manifold Filter-Combine Networks


## Install

You need poetry:

You will first need to install a python environment.

You will need to install conda (python and virtual environment manager) and poetry (packages manager)

Link for conda installation[https://docs.conda.io/en/latest/miniconda.html]

Link for poetry installation[https://python-poetry.org/docs/#installation]

Once they are both installed, you can just do

`conda create -n mfcn python=3.9`

Then,

`poetry install` 

It should now be installed.


## Example

We use hydra for the configuration of the runs. To run legs on the Modelnet dataset with a dense graph construction, you can use

`poetry run python train.py model=legs data=modelnet graph_construct=dense_graph`

