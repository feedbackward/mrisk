
# mrisk: learning with risks based on M-location

This repository houses software for recreating the numerical examples and experiments included in the following paper:

- Learning with risks based on M-location. Matthew J. Holland. *Preprint*.

The software here can be used to faithfully reproduce all the experimental results given in the above paper, and can also be easily applied to more general machine learning tasks, going well beyond the examples considered here.

A table of contents for this README file:

- Setup: initial software prepartion
- Setup: preparing the benchmark data sets
- Getting started
- Demos and visualization
- Safe hash value


<a id="setup_init"></a>
## Setup: initial software preparation

To begin, please ensure you have the prerequisite software used in the setup of our `mml` repository.

Next, make a local copy of the repository and create a virtual environment for working in as follows:

```
$ git clone https://github.com/feedbackward/mml.git
$ git clone https://github.com/feedbackward/mrisk.git
$ conda create -n mrisk python=3.8 jupyter matplotlib pip pytables scipy
$ conda activate mrisk
```

Having made (and activated) this new environment, we would like to use pip to install the supporting libraries for convenient access. This is done easily, by simply running

```
(mrisk) cd [mml path]/mml
(mrisk) pip install -e ./
```

with the `[mml path]` placeholder replaced with the path to wherever you placed the repositories. If you desire a safe, tested version of `mml`, just run

```
(mrisk) $ git checkout [safe hash mml]
```

and then do the `pip install -e ./` command mentioned above. The `[safe hash mml]` placeholder is to be replaced using the safe hash value given at the end of this document.


<a id="setup_data"></a>
## Setup: preparing the benchmark data sets

Please follow the instructions under <a href="https://github.com/feedbackward/mml#data">"Acquiring benchmark datasets"</a> using our `mml` repository. The rest of this README assumes that the user has prepared any desired benchmark datasets, stored in a local data storage directory (default path is `[path to mml]/mml/mml/data` as specified by the variable `dir_data_towrite` in `mml/mml/config.py`.

One __important__ step is to ensure that once you've acquired the benchmark data using `mml`, you must ensure that `mrisk` knows where that data is. To do this, set `dir_data_toread` in `setup_data.py` to the directory housing the HDF5 format data sub-directories (default setting: your home directory).


<a id="start"></a>
## Getting started

We have basically three types of files:

- __Setup files:__ these take the form `setup_*.py`.
  - Configuration for all elements of the learning process, with one setup file for each of the following major categories: learning algorithms, data preparation, learned model evaluation, loss functions, models, result processing, and general-purpose training functions.

- __Driver scripts:__ just one at present, called `learn_driver.py`.
  - This script controls the flow of the learning procedure and handle all the clerical tasks such as organizing, naming, and writing numerical results to disk. No direct modification to this file is needed to run the experiments in the above paper.

- __Execution scripts:__ these take the form `*_run.sh`.
  - The choice of algorithm, model, data generation protocol, among other key parameters is made within these simple shell scripts. Basically, parameters are specified explicitly, and these are then passed to the driver script as options.

Most of the demo notebooks are self-contained and can be run as-is. The experiments using real-world datasets require the user to run the driver script themselves; this is described in more detail in the relevant demo notebook.


<a id="demos"></a>
## List of demos

This repository includes detailed demonstrations to walk the user through re-creating the results in the paper cited at the top of this document. Below is a list of demo links which give our demos (constructed in Jupyter notebook form) rendered using the useful <a href="https://github.com/jupyter/nbviewer">nbviewer</a> service.

- <a href="https://nbviewer.jupyter.org/github/feedbackward/mrisk/blob/main/mrisk/demo_dev.ipynb">Visualizing key underlying functions</a>
- <a href="https://nbviewer.jupyter.org/github/feedbackward/mrisk/blob/main/mrisk/demo_sanity.ipynb">Demo: sanity check in one dimension</a>
- <a href="https://nbviewer.jupyter.org/github/feedbackward/mrisk/blob/main/mrisk/demo_linreg_1d.ipynb">Demo: linear regression in one dimension</a>
- <a href="https://nbviewer.jupyter.org/github/feedbackward/mrisk/blob/main/mrisk/demo_real_data.ipynb">Tests using benchmark datasets</a>


<a id="safehash"></a>
## Safe hash value

- Replace `[safe hash mml]` with `4e1735382c874eb639f5b8f0ea217bf48453b499`.

__Date of safe hash test:__ 2021/04/05.
