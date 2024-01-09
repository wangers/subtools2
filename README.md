

-------------------------------------------------------------------------------------------------------------------------------------------------------
# Egrecho: ASV-subtools2 project


## Introduction

## Installation

This repository is currently tested on Python 3.8+ and Linux OS.

### 1. Virtual Environment

It is **strongly** recommended to install repo in a virtual environment using [conda](https://www.anaconda.com/).
```bash
conda create -n egrecho python=3.8
conda activate egrecho
```

### 2. Pre-requirement
Refer [pytorch](https://pytorch.org/) to install torch and torchaudio. An example could be:

```bash
# install torch which is cuda available
pip install torch==1.13.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
Additionally, we recommond to install
lightning manually refer to [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).

### 3. Install Repo
#### Developer Installation

To set up a development environment, run the following commands to clone and install:

```bash
git clone https://github.com/wangers/subtools2.git subtools2
cd subtools2
pip install -e .

or

# Format code tidy
pip install -e .[dev]
pre-commit install
```
After installation, use `egrecho -h` to show some commands.
#### Python package Installation

Installation package without modifications:

```bash
pip install https://github.com/wangers/subtools2.git
```

## Documentation
 - [Home page](https://wangers.github.io/subtools2)
 - [Installation](https://wangers.github.io/subtools2/tutorial_installation.html)
 - [How to develop dynamic project](https://wangers.github.io/subtools2/tutorial_dynamic_project.html)
 - [API Reference](https://wangers.github.io/subtools2/api/api.html)

## Code Structure
Egrecho is organized into the following **main directories**:
+ **egrecho**: Source python package.
    <details>
    <summary>Package list</summary>
    <!-- following section will be skipped from PyPI description -->

    + **core**: Primary code for the framework, defines the base data builder, model, fit teacher, optimizer, scheduler, etc.
    + **data**: Code related to building datasets.
    + **models**: Contains code of various models.
    + **nn**: Contains some custom components/layers.
    + **pipeline**: Interacts with model in inference mode.
    + **score**: Metrics the performance.
    + **training**: Training-related code, including optimizations, callbacks, etc.
    + **utils**: Contains a number of frequently used utility methods.
    </details>
    <!-- end skipping PyPI description -->

+ **egrecho_cli**: Includes a number of useful one-step command line scripts, interactive with user directly for training, extract embedding, scores, etc.
+ **recipes**: Provides examples of projects.
+ **runtime**: Designed for deployment.
+ **scripts**: Houses shell script utilities (most of which are based on Kaldi).
