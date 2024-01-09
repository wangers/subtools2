# Installation

This repository is currently tested on Python 3.8+ and Linux OS.

## 1. Virtual Environment

It is **strongly** recommended to install repo in a virtual environment using [conda](https://www.anaconda.com/).
```bash
conda create -n egrecho python=3.8
conda activate egrecho
```

## 2. Pre-requirement
Refer [pytorch](https://pytorch.org/) to install torch and torchaudio. An example could be:

```bash
# install torch which is cuda available
pip install torch==1.13.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
Additionally, we recommond to install
lightning manually refer to [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).

## 3. Install Repo
### Developer Installation

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
### Python package Installation

Installation package without modifications:

```bash
pip install https://github.com/wangers/subtools2.git
```
