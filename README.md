

-------------------------------------------------------------------------------------------------------------------------------------------------------
# Egrecho: ASV-Subtools2 Project
![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![pytorch version](https://img.shields.io/badge/pytorch-2.0+-lightgreen.svg)
![Lightning version](https://img.shields.io/badge/Lightning-2.0+-lightgreen.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-lightblue.svg)](https://opensource.org/licenses/Apache-2.0)
![sys](https://img.shields.io/badge/sys-Linux-9cf)

## Introduction
Egrecho is a Python-based open-source toolkit that reimplements the [asv-subtools](https://github.com/Snowdar/asv-subtools). It is designed to help you efficiently create, customize, and develop new models within sub-projects, referred to as [**recipes**][recipes]. See [here](https://wangers.github.io/subtools2) for more introduction.

## Installation

This repository is currently tested on Python 3.8+ and Linux OS.
<details>
<summary>Installation Setup</summary>

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

</details>

## Recipes
The following is a list of available speech-related sub-projects:

1. [VoxcelebSRC][voxcelebSRC]

    Task: **Automatic Speaker Verification** (ASV)

    Speaker recognition on the popular Voxceleb [dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html#about).



    <details>
    <summary>Features</summary>

    * Online Datasets, Online feature extractor + online augmentation.
    * Models
        + [x] [ECAPA X-vector](https://arxiv.org/abs/2005.07143)
        + [x] [CamPPlus](https://arxiv.org/abs/2303.00332)
    * Back-End
        + [x] Cosine Similarity
        + [x] Score Normalization: [S-Norm](http://www.crim.ca/perso/patrick.kenny/kenny_Odyssey2010_presentation.pdf), [AS-Norm](https://www.researchgate.net/profile/Daniele_Colibro/publication/221480280_Comparison_of_Speaker_Recognition_Approaches_for_Real_Applications/links/545e4f6e0cf295b561602c42/Comparison-of-Speaker-Recognition-Approaches-for-Real-Applications.pdf)
        + [x] Metric: EER, Cavg, minDCF
    * Large-Margin Finetune
    </details>

2. [Whisper Finetune][whisper_finetune]

    Task: **Automatic Speech Recognition** (ASR)

    Fine-tune openai [Whisper](https://arxiv.org/abs/2212.04356)  more conveniently.


    <details>
    <summary>Features</summary>

    - [x]  [Lhotse](https://github.com/lhotse-speech/lhotse) dataset
    - [x]  Full/lora tune
    - [x]  [Transformers](https://github.com/huggingface/transformers) trainer
    - [x]  Multi-GPU traning based on accelerate, including DDP and deepspeed zero
    - [x]  Multilingual training
    - [ ]  New language fine-tune
    </details>
3. [VALL-E][vall-e]

    Task: **Text-To-Speech** (TTS)

    [Vall-e](https://arxiv.org/abs/2301.02111) is a pioneering GPT-like framework that treats TTS as a language modeling task. This recipe is an unofficial PyTorch implementation, which can serve as a baseline model for further refinements and advancements.
    <details>
    <summary>Features</summary>

   - [x]  Llama framework upgraded
   - [x]  TTS Demo & Inference
   - [x]  Multi-GPU traning based on [Lightning](https://github.com/Lightning-AI/pytorch-lightning) trainer
   - [x]  Automatic metrics (SV + ASR)
   - [ ]  Webui
   </details>

## Documentation
 - [Home page](https://wangers.github.io/subtools2)
 - [Installation](https://wangers.github.io/subtools2/tutorial_installation.html)
 - [How to develop dynamic project](https://wangers.github.io/subtools2/tutorial_dynamic_project.html)
 - [API Reference](https://wangers.github.io/subtools2/api/api.html)


## Code Structure
Egrecho is organized into the following **main directories**:
+ [**egrecho**][egrecho]: Source python package.
    <details>
    <summary>Click to show package list</summary>
    <!-- following section will be skipped from PyPI description -->

    + [**core**][egrecho/core]: Primary code for the framework, defines the base data builder, model, fit teacher, optimizer, scheduler, etc.
    + [**data**][egrecho/data]: Code related to building datasets.
    + **models**: Contains code of various models.
    + [**nn**][egrecho/nn]: Contains some custom components/layers.
    + [**pipeline**][egrecho/pipeline]: Interacts with model in inference mode.
    + [**score**][egrecho/score]: Metrics the performance.
    + [**training**][egrecho/training]: Training-related code, including optimizations, callbacks, etc.
    + [**utils**][egrecho/utils]: Contains a number of frequently used utility methods.
    </details>
    <!-- end skipping PyPI description -->

+ [**egrecho_cli**][egrecho_cli]: Includes a number of useful one-step command line scripts, interactive with user directly.
+ [**recipes**][recipes]: Provides examples of projects.
+ **runtime**: Designed for deployment.
+ [**scripts**][scripts]: Houses some common shell script utilities.



[egrecho]: egrecho
[egrecho/core]: egrecho/core
[egrecho/data]: egrecho/data
[egrecho/models]: egrecho/models
[egrecho/nn]: egrecho/nn
[egrecho/pipeline]: egrecho/pipeline
[egrecho/score]: egrecho/score
[egrecho/training]: egrecho/training
[egrecho/utils]: egrecho/utils
[egrecho_cli]: egrecho_cli
[recipes]: recipes
[scripts]: scripts

[voxcelebSRC]: recipes/voxcelebSRC
[whisper_finetune]: recipes/whisper_finetune
[vall-e]: recipes/vall-e
