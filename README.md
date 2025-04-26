
# MagLearn

## Award-winning solution to MagNet Challenge 2023 (3rd Place Model Performance)

MagLearn is a Python library designed for predicting volumetric power loss from magnetic materials using the MagNet dataset. The library leverages PyTorch with CUDA for training and model verification.

[MagNet Challenge 2023](https://github.com/minjiechen/magnetchallenge)

Paper: [MagLearn – Data-driven Machine Learning Framework with Transfer and Few-shot Training for Modeling Magnetic Core Loss](https://research-information.bris.ac.uk/en/publications/maglearn-data-driven-machine-learning-framework-with-transfer-and)

## Features

- Sequence-to-scalar prediction of volumetric power loss from B timeseries, temperature and flux frequency. 
- Data preprocessing for standardizing and splitting datasets.
- Training pipeline compatible with both local machines and Google Colab.
- Model verification and error analysis.

## Requirements

- Python 3.x
- PyTorch with CUDA
- Anaconda (Optional if you have already installed your project's environment)

## Overview 

This figure illustrates an example inference workflow: the trained model takes the preprocessed B waveform together with scalar frequency and temperature inputs, and outputs the predicted volumetric power loss.

![Pipeline Overview](<Pipeline Overview-1.jpg>)

## Depolyment

The Deployment section focuses on the verification process `(User_verify.ipynb)`:

- Loads the trained model checkpoint and compare with the verification dataset to evaluate model performance.

- Computes relative errors and generates visualizations of error distributions.


## Pipeline

The Pipeline encompasses the entire workflow, including:

- Data Transformation `(User_dataTransform.ipynb)`:

    Preprocesses raw data by reshaping, random splitting, and standardizing.  Saves processed dataset and standardization parameters for later verification.

- Model Training `(User_training.ipynb)`:

    Train the machine learning model. Allows adjustment of learning rate and number of epochs for optimal convergence.

- Model Verification `(User_verify.ipynb)`:

    Verifies the model's performance using the saved checkpoint and computes error metrics.



## Cite As

MagLearn - University of Bristol 2024

## References

MagNet Challenge 2023 https://github.com/minjiechen/magnetchallenge