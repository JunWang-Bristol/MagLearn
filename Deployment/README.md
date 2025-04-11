# Deployment

This project is for users who do not wish to train a model but want to see the verification results using the trained checkpoint file provided. Follow these steps to run the verification process and review the output.

## Prerequisites

1. Environment Setup:

    Make sure you have set up the environment using the provided environment.yml file. If you haven’t done so, please follow these steps:

    - Clone the repository.

    - Create the environment by running:
        ```bash
        conda env create -f environment.yml
        conda activate MagLearn
        ```


2. Pre-Trained Model: `mdoel_colab.ckpt`


## How to Use

Lauch `Deployment_testing.ipynb` to generate `3C90_pred_loss.csv`, which can be checked against the `\Testing Data\Volumetri_Loss.csv`. This notebook also generates an error distribution plot to visualize the prediction error.

## Note

No Training Required: These files let you directly verify model outputs using the pre-trained checkpoint, without running the full training pipeline.