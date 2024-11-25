
# MagLearn
## Award-winning solution to MagNet Challenge 2023 (3rd Place Model Performance)

MagLearn is a Python library designed for predicting volumetric power loss from magnetic materials using the MagNet dataset. The library leverages PyTorch with CUDA for training and model verification.

[MagNet Challenge](https://github.com/minjiechen/magnetchallenge]

Paper: [MagLearn – Data-driven Machine Learning Framework with Transfer and Few-shot Training for Modeling Magnetic Core Loss](https://ieeexplore.ieee.org/document/10751860)

## Features

- Sequence-to-scalar prediction of volumetric power loss from B timeseries, temperature and flux frequency. 
- Data preprocessing for standardizing and splitting datasets.
- Training pipeline compatible with both local machines and Google Colab.
- Model verification and error analysis.

## Requirements

- Python 3.x
- PyTorch with CUDA

## Installation

1. Install PyTorch with CUDA: [PyTorch Installation](https://pytorch.org/get-started/locally/)
2. Clone the repository and navigate to the directory:
    ```bash
    git clone <https://github.com/JunWang-Bristol/MagLearn>
    cd <repository-directory>
    ```

## Directory Structure

- **Working Directory**: Contains Jupyter notebooks and Python files.
- **raw_data_path**: The raw data directory housing the unprocessed MagNet dataset referenced by `batch_pre_process.ipynb`.
- **data_dir**: The output data directory that stores processed training data, trained models, and validation plots, referenced by all notebooks.

This absolute path structure is designed to accommodate large datasets stored on separate drives or Google Drive when running on Google Colab, allowing the working directory to remain lightweight. For simplicity, when using small datasets, these two data directories could also be stored within the working directory.

## Data Preprocessing

1. Place the raw data in the following structure, refer to example_material for csv format of each material dataset:
    ```
    raw_data_path >
        Material_0
            B_waveform[T].csv
            Frequency[Hz].csv
            Temperature[C].csv
            Volumetric_losses[Wm-3].csv
        ...
        Material_N
    ```
2. Run `batch_pre_process.ipynb` to preprocess data. This script:
    - Imports raw data using `Maglib.py`.
    - Resamples, standardizes, and exports processed data to `{data_dir}/Processed Training Data/{Material}/`.
    - Splits data into training, validation, and testing sets.

## Training Pipeline

1. **Google Colab**:
    - Upload the working directory and data directory to Google Drive.
    - Open `batch_training.ipynb` in Google Colab.
    - Set `data_dir` and `working_dir` paths to the respective directories in Google Drive.
    - Ensure GPU acceleration is enabled and run the notebook.

2. **Local Compute**:
    - Ensure a CUDA-compatible GPU is available.
    - Install necessary CUDA packages.
    - Run `batch_training.ipynb`, ensuring the correct 'data_dir' is used.

## Transfer Learning

- To enable transfer learning, set the `base_mat` variable to the desired base material. If transfer learning is not desired, set `base_mat` to an empty string (`''`).
- Adjust epochs and batch sizes as needed. Larger batch sizes can speed up training but may lead to overfitting if not carefully managed.
- Trained weights for each material are stored in respective `{material}.ckpt` files within the `Trained Weights` subfolder in the data directory.

## Verification Pipeline

1. Run `batch_verify.ipynb` to verify the model.
2. This script:
    - Uses `MagLoss()` from `MagNet.py` to infer losses from validation data.
    - Plots error distribution and exports error metrics.
    - Uses a maximum of 2000 samples from the validation dataset by default, adjustable by changing `max_samples`.

Output:

- Trained weights saved in the `Trained Weights` subfolder.
- Verification results saved as `model_errors.csv` and error histograms in the `Validation` subfolder.

## Loss Inference

The model can be deployed to predict core loss densities using the `MagNet.MagLoss_shiftflip()` function, which utilizes a model checkpoint, B waveform, temperature, and waveform frequency. This function ensures robustness to phase-shifted input data by performing inference multiple times, randomly shifting and flipping the B waveform.

The `deploy.ipynb` script provides a straightforward way to deploy the model, handling waveform timeseries import from a CSV file along with setting the temperature and frequency.

## Cite As

MagLearn - University of Bristol 2024

## References

MagNet Challenge 2023 https://github.com/minjiechen/magnetchallenge
