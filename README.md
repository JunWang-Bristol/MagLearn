
# MagLearn

MagLearn is a Python library designed for predicting volumetric power loss from magnetic materials using the MagNet dataset. The library leverages PyTorch with CUDA for training and model verification.

## Features

- Sequence-to-scalar prediction of volumetric power loss.
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

1. Place the raw data in the following structure, refer to example_material for csv format of each dataset:
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
    - Run `batch_training.ipynb` and verify CUDA GPU recognition.

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

## Output

- Trained weights saved in the `Trained Weights` subfolder.
- Verification results saved as `model_errors.csv` and error histograms in the `Validation` subfolder.

## References

[Specify References Here]