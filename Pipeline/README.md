
# Pipeline

This project implements an end-to-end pipeline for predicting power loss from raw measurements of magnetic flux density, frequency and temperature. It is structured into three stages: data pre-processing, model training, and performance verification.

## Key Files

1. `User_dataTransform.ipynb`

    - Reads raw data files and convert all parameters into .mat format, which needed for the model by reshaping, random splitting and standardization. 

    - Save standardization parameters for later use in unstandardizing model predictions during verification.

2. `User_training.ipynb`

    - Train machine learning model.

    - Learning rate and number of epoches can be adjusted to optimize convergence.

3. `User_verify.ipynb`

    - Load saved checkpoint abd verfivation dataset. Computes relative error and visualizes performance statistics.


## Installation

1. Install PyTorch with CUDA: [PyTorch Installation](https://pytorch.org/get-started/locally/)

2. Clone the repository and navigate to the directory:
    ```bash
    git clone <https://github.com/JunWang-Bristol/MagLearn2.git>
    cd <repository-directory>
    ```

3. Install Anaconda: [Anaconda Installation](https://www.anaconda.com/download) (Optional)

4. Download Anaconda environment `environment.yml` from Github and import into your project via Conda Navigator: (Optional)
    ```bash
    conda env create -f environment.yml
    conda activate Maglearn
    ```


## How to use

1. Prepare your raw data as CSV files with headers and match the following column headers:
    ```bash
    B_waveform[T]
    H_waveform[Am-1]
    Temperature[C]
    Frequency[Hz]
    Volumetric_losses[Wm-3]
    ```

    Additional samples can be downloadeed from [MagNet Challenge 1](https://www.princeton.edu/~minjie/magnet.html).

2. Configure paths:
    In `path_config.py`, set your raw dataset paths (`rawDataPath`) to match your local environment and material name (`material_name`).

3. Data transformation:

    Run `User_dataTransform.ipynb` to preprocess data. This notebook automatically generates a folder for each material, where all processed files (like standardized datasets, checkpoint, etc.) are stored.

4. Training:

    We highly recommend using a pre-trained model to improve the convergence and accuracy of your checkpoint file. For instance, `model_colab.ckpt` for 3C90 and 3C94 are provided in their respective folders.

    1. **Local computer**:

        Run `User_training.ipynb`.

    2. **Google Colab**:
        - Download Google Drive: [Google Drive](https://support.google.com/a/users/answer/13022292?hl=en)
        - Upload your project to Google Drive.
        - (Option: First-time use in Google Colab)

            Open `mount_drive.ipynb` and set the path variable to the folder in Google Drive (`/content/drive/MyDrive/[YourFolderName]`) where `User_training.ipynb` is located. Run to mount your Google Drive, 

        - In your Google Drive, select Google Colaboratory as the opening method of `User_training.ipynb`. Run this notebook.


5. Verification:

    Run `User_verify.ipynb` to verify the model. This notebook infer losses from validation data and plot error distribution.


## Output:

In each material subfolder:

- Plot of error distributions in the `plot` subfolder.
- The trained model checkpoint `model_colab.ckpt`.
