import os
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import scipy.io as sio

import NW_LSTM 
import NN_DataLoader
import linear_std

def train_model(
        data_dir,
        material, 
        base_model_path, 
        device, 
        epochs, 
        valid_batch_size, 
        verbose=False,
        ):
        
    training_data_dir = os.path.join(data_dir, 'Processed Training Data') # Directory of pre-processed training data
    weight_dir = os.path.join(data_dir, 'Trained Weights') # Directory where trained weights checkpoint fles will be stored
    progress_dir = os.path.join(data_dir, 'Training Progress') # Directory that stores training loss plots and associated .MAT files to compare training regimes
    
    # Instantiate the model with appropriate dimensions
    model = NW_LSTM.get_global_model().to(device)

    # Print the model architecture and number of parameters
    if verbose:
        print(model)
        print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))

    # Load the pre-trained model if specified
    if base_model_path != '':
        model.load_state_dict(torch.load(base_model_path, map_location=torch.device(device)))
        print("Pre-trained model loaded")

    # Encode database specific standardisation coeffs to model .ckpt for use in future inferrence
    std_b = linear_std.linear_std()
    std_freq = linear_std.linear_std()
    std_temp = linear_std.linear_std()
    std_loss = linear_std.linear_std()
    std_b.load(os.path.join(training_data_dir, material, "std_b.stdd"))
    std_freq.load(os.path.join(training_data_dir, material, "std_freq.stdd"))
    std_temp.load(os.path.join(training_data_dir, material, "std_temp.stdd"))
    std_loss.load(os.path.join(training_data_dir, material, "std_loss.stdd"))
    model.std_b = (std_b.k, std_b.b)
    model.std_freq = (std_freq.k, std_freq.b)
    model.std_temp = (std_temp.k, std_temp.b)
    model.std_loss = (std_loss.k, std_loss.b)


    # Define the loss function and optimizer
    # loss_fn = nn.MSELoss()
    loss_fn = NW_LSTM.RelativeLoss()
    # loss_fn = NW_LSTM.RelativeLoss_abs()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0, last_epoch=-1)

    # Get training data loader
    train_dataloader = NN_DataLoader.get_dataLoader(os.path.join(training_data_dir, material, "train.mat"), batch_size=128)

    # Get validation data loader
    valid_dataloader = NN_DataLoader.get_dataLoader(os.path.join(training_data_dir, material, "valid.mat"), batch_size=valid_batch_size)
    valid_inputs, valid_targets = next(iter(valid_dataloader))
    valid_inputs, valid_targets = valid_inputs.to(device), valid_targets.to(device)

    # Estimate time used for training
    t0 = time.perf_counter()

    # Save the model with the lowest validation loss
    with torch.no_grad():
        valid_outputs = model(valid_inputs)
        # Compute loss
        minium_loss = loss_fn(valid_outputs, valid_targets)

    # Initialize the live plot
    train_losses = []
    valid_losses = []
    epochs_list = []
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim(0, epochs)

    def update_plot(epoch, train_loss, valid_loss):
        epochs_list.append(epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        ax.clear()
        ax.plot(epochs_list, train_losses, label='Training Loss')
        ax.plot(epochs_list, valid_losses, label='Validation Loss')
        plt.title(f'Material: {material}')
        ax.set_ylim(0, 1)
        ax.legend()
        clear_output(wait=True)
        display(fig)

    # Train the model
    for epoch in range(epochs):
        # Estimate time used for one epoch
        t_epoch = time.perf_counter() - t0
        t0 = time.perf_counter()

        # Train one epoch
        for i, (train_inputs, train_targets) in enumerate(train_dataloader):
            # Move data to device
            train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)

            # Forward pass
            train_outputs = model(train_inputs)

            # Compute loss
            loss = loss_fn(train_outputs, train_targets)

            # Backward pass and optimise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute validation loss
        if epoch > 0:
            with torch.no_grad():
                valid_outputs = model(valid_inputs)
                
                # Compute loss
                valid_loss = loss_fn(valid_outputs, valid_targets)

            if valid_loss < minium_loss:
                minium_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(weight_dir, f'{material}.ckpt')) # Saves improved model in folder specified by weight_dir
                print(f"  {material} Model saved , Validation Loss: {valid_loss.item():.3e}, lr: {optimizer.param_groups[0]['lr']:.3e}")
            update_plot(epoch + 1, loss.item(), valid_loss.item())  # Update live plot with new losses
        
        # Update LR
        scheduler.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0 and verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.3e}, "
                  f"Remaining time for material: {t_epoch / 60 * (epochs - epoch - 1):.1f} min")

    # Save and close the figure
    figname = os.path.join(progress_dir, f'{material}.pdf')
    plt.savefig(figname) # Save training loss plot as a PDF
    plt.ioff()
    plt.close()

    # Save the loss evolution for future analysis of training optimisation
    losses = {
        'train_losses': train_losses,
        'validation_losses': valid_losses
    }
    sio.savemat(os.path.join(progress_dir, f'{material}.mat'), losses) # Save .MAT file containing datapoints used in loss plot for future replotting
    
    