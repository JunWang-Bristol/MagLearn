import os
import torch
import torch.optim as optim
import time

import NW_LSTM
import NN_DataLoader

def train_model(preprocessed_training_dataset_path, material, base_mat, model_saved_name, platform, device, epochs, valid_batch_size, verbose=False, load_pretrained=False):
    # Instantiate the model with appropriate dimensions
    model = NW_LSTM.get_global_model().to(device)

    # Print the model architecture and number of parameters
    if verbose:
        print(model)
        print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))

    # Load the pre-trained model if specified
    if load_pretrained:
        try:
            model.load_state_dict(torch.load(os.path.join(preprocessed_training_dataset_path, base_mat, model_saved_name)))
            print("Pre-trained model loaded")
        except FileNotFoundError:
            print(f"No pre-trained model found for {base_mat}, starting from scratch")
    
    # Define the loss function and optimizer
    # loss_fn = nn.MSELoss()
    # loss_fn = NW_LSTM.RelativeLoss()
    # loss_fn = NW_LSTM.RelativeLoss_abs()
    loss_fn = NW_LSTM.RelativeLoss_95()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)

    # Get training data loader
    train_dataloader = NN_DataLoader.get_dataLoader(os.path.join(preprocessed_training_dataset_path, material, "train.mat"), batch_size=128)

    # Get validation data loader
    valid_dataloader = NN_DataLoader.get_dataLoader(os.path.join(preprocessed_training_dataset_path, material, "valid.mat"), batch_size=valid_batch_size)
    valid_inputs, valid_targets = next(iter(valid_dataloader))
    valid_inputs, valid_targets = valid_inputs.to(device), valid_targets.to(device)

    # Estimate time used for training
    t0 = time.perf_counter()

    # Save the model with the lowest validation loss
    with torch.no_grad():
        valid_outputs = model(valid_inputs)
        # Compute loss
        minium_loss = loss_fn(valid_outputs, valid_targets)

    # Train the model
    for epoch in range(epochs):
        t_epoch = time.perf_counter() - t0
        t0 = time.perf_counter()

        # Train one epoch
        for i, (train_inputs, train_targets) in enumerate(train_dataloader):
            train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)

            train_outputs = model(train_inputs)
            loss = loss_fn(train_outputs, train_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute validation loss
        if epoch > 0:
            with torch.no_grad():
                valid_outputs = model(valid_inputs)
                valid_loss = loss_fn(valid_outputs, valid_targets)

            if valid_loss < minium_loss:
                minium_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(preprocessed_training_dataset_path, material, model_saved_name))
                print(f"  {material} Model saved , Validation Loss: {valid_loss.item():.3e}, lr: {optimizer.param_groups[0]['lr']:.3e}")

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.3e}, "
                  f"Remaining time for material: {t_epoch / 60 * (epochs - epoch - 1):.1f} min")
