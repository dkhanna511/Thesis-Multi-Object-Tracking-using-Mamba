import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# import sys
# sys.path.append('/home/dheerajk/Research_DK/Mamba-MOT')

from datasets import MOT20DatasetOffset, MOT20DatasetBB
from torch.utils.data import DataLoader, random_split
# Initialize the dataset
dataset = MOT20DatasetOffset(path='MOT17/train', window_size=11)


print(" dataset[0] : ", dataset[0])
# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# Model parameters
input_size = 4  # Bounding box has 4 coordinates: [x, y, width, height]
hidden_size = 64 ## This one is used for LSTM NEtwork which I tried
output_size = 4  # Output also has 4 coordinates
num_layers = 3## For LSTM
embedding_dim = 128 ## For Mamba
num_blocks = 3 ## For Mamba
num_epochs = 200

warmup_steps = 4000 ## This is for custom warmuo schedular


# Define the split ratio
train_ratio = 0.8
val_ratio = 1 - train_ratio


from models_mamba import FullModelMambaOffset, BBoxLSTMModel
from schedulars import CustomWarmupScheduler

# torch.manual_seed(3000)  ## Setting up a seed where to start the weights (somewhat)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Initialize model

# Load data

model_used = "Mamba" ## OR LSTM

if model_used == "Mamba":
    type_mamba = "bi mamba" ## OR vanilla mamba
    model = FullModelMambaOffset(input_size,embedding_dim, num_blocks, output_size, model_used).to(device)
elif model_used == "LSTM":
    model = BBoxLSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

    
dataset = MOT20DatasetOffset(path='MOT17/train', window_size=10)


# for input, targets in dataset:
#     print(" input is : ", input)
# Calculate the number of samples for training and validationwindow_size
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


criterion = nn.SmoothL1Loss()  # Smooth L1 Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas = (0.9, 0.98), )



# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
scheduler = CustomWarmupScheduler(optimizer, d_model = embedding_dim, warmup_steps = warmup_steps)


# scheduler_after_warmup = StepLR(optimizer, step_size=30, gamma=0.1)

# warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=4000, initial_lr=0.001, warmup_lr=1e-6)

import time
print(" dataloader length is :", len(train_loader))
# exit(0)

best_model_path = 'best_model_offset.pth'

best_loss = float('inf')
# Training loop
print(" Model used to training: ", model_used) ## This is just a sanity printing check so that I dont have to see which loss came from which model later on or re-train it
for epoch in range(num_epochs):
    start_time = time.time()
    epoch_loss = 0.0  # Initialize epoch_loss
    model.train()
    for inputs, targets in train_loader:
        # Move tensors to the configured device
        inputs, targets = inputs.to(device), targets.to(device)
        # print("shape of inputs is : ", inputs.shape)
        targets = targets.float()
        # Forward pass
        outputs = model(inputs.float())
        # print(" outputs are :", outputs)
        
        # print(" shape of outputs is : ", outputs.shape)
        # print(" shape of targets is : ", targets.shape)
        loss = criterion(outputs, targets)
        # if torch.isnan(loss):
        #     continue
        epoch_loss += loss.item()  # Accumulate loss
        
        # print(" loss is :", loss.item())

        # print("output is : ", outputs)
        # print("target is : ", targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Step the warmup scheduler
        # if warmup_scheduler.current_step < warmup_scheduler.warmup_steps:
        #     warmup_scheduler.step()
        # else:
        #     # Step the standard scheduler after warmup
        #     scheduler_after_warmup.step()
        
        # Update the learning rate
        scheduler.step()
    
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():  ## No gradient so we dont update the weights and biases with test
        for data_valid, targets_valid in val_loader:
            data_valid, targets_valid  = data_valid.to(device), targets_valid.to(device)
            
            targets_valid = targets_valid.float()
            
            prediction_offset = model(data_valid.float()).to(device)
            # print(" prediction offset is : ", prediction_offset)
            loss = criterion(prediction_offset, targets_valid)
            # if torch.isnan(loss):
            #     continue
            validation_loss +=loss.item()
    
            
            
    print("Accumulated training loss is : ", epoch_loss)
    avg_loss = epoch_loss / len(train_loader)  # Calculate average loss for smooth epoch
    avg_valid_loss = validation_loss / len(val_loader)
    if avg_valid_loss < best_loss:
        best_loss = avg_valid_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model saved with loss: {best_loss:.4f}')
        
    end_time = time.time()
    time_taken = end_time - start_time
    print('Epoch [{}/{}], Train Loss: {} , Validation Loss : {} , Time Taken : {}'.format(epoch+1, num_epochs, avg_loss, avg_valid_loss, time_taken))
