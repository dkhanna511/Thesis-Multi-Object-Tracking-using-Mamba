import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, random_split

from models_mamba import FullModelMambaBBox, BBoxLSTMModel
from schedulars import CustomWarmupScheduler
from datasets import MOT20DatasetBB
# torch.manual_seed(3000)  ## Setting up a seed where to start the weights (somewhat)
from torchvision.ops import generalized_box_iou_loss as GIOU_Loss

import iou_calc


# Model parameters
input_size = 4  # Bounding box has 4 coordinates: [x, y, width, height]
hidden_size = 64 ## This one is used for LSTM NEtwork which I tried
output_size = 4  # Output also has 4 coordinates
num_layers = 1## For LSTM
embedding_dim = 128 ## For Mamba
num_blocks = 3 ## For Mamba
num_epochs = 20

warmup_steps = 4000 ## This is for custom warmuo schedular


# Define the split ratio
train_ratio = 0.8
val_ratio = 1 - train_ratio


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Initialize model

# Load data

model_used = "Mamba" ## OR LSTM

if model_used == "Mamba":
    type_mamba = "bi mamba" ## OR vanilla mamba
    model = FullModelMambaBBox(input_size,embedding_dim, num_blocks, output_size).to(device)
elif model_used == "LSTM":
    model = BBoxLSTMModel(input_size, hidden_size, output_size, num_layers).to(device)



dataset_mot_bbox = MOT20DatasetBB(path='MOT17/train', window_size=11)


print(" dataset[0] : ", dataset_mot_bbox[0])
# Create a DataLoader
dataloader = DataLoader(dataset_mot_bbox, batch_size=64, shuffle=True)


criterion = nn.MSELoss()  # Mean squared error loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas = (0.9, 0.98), )
# criterion2 = GIOU_Loss
# Initialize model


# Calculate the number of samples for training and validationwindow_size
train_size = int(train_ratio * len(dataset_mot_bbox))
val_size = len(dataset_mot_bbox) - train_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset_mot_bbox, [train_size, val_size])

# Create DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# for data, targets in train_loader:
#     print(" data are : ", data)
# dataset


# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
scheduler = CustomWarmupScheduler(optimizer, d_model = embedding_dim, warmup_steps = warmup_steps)
lambda1 = 0.4
lambda2 = 0.6

# scheduler_after_warmup = StepLR(optimizer, step_size=30, gamma=0.1)

# warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=4000, initial_lr=0.001, warmup_lr=1e-6)

import time
print(" dataloader length is :", len(train_loader))
# exit(0)
best_model_path = "best_model_bbox.pth"
best_loss = float('inf')


# Training loop
num_epochs = 150
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
        # print(" targets are : ", targets)
        # Forward pass
        outputs = model(inputs.float())
        # print(" outputs are : ", outputs)
        # print(" shape of outputs is : ", outputs.shape)
        # print(" shape of targets is : ", targets.shape)
        loss_mse = criterion(outputs, targets)
        # loss_giou = GIOU_Loss(outputs, targets)
        loss_giou = GIOU_Loss(outputs, targets)
        
        total_loss = loss_mse + loss_giou.mean()
        # combined_loss = giou_weight * giou + mse_weight * mse

        epoch_loss += total_loss  # Accumulate loss
        # print(" loss is :", loss.item())

        # print("output is : ", outputs)
        # print("target is : ", targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        # loss_smooth_l1.backward()
        loss_mse.backward()
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
            
            # loss = criterion_bbox(prediction_offset, targets_valid)
            loss_mse = criterion(prediction_offset, targets_valid)
            # loss_giou = GIOU_Loss(outputs, targets)
            loss_giou = GIOU_Loss(prediction_offset, targets_valid)
        
            total_loss = loss_mse + loss_giou.mean()
        
            validation_loss += total_loss
            
            
    print("Accumulated training loss is : ", epoch_loss)
    avg_loss = epoch_loss / len(train_loader)  # Calculate average loss for the epoch
    avg_valid_loss = validation_loss / len(val_loader)
    
    if avg_valid_loss < best_loss:
        best_loss = avg_valid_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model saved with loss: {best_loss:.4f}')
    end_time = time.time()
    time_taken = end_time - start_time
    print('Epoch [{}/{}], Train Loss: {} , Validation Loss : {} , Time Taken : {}'.format(epoch+1, num_epochs, avg_loss, avg_valid_loss, time_taken))dataloader