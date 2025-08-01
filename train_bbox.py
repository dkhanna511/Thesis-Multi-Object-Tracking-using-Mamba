import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, random_split

import time
from models_mamba import FullModelMambaBBox, BBoxLSTMModel
from schedulars import CustomWarmupScheduler
from datasets import MOTDatasetBB
# torch.manual_seed(3000)  ## Setting up a seed where to start the weights (somewhat)
from torchvision.ops import generalized_box_iou_loss as GIOU_Loss
import iou_calc
import wandb


### Dataset parameters
window_size = 10


# Model parameters
input_size = 4  # Bounding box has 4 coordinates: [x, y, width, height]
hidden_size = 64 ## This one is used for LSTM NEtwork which I tried
output_size = 4  # Output also has 4 coordinates
num_layers = 1## For LSTM
embedding_dim = 128 ## For Mamba
num_blocks = 3 ## For Mamba

# Training loop
num_epochs = 20
warmup_steps = 4000 ## This is for custom warmuo schedular
batch_size = 64
loss_fn = "MSE LOSS, GIOU LOSS" ## Just mentioning it here coz I have to add it in log file of wandb
learning_rate = 0.001
betas_adam = (0.9, 0.98)
# optimizer_name = "Adam"
# Define the split ratio
train_ratio = 0.8
val_ratio = 1 - train_ratio

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# Initialize model
model_used = "Mamba" ## OR LSTM

if model_used == "Mamba":
    type_mamba = "bi mamba" ## OR vanilla mamba
    model = FullModelMambaBBox(input_size,embedding_dim, num_blocks, output_size).to(device)
elif model_used == "LSTM":
    model = BBoxLSTMModel(input_size, hidden_size, output_size, num_layers).to(device)



dataset_mot_bbox = MOTDatasetBB(path='MOT17/train', window_size=window_size)


print(" dataset[0] : ", dataset_mot_bbox[0])
# Create a DataLoader
dataloader = DataLoader(dataset_mot_bbox, batch_size=64, shuffle=True)


criterion = nn.MSELoss()  # Mean squared error loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas = betas_adam, )
# criterion2 = GIOU_Loss
# Initialize model


# Calculate the number of samples for training and validation
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

print(" dataloader length is :", len(train_loader))
# exit(0)
best_model_path = "best_model_bbox_MOT17.pth"
best_loss = float('inf')


# Initialize W&B
wandb.init(
    project='mamba-mot17-bbox',   # Set your project name
    config={                # Optional: set configurations
        'epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'betas' : betas_adam,
        'optimizer' : optimizer.__class__.__name__,
        'architecture': model.__class__.__name__,
        "loss_function": (criterion.__class__.__name__, "GIOU Loss"),
        "lambda_mse" : 1, 
        "lambda_giou" : 1 ,
        }
    )   




print(" Model used to training: ", model_used) ## This is just a sanity printing check so that I dont have to see which loss came from which model later on or re-train it
for epoch in range(num_epochs):
    start_time = time.time()
    epoch_loss = 0.0  # Initialize epoch_loss
    epoch_giou_loss = 0.0
    epoch_mse_loss = 0.0
    
    model.train()
    for inputs, targets, sequences in train_loader:
        # Move tensors to the configured device
        inputs, targets = inputs.to(device), targets.to(device)
        # print("shape of inputs is : ", inputs.shape)
        targets = targets.float()
        # Forward pass
        outputs = model(inputs.float())
        
        loss_mse = criterion(outputs, targets)
        loss_giou = iou_calc.giou_loss(outputs, targets)
        
        total_loss = loss_mse + loss_giou
        
        epoch_giou_loss += loss_giou
        epoch_mse_loss +=loss_mse
        
        epoch_loss += total_loss  # Accumulate loss
        
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss_giou.backward(retain_graph = True)
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
        for data_valid, targets_valid, sequences in val_loader:
            data_valid, targets_valid  = data_valid.to(device), targets_valid.to(device)
            
            targets_valid = targets_valid.float()
            
            prediction_offset = model(data_valid.float()).to(device)
            
            loss_mse = criterion(prediction_offset, targets_valid)
            loss_giou = iou_calc.giou_loss(prediction_offset, targets_valid)
        
            total_loss = loss_mse + loss_giou
        
            validation_loss += total_loss
            
    print("GIOU Loss : ", epoch_giou_loss/len(train_loader))
    print("MSE Loss : ", epoch_mse_loss/len(train_loader))
    print("Accumulated training loss is : ", epoch_loss)
    avg_loss = epoch_loss / len(train_loader)  # Calculate average loss for the epoch
    avg_valid_loss = validation_loss / len(val_loader)
    
    if abs(avg_valid_loss) < abs(best_loss):
        best_loss = avg_valid_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model saved with loss: {best_loss:.4f}')
    end_time = time.time()
    time_taken = end_time - start_time  
    wandb.log({'epoch': epoch + 1, 'training loss': avg_loss, 'validation loss': avg_valid_loss})

    print('Epoch [{}/{}], Train Loss: {} , Validation Loss : {} , Time Taken : {}'.format(epoch+1, num_epochs, avg_loss, avg_valid_loss, time_taken))
    
    
wandb.finish()