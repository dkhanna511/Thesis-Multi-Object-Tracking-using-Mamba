import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, random_split

from models_mamba import FullModelMambaBBox, BBoxLSTMModel
from schedulars import CustomWarmupScheduler
from datasets import MOTDatasetBB
# torch.manual_seed(3000)  ## Setting up a seed where to start the weights (somewhat)
from torchvision.ops import generalized_box_iou_loss as GIOU_Loss
import iou_calc
import wandb
import time
import argparse
from torch.nn.utils.rnn import pad_sequence

# Custom collate function for handling batches with variable-length inputs
def custom_collate_fn(batch):
    # Split the batch into inputs, sequence_names, and targets
    inputs, targets, sequence_names = zip(*batch)
    # print("input shape is : ", inputs.shape)
    # Pad the inputs (they are assumed to be lists of tensors or tensors themselves)
    # Find the maximum length of the inputs
    padded_inputs = pad_sequence(inputs, batch_first=True)
    # print(" padded inputs", padded_inputs.shape)
    # Convert the sequence_names and targets back to tensors if they are lists
    # sequence_names = torch.stack(sequence_names)  # Assuming sequence_names are tensors
    targets = torch.stack(targets)  # Assuming targets are tensors
    
    return padded_inputs, targets, sequence_names
    

def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple argument parser example.")

    # Add arguments
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--window_size', type = str, default = 10, required = False, help = "Window size of sequence for tracklets")
    parser.add_argument('--model_type', type=str, choices = ["bi-mamba", "vanilla-mamba", "LSTM"], required = True, help = "model selection for testing" )
    parser.add_argument('--epochs', type=int,  default = 50, required = False, help = "number of epochs")
    parser.add_argument('--batch_size', type=int,  default = 64, required = False, help = "Batch size")
    parser.add_argument('--run_wandb', action="store_true",  help = "Log the training in wandb or not")
    parser.add_argument('--save_model', action="store_true",  help = "Save the model or not(no in case you're just testing something)")
    
        
    # Parse the arguments
    args = parser.parse_args()
    
        
    ## dataset parameters
    window_size = args.window_size



    # Model parameterst
    input_size = 4  # Bounding box has 4 coordinates: [x, y, width, height]
    hidden_size = 64 ## This one is used for LSTM NEtwork which I tried
    output_size = 4  # Output also has 4 coordinates
    num_layers = 1## For LSTM
    embedding_dim = 128 ## For Mamba
    num_blocks = 3 ## For Mamba


    # Training loop
    num_epochs = args.epochs
    warmup_steps = 4000 ## This is for custom warmup schedular
    batch_size = 64
    learning_rate = 0.001
    betas_adam = (0.9, 0.98)


    # Define the split ratio
    train_ratio = 0.8
    val_ratio = 1 - train_ratio

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # Initialize model
    model_used = args.model_type

    if model_used == "bi-mamba" or model_used == "vanilla-mamba":
        # type_mamba = "bi-mamba" ## OR vanilla mamba
        model = FullModelMambaBBox(input_size,embedding_dim, num_blocks, output_size, mamba_type = model_used).to(device)
    elif model_used == "LSTM":
        model = BBoxLSTMModel(input_size, hidden_size, output_size, num_layers).to(device)


    ## Define the dataset
    dataset_train_bbox = MOTDatasetBB(path='datasets/dancetrack/train')
    dataset_val_bbox = MOTDatasetBB(path="datasets/dancetrack/val")

    print(" dataset[0] : ", dataset_train_bbox[0])


    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(dataset_train_bbox, batch_size=batch_size, shuffle=True, collate_fn = custom_collate_fn)
    val_dataloader = DataLoader(dataset_val_bbox, batch_size = batch_size, shuffle = False, collate_fn = custom_collate_fn)


    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate, betas = betas_adam, )

    ### We're not using the random split as we do have seperate folder of validation of dancetrack in the datasets


    # for data, targets in train_loader:
    #     print(" data are : ", data)
    # dataset


    scheduler = CustomWarmupScheduler(optimizer, d_model = embedding_dim, warmup_steps = warmup_steps)
    # lambda1 = 0.4
    # lambda2 = 0.6

    # scheduler_after_warmup = StepLR(optimizer, step_size=30, gamma=0.1)

    # warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=4000, initial_lr=0.001, warmup_lr=1e-6)


    print(" dataloader length is :", len(train_loader))
    # exit(0)
    best_model_path = "best_model_bbox_dancetrack_{}.pth".format(model_used)
    best_loss = float('inf')

    # Initialize W&B
    if args.run_wandb:
        wandb.init(
            project='mamba-dancetrack-bbox',   # Set your project name
            name =  model_used + "_ep_" + str(num_epochs) + "_ws_" +  window_size,
            config={                # Optional: set configurations
                'epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'betas' : betas_adam,
                'optimizer' : optimizer.__class__.__name__,
                'architecture': model.__class__.__name__,
                'model_type' : model_used,
                "loss_function": (criterion.__class__.__name__, "GIOU Loss"),
                "lambda_mse" : 1, 
                "lambda_giou" : 1 ,
                }
            )   


    print(" Model used to training: ", model_used) ## This is just a sanity printing check so that I dont have to see which loss came from which model later on or re-train it
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0  # Initialize epoch_loss
        model.train()
        epoch_loss_mse = 0.0
        epoch_loss_giou =0.0
        for inputs, targets, sequences in train_loader:
            # Move tensors to the configured device
            # print("inputs shape is :", inputs)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # inputs, targets = inputs.to(device), targets.to(device)
            # print("shape of inputs is : ", inputs.shape)
            targets = targets.float()
            # print(" targets are : ", targets)
            # Forward pass
            outputs = model(inputs.float())
            # print(" outputs are : ", outputs)
            # print(" shape of outputs is : ", outputs.shape)
            # print(" shape of targets is : ", targets.shape)
            loss_mse = criterion(outputs, targets)
            loss_giou_func = iou_calc.giou_loss(outputs, targets)
            
            total_loss = loss_giou_func + loss_mse
            # combined_loss = giou_weight * giou + mse_weight * mse
            
            epoch_loss_giou +=loss_giou_func    
            epoch_loss_mse +=loss_mse
            epoch_loss += total_loss# Accumulate loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            # loss_smooth_l1.backward()
            loss_giou_func.backward(retain_graph = True)
            loss_mse.backward()
            
            # total_loss.backward()
            
            optimizer.step()
            scheduler.step()
            # Step the warmup scheduler
            # if warmup_scheduler.current_step < warmup_scheduler.warmup_steps:
            #     warmup_scheduler.step()
            # else:
            #     # Step the standard scheduler after warmup
            #     scheduler_after_warmup.step()
            
            # Update the learning rate
            
        print(" MSE Loss : ", epoch_loss_mse/len(train_loader))
        print(" GIOU Loss: ", epoch_loss_giou/len(train_loader))
        
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():  ## No gradient so we dont update the weights and biases with test
            for data_valid, targets_valid, sequences in val_dataloader:
                data_valid, targets_valid  = data_valid.to(device), targets_valid.to(device)
                
                targets_valid = targets_valid.float()
                
                prediction_offset = model(data_valid.float()).to(device)
                
                loss_mse = criterion(prediction_offset, targets_valid)
                loss_giou = iou_calc.giou_loss(prediction_offset, targets_valid)
            
                total_loss = loss_mse + loss_giou
                validation_loss += total_loss
                
        print("Accumulated training loss is : ", epoch_loss)
        avg_loss = epoch_loss / len(train_loader)  # Calculate average loss for the epoch
        avg_valid_loss = validation_loss / len(val_dataloader)
        
        if abs(avg_valid_loss) < abs(best_loss):
            best_loss = avg_valid_loss
            if args.save_model:
                torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with loss: {best_loss:.4f}')
        end_time = time.time()
        time_taken = end_time - start_time
        if args.run_wandb:
            wandb.log({'epoch': epoch + 1, 'training loss': avg_loss, 'validation loss': avg_valid_loss})

        print('Epoch [{}/{}], Train Loss: {} , Validation Loss : {} , Best Loss : {}, Time Taken : {}'.format(epoch+1, num_epochs, avg_loss, avg_valid_loss, best_loss, time_taken))
        
    if args.run_wandb:
        wandb.finish()
        
        
        
        

    
    
if __name__ == "__main__":
    main()