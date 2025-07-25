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
import training_utils


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
    parser.add_argument('--device', type = str, required = True ,  help = "Mention cuda device : cuda:0 Or cuda:1)")

        
    # Parse the arguments
    args = parser.parse_args()
    
        
    ## dataset parameters
    window_size = args.window_size

    # Model parameters
    input_size = 4  # Bounding box has 4 coordinates: [x, y, width, height]
    hidden_size = 64 ## This one is used for LSTM NEtwork which I tried
    output_size = 4  # Output also has 4 coordinates
    num_layers = 4## For LSTM
    embedding_dim = 128 ## For Mamba
    num_blocks = 3 ## For Mamba


    # Training loop
    num_epochs = args.epochs
    warmup_steps = 4000 ## This is for custom warmup schedular
    batch_size = 64
    learning_rate = 0.001
    lambda_criterion, lambda_criterion_2 = 50, 1
    betas_adam = (0.9, 0.98)


    # Define the split ratio
    train_ratio = 0.8
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    # Initialize model
    model_used = args.model_type
    if model_used == "bi-mamba" or model_used == "vanilla-mamba":
        # type_mamba = "bi-mamba" ## OR vanilla mamba
        model = FullModelMambaBBox(input_size,embedding_dim, num_blocks, output_size, mamba_type = model_used).to(device)
    elif model_used == "LSTM":
        model = BBoxLSTMModel(input_size, hidden_size, output_size, num_layers).to(device)


    ## Define the dataset
    dataset_mot_bbox = MOTDatasetBB(path='datasets/MOT20/train', window_size=window_size)

    criterion = nn.MSELoss()  # Mean squared error loss
    criterion_2 = iou_calc.CIOU_Loss_Perplexity

    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate, betas = betas_adam, )
    # Initialize model


    # Calculate the number of samples for training and validationwindow_size
    train_size = int(train_ratio * len(dataset_mot_bbox))
    val_size = len(dataset_mot_bbox) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset_mot_bbox, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=training_utils.custom_collate_fn_fixed)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn= training_utils.custom_collate_fn_fixed)
    # for data, targets in train_loader:
    #     print(" data are : ", data)
    # dataset


    scheduler = CustomWarmupScheduler(optimizer, d_model = embedding_dim, warmup_steps = warmup_steps)

    # scheduler_after_warmup = StepLR(optimizer, step_size=30, gamma=0.1)

    # warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=4000, initial_lr=0.001, warmup_lr=1e-6)


    print(" dataloader length is :", len(train_dataloader))
    # exit(0)
    
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
                "loss_function": (criterion.__class__.__name__, criterion_2.__name__),
                "lambda_criterion_1" : lambda_criterion, 
                "lambdacriterion_2" : lambda_criterion_2 ,
                }
            )   


    configs = {'epochs': num_epochs, 'optimizer': optimizer, 'criterion' : (criterion, criterion_2), 'scheduler' : scheduler, 
               'lambda_criterion_1' : lambda_criterion, 'lambda_criterion_2': lambda_criterion_2, 'device':device}
    


    if args.window_size =="variable":
        training_utils.train_var_window(args, model, train_dataloader, val_dataloader, configs)
    else:
        training_utils.train_const_window(args, model, train_dataloader, val_dataloader, configs)

    # print(" Model used to training: ", model_used) ## This is just a sanity printing check so that I dont have to see which loss came from which model later on or re-train it
        

    if args.run_wandb:    
        wandb.finish()
    
    
if __name__ == "__main__":
    main()