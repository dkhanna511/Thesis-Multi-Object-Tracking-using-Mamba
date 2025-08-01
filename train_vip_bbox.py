import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, random_split

from models_mamba import FullModelMambaBBox, BBoxLSTMModel
from mambaAttention import FullModelMambaBBoxAttention
from schedulars import CustomWarmupScheduler
from datasets import MOTDatasetBB
# torch.manual_seed(3000)  ## Setting up a seed where to start the weights (somewhat)
from torchvision.ops import generalized_box_iou_loss as GIOU_Loss
import iou_calc
import wandb
import argparse
import training_utils

def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple argument parser example.")

    # Add arguments
    parser.add_argument('--dataset', type=str, required=True, choices = ["MOT20", "dancetrack", "sportsmot_publish", "VIP_HTD"],  help="Path to the dataset file.")
    parser.add_argument('--window_size', type = str, default = "variable", required = False, help = "Window size of sequence for tracklets")
    parser.add_argument('--max_window', type = int, default = 10, required = False, help = "Max Window Size")
    parser.add_argument('--num_blocks', type = int, default = 3, required = False, help = "Max Window Size")
    
    parser.add_argument('--model_type', type=str, choices = ["bi-mamba", "vanilla-mamba", "LSTM", "attention-mamba"], required = True, help = "model selection for testing" )
    parser.add_argument('--epochs', type=int,  default = 50, required = False, help = "number of epochs")
    parser.add_argument('--batch_size', type=int,  default = 64, required = False, help = "Batch size")
    parser.add_argument('--run_wandb', action="store_true",  help = "Log the training in wandb or not")
    parser.add_argument('--save_model', action="store_true",  help = "Save the model or not(no in case you're just testing something)")
    parser.add_argument('--device', type = str, required = True ,  help = "Mention cuda device : cuda:0 Or cuda:1)")
    parser.add_argument('--output_dir', type = str, required = True, help = "Saving DIR")

        
    # Parse the arguments
    args = parser.parse_args()
    
    output_dir = args.output_dir
    ## dataset parameters
    window_size = args.window_size



    # Model parameterst
    input_size = 4  # Bounding box has 4 coordinates: [x, y, width, height]
    hidden_size = 64 ## This one is used for LSTM NEtwork which I tried
    output_size = 4  # Output also has 4 coordinates
    num_layers = 1## For LSTM
    embedding_dim = 128 ## For Mamba
    num_blocks = args.num_blocks ## For Mamba


    # Training
    num_epochs = args.epochs
    warmup_steps = 4000 ## This is for custom warmup schedular
    batch_size = 64
    learning_rate = 0.0001
    lambda_criterion, lambda_criterion_2 = 50, 1
    betas_adam = (0.9, 0.98)
    augment = True
    augment_ratio = 0.3
    num_heads  = 4

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = args.device

    from functools import partial
    collate_fn_with_padding = partial(training_utils.custom_collate_fn_fixed, context_length= args.max_window)
    # Initialize model


    model_used = args.model_type

    if model_used == "bi-mamba" or model_used == "vanilla-mamba":
        # type_mamba = "bi-mamba" ## OR vanilla mamba
        model = FullModelMambaBBox(input_size,embedding_dim, num_blocks, output_size, mamba_type = model_used).to(device)
    elif model_used == "attention-mamba":
        # type_mamba = "bi-mamba" ## OR vanilla mamba
        model = FullModelMambaBBoxAttention(input_size,embedding_dim, num_blocks, output_size, num_heads = num_heads, mamba_type = model_used).to(device)
        print(" yes im here")
    elif model_used == "LSTM":
        model = BBoxLSTMModel(input_size, hidden_size, output_size, num_layers).to(device)


    
    ## Define the dataset
    dataset_train_bbox = MOTDatasetBB(path='datasets/VIP-HTD/train', window_size = window_size, max_window= args.max_window, augment = augment, augment_ratio = augment_ratio )
    dataset_val_bbox = MOTDatasetBB(path="datasets/VIP-HTD/val", window_size = window_size, max_window = args.max_window,  augment = False)

    print(" dataset[0] : \n", dataset_train_bbox[0])

    # exit(0)

    # Create DataLoaders for training and validation sets
    if window_size !="variable":
        print("here?>??????")
        train_dataloader = DataLoader(dataset_train_bbox, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset_val_bbox, batch_size = batch_size, shuffle = False)
        # exit(0)
    else:
        train_dataloader = DataLoader(dataset_train_bbox, batch_size=batch_size, shuffle=True, collate_fn = collate_fn_with_padding)
        val_dataloader = DataLoader(dataset_val_bbox, batch_size = batch_size, shuffle = False, collate_fn = collate_fn_with_padding)


    # criterion = nn.MSELoss()  # Mean squared error loss
    criterion = nn.SmoothL1Loss()
    criterion_2 = iou_calc.CIOU_Loss_Perplexity

    optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate, betas = betas_adam, weight_decay=1e-5)

    ### We're not using the random split as we do have seperate folder of validation of dancetrack in the datasets


    # for data, targets in train_dataloader:
    #     print(" data are : ", data)
    # dataset


    scheduler = CustomWarmupScheduler(optimizer, d_model = embedding_dim, warmup_steps = warmup_steps)
    # lambda1 = 0.4
    # lambda2 = 0.6

    # scheduler_after_warmup = StepLR(optimizer, step_size=30, gamma=0.1)

    # warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=4000, initial_lr=0.001, warmup_lr=1e-6)


    # exit(0)
    
    if args.run_wandb:
        wandb.init(
            project='mamba-VIP-HTD-bbox',   # Set your project name
            name =  model_used + "_ep_" + str(num_epochs) + "_ws_" +  str(args.max_window) + "_block_" +  str(args.num_blocks)  ,
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
                "augment_ratio" : augment_ratio,
                "max_window" : args.max_window,
                }
            )   
    print(" dataloader length is :", len(train_dataloader))


    configs = {'epochs': num_epochs, 'optimizer': optimizer, 'criterion' : (criterion, criterion_2), 'scheduler' : scheduler, 
               'lambda_criterion_1' : lambda_criterion, 'lambda_criterion_2': lambda_criterion_2, 'device':device, 
               "max_window" : args.max_window, "num_blocks" : num_blocks}
    

    if args.window_size =="variable":
        training_utils.train_var_window(args, model, train_dataloader, val_dataloader, configs, output_dir)
    else:
        training_utils.train_const_window(args, model, train_dataloader, val_dataloader, configs)

    # print(" Model used to training: ", model_used) ## This is just a sanity printing check so that I dont have to see which loss came from which model later on or re-train it
        
    if args.run_wandb:
        wandb.finish()
        
        
            
if __name__ == "__main__":
    main()