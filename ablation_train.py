import subprocess

import time
import torch
import gc

# max_window = ["5", "7", "10", "12", "15"]
num_blocks = ["4", "5"]
# num_blocks = ["3", "4", "5"]

# max_window = ["10", "12", "15"]
max_window  = ["15", "10"]

# max_window = ["5", "7"]

command_default_dance = "python train_dancetrack_bbox.py --epochs 60 --model_type attention-mamba --window_size variable --device cuda:1 --save_model --dataset dancetrack --run_wandb"

command_default_sports ="python train_sportsmot_bbox.py  --epochs 60 --model_type attention-mamba --window_size variable --dataset sportsmot_publish --save_model --device cuda:0 --run_wandb"

count = 0
for window in max_window:
    for blocks in num_blocks:
        command_dance = command_default_dance + " --max_window " + window + " --num_blocks " + blocks + " --output_dir dancetrack_ablations"
        command_sports = command_default_sports + " --max_window " + window + " --num_blocks " + blocks + " --output_dir sportsmot_ablation"
        
        if window == "15" and blocks == "4": 
            print(" command is : ", command_dance)
            subprocess.call(command_dance, shell = True)
            torch.cuda.empty_cache()
            gc.collect()
        
        # if window == "10" and blocks == "4": 
        #     print(" command is : ", command_sports)
        #     subprocess.call(command_sports, shell = True)
        #     torch.cuda.empty_cache()
        #     gc.collect()
            
        #     time.sleep(2)
        # if window == "7" and blocks == "4":
        #     print(" command is : ", command_sports)
        
        #     subprocess.call(command_sports, shell = True)
        
        #     torch.cuda.empty_cache()
        #     gc.collect()
            
       
    
    