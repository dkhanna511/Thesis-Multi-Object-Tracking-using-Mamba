#!/bin/bash

# Define the arguments for each Python script
args1="--dataset MOT20 --model_type LSTM --save_model --epochs 50 --run_wandb"
args2="--dataset dancetrack --model_type LSTM --save_model --epochs 50 --run_wandb"
# args3="arg1_for_script3 arg2_for_script3"

# Run the first Python script with its arguments
echo "Running train_mot20_bbox.py $args1"
python train_mot20_bbox.py $args1

# Check if the first script executed successfully
if [ $? -ne 0 ]; then
  echo "train_mot20_bbox.py failed. Exiting."
  exit 1
fi

# Run the second Python script with its arguments
echo "Running train_dancetrack_bbox.py $args2"
python train_dancetrack_bbox.py $args2

# Check if the second script executed successfully
if [ $? -ne 0 ]; then
  echo "train_dancetrack_bbox.py failed. Exiting."
  exit 1
fi

# # Run the third Python script with its arguments
# echo "Running script3.py with arguments: $args3"
# python script3.py $args3

# # Check if the third script executed successfully
# if [ $? -ne 0 ]; then
#   echo "script3.py failed. Exiting."
#   exit 1
# fi

echo "All scripts ran successfully."