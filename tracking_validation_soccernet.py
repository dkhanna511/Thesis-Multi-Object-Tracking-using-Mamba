import os
import subprocess
import time
import argparse
import shutil
import glob
# Path to the script you want to run

parser = argparse.ArgumentParser(description="A simple argument parser example.")
script_to_run_eval = ["python TrackEval/scripts/run_mot_challenge.py", "--SPLIT_TO_EVAL val", "--METRICS HOTA CLEAR Identity", "--GT_FOLDER datasets/sportsmot_publish/val" , "--SEQMAP_FILE datasets/sportsmot_publish/sportsmot-val.txt", 
                      "--SKIP_SPLIT_FOL True",  "--TRACKERS_TO_EVAL '' ", "--TRACKER_SUB_FOLDER ''  " , "--USE_PARALLEL True" , "--NUM_PARALLEL_CORES 8" , "--PLOT_CURVES True", "--TRACKERS_FOLDER"]


    # Add arguments
parser.add_argument('--expn', type=str, required=True, help="Path to the dataset file.")
parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)
    # Parse the arguments
args = parser.parse_args()

if args.test:
    script_to_run = 'tools/run_mamba_tracker.py --test'
else:
    script_to_run = 'tools/run_mamba_tracker.py'


# Base directory to save results
base_output_dir = 'RESULTS_Soccernet'

current_time = time.localtime()
formatted_date = time.strftime("%d %B", current_time)
formatted_date = str(formatted_date.split(' ')[0])+ "_" + str(formatted_date.split(' ')[1])
    
if args.test:
    output_dir = os.path.join(base_output_dir, formatted_date, args.expn, "test")
else:
    output_dir = os.path.join(base_output_dir, formatted_date, args.expn, "val")


# Ensure the base output directory exists
os.makedirs(output_dir, exist_ok=True)


# Define a list of different parameter sets
parameter_sets = [
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(1), 'track_buffer' : 30},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(2), 'track_buffer' : 30},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(3), 'track_buffer' : 30},
    {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer' : str(7), 'track_buffer' : 30},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(10), 'track_buffer' : 30},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': 5, 'track_buffer' : 40},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer' : 7, 'track_buffer' : 40},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': 10, 'track_buffer' : 40},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': 5, 'track_buffer' : 50},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer' : 7, 'track_buffer' : 50},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': 10, 'track_buffer' : 50}
    
    # Add more parameter sets as needed
]


if args.test:
    run_file_name = "yolox_soccernet_test.py"
else:
    run_file_name = "yolox_soccernet_val.py"


# Iterate over each parameter set and run the script
for i, params in enumerate(parameter_sets):
    # Create a unique directory for each run
    final_output_dir = os.path.join(output_dir, f'run_{i+1}')
    output_files_dir = os.path.join(final_output_dir, "output_files/")
    if not os.path.exists(output_files_dir):
        os.makedirs(output_files_dir, exist_ok=True)
    
    viz_dir = os.path.join(final_output_dir, "visualizations/")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
   
    # Construct the command with parameters
    experiment_folder = args.expn + "_run_{}".format(i+1)

    experiment_dir = "YOLOX_outputs/{}".format(experiment_folder)
    
    command = [
        'python',script_to_run,
        '-f exps/example/mot/{}'.format(run_file_name),
        '-c pretrained/SportsMOT_yolox_x_mix.tar',
        '--expn',  experiment_folder, 
        '-b 1',
        '-d 1' ,
        '--fp16 --fuse',
        '--dataset_name soccernet',
        '--model_path running_models/best_model_bbox_dancetrack_variable_bi-mamba_14_October.pth',
        '--association' , params['association'], 
        '--model_type', params['model_type'],
        '--virtual_track_buffer', params['virtual_track_buffer']
    ]
    
    # Run the command
    print(f"Running: {' '.join(command)}")
    
    overall_command = ' '.join(command)
    
    print("overall command is : ", overall_command)
    subprocess.call(overall_command, shell = True)
    
    # exit(0)
    if args.test:
        results_dir = "YOLOX_outputs/{}/{}_{}/".format(experiment_folder,  experiment_folder, "test")
    else:
        results_dir = "YOLOX_outputs/{}/{}_{}/".format(experiment_folder,  experiment_folder, "val")
    
    visualization_dir = "YOLOX_outputs/{}/{}".format(experiment_folder,  "visualizations")
    
    for files in glob.glob(results_dir +  "/*.txt"):
        file_name = files.split("/")[-1]
        if os.path.exists(os.path.join(output_files_dir, file_name)):
            os.remove(os.path.join(output_files_dir, file_name))
        shutil.move(files,  output_files_dir)
    
    
    for files in glob.glob(visualization_dir + "/*.mp4"):
        file_name = files.split("/")[-1]
        if os.path.exists(os.path.join(viz_dir, file_name)):
            os.remove(os.path.join(viz_dir, file_name))
        shutil.move(files,  viz_dir)

        
    script_eval = ' '.join(script_to_run_eval) + ' ' + output_files_dir
    if os.path.exists(os.path.join(final_output_dir, "val_log.txt")):
        os.remove(os.path.join(final_output_dir, "val_log.txt"))
    shutil.move(os.path.join(experiment_dir, "val_log.txt"), final_output_dir)
    # script_to_run_eval.append(output_files_dir)
    # overall_command_val = ' '.join(script_to_run_eval)
    print(" Overall command Evaluation :", script_to_run_eval)
    if not args.test:
        subprocess.call(script_eval, shell = True)

    # experiment_dir = "YOLOX_outputs/{}".format(experiment_folder)
    
    if len(os.listdir(results_dir)) == 0:
        os.rmdir(results_dir)
        # os.remove(os.path.join(experiment_dir, "val_log.txt"))
        if os.path.exists(visualization_dir):
            os.rmdir(visualization_dir)
        os.rmdir(experiment_dir)


print("All runs completed.")