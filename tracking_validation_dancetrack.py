import os
import subprocess
import time
import argparse
import shutil
import glob
# Path to the script you want to run
script_to_run_eval = ["python TrackEval/scripts/run_mot_challenge.py", "--SPLIT_TO_EVAL val", "--METRICS HOTA CLEAR Identity", "--GT_FOLDER datasets/dancetrack/val" , "--SEQMAP_FILE datasets/dancetrack/val_seqmap.txt", 
                      "--SKIP_SPLIT_FOL True",  "--TRACKERS_TO_EVAL '' ", "--TRACKER_SUB_FOLDER ''  " , "--USE_PARALLEL True" , "--NUM_PARALLEL_CORES 8" , "--PLOT_CURVES True", "--TRACKERS_FOLDER"]
parser = argparse.ArgumentParser(description="A simple argument parser example.")

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
base_output_dir = 'RESULTS_Dancetrack'

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
    
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 't' : str(101),  'up_ft_index' : str(5)},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 't' : str(151),  'up_ft_index' : str(5)},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 't' : str(201),  'up_ft_index' : str(5)},
    # # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 't' : str(21),  'up_ft_index' : str(5)},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 't' : str(21),  'up_ft_index' : str(4)},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 't' : str(21),  'up_ft_index' : str(2)},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5),'track_buffer' : 30, 't' : str(41),  'up_ft_index' : str(5)},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5),'track_buffer' : 30, 't' : str(41),  'up_ft_index' : str(4)},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5),'track_buffer' : 30, 't' : str(41),  'up_ft_index' : str(2)},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 't' : str(51),  'up_ft_index' : str(5)},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(7), 'track_buffer' : 30, 't' : str(51),  'up_ft_index' : str(4)},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(10), 'track_buffer' : 30, 't' : str(51),  'up_ft_index' : str(2)},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': 5, 'track_buffer' : 50, 't' : str(21),  'up_ft_index' : 2},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer' : 7, 'track_buffer' : 50, 't' : str(21),  'up_ft_index' : 2},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': 10, 'track_buffer' : 50}
    
    
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 5, 'num_blocks' : 2},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 5, 'num_blocks' : 3},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 5, 'num_blocks' : 4},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 5, 'num_blocks' : 5},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 7, 'num_blocks' : 2},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 7, 'num_blocks' : 3},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 7, 'num_blocks' : 4},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 7, 'num_blocks' : 5},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 10, 'num_blocks' : 2},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 10, 'num_blocks' : 3},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 10, 'num_blocks' : 4},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 10, 'num_blocks' : 5},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 12, 'num_blocks' : 2},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 12, 'num_blocks' : 3},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 12, 'num_blocks' : 4},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 12, 'num_blocks' : 5},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 15, 'num_blocks' : 2},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 15, 'num_blocks' : 3},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 15, 'num_blocks' : 4},
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 15, 'num_blocks' : 5},
    
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 5, 'num_blocks' : 2},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 5, 'num_blocks' : 3},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 5, 'num_blocks' : 4},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 5, 'num_blocks' : 5},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 7, 'num_blocks' : 2},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 7, 'num_blocks' : 3},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 7, 'num_blocks' : 4},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 7, 'num_blocks' : 5},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 10, 'num_blocks' : 2},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 10, 'num_blocks' : 3},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 10, 'num_blocks' : 4},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 10, 'num_blocks' : 5},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 12, 'num_blocks' : 2},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 12, 'num_blocks' : 3},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 12, 'num_blocks' : 4},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 12, 'num_blocks' : 5},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 15, 'num_blocks' : 2},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 15, 'num_blocks' : 3},
    {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 15, 'num_blocks' : 4},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 15, 'num_blocks' : 5},
    
    
    #### TESTING STUFF
    # {'association' : 'diffmot', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 15, 'num_blocks' : 4},
    # {'association' : 'diffmot_without_virtual', 'model_type' : 'attention-mamba', 'virtual_track_buffer': str(5), 'track_buffer' : 30, 'max_window': 7, 'num_blocks' : 3},



    # Add more parameter sets as needed
]


if args.test:
    run_file_name = "yolox_dancetrack_test.py"
else:
    run_file_name = "yolox_dancetrack_val.py"


# Iterate over each parameter set and run the script
for i, params in enumerate(parameter_sets):
    # Create a unique directory for each run
    # final_output_dir = os.path.join(output_dir, f'run_{i+1}')
    final_output_dir = os.path.join(output_dir, 'association_{}_ws_{}_blocks_{}'.format(params['association'], params['max_window'], params['num_blocks']))
    output_files_dir = os.path.join(final_output_dir, "output_files/")
    if not os.path.exists(output_files_dir):
        os.makedirs(output_files_dir, exist_ok=True)
    
    viz_dir = os.path.join(final_output_dir, "visualizations/")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)


    # Construct the command with parameters
    # experiment_folder = args.expn + "_run_{}".format(i+1)
    experiment_folder = args.expn

    experiment_dir = "YOLOX_outputs/{}".format(experiment_folder)
    # model_name = "dancetrack_ablations/best_model_bbox_dancetrack_variable_attention-mamba_max_window_{}_num_blocks_{}.pth".format(params['max_window'], params['num_blocks'])
    # model_name = "dancetrack_ablations/best_model_bbox_dancetrack_variable_attention-mamba_max_window_{}_num_blocks_{}.pth".format(params['max_window'], params['num_blocks'])
    
    # model_name = "dancetrack_ablations/best_model_bbox_dancetrack_variable_attention-mamba_max_window_15_num_blocks_4_25_December_NO_AUGMENT.pth"
    model_name = "dancetrack_ablations/best_model_bbox_dancetrack_variable_attention-mamba_max_window_15_num_blocks_4_24_December.pth"
    
    print(" model name is : ", model_name)
    command = [
        'python',script_to_run,
        '-f exps/example/mot/{}'.format(run_file_name),
        '-c pretrained/ocsort_dance_model.pth.tar',
        '--expn',  experiment_folder, 
        '-b 1',
        '-d 1' ,
        '--fp16 --fuse',
        '--dataset_name dancetrack',
        '--association' , params['association'], 
        '--model_type', params['model_type'],
        '--num_blocks', str(params['num_blocks']),
        '--max_window', str(params['max_window']),
        '--virtual_track_buffer', params['virtual_track_buffer'],
        '--model_path', model_name,
        '--b1 0.0',
        '--b2 0.0' 
        # '--t', params['t'],
        # '--up_ft_index', params['up_ft_index'],  
    ]
    
    # Run the command
    print(f"Running: {' '.join(command)}")
    
    overall_command = ' '.join(command)
    
    print("overall command is : ", overall_command)
    subprocess.call(overall_command, shell = True)
    

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