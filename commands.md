
## dancetrack

### For training dancetrack
`python train_dancetrack_bbox.py --epoch 100 --model_type bi-mamba --window_size variable --device cuda:0 --dataset dancetrack --save_model --run_wandb`


### For running the tracker for dancetrack [validation] 
`python tools/run_mamba_tracker.py -f exps/example/mot/yolox_dancetrack_val.py -c pretrained/ocsort_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse --expn fixed_scaling_issue_oct16_test --model_type bi-mamba --dataset_name dancetrack --model_path running_models/best_model_bbox_dancetrack_variable_bi-mamba_14_October.pth --association bytetrack`



### For running the tracker for dancetrack [validation] [with reid] 
`python tools/run_mamba_tracker.py -f exps/example/mot/yolox_dancetrack_val.py -c pretrained/ocsort_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse --expn fixed_scaling_issue_oct16_test --model_type bi-mamba --dataset_name dancetrack --model_path running_models/best_model_bbox_dancetrack_variable_bi-mamba_14_October.pth --association diffmot --with-reid --virtual_track_buffer 5`
`


### For running the tracker for dancetrack [test] 
`python tools/run_mamba_tracker.py -f exps/example/mot/yolox_dancetrack_test.py -c pretrained/ocsort_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse --expn fixed_scaling_issue_oct16_test --model_type bi-mamba --dataset_name dancetrack --model_path running_models/best_model_bbox_dancetrack_variable_bi-mamba_14_October.pth --association bytetrack --test`



#### For running the evaluation script for dancetrack
`python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val  --METRICS HOTA CLEAR Identity  --GT_FOLDER datasets/dancetrack/val --SEQMAP_FILE datasets/dancetrack/val_seqmap.txt --SKIP_SPLIT_FOL True   --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --TRACKERS_FOLDER YOLOX_outputs/fixed_scaling_issue_oct16/fixed_scaling_issue_oct16_val/`


## SportsMOT

### For training sportsmot_publish
`python train_sportsmot_bbox.py --epochs 100 --model_type bi-mamba --window_size variable --dataset sportsmot_publish --device cuda:0 --save_model --run_wandb`


### For running the tracker for sportsmot_publish  [validation]
`python tools/run_mamba_tracker.py -f exps/example/mot/yolox_sportsmot_val.py -c pretrained/SportsMOT_yolox_x_mix.tar -b 1 -d 1 --fp16 --fuse --expn experiment_name --model_type attention-mamba --dataset_name sportsmot_publish --model_path running_models/best_model_bbox_sportsmot_publish_variable_attention-mamba_1_November.pth --association diffmot --with-reid`


### For running the tracker for sportsmot_publish [test]
`python tools/run_mamba_tracker.py -f exps/example/mot/yolox_sportsmot_test.py -c pretrained/SportsMOT_yolox_x_mix.tar -b 1 -d 1 --fp16 --fuse --expn oct_16_sportsmot --model_type attention-mamba --dataset_name sportsmot_publish --model_path running_models/best_model_bbox_sportsmot_publish_variable_bi-mamba_15_October.pth --test`

### For running the evaluation script on sportsmot [validation]
`python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val  --METRICS HOTA CLEAR Identity  --GT_FOLDER datasets/sportsmot_publish/val --SEQMAP_FILE datasets/sportsmot_publish/sportsmot-val.txt --SKIP_SPLIT_FOL True   --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --TRACKERS_FOLDER YOLOX_outputs/oct16_val/oct16_val_val/`
