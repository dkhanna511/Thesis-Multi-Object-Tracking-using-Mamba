import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from models_mamba import FullModelMambaBBox, BBoxLSTMModel
from mambaAttention import FullModelMambaBBoxAttention
import torch
import torch.nn as nn
from datasets import MOTDatasetBB
from PIL import Image
import os
import cv2
import argparse
import glob
import training_utils

import time
def load_image(image_path):
    """Load an image from a given path."""
    return Image.open(image_path).convert('RGB')


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    box1 and box2 should be in the format [x1, y1, width, height].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    # Calculate intersection coordinates
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    
    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    # IoU calculation
    return intersection / union if union > 0 else 0
def calculate_map(predictions, targets, iou_thresholds=[0.5]):
    """
    Calculate mean Average Precision (mAP) for a single batch over multiple IoU thresholds.
    predictions: List of predicted bounding boxes [[x1, y1, width, height], ...]
    targets: List of ground truth bounding boxes [[x1, y1, width, height], ...]
    iou_thresholds: List of IoU thresholds for mAP calculation.
    Returns:
        - A dictionary with IoU thresholds as keys and their corresponding mAP as values.
    """
    if len(predictions) == 0 or len(targets) == 0:
        return {iou: 0.0 for iou in iou_thresholds}

    results = {}
    for threshold in iou_thresholds:
        tp, fp, fn = 0, 0, len(targets)
        matched_targets = set()

        for pred in predictions:
            best_iou = 0
            best_idx = -1
            for idx, target in enumerate(targets):
                if idx not in matched_targets:
                    iou = calculate_iou(pred, target)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

            if best_iou >= threshold:
                tp += 1
                matched_targets.add(best_idx)
                fn -= 1
            else:
                fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        results[threshold] = precision * recall

    return results


def denormalize_bbox(bbox, image_width, image_height):
    """
    Denormalize bounding boxes to original scale.
    """
    bbox[:, 0] *= image_width  # Denormalize center_x
    bbox[:, 1] *= image_height  # Denormalize center_y
    bbox[:, 2] *= image_width  # Denormalize width
    bbox[:, 3] *= image_height  # Denormalize height
    
    # Convert back from (center_x, center_y, width, height) to (left, top, width, height)
    bbox[:, 0] -= bbox[:, 2] / 2  # left = center_x - width/2
    bbox[:, 1] -= bbox[:, 3] / 2  # top = center_y - height/2
    
    return bbox

def visualize_tracking(dataloader, model, root_dir, device, window_size, model_type, image_dims=(1920, 1080)):
    """
    Visualize tracking predictions alongside ground truth.
    
    :param dataloader: DataLoader object for the dataset.
    :param model: Trained PyTorch model.
    :param root_dir: Root directory of the MOT20 dataset.
    :param image_dims: Tuple of image dimensions (width, height).
    """
    model.eval()
    print(" root dir is : ", root_dir)
    image_width, image_height = image_dims
    num_batches = len(dataloader)
    print(" window_size is ", window_size)
    # exit(0)
    total_iou = 0.0
    total_map_50 = 0.0
    total_map_75 = 0.0
    total_map_90 = 0.0
    total_map_95 = 0.0
    # total_map = 0.0
    iou_thresholds = [0.5, 0.75] + [i / 100 for i in range(50, 100, 5)]  # 0.50, 0.75, and 0.50:0.95 (in steps of 0.05)
    total_map = {iou: 0.0 for iou in iou_thresholds}
    total_frames = 0
    with torch.no_grad():
        frame_data = {}
        for batch_index, (inputs, targets, seq_info, _, _ ) in enumerate(dataloader):
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            # print(" shape of inputs is :", inputs.shape)
            # print("input is : ", inputs)
            # Predict using the model
            # print(" seq info is : ", seq_info)
            # print(" inputs shape is : ", inputs.shape)    
            predictions = model(inputs)
            # print("predictions are : ", predictions)
            # print("targets are : ", targets)
            if batch_index%5000 == 0:
                print("batches done : {} / {}".format(batch_index, num_batches))
            # print("predictions shape if : ", predictions.shape)
            # for i, (seq_name, frames) in enumerate(seq_info):
            seq_name  = seq_info[0][0]
            # print("sequence name is : ", seq_name)
            # frames = seq_info[1]
                # Denormalize bounding boxes for visualization
            # predicted_bbox = predictions.detach().cpu().unsqueeze(0).numpy()     
            # print("predictions are : ", predictions)
            # print("target is : ", targets)
            # print(" sequence_name is : ", seq_name)
            seq_info_ini = os.path.join(root_dir, seq_name, "seqinfo.ini")
            
            
            with open(seq_info_ini, 'r') as file:
                for line in file.readlines():
                    if "imWidth" in line:
                        # print(" line is : ", line)
                    
                        image_width = int(line.split("=")[1])
                    if "imHeight" in line:
                        image_height = int(line.split("=")[1])
            
            # print(" image width is : ", image_width)
            # print(" image height is : ", image_height)

            # print("seq_info[0] : ", seq_info[0])
            for i, frames in enumerate(seq_info):
                # print(" frames are : ", frames)
                # print("inputs[i] are : ", inputs[i])
                # print(" inputs are : ", inputs)
                # print("i is : ", i)
                input_bboxes = denormalize_bbox(inputs[i].cpu().numpy(), image_width, image_height)
                target_bbox = denormalize_bbox(targets[i].cpu().unsqueeze(0).numpy(), image_width, image_height)
                predicted_bbox = denormalize_bbox(predictions[i].cpu().unsqueeze(0).numpy(), image_width, image_height)
                
                iou = calculate_iou(predicted_bbox[0], target_bbox[0])
                total_iou += iou

                # total_map_50 += calculate_map([predicted_bbox[0]], [target_bbox[0]], iou_threshold=0.5)
                # total_map_75 += calculate_map([predicted_bbox[0]], [target_bbox[0]], iou_threshold=0.75)
                # total_map_90 += calculate_map([predicted_bbox[0]], [target_bbox[0]], iou_threshold=0.90)
                # total_map_95 += calculate_map([predicted_bbox[0]], [target_bbox[0]], iou_threshold=0.95)
                map_results = calculate_map([predicted_bbox[0]], [target_bbox[0]], iou_thresholds=iou_thresholds)
                for iou, value in map_results.items():
                    total_map[iou] += value
                total_frames += 1

                # print(" predictions shape is ", predictions[i].shape)
                # print(" target bbox is : ", target_bbox)
                # print(" prediction bbox is : ", predicted_bbox)
                # seq_path = os.path.join(root_dir, seq_name, "img1")  # Assuming images are in img1 folder
                frame_start = int(frames[1][0].item())  # Starting frame number
                # print("frame start is : ", frame_start)
                 # Use (sequence, frame_number) as a unique key for the frame data
                for j, bbox in enumerate(input_bboxes):
                    frame_number = frame_start + j
                    frame_key = (seq_name, frame_number)
                    if frame_key not in frame_data:
                        frame_data[frame_key] = []

                    frame_data[frame_key].append(('input', bbox))

                # Add target and predicted bounding boxes to the frame data
                frame_number = frame_start + len(input_bboxes) - 1  # Last frame in the sequence
                frame_key = (seq_name, frame_number)
                frame_data[frame_key].append(('target', target_bbox[0]))
                frame_data[frame_key].append(('predicted', predicted_bbox[0]))
                # print("frame data is : ", frame_data)frame_data

            # print("\n\n")
            
        
        
        # print("frame_data is : ", frame_data)
        # Visualization of all frames with their respective bounding boxes
        # for (seq_name, frame_number), bboxes in sorted(frame_data.items()):
        #     # print(" root dir is :", root_dir)
        #     if "dancetrack" in root_dir.split("/"):
        #         frame_path = os.path.join(root_dir, seq_name, "img1", f"{frame_number:08d}.jpg")
        #     else:
        #         frame_path = os.path.join(root_dir, seq_name, "img1", f"{frame_number:06d}.jpg")
        #     # print(" frame path is : ", frame_path)
        #     image = cv2.imread(frame_path)

        #     if image is None:
        #         continue  # Skip missing frames

        #     # Draw all bounding boxes for the current frame
        #     for bbox_type, bbox in bboxes:
        #         left, top, width, height = bbox

        #         if bbox_type == 'input':  # Green for historical bboxes
        #             color = (0, 255, 0)
        #             # cv2.rectangle(image, (int(left), int(top)),  (int(left + width), int(top + height)), color, 1)

        #         elif bbox_type == 'target':  # Blue for ground truth
        #             color = (255, 0, 0)
        #             cv2.rectangle(image, (int(left), int(top)),  (int(left + width), int(top + height)), color, 2)

        #         elif bbox_type == 'predicted':  # Black for predicted
        #             color = (0, 0, 0)

        #             cv2.rectangle(image, (int(left), int(top)), 
        #                         (int(left + width), int(top + height)), color, 2)

        #     # Display or save the visualized frame
        #     # cv2.imshow(f"Tracking Visualization: {seq_name}", image)
        #     # cv2.waitKey(100)  # Adjust for slower or faster visualization
        #     # Optionally, save the frame to disk:
            
        #     save_dir = f"visualizations/{model_type}_{window_size}/{root_dir.split('/')[1]}/{seq_name}"
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
                
        #     # print("here??")
                
        #     cv2.imwrite(f"{save_dir}/{frame_number:06d}.jpg", image)
        
        
        # # saved_dir = os.listdir("visualizations/{}_{}/{}")
        # sequence_files = "visualizations/{}_{}/{}".format(model_type, window_size, root_dir.split("/")[1])
        # # saved_dir = os.listdir(sequence_main)
        # sequence_main = [entry for entry in sequence_files if os.path.isdir(entry)]
        # print("sequence_main is : ", sequence_main)
        # # print(saved_dir)
        # video_path = "visualizations/{}_{}/{}".format(model_type, window_size, root_dir.split("/")[1])
        # sequence_file = glob.glob("visualizations/{}_{}/{}".format(model_type, window_size, root_dir.split("/")[1] + '/*'))

        # sequence_main = [entry for entry in sequence_file if os.path.isdir(entry)]
        # print("sequence_main is : ", sequence_main)

        # exit(0)
        # for sequence in sequence_main:
        #     sequence_name = sequence.split("/")[-1]
        #     print(" Reaching sequence : ", sequence)
        #     sequence_images = sorted(glob.glob(sequence + "/" +  "*.jpg"))
        #     # print("sequences images are : ", sequence_images)
        #     # break
        #     print(" sequence images are : ", sequence_images[0])
        #     frame = cv2.imread(sequence_images[0])
        #     height, width, layers = frame.shape
        #     # print("frame is : ", frame)
        #     size = (width, height)
        #     output_video_name = "{}.mp4".format(sequence_name)
        #     print("output video name : ", output_video_name)
        #     output_video_path = os.path.join(video_path, output_video_name)
        #     print("output video path : ", output_video_path)
            
        #     # Define the video writer object
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        #     fps = 20  # Frames per second
        #     out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
            
        #     for image in sequence_images:
        #         img = cv2.imread(image)
        #         out.write(img)
        
        #     out.release()     
    average_map = {iou: total_map[iou] / total_frames if total_frames > 0 else 0 for iou in iou_thresholds}

    average_iou = total_iou / total_frames if total_frames > 0 else 0
    # average_map50 = total_map_50 / total_frames if total_frames > 0 else 0
    # average_map75 = total_map_75 / total_frames if total_frames > 0 else 0
    # average_map90 = total_map_90 / total_frames if total_frames > 0 else 0
    # average_map95 = total_map_95 / total_frames if total_frames > 0 else 0
    map_50 = average_map[0.5]
    map_75 = average_map[0.75]
    map_95 = average_map[0.95]
    map_50_95 = sum(average_map.values()) / len(average_map) if total_frames > 0 else 0 
    # print(f"Average IoU: {average_iou:.4f}")
    # print(f"Average mAP50: {average_map50:.4f}")
    # print(f"Average mAP75: {average_map75:.4f}")
    # print(f"Average mAP90: {average_map90:.4f}")
    # print(f"Average mAP95: {average_map95:.4f}")

    print(f"mAP@50: {map_50:.4f}")
    print(f"mAP@75: {map_75:.4f}")
    print(f"mAP@95: {map_95:.4f}")
    print(f"mAP@50:95: {map_50_95:.4f}")

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple argument parser example.")

    # Add arguments
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--window_size', type = str, default = "variable", required = False, help = "Window size of sequence for tracklets")
    parser.add_argument('--model_type', type=str, choices = ["bi-mamba", "vanilla-mamba", "LSTM", "attention-mamba"], required = True, help = "model selection for testing" )
    parser.add_argument('--device', type=str,  required = True, help = "device for inferencing" )
    # parser.add_argument('--device', type=str,  required = True, help = "device for inferencing" )
    
    # Parse the arguments
    args = parser.parse_args()

    ## Dataset parameters
    
    # Model parameters
    input_size = 4  # Bounding box has 4 coordinates: [x, y, width, height]
    hidden_size = 64 ## This one is used for LSTM NEtwork which I tried
    output_size = 4  # Output also has 4 coordinates
    num_layers = 4## For LSTM
    embedding_dim = 128 ## For Mamba
    num_blocks = 2 ## For Mamba
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
 
    device = args.device
    # Load dataset and dataloader
    root_dir = args.dataset
    window_size = args.window_size
    model_type = args.model_type
    
    if args.dataset == "dancetrack":
        context_length = 5
    elif args.dataset == "sportsmot_publish":
        context_length = 10
    
    
    train_path = os.path.join("datasets", root_dir, "val")
    # best_model_name = "best_model_bbox_{}_{}.pth".format(root_dir, model_type)
    # best_model_name = "running_models/best_model_bbox_sportsmot_publish_variable_vanilla-mamba_20_October.pth"
    # best_model_name = "best_model_bbox_dancetrack_variable_attention-mamba_30_October.pth"
    best_model_name = "best_model_bbox_dancetrack_variable_attention-mamba_max_window_5_num_blocks_4_16_December.pth"
    # print("best model name is : ", best_model_name)
    print("train path is : ", train_path)
    from functools import partial
    collate_fn_with_padding = partial(training_utils.custom_collate_fn_fixed, context_length= context_length)
    # exit(0)
    # Adjust path to your dataset
    dataset = MOTDatasetBB(train_path, window_size=window_size, max_window = 5, augment = False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn= collate_fn_with_padding)
    # for inputs, targets, sequences in dataloader:
    #     print("sequences are :", sequences)

    # exit(0)
    # Load your trained model (ensure it's on the same device as your data)
    
    if model_type == "bi-mamba" or model_type == "vanilla-mamba":
        model = FullModelMambaBBox(input_size,embedding_dim, num_blocks, output_size, mamba_type =  model_type).to(device)
    # exit(0)
    
    if model_type == "attention-mamba":
        model = FullModelMambaBBoxAttention(input_size,embedding_dim, num_blocks, output_size, num_heads = 4, mamba_type =  model_type).to(device)
    else:
        model = BBoxLSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

    model.load_state_dict(torch.load(best_model_name))  # Load the best model

    # Visualize the tracking
    visualize_tracking(dataloader, model, train_path, device, window_size, model_type)


    
if __name__ == "__main__":
    main()