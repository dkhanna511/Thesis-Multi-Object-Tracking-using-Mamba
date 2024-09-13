import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from models_mamba import FullModelMambaBBox, BBoxLSTMModel
import torch
import torch.nn as nn
from datasets import MOTDatasetBB
from PIL import Image
import os
import cv2
import argparse


def load_image(image_path):
    """Load an image from a given path."""
    return Image.open(image_path).convert('RGB')


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
    image_width, image_height = image_dims
    num_batches = len(dataloader)
    print(" window_size is ", window_size)
    # exit(0)
    
    with torch.no_grad():
        frame_data = {}
        for batch_index, (inputs, targets, seq_info) in enumerate(dataloader):
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            # print(" shape of inputs is :", inputs.shape)
            # Predict using the model
            predictions = model(inputs)
            print("batches done : {} / {}".format(batch_index, num_batches))
            # for i, (seq_name, frames) in enumerate(seq_info):
            seq_name  = seq_info[0][0]
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

            for i, frames in enumerate(seq_info[1]):
                # print("inputs are : ", inputs[i])
                input_bboxes = denormalize_bbox(inputs[i].cpu().numpy(), image_width, image_height)
                target_bbox = denormalize_bbox(targets[i].cpu().unsqueeze(0).numpy(), image_width, image_height)
                predicted_bbox = denormalize_bbox(predictions[i].cpu().unsqueeze(0).numpy(), image_width, image_height)

                # print(" target bbox is : ", target_bbox)
                # print(" prediction bbox is : ", predicted_bbox)
                # seq_path = os.path.join(root_dir, seq_name, "img1")  # Assuming images are in img1 folder
                frame_start = int(frames[0].item())  # Starting frame number

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

        
        # Visualization of all frames with their respective bounding boxes
        for (seq_name, frame_number), bboxes in sorted(frame_data.items()):
            if root_dir.split('/')[0] == "dancetrack":
                frame_path = os.path.join(root_dir, seq_name, "img1", f"{frame_number:08d}.jpg")
            else:
                frame_path = os.path.join(root_dir, seq_name, "img1", f"{frame_number:06d}.jpg")
            print(" frame path is : ", frame_path)
            image = cv2.imread(frame_path)

            if image is None:
                continue  # Skip missing frames

            # Draw all bounding boxes for the current frame
            for bbox_type, bbox in bboxes:
                left, top, width, height = bbox

                if bbox_type == 'input':  # Green for historical bboxes
                    color = (0, 255, 0)
                    # cv2.rectangle(image, (int(left), int(top)),  (int(left + width), int(top + height)), color, 1)

                elif bbox_type == 'target':  # Blue for ground truth
                    color = (255, 0, 0)
                    cv2.rectangle(image, (int(left), int(top)),  (int(left + width), int(top + height)), color, 2)

                elif bbox_type == 'predicted':  # Red for predicted
                    color = (0, 0, 0)

                    cv2.rectangle(image, (int(left), int(top)), 
                                (int(left + width), int(top + height)), color, 2)

            # Display or save the visualized frame
            # cv2.imshow(f"Tracking Visualization: {seq_name}", image)
            # cv2.waitKey(100)  # Adjust for slower or faster visualization
            # Optionally, save the frame to disk:
            save_dir = f"visualizations_{model_type}_{window_size}/{root_dir.split('/')[0]}/{seq_name}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            print("here??")
                
            cv2.imwrite(f"{save_dir}/{frame_number:06d}.jpg", image)



def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple argument parser example.")

    # Add arguments
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--window_size', type = int, default = 10, required = False, help = "Window size of sequence for tracklets")
    parser.add_argument('--model_type', type=str, choices = ["bi-mamba", "vanilla-mamba", "LSTM"], required = True, help = "model selection for testing" )
    # Parse the arguments
    args = parser.parse_args()

    ## Dataset parameters
    
    # Model parameters
    input_size = 4  # Bounding box has 4 coordinates: [x, y, width, height]
    hidden_size = 64 ## This one is used for LSTM NEtwork which I tried
    output_size = 4  # Output also has 4 coordinates
    num_layers = 4## For LSTM
    embedding_dim = 128 ## For Mamba
    num_blocks = 3 ## For Mamba
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load dataset and dataloader
    root_dir = args.dataset
    window_size = args.window_size
    model_type = args.model_type
    
    train_path = os.path.join("datasets", root_dir, "train_copy_testing")
    best_model_name = "best_model_bbox_{}_{}.pth".format(root_dir, model_type)
    
    print("best model name is : ", best_model_name)
    print("train path is : ", train_path)
    
    exit(0)
    # Adjust path to your dataset
    dataset = MOTDatasetBB(train_path, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    # for inputs, targets, sequences in dataloader:
    #     print("sequences are :", sequences)

    # exit(0)
    # Load your trained model (ensure it's on the same device as your data)
    
    if model_type == "bi-mamba" or model_type == "vanilla-mamba":
        model = FullModelMambaBBox(input_size,embedding_dim, num_blocks, output_size, mamba_type =  model_type).to(device)
    # exit(0)
    else:
        model = BBoxLSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

    model.load_state_dict(torch.load(best_model_name))  # Load the best model

    # Visualize the tracking
    visualize_tracking(dataloader, model, train_path, device, window_size, model_type)


    
if __name__ == "__main__":
    main()