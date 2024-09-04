import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset
import glob


class MambaMOTDataset(Dataset):
    def __init__(self, path, interval=None):
        # self.config = 
        self.interval = interval + 1 ## Changed config interval to interval only

        self.trackers = {}
        self.images = {}
        self.nframes = {}
        self.ntrackers = {}

        self.nsamples = {}
        self.nS = 0

        self.nds = {}
        self.cds = {}
        if os.path.isdir(path):
            # if 'MOT' in path:
            self.seqs = ['MOT17-02-SDP', 'MOT17-11-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-DPM', 'MOT17-11-FRCNN', 
                            'MOT17-10-DPM', 'MOT17-10-FRCNN', 'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-DPM', 'MOT17-13-SDP', 
                            'MOT17-02-DPM', 'MOT17-13-FRCNN', 'MOT17-02-FRCNN', 'MOT17-13-DPM', 'MOT17-04-DPM', 'MOT17-05-FRCNN', 
                            'MOT17-09-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-DPM']
                
                
            # else:
                # self.seqs = [s for s in os.listdir(path)]
            self.seqs.sort()
            lastindex = 0
            for seq in self.seqs:
                
                # path_yo = os.path.join(path + "/" + seq, "img1/"))
                # print(path_yo)
                seq_path = glob.glob(os.path.join(path, seq, "img1", "*.txt"))
                # print(os.listdir(seq_path))
                # trackerPath = glob.glob(os.path.join(path + "/" + seq, "/img1/*.txt"))
                # print(seq_path)
                # print(" tracker path fiels are : ", seq_path)
                self.trackers[seq] = sorted((seq_path))
                self.ntrackers[seq] = len(self.trackers[seq])
                print(" number of tracks in sequence : {} : {}".format(seq, self.ntrackers[seq]))
                if 'MOT' in seq:
                    # print(" Yes Coming her?")
                    imagePath = os.path.join(path, '../../train', seq, "img1/*.*")
                else:
                    # print(" else")
                    imagePath = os.path.join(path, '../train', seq, "img1/*.*")
                self.images[seq] = sorted(glob.glob(imagePath))
                self.nframes[seq] = len(self.images[seq])
                print(" number of frames in sequence : {} : {}".format(seq, self.nframes[seq]))
                
                # print("images are : ", self.images)

                self.nsamples[seq] = {}
                for i, pa in enumerate(self.trackers[seq]):
                    self.nsamples[seq][i] = len(np.loadtxt(pa, dtype=np.float32).reshape(-1,7)) - self.interval
                    self.nS += self.nsamples[seq][i]


                self.nds[seq] = [x for x in self.nsamples[seq].values()]
                self.cds[seq] = [sum(self.nds[seq][:i]) + lastindex for i in range(len(self.nds[seq]))]
                lastindex = self.cds[seq][-1] + self.nds[seq][-1]

        print('=' * 80)
        print('dataset summary')
        print(self.nS)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, seq in enumerate(self.cds):
            if files_index >= self.cds[seq][0]:
                ds = seq
                for j, c in enumerate(self.cds[seq]):
                    if files_index >= c:
                        trk = j
                        start_index = c
                    else:
                        break
            else:
                break

        track_path = self.trackers[ds][trk]
        track_gt = np.loadtxt(track_path, dtype=np.float32)

        init_index = files_index - start_index

        cur_index = init_index + self.interval
        cur_gt = track_gt[cur_index]
        cur_bbox = cur_gt[2:6]

        boxes = [track_gt[init_index + tmp_ind][2:6] for tmp_ind in range(self.interval)]
        delt_boxes = [boxes[i+1] - boxes[i] for i in range(self.interval - 1)]
        conds = np.concatenate((np.array(boxes)[1:], np.array(delt_boxes)), axis=1)

        delt = cur_bbox - boxes[-1]
        ret = {"cur_gt": cur_gt, "cur_bbox": cur_bbox, "condition": conds, "delta_bbox": delt}

        return ret

    def __len__(self):
        return self.nS


class MOTDatasetBB(Dataset):
    def __init__(self, path, window_size=10, image_dims=(1920, 1080)):
        self.window_size = window_size
        # self.image_width, self.image_height = image_dims

        # Initialize data storage
        self.data = []
        self.targets = []
        self.sequence_info = []  # To store sequence and frame info

        # Load the dataset
        self._load_data(path)

    def _load_data(self, path):
        """
        Load data from the provided path and collect bounding boxes.
        """
        sequences = [seq for seq in os.listdir(path) if os.path.isdir(os.path.join(path, seq))]
        sequences.sort()
        
        for seq in sequences:
            gt_path = os.path.join(path, seq, "gt", "gt.txt")  # Path to ground truth file
            seq_info_path = os.path.join(path, seq, "seqinfo.ini")
            
            with open(seq_info_path, 'r') as file:
                for line in file.readlines():
                    if "imWidth" in line:
                        # print(" line is : ", line)
                    
                        image_width = int(line.split("=")[1])
                    if "imHeight" in line:
                        image_height = int(line.split("=")[1])
                        
            # print(" image height is  : {}, and type is : {} ".format(image_height, type(image_height)))
            
            
            if not os.path.exists(gt_path):
                continue
            print("here")
            # Load ground truth data for the sequence
            gt_data = np.loadtxt(gt_path, delimiter=',')
            
            # Filter for specific object IDs, sorting by frame number
            for obj_id in np.unique(gt_data[:, 1]):
                obj_data = gt_data[gt_data[:, 1] == obj_id]
                obj_data = obj_data[obj_data[:, 0].argsort()]  # Sort by frame number
                
                # Extract bounding boxes (columns: [frame, id, left, top, width, height, conf, x, y, z])
                bboxes = obj_data[:, 2:6]  # [left, top, width, height]
                frame_nums = obj_data[:, 0]  # Extract frame numbers
                
                ### Converting the dataset format from MOT to YOL O , i.e. ( Center x, Center y, Width, Height)
                bboxes[:, 0] = bboxes[:, 0] + bboxes[:,2]/2
                bboxes[:, 1] = bboxes[:, 1] + bboxes[:,3]/2
                # Normalize bounding boxes
                bboxes[:, 0] /= image_width  # Normalize center x
                bboxes[:, 1] /= image_height  # Normalize center_y
                bboxes[:, 2] /= image_width  # Normalize width
                bboxes[:, 3] /= image_height  # Normalize height
                
                # Skip sequences that are too short
                if len(bboxes) <= self.window_size:
                    continue  

                # Collect bounding boxes for input and target pairs
                for i in range(len(bboxes) - self.window_size):
                    input_bboxes = bboxes[i:i + self.window_size]  # Bounding boxes for the input window
                    target_bbox = bboxes[i + self.window_size]    # Next frame's bounding box as target
                    frames_in_window = frame_nums[i:i + self.window_size + 1]  # Corresponding frames for input and target
                    
                    self.data.append(input_bboxes)
                    self.targets.append(target_bbox)
                    self.sequence_info.append((seq, frames_in_window))  # Store sequence name and frame range
                    # print("yoooo")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the input-target pair at index `idx`.
        """
        input_data = self.data[idx]
        target_data = self.targets[idx]
        seq_info = self.sequence_info[idx]
        return torch.from_numpy(input_data.astype(float)), torch.from_numpy(target_data.astype(float)), seq_info    

class MOT20DatasetOffset(Dataset):
    def __init__(self, path, window_size=10, image_dims = (1920, 1080)):
        self.window_size = window_size
        self.image_width, self.image_height = image_dims

        # Initialize data storage
        self.data = []
        self.targets = []

        # Load the dataset
        self._load_data(path)

    def _load_data(self, path):
        """
        Load data from the provided path and compute the bounding box differences.
        """
        sequences = [seq for seq in os.listdir(path) if os.path.isdir(os.path.join(path, seq))]
        # sequences = ["MOT17-02-DPM"]
        sequences.sort()
        
        for seq in sequences:
            gt_path = os.path.join(path, seq, "gt", "gt.txt")  # Path to ground truth file
            if not os.path.exists(gt_path):
                continue
            
            # Load ground truth data for the sequence
            gt_data = np.loadtxt(gt_path, delimiter=',')
            
            # Filter for specific object IDs, sorting by frame number
            for obj_id in np.unique(gt_data[:, 1]):
                obj_data = gt_data[gt_data[:, 1] == obj_id]
                obj_data = obj_data[obj_data[:, 0].argsort()]  # Sort by frame number
                # print(" object data is : ", obj_data)
                # Extract bounding boxes (columns: [frame, id, left, top, width, height, conf, x, y, z])
                bboxes = obj_data[:, 2:6]  # [left, top, width, height]
                
                
                # Normalize bounding boxes
                # bboxes[:, 0] /= self.image_width  # Normalize left
                # bboxes[:, 1] /= self.image_height  # Normalize top
                # bboxes[:, 2] /= self.image_width  # Normalize width
                # bboxes[:, 3] /= self.image_height  # Normalize height
                
                
                if len(bboxes) <= self.window_size:
                    continue  # Skip sequences that are too short

                
                # Compute differences and form input-target pairs
                for i in range(len(bboxes) - self.window_size):
                    input_diffs = np.diff(bboxes[i:i + self.window_size + 1], axis=0)
                    input_data = input_diffs[:-1]  # Differences for the input window
                    target_data = input_diffs[-1]  # Difference for the target frame
                    
                    self.data.append(input_data)
                    self.targets.append(target_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the input-target pair at index `idx`.
        """
        input_data = self.data[idx]
        target_data = self.targets[idx]
        return torch.from_numpy(input_data.astype(float)), torch.from_numpy(target_data.astype(float))
    
    
    
    
# Usage example

if __name__  == '__main__':
    root_dir = 'MOT20/train'
    
    # Sample verification
    dataset = MOTDatasetBB(path=root_dir)
    # for i in range(len(dataset)):
        # try:
        # input_frames, target_frame, sequences = dataset[i]
            # print(f"Sample {i}: input_frames shape: {input_frames.shape}, target_frame shape: {target_frame.shape}")
        # except Exception as e:
            # print(f"Error at index {i}: {e}")# print(input_frames.shape, target_frame.shape)
    # print("target is : ", target_frame)
        # print(" input shape is : ", input_frames.shape)
        # print("target shape is : ", target_frame.shape)
        # if i == 1200:
        #     print(input_frames)
        #     print(target_frame)
        #     exit(0)


