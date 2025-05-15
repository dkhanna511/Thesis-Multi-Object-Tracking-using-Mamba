from collections import defaultdict
from loguru import logger
from tqdm import tqdm
# import models_mamba
# from models_mamba import FullModelMambaBBox
# from datasets import MambaMOTDataset, MOTDatasetBB
import gc
from pathlib import Path

import os.path as osp

import torch
import numpy as np

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from trackers.byte_tracker.byte_tracker import BYTETracker
from trackers.ocsort_tracker.ocsort import OCSort
# from trackers.mamba_tracker.mamba_tracker import MambaTracker
import cv2
# from trackers.mamba_tracker.mamba_tracker_botsort import MambaTrackerBot
# from trackers.mamba_tracker.mamba_tracker_diffmot import MambaTrackerDiffMOT
from trackers.mamba_tracker.mamba_tracker_diffmot_without_virtual import MambaTrackerDiffMOTWithoutVirtualDets

# from trackers.mamba_tracker.mamba_tracker_dift import MambaTrackerDift

# from trackers.mamba_tracker.mamba_tracker_reid import MambaTrackerReID
from trackers.deepsort_tracker.deepsort import DeepSort
# from trackers.motdt_tracker.motdt_tracker import OnlineTracker

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
from utils.utils import write_results, write_results_no_score, visualize_tracking_to_video

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    
    def get_image_list(path):
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = osp.join(maindir, filename)
                ext = osp.splitext(apath)[1]
                if ext in IMAGE_EXT:
                    image_names.append(apath)
        return image_names

    def evaluate_mamba_track(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None, 
        model_type = None,
        padding_window = None,
        model_path = None,
        gtfiles = None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            print("coming here??")

        print(' OR Here?')
        # exit(0) 
        # tracker = MambaTracker(self.args)
        tracker = None
        print(" tracker is :", tracker)
        # exit(0)
        count = 0
        start_time = time.time()
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                count +=1
                frame_id = info_imgs[2].item()
                # print("frame ID is : ", frame_id)
                video_id = info_imgs[3].item()
                # print("video id is : ", video_id)
                # print(" frame id  is : ", frame_id)
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                # print(" video starting to process is ; {}".format(video_name))
                # gt_file =  os.path.join("datasets/VIP-HTD/test/", video_name, "gt/gt.txt")
                # print("gt file is : ", gt_file)
                # output_gt = self.extract_gt_data(gt_file, frame_id)
                # output_gt = []

                # with open(gt_file, "r") as file:
                #     for line in file:
                #         values = line.strip().split(",")  # Assuming CSV format
                        
                        # Ensure the line has enough values before accessing indices
                        # if len(values) < 7:
                        #     continue

                        # Extract relevant values
                        # frame_number = int(values[0])  # Convert frame number to int
                        
                        # if frame_id == frame_number:
                        #     x, y, width, height, confidence = map(float, values[2:7])  # Extract relevant columns
                        #     output_gt.append((x, y, width+x, height+y, confidence))  # Store as a tuple
                # output_gt = torch.tensor(output_gt)
                # print("output_gt is : ", output_gt)
                # gt_file = glob.glob(gtfiles,
                # list_done = ["CAR_VS_BOS_001", "CAR_VS_NYR_001"]# "dancetrack0030", "dancetrack0034", "dancetrack0035", "dancetrack0041", "dancetrack43" , "dancetrack0047", "dancetrack0063", "dancetrack65" ]#, "dancetrack0010", "dancetrack0014","dancetrack0018"
                            #  , "dancetrack0019", "dancetrack0025"   , "dancetrack0026", "dancetrack0030", "dancetrack0034", "dancetrack0035","dancetrack0041", "dancetrack0043", "dancetrack0047"]
                            #   , "dancetrack0058", "dancetrack0063", "dancetrack0065",
                #              "dancetrack0073", "dancetrack0077"]
                            #  , "dancetrack0079"  , "dancetrack0081"]
                            
                # list_done = ['v_9MHDmAMxO5I_c004', 'v_G-vNjfx1GGc_c601']#, 'v_9MHDmAMxO5I_c006', 'v_4-EmEtrturE_c009', 'v_5ekaksddqrc_c004',
                #               'v_00HRwkvvjtQ_c007', 'v_ITo3sCnpw_k_c010', 'v_0kUtTtmLaJA_c010', 'v_ITo3sCnpw_k_c007', 'v_0kUtTtmLaJA_c007',
                #                 'v_00HRwkvvjtQ_c005', 'v_dw7LOz17Omg_c067', 'v_0kUtTtmLaJA_c004', 'v_5ekaksddqrc_c002', 'v_5ekaksddqrc_c003', 
                #                 'v_cC2mHWqMcjk_c008', 'v_4r8QL_wglzQ_c001', 'v_G-vNjfx1GGc_c008', 'v_5ekaksddqrc_c001', 'v_ITo3sCnpw_k_c011', 
                #                 'v_ITo3sCnpw_k_c012', 'v_00HRwkvvjtQ_c008', 'v_5ekaksddqrc_c005', 'v_G-vNjfx1GGc_c004', 'v_00HRwkvvjtQ_c003',
                #                   'v_9MHDmAMxO5I_c002', 'v_0kUtTtmLaJA_c008', 'v_9MHDmAMxO5I_c003', 'v_BgwzTUxJaeU_c008', 
                #                   'v_cC2mHWqMcjk_c007', 'v_00HRwkvvjtQ_c011', 'v_0kUtTtmLaJA_c006', 'v_dw7LOz17Omg_c053', 'v_i2_L4qquVg0_c006', 'v_cC2mHWqMcjk_c009', 
                #                   'v_2QhNRucNC7E_c017', 'v_0kUtTtmLaJA_c005', 'v_BgwzTUxJaeU_c014', 'v_i2_L4qquVg0_c007', 'v_9MHDmAMxO5I_c009', 'v_BgwzTUxJaeU_c012',
                # #                     'v_00HRwkvvjtQ_c001', 'v_G-vNjfx1GGc_c600']
                # if video_name in  list_done:
                    # continue
                # image = cv2.imread()
                # if self.args.association == "botsort" or self.args.association == "bytetrack" or self.args.association == "diffmot":
                if self.args.test:
                    image_path = os.path.join("datasets", self.args.dataset_name, "test", img_file_name[0])
                else:
                    image_path = os.path.join("datasets", self.args.dataset_name, "val", img_file_name[0])
                    # print("image file name is :", image_path)
                image =  cv2.imread(image_path)

                
                # print("video name is ", video_name)
                # print("image is " , image)
                # exit(0)
                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    
                    end_time = time.time()
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                    # print(" video starting to process is ; {}".format(video_name))

                    # print(" \nimage file name is : ", img_file_name)
                    if self.args.association == "bytetrack":
                        print("\nvideo name is : ", video_names)
                        tracker = BYTETracker(self.args)
                    elif self.args.association == "deepsort":
                        print("\nvideo name is : ", video_names)
                        tracker = DeepSort(self.args)
                    
                    elif self.args.association == "ocsort":
                        print("\nvideo name is : ", video_names)
                        print(" tracker is OCSORT")
                        tracker = OCSort(det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh,
            asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte)
                    
                    # elif self.args.association == "botsort":
                    #     tracker = MambaTrackerBot(self.args, padding_window)
                    # if self.args.association == "diffmot":
                        # tracker = MambaTrackerDiffMOT(self.args, padding_window)
                    elif self.args.association == "diffmot_without_virtual":
                        tracker = MambaTrackerDiffMOTWithoutVirtualDets(self.args, padding_window)
                    # print("here?>?>>>>")
                    # if self.args.association == "diffusion":
                    #     tracker = MambaTrackerDift(self.args, padding_window)
                    
                    print(" FPS is : ", (count - 1)/(end_time - start_time))
                    count = 0
                    start_time = time.time()
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        video_dir = os.path.join(Path(result_folder).parent, "visualizations")
                        video_filename = os.path.join(Path(result_folder).parent, "visualizations", '{}.mp4'.format(video_names[video_id -1 ]))
                        if not os.path.exists(video_dir):
                            os.makedirs(video_dir)
                        if self.args.test:
                            GT_Seq_path = os.path.join("datasets", self.args.dataset_name, "test", "{}".format(video_names[video_id-1]), "img1")
                        else:
                            GT_Seq_path = os.path.join("datasets", self.args.dataset_name, "val", "{}".format(video_names[video_id-1]), "img1")
                        
                        # visualize_tracking_to_video(GT_Seq_path,  result_filename, video_filename, self.args.dataset_name)
                        results = []

                    # print("\ntracked vals are :", tracker.frame_id)
                    # print('\nresults are :', results)
                
                if frame_id %100==0:
                    torch.cuda.empty_cache()
                    gc.collect()
                imgs = imgs.type(tensor_type)
                # print(" images shape is : ", imgs.shape) ### This gives shape of 800, 1440. Which means that in this img height and width, yolo returns its thing
                # exit(0)
                
                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                # print("imgs shape is ", imgs.shape)
                # exit(0)
                outputs = model(imgs)
                
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                ### This output is is normal MOT Format after preprocessing, 
                # but the results would be (800, 1440) width height result
                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
    
            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            # print(" output results are : ", output_results)
            # output_results = self.convert_to_coco_format(outputs, info_imgs, ids)

            # num_detections = len(output_results)
            # num_gt = len(output_gt)

            # # Update existing detections
            # for i in range(min(num_detections, num_gt)):
            #     output_results[i]['bbox'] = output_gt[i][:4]  # Replace bbox
            #     output_results[i]['score'] = 1.0  # Set score to 1.0

            # # Append additional ground truth boxes if more exist
            # for i in range(num_detections, num_gt):
            #     output_results.append({
            #         'image_id': output_results[0]['image_id'],  # Keep the same image ID
            #         'category_id': output_results[0]['category_id'],  # Set category ID to 1
            #         'bbox': output_gt[i][:4],  # Add new bbox
            #         'score': 1.0,  # Set score to 1.0
            #         'segmentation': output_results[0]['segmentation']  # Empty segmentation
            #     })


            data_list.extend(output_results)
            
            # run tracking
            # online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            # image_size = imgs.shape[2:].cpu().to_
            tensor_shape = imgs.shape
            image_height = int(tensor_shape[3])  
            image_width = int(tensor_shape[2])   
            # image_size  = (image_width, image_height)   ## MAybe this is not used
            # print(" outputs is :", outputs[0])
            # print(" image size is {}, {}".format(info_imgs[0], info_imgs[1]))
            # print("image_shape is : ", self.img_size)
            # print(" img shape is : ", img.shape)
            # exit(0)
            tag = f"{video_name}:{frame_id}"
            # print(" outputs[0] is :  ", outputs[0])
            if self.args.association == "bytetrack" or self.args.association == "ocsort" or self.args.association == "deepsort":
                # output_gt = np.array(output_gt)

                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                # online_targets = tracker.update(output_gt, info_imgs, self.img_size)
            
            
            elif (self.args.association == "botsort" or self.args.association == "diffmot" or self.args.association == "diffusion" or self.args.association == "diffmot_without_virtual"): 
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, image, tag)
                # output_gt = np.array(output_gt)

                # print(" utput type is      : ", outputs[0].dtype)
                # print(outputs[0].shape)
                # print(" utput gt type is      : ", output_gt.dtype)
                # # # output_gt.to(torch.float32)
                # print(" output gt is : ", output_gt)
                # print(output_gt.shape)

                # online_targets = tracker.update(output_gt, info_imgs, self.img_size, image, tag)
                    
            # gc.collect()
            # torch.cuda.empty_cache()
            # print("online targets are :", online_targets)
            # exit(0)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                # print(" t is : ", t)
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))
            
            # for t in online_targets:
            #     """
            #         Here is minor issue that DanceTrack uses the same annotation
            #         format as MOT17/MOT20, namely xywh to annotate the object bounding
            #         boxes. But DanceTrack annotation is cropped at the image boundary, 
            #         which is different from MOT17/MOT20. So, cropping the output
            #         bounding boxes at the boundary may slightly fix this issue. But the 
            #         influence is minor. For example, with my results on the interpolated
            #         OC-SORT:
            #         * without cropping: HOTA=55.731
            #         * with cropping: HOTA=55.737
            #     """
            #     tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            #     tid = t[4]
            #     if tlwh[2] * tlwh[3] > self.args.min_box_area:
            #         online_tlwhs.append(tlwh)
            #         online_ids.append(tid)
            # # save results
            # results.append((frame_id, online_tlwhs, online_ids))
            
            # if frame_id ==1:

                # print("\nonline ids", online_ids)
                # print("\nonline scores : ", online_scores)
                # print("\nfram ID : ", frame_id)
                # print("onlin targets are : ", online_targets)
            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)
                video_dir = os.path.join(Path(result_folder).parent, "visualizations")
                video_filename = os.path.join(Path(result_folder).parent, "visualizations", '{}.mp4'.format(video_names[video_id]))
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)
                if self.args.test:
                    GT_Seq_path = os.path.join("datasets", self.args.dataset_name, "test", "{}".format(video_names[video_id]), "img1")
                else:
                    GT_Seq_path = os.path.join("datasets", self.args.dataset_name, "val", "{}".format(video_names[video_id]), "img1")
                # visualize_tracking_to_video(GT_Seq_path, result_filename, video_filename)
                # visualize_tracking_to_video(GT_Seq_path,  result_filename, video_filename, self.args.dataset_name)

                
            # gc.collect()
            # torch.cuda.empty_cache()
            if self.args.association == "bytetrack" or self.args.association == "diffmot" or self.args.association == "botsort" or self.args.association == "diffusion" or self.args.association == "diffmot_without_virtual":
                del image
        
        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results


    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list



    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info