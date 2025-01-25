import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from .mamba_predictor import MambaPredictor
from trackers.mamba_tracker import matching
from .basetrack import BaseTrack, TrackState       ######## THIS IS REALLY IMPORTANT, THIS KEEPS TRACK OF ALL THE TRACKLETS

# from fast_reid.fast_reid_interfece import FastReIDInterface
global args_global
args_global = None


class STrack(BaseTrack):

    # mamba_predictor = MambaPredictor(model_type = "bi-mamba", dataset_name = "MOT20", model_path = None)
    mamba_predictor = MambaPredictor(args = args_global)

    # mamba_predictor = 
    def __init__(self, tlwh, score, padding_window):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        # BaseTrack.reset_id()
        # self.mamba_predictor = MambaPredictor(model_type = "bi-mamba", dataset_name = "MOT20", model_path= model_path)
        self.is_activated = False

        ### Mamba prediction parameters
        self.prediction = None
        # self.track_id = 0
        self.score = score
        self.tracklet_len = 0
        self.padding_window = padding_window
        self.tracklet = [] ## Store the sequence of past detections
        self.tracklet_predictions = []
        self.add_detection(self._tlwh, self.score)
    
    
    def add_detection(self, bbox, score):

        # print(" add detections function working")
        # print("padding window add detection is : ", self.padding_window)
        self.tracklet.append((bbox, score))
        if len(self.tracklet) > self.padding_window:
            self.tracklet.pop(0)

    def add_prediction(self, bbox):

        # print(" add detections function working")
        # print("padding window add detection is : ", self.padding_window)
        self.tracklet_predictions.append((bbox))
        if len(self.tracklet_predictions) > self.padding_window:
            self.tracklet_predictions.pop(0)



    def add_guassian_noise(self, sequence):
        
        mean = 0
        std_dev = 1
        # Generate Gaussian noise
        noise = np.random.normal(mean, std_dev, sequence.shape)
        
        # Add the noise to the sequence
        noisy_sequence = sequence + noise

        return noisy_sequence

    def get_sequence_input(self, tracklet_info):
        
        sequence = []
        
        for tlwh, score in tracklet_info:
            
            input_vector = list(tlwh)
            # print("input vector is :", input_vector)
            sequence.append(input_vector)

        ## Convert the list to a Tensor
        sequence = np.array(sequence)
        # print(sequence)
        # exit(0)
        
        # sequence_tensor = torch.Tensor(sequence)
        
        ## Get the current length of the sequence
        current_length = len(sequence)
        
        ## If the sequence is shorter than max length, pad it with zeros
        
        max_length = self.padding_window
        # print(" max length is : ", max_length)
        flag = False
        input_dim = 4 ## Bounding box shape
        if current_length == 1:
            noisy_sequence = self.add_guassian_noise(sequence)
            sequence  = np.concatenate((sequence, noisy_sequence), axis = 0)
            flag = True
            
            
        if current_length < max_length:
            padding = np.zeros((max_length - current_length, input_dim))
            if flag:
                padding = np.zeros((max_length - (current_length + 1), input_dim))
            sequence  = np.concatenate((sequence, padding), axis = 0)
            # print()
            
            
        if current_length > max_length:
            sequence = sequence[-max_length:]
            
        return sequence
            
            
        
        
    @staticmethod
    def predict_mamba(stracks, img_size):
        # print("strack is : ", stracks)
        if len(stracks)  > 0:
            
        
            tracklets = []
            # print("\n\ntracklet before prediction is : \n")
            # print("padding window is : ", padding_window)
            for track in stracks:
            #     print("track is : ", track)
                # if len(track.tracklet) ==1:
                # print(np.asarray(track.tracklet))
                sequence_input = track.get_sequence_input(track.tracklet)
                tracklets.append(sequence_input)
                    # tracklets = np.asarray([st.prediction.copy() for st in stracks])
            #         # tracklets = torch.Tensor(tracklets)
                
                
            tracklets = np.asarray(tracklets)
            # print(" tracklets shape is : ", tracklets.shape)
            # print("tracklet before prediction after padding : \n", tracklets)
            # print(" image size passed into the predictor is :", img_size)
            multi_predictions = STrack.mamba_predictor.multi_predict(tracklets, img_size)
            # print(" \npredicted tracklet is  : \n", multi_prediction)
            for i, multi_prediction in enumerate(multi_predictions):
                stracks[i].add_prediction(multi_prediction)
                stracks[i].prediction = multi_prediction
                    

   

    def activate_mamba(self, mamba_prediction, frame_id):
        """Start a new tracklet"""
        self.mamba_prediction = mamba_prediction
        
      
        ## The format is already top, left, width, height. No need to change it further to aspect ratio thing as done in kalman filter code.
        self.prediction = self.mamba_prediction.initiate(self._tlwh)
        # print("prediction initiation is :", self.prediction)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            # self.reset_id()
            # self.track_id = self.next_id()
            self.is_activated = True
            # print(" track ID is : ", self.track_id)
            # self.track_id = 0
            
            # self.state = TrackState.New

        self.track_id = self.next_id()    
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id




    def re_activate_mamba(self, new_track, frame_id, new_id=False):
        
        # self.prediction = self.mamba_prediction.prediction

        # print("inside reactive mamba")
        # print(" new track tlwh is : ", new_track.tlwh)
        self.prediction = new_track.tlwh
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score


    
    def update_mamba(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        # print("inside update mamba")
        # print(" new tlwh is : ", new_track.tlwh)
        new_tlwh = new_track.tlwh
        
        # self.prediction = self.mamba_prediction.update(self.prediction)
        
        self.prediction = new_tlwh
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score



    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
       

        if self.prediction is None:
            return self._tlwh.copy()    
        
        # print(" prediction inside tlwh is : ", self.prediction)
        ret = self.prediction[:4].copy()
        # ret[2] *= ret[3]
        # ret[:2] -= ret[2:] / 2

        # print(" result in tlwh func is :",ret)
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class MambaTracker(object):
    def __init__(self, args, padding_window, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        # print(self.args)
        self.model_type  = self.args.model_type
        self.dataset_name = self.args.dataset_name
        #self.det_thresh = args.track_thresh
        self.model_path = args.model_path
        self.padding_window = padding_window
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.model_path = args.model_path
        # self.mamba_prediction_cl = MambaPredictor(model_type = self.model_type, dataset_name  = self.dataset_name, model_path = self.model_path)
        self.mamba_prediction_cl = MambaPredictor(args = self.args)
        
        # self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)
        # self.STrack =- 
        BaseTrack.reset_id()

    # Function to scale bounding boxes to the new aspect ratio
    def scale_bounding_boxes(self, bounding_boxes,scale):
        # scale_x = scale[1]
        # scale_y = scale[0]
        scale_old = scale
        # print(" scale x is : ", scale_x)
        # print("scale y is : ", scale_y)
        scaled_boxes = []
        for (x_min, y_min, x_max, y_max) in bounding_boxes:
            # Scale the coordinates
            new_x_min = int(x_min * scale_old)
            new_y_min = int(y_min * scale_old)
            new_x_max = int(x_max * scale_old)
            new_y_max =  int(y_max * scale_old)
            scaled_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
        
        
        return np.array(scaled_boxes)

        
    def update(self, output_results, img_info, img_size, image):
        self.frame_id += 1
        global args_global
        args_global = self.args
        
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        if img_info[2].item() == 1:
            print("frame ID getting accessed is :", self.frame_id)
            # print(" activated stracks : ", activated_stracks)
            # print("refind tracks : ", refind_stracks)
            # print("los tracks : ", lost_stracks)

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]            ### THIS IS THE ACTUAL IMAGE INFO
        
        new_height, new_width  = img_h, img_w
        old_height, old_width = img_size[0], img_size[1]
        
        # print("img info is :", img_info)
        scale_old = min(img_size[0] / float(img_h), img_size[1] / float(img_w)) ### This is the bytetrack one
        # scale_old = min(img_h / float(img_size[0]),  img_w/ float(img_size[1]))
        # print(" old image height width is :{}, {}".format(old_height, old_width))
        # print(" new image height width :  {}, {}".format(new_height, new_width))
        # print(" bounding boxes before :", bboxes)
        # scale_new = min(new_height/float(old_height), new_width/float(old_width))
        # print(" scale is : ", scale)
        # print("scale_new is : ", scale_old)
        # bboxes = self.scale_bounding_boxes(bboxes, scale)
        # bboxes = self.scale_bounding_boxes(bboxes, scale_new)
        

        # print("outputs for the next frame are:", bboxes)
        
        bboxes /= scale_old

        # print("boxes after scaling are :", bboxes)

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh


        ## DETECTIONS ARE IN THE FORMAT : TOP LEFT, BOTTOM RIGHT
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        # print(" detections before the class thing : \n", dets)
        if len(dets) > 0:
            '''Detections'''
            ###  DETECTIONS ARE GETTING CONVERTED TO TOP LEFT WIDTH HEIGHT FORMAT
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.padding_window) for
                          (tlbr, s) in zip(dets, scores_keep)]
            # print(" detections are : \n", detections)
            # detections_second = [STrack(STrack.tlwh, s) for (tlwh, s) in zip(dets, scores_keep)]

        else:
            detections = []

        #####  KEEP IN MIND THESE ARE DETS, NOT DETECTIONS, SO THEY ARE IN TOP LEFT BOTTOM RIGHT FORMAT 
        # print(" detections are : ", dets)
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        

        STrack.predict_mamba(strack_pool, img_info)
        
        dists, multiple_matched_detections, _ = matching.iou_distance(strack_pool, detections, image, "first_association")
        # print(" shape of cost matrix is : ", dists.shape)
        # print("dists is :", dists)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        #     min_cost_thresh =  0.08
        #     for i in range(dists.shape[0]):
        #         # min_cost = np.where(dists[i] - min_cost_thr)
        #         # Check if the tracklet has multiple similar IoU costs with detections
        #         min_cost = np.min(dists[i])
        #         similar_costs = np.where((dists[i] - min_cost) <= min_cost_thresh)[0]
                # if similar_costs > 1:
                #     # dists  = matching.re_evaluate_score(dists, detections)
                #     features_keep = self.encoder.inference(img, dets)
                #     ious_dists_mask = (ious_dists > self.proximity_thresh)
           


        
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # u_detection.extend(multiple_matched_detections)
        # u_detection = u_detection + multiple_matched_detections
        # print(" FRAME NUMBER IS : ", self.frame_id)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            # print("tracked part is : \n", track)
            # print(" track is : ", track.tlwh)
            # print(" detections part is : \n", det)
            # print("detecetion is : ", det.tlwh)
            
            
            if track.state == TrackState.Tracked:
                # print(" detections are : {}".format(detections[idet].tlwh))
                # track.add_detection(detections[idet].tlbr, detections[idet].score)
                track.add_detection(detections[idet].tlwh, detections[idet].score)
                track.update_mamba(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate_mamba(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        # print("SECOND ASSOCIATION WITH MEDIUM CONFIDENCE SCORES")
        if multiple_matched_detections is not None:
            multiple_matched_detections_tlbr  = [detections.tlbr.tolist() for detections in  multiple_matched_detections]
            multiple_matched_detections_score = [detections.score.tolist() for detections in  multiple_matched_detections]
        # print(" dets second is ", dets_second )
        # dets_second = np.append(dets_second, np.array(multiple_matched_detections_tlbr))
            if len(multiple_matched_detections_tlbr) > 0:
                print("\nmutiple matches detection is : \n", multiple_matched_detections_tlbr)
                print("\nunmatched trracklets are : \n", u_track)
                print("unmatched detections include : \n", u_detection)
                scores_second = np.concatenate((scores_second, multiple_matched_detections_score))
                dets_second  = np.concatenate((dets_second, multiple_matched_detections_tlbr))
                print("\ndets second is : \n", dets_second)
    
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.padding_window) for
                          (tlbr, s) in zip(dets_second, scores_second)]
            # detections_second = [STrack(STrack.tlwh, s) for (tlwh, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        

        # detections_second = detections_second + multiple_matched_detections

        # print("MATCHING FOR SECOND ASSOCIATIONS")
      

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists, _, _ = matching.iou_distance(r_tracked_stracks, detections_second, image, "second_association")
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        
        # print("SECOND TIME ASSOCIATIONS")
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]

        
            if track.state == TrackState.Tracked:
                # track.add_detection(detections_second[idet].tlbr, detections_second[idet].score)
                track.add_detection(detections_second[idet].tlwh, detections_second[idet].score)
           
                track.update_mamba(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate_mamba(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # print("MATCHING FOR UNCONFIRMED TRACKS")
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # print(" \nun confirmed tracks :", unconfirmed)
        detections = [detections[i] for i in u_detection]
        dists, _, _ = matching.iou_distance(unconfirmed, detections, image)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        # print("\ndistance is :", dists)

        
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        # print("\nstrack pool is :", strack_pool)
        # print("\ndetections are : ", detections)
        # print(" \nmatches is :", matches)
        # print("\nun confirmed track is :", u_unconfirmed)
        # print("\nun confirmed detection is ", u_detection)
        # print(' matches for unconfirmed track are  : ', matches)
        for itracked, idet in matches:
            # unconfirmed[itracked].update(detections[idet], self.frame_id)
            # print(" frame id is :", self.frame_id)
            unconfirmed[itracked].update_mamba(detections[idet], self.frame_id)
            # print(' i tracked here is : ', itracked)
            # print('strack pool is ', strack_pool)
            # track = strack_pool[itracked]
            # print("after add detections")
            # track.add_detection(detections[idet].tlbr, detections[idet].score)
            # track.add_detection(detections[idet].tlwh, detections[idet].score, padding_window)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)


        # print(" INITIALIZING NEW TRACKS")
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            # track.activate(self.kalman_filter, self.frame_id)
            track.activate_mamba(self.mamba_prediction_cl, self.frame_id)
            activated_stracks.append(track)
        """ Step 5: Update state"""
        
        # print("UPDATE STATS (DOING SOMETHING WITH LOST TRACKS)")
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks, image)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        # print("output stracks at the end of function is : ", output_stracks)
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb, image):
    # print(" MATCHING FOR REMOVING DUPLICATE TRACKS")
    pdist, _, _ = matching.iou_distance(stracksa, stracksb, image)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
