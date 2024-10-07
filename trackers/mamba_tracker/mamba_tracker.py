import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from .mamba_predictor import MambaPredictor
from trackers.byte_tracker import matching
from .basetrack import BaseTrack, TrackState       ######## THIS IS REALLY IMPORTANT, THIS KEEPS TRACK OF ALL THE TRACKLETS

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    mamba_predictor = MambaPredictor(model_type = "bi-mamba", dataset_name = "MOT20")
    # mamba_predictor = 
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        ### Mamba prediction parameters
        self.prediction = None
        
        self.score = score
        self.tracklet_len = 0
        self.tracklet = [] ## Store the sequence of past detections
        self.add_detection(self._tlwh, self.score)
        
    
    def add_detection(self, bbox, score):
        self.tracklet.append((bbox, score))
        if len(self.tracklet) > 10:
            self.tracklet.pop(0)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def get_sequence_input(self, tracklet_info):
        
        sequence = []
        
        for tlwh, score in tracklet_info:
            
            input_vector = list(tlwh)
            # print("input vector is :", input_vector)
            sequence.append(input_vector)

        ## Convert the list to a Tensor
        sequence = np.array(sequence)
        
        # sequence_tensor = torch.Tensor(sequence)
        
        ## Get the current length of the sequence
        current_length = len(sequence)
        
        ## If the sequence is shorter than max length, pad it with zeros
        
        max_length = 10
        input_dim = 4 ## Bounding box shape
        if current_length < max_length:
            padding = np.zeros((max_length - current_length, input_dim))
            sequence  = np.concatenate((sequence, padding), axis = 0)
            
            
        if current_length > max_length:
            sequence = sequence[-max_length:]
            
        return sequence
            
            
        
        
    @staticmethod
    def predict_mamba(stracks, img_size):
        # print("strack is : ", stracks)
        if len(stracks)  > 0:
            
            # tracklets = np.asarray([st.prediction.copy() for st in stracks])
            # for st in stracks:
                # print(" st is : {} and prediction for it is : {}, and the tracklet length is : {}, and tracklet length is : {}".format(st, st.prediction ,st.tracklet_len, len(st.tracklet)))
                
            # multi_prediction = STrack.mamba_predictor.multi_predict(tracklets, img_size)
            # print(" multi prediction is :", multi_prediction)
            # for i, multi_prediction in enumerate(multi_prediction):
                # stracks[i].prediction = multi_prediction
                
        #     prediction_state = self.prediction_val.copy()
        # if self.state !=TrackState.Tracked:
        #     prediction_state[7] = 0
        # self.prediction = self.mamba_predictor.predict(prediction_state)
        
            tracklets = []
        
            for track in stracks:
            #     print("track is : ", track)
                # if len(track.tracklet) ==1:
                # print(" tracklet before prediction is : \n", track.tracklet)
                sequence_input = track.get_sequence_input(track.tracklet)
                # print("sequence_input shape is :", sequence_input.shape)
                tracklets.append(sequence_input)
                    # tracklets = np.asarray([st.prediction.copy() for st in stracks])
            #         # tracklets = torch.Tensor(tracklets)
                    ### IN THIS CASE WE'RE PASSING JUST ONE SINGLE INPUT TO THE PREDICTOR TO GET THE RESULTS
                
                # elif len(track.tracklet) >=2:
                    # print(" sequence input shape is : ")
                    # print("tracklet is : ", track.tracklet)
                    # predicted_bboxes = STrack.mamba_predictor.multi_predict(sequence_input, img_size)
            #         # print("predicted_bbox is : ", predicted_bboxes)
                    # for i, predicted_bbox in enumerate(predicted_bboxes):
                        # stracks[i].prediction = predicted_bbox
                    # track.update_predicted_state(predicted_bbox)
            tracklets = np.asarray(tracklets)
            # print(" tracklets shape is : ", tracklets.shape)
            # print("tracklet before prediction after padding : \n", tracklets)
            # print(" image size passed into the predictor is :", img_size)
            multi_prediction = STrack.mamba_predictor.multi_predict(tracklets, img_size)
            print(" predicted tracklet is  : \n", multi_prediction)
            for i, multi_prediction in enumerate(multi_prediction):
                stracks[i].prediction = multi_prediction
                    

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks]) ### IT containes the tracklet information upto previous frame. 
                                                                            #if prev frame had 5 tracklets, it'll show (5,8) shape
            
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            print("multi mean shape is :", multi_mean.shape)
            # exit(0)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov


    def activate_mamba(self, mamba_prediction, frame_id):
        """Start a new tracklet"""
        self.mamba_prediction = mamba_prediction
        self.track_id = self.next_id()
        # self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        # print("This top left width height is : ", self._tlwh)
        # print(self.)
        ## The format is already top, left, width, height. No need to change it further to aspect ratio thing as done in kalman filter code.
        self.prediction = self.mamba_prediction.initiate(self._tlwh)
        # print("prediction initiation is :", self.prediction)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id


    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score


    def re_activate_mamba(self, new_track, frame_id, new_id=False):
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        # )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score


    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        
        self.state = TrackState.Tracked
        self.is_activated = True

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

        new_tlwh = new_track.tlwh
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        # self.prediction = self.mamba_prediction.update(self.prediction)
    
        
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score




    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
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
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        print(self.args)
        self.model_type  = self.args.model_type
        self.dataset_name = self.args.dataset_name
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.mamba_prediction_cl = MambaPredictor(model_type = self.model_type, dataset_name  = self.dataset_name)

    def normalize_bounding_boxes(self, bboxes, img_w, img_h):
        
        #### This function first converts the data from xywh format to YOLO Format, then nomralize it to process the model
        
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:,2]/2
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:,3]/2
        # Normalize bounding boxes
        
        bboxes[:, 0] /= img_w  # Normalize center x
        bboxes[:, 1] /= img_h  # Normalize center_y
        bboxes[:, 2] /= img_w  # Normalize width
        bboxes[:, 3] /= img_h  # Normalize height
                
        return bboxes
    
        
        
     
    # Function to scale bounding boxes to the new aspect ratio
    def scale_bounding_boxes(self, bounding_boxes,scale):
        scale_x = scale[1]
        scale_y = scale[0]
        
        print(" scale x is : ", scale_x)
        print("scale y is : ", scale_y)
        scaled_boxes = []
        for (x_min, y_min, x_max, y_max) in bounding_boxes:
            # Scale the coordinates
            new_x_min = int(x_min * scale_x)
            new_y_min = int(y_min * scale_y)
            new_x_max = int(x_max * scale_x)
            new_y_max =  int(y_max * scale_y)
            scaled_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
        
        
        return np.array(scaled_boxes)

        
    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]            ### THIS IS THE ACTUAL IMAGE INFO
        # print("boxes before scaling are :", bboxes)
        print("image size for the actual image is : ", img_h, img_w)
        print("img_height, img_width  for the yolo thing is : {} {}".format(img_size[0], img_size[1]))
        # print(" image height 2, image width 2 : {} {}".format(img_info[0], img_info[1]))
        # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    
        print(" bounding boxes before scaling : \n", bboxes)
        scale = (img_h/img_size[0], img_w/float(img_size[1]))
        bboxes = self.scale_bounding_boxes(bboxes, scale)
        
        print("bounding boxes after scaling : ", bboxes)
        
        # exit(0)
        #### THESE ARE NORMALIZED YOLO FORMAT BOUNDING BOXES
        # bboxes = self.normalize_bounding_boxes(bboxes, img_w, img_h)
        # bboxes /= scale

        # print("boxes after scaling are :", bboxes)

        
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            ###  DETECTIONS ARE GETTING CONVERTED TO TOP LEFT WIDTH HEIGHT FORMAT
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
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
        
        ########### WILL HAVE THE MAKE THE CHANGES HERE ############ REPLACE THE MULTI PREDICT FUNCTIONS WITH MY THINGS
        STrack.predict_mamba(strack_pool, img_info)
        
        # STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)


        # print(" matches : {}, unmatched tracked : {}, unmatched detections : {}".format(matches, u_track, u_detection))
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            
            
            if track.state == TrackState.Tracked:
                track.add_detection(detections[idet].tlbr, detections[idet].score)

                track.update_mamba(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate_mamba(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        # print("SECOND ASSOCIATION WITH LOW CONFIDENCE SCORES")
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
            
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        
        print("SECOND TIME ASSOCIATIONS")
        # print(" matches : {}, unmatched tracked : {}, unmatched detections : {}".format(matches, u_track, u_detection))

        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]

        
            if track.state == TrackState.Tracked:
                track.add_detection(detections_second[idet].tlbr, detections_second[idet].score)

                track.update_mamba(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate_mamba(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            # unconfirmed[itracked].update(detections[idet], self.frame_id)
            track = strack_pool[itracked]


            track.add_detection(detections[idet].tlbr, detections[idet].score)
            unconfirmed[itracked].update_mamba(detections[idet], self.frame_id)
            
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)


        print(" INITIALIZING NEW TRACKS")
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            # track.activate(self.kalman_filter, self.frame_id)
            track.activate_mamba(self.mamba_prediction_cl, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        
        print("UPDATE STATS (DOING SOMETHING WITH LOST TRACKS)")
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
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


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
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
