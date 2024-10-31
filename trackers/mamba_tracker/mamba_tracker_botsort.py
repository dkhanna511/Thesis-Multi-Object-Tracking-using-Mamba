import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import gc
from .kalman_filter import KalmanFilter
from .mamba_predictor import MambaPredictor
from trackers.mamba_tracker import matching
from .basetrack import BaseTrack, TrackState       ######## THIS IS REALLY IMPORTANT, THIS KEEPS TRACK OF ALL THE TRACKLETS
from .gmc import GMC
# from fast_reid.fast_reid_interfece import FastReIDInterface
from .embedding import EmbeddingComputer

torch.set_num_threads(16)  # Reduce the number of threads
from .cmc import CMCComputer
class STrack(BaseTrack):

    mamba_predictor = MambaPredictor(model_type = "bi-mamba", dataset_name = "MOT20", model_path = None)
    # mamba_predictor = 
    def __init__(self, tlwh, score, padding_window, feat=None, feat_history = 50):

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
        self.add_detection(self._tlwh, self.score)
        

        ##### BOT-SORT Trial CODE
        self.smooth_feat = None
        self.curr_feat = None

        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9
    


    ####################3 Bot-SORT CODE ##########################3
    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)
    #############################################################

    def add_detection(self, bbox, score):

        # print(" add detections function working")
        # print("padding window add detection is : ", self.padding_window)
        self.tracklet.append((bbox, score))
        if len(self.tracklet) > self.padding_window:
            self.tracklet.pop(0)

   

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
            multi_prediction = STrack.mamba_predictor.multi_predict(tracklets, img_size)
            # print(" \npredicted tracklet is  : \n", multi_prediction)
            for i, multi_prediction in enumerate(multi_prediction):
                stracks[i].prediction = multi_prediction
                    

        @staticmethod
        def multi_gmc(stracks, H=np.eye(2, 3)):
            if len(stracks) > 0:
                multi_prediction = np.asarray([st.prediction.copy() for st in stracks])
                
                R = H[:2, :2]
                R8x8 = np.kron(np.eye(4, dtype=float), R)
                t = H[:2, 2]

                for i, mean in enumerate(multi_prediction):
                    prediction = R8x8.dot(mean)
                    prediction[:2] += t
                    # cov = R8x8.dot(cov).dot(R8x8.transpose())

                    stracks[i].prediction = prediction



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

        ############################### BOT SORT ####################
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        ###########################################
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
        

        ############################ BOT SORT ###################

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        ########################################################

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


class MambaTrackerBot(object):
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
        self.model_path = args.model_path
        self.mamba_prediction_cl = MambaPredictor(model_type = self.model_type, dataset_name  = self.dataset_name, model_path = self.model_path)
        self.cmc = CMCComputer()
        ################################3 BOT SORT ###########################
        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

         # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        # self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

        if args.with_reid:
            self.embedder = EmbeddingComputer(args.dataset_name, False, grid_off = True)

            # self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        #######################################################################
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

        
    def update(self, output_results, img_info, img_size, img, tag):
        self.frame_id += 1
       
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        img_h, img_w = img_info[0], img_info[1]            ### THIS IS THE ACTUAL IMAGE INFO
        
        scale_old = min(img_size[0] / float(img_h), img_size[1] / float(img_w)) ### This is the bytetrack one

        if img_info[2].item() == 1:
            print("frame ID getting accessed is :", self.frame_id)
            # print(" activated stracks : ", activated_stracks)
            # print("refind tracks : ", refind_stracks)
            # print("los tracks : ", lost_stracks)
        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
            else:
                output_results = output_results.cpu().numpy()
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2


            bboxes /=scale_old
            ##########################3 BOT SORT ######################
            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]

        else:
            bboxes = []
            scores = []
            dets = []
            scores_keep = []
  

        # dets /=scale_old
        '''Extract embeddings '''
        if self.args.with_reid:
            ############### THIS IS FROM BOTSORT DEFAULT CODE ########################
            # features_keep = self.encoder.inference(img, dets)
            
            ##################3 THIS ONE IS FROM DEEP OC SORT code ####################################
            features_keep = self.embedder.compute_embedding(img, dets[:, :4], tag)
            
            # print(" features keep is : \n", features_keep)
            # np.savetxt("botsort.txt", features_keep)
        # print(" detections before the class thing : \n", dets)
        if len(dets) > 0:
            '''Detections'''
            ###  DETECTIONS ARE GETTING CONVERTED TO TOP LEFT WIDTH HEIGHT FORMAT
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.padding_window, f) for
                            (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            # print(" detections are : \n", detections)
            # detections_second = [STrack(STrack.tlwh, s) for (tlwh, s) in zip(dets, scores_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s,  self.padding_window) for
                              (tlbr, s) in zip(dets, scores_keep)]

        else:
            detections = []
        
        # exit(0)
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
        
        #  Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)      

        ious_dists, multiple_matched_detections = matching.iou_distance(strack_pool, detections, img)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        # print(" ious distsances are : \n", ious_dists)
        # print(ious_dists_mask)


        # if not self.args.mot20:
        #     dists = matching.fuse_score(ious_dists, detections)
        

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            # if self.args.with_reid and dets.shape[0] != 0:
            # Shape = (num detections, 3, 512) if grid
            # print(" embedded distances : \n",  emb_dists)
            raw_emb_dists = emb_dists.copy()
            # print("appearance threshold : ", self.appearance_thresh )
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists



        
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        
       
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]   
            
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
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            
        else:
            dets_second = []
            scores_second = []



        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.padding_window) for
                          (tlbr, s) in zip(dets_second, scores_second)]
            # detections_second = [STrack(STrack.tlwh, s) for (tlwh, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists, _ = matching.iou_distance(r_tracked_stracks, detections_second, img)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        
        
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



        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''

        detections = [detections[i] for i in u_detection]
        dists, _ = matching.iou_distance(unconfirmed, detections, img)
        ious_dists, _ = matching.iou_distance(unconfirmed, detections, img)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)
  
        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists


        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)


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

     

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
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
    # print(" MATCHING FOR REMOVING DUPLICATE TRACKS")
    pdist , _= matching.iou_distance(stracksa, stracksb, None)
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
