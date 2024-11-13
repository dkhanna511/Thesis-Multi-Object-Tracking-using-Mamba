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
from trackers.mamba_tracker import matching, matching_hybrid
from .basetrack import BaseTrack, TrackState       ######## THIS IS REALLY IMPORTANT, THIS KEEPS TRACK OF ALL THE TRACKLETS
from .gmc import GMC

from .cmc import CMCComputer
from .gmc import GMC
# from fast_reid.fast_reid_interfece import FastReIDInterface
from .embedding import EmbeddingComputer
# import associations

torch.set_num_threads(16)  # Reduce the number of threads
from .cmc import CMCComputer
class STrack(BaseTrack):

    mamba_predictor = MambaPredictor(model_type = "bi-mamba", dataset_name = "MOT20", model_path = None)
    # mamba_predictor = 
    def __init__(self, tlwh, score, padding_window, feat=None, temp_feat = None,  feat_history = 30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        # BaseTrack.reset_id()
        # self.mamba_predictor = MambaPredictor(model_type = "bi-mamba", dataset_name = "MOT20", model_path= model_path)
        self.is_activated = False
        self.is_virtual = False

        ### Mamba prediction parameters
        self.prediction = None
        
        # self.track_id = 0
        self.score = score
        self.tracklet_len = 0
        self.padding_window = padding_window
        self.virtual_frames = 0
        self.tracklet = [] ## Store the sequence of past detections
        self.add_detection(self._tlwh, self.score)
        

        ##### BOT-SORT Trial CODE
        self.smooth_feat = None
        self.curr_feat = None

        ## DifFMOT/Deep-OC-SORT Parameters
        self.emb = temp_feat

          # wait activate
        self.xywh_omemory = deque([], maxlen=feat_history)
        self.xywh_pmemory = deque([], maxlen=feat_history)
        self.xywh_amemory = deque([], maxlen=feat_history)



        # if feat is not None:
        #     self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.pose_history = deque([], maxlen = feat_history)
        self.alpha = 0.9
    


    ####################3 Bot-SORT CODE ##########################3
    # def update_features(self, feat):
    #     feat /= np.linalg.norm(feat)
    #     self.curr_feat = feat
    #     if self.smooth_feat is None:
    #         self.smooth_feat = feat
    #     else:
    #         self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
    #     self.features.append(feat)
    #     self.smooth_feat /= np.linalg.norm(self.smooth_feat)
    # #############################################################


    ################################  DiffMOT #######################

    # def update_pose(self, new_pose, alpha=0.9):
    #     if self.current_pose is None:
    #         self.current_pose = new_pose
    #     else:
    #         self.current_pose = alpha * self.current_pose + (1 - alpha) * new_pose
    #     self.pose_history.append(self.current_pose)
    
    def update_pose(self, new_pose, max_history=10):
        self.pose_history.append(new_pose)
        if len(self.pose_history) > max_history:
            self.pose_history.popleft()
        
        # Calculate pose velocity
        if len(self.pose_history) > 1:
            pose_velocity = np.mean([np.linalg.norm(self.pose_history[i+1] - self.pose_history[i]) 
                                    for i in range(len(self.pose_history)-1)])
            
            # Adjust alpha based on velocity
            alpha = max(0.1, min(0.9, 1.0 - pose_velocity / 100))  # Adjust scaling as needed
        else:
            alpha = 0.5  # Default value
        
        # Update current pose
        if self.current_pose is None:
            self.current_pose = new_pose
        else:
            self.current_pose = alpha * self.current_pose + (1 - alpha) * new_pose

    def update_features(self, feat, alpha=0.95):
        self.curr_feat = feat
        # alpha = 0.8 if self.virtual_frames > 0 else alpha 
        self.emb = alpha * self.emb + (1 - alpha) * feat
        self.emb /= np.linalg.norm(self.emb)
    ################################################################
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
            for i, multi_pred in enumerate(multi_prediction):
                stracks[i].prediction = multi_pred
         
            
            for i, st in enumerate(stracks):
                st._tlwh = multi_prediction[i]
                st.xywh_pmemory.append(st.xywh.copy())
                st.xywh_amemory.append(st.xywh.copy())

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


    ''' THIS FUNCTIONS IS FOR INITIATION OF A TRACK WITH NEW DETECTIONS'''
    def activate_mamba(self, mamba_prediction, frame_id):
        """Start a new tracklet"""
        self.mamba_prediction = mamba_prediction
        
      
        ## The format is already top, left, width, height. No need to change it further to aspect ratio thing as done in kalman filter code.
        self.prediction = self.mamba_prediction.initiate(self._tlwh)
        # print("prediction initiation is :", self.prediction)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
            
        self.track_id = self.next_id()    
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_pmemory.append(self.xywh.copy())
        self.xywh_amemory.append(self.xywh.copy())




    ''' THIS FUNCTIONS IS FOR RE-ACTIVATING THE LOST TRACKS'''
    def re_activate_mamba(self, new_track, frame_id, new_id=False, state = None):
        
        # self.prediction = self.mamba_prediction.prediction

        # print("inside reactive mamba")
        # print(" new track tlwh is : ", new_track.tlwh)

        ############################### BOT SORT ####################
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        ###########################################


        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_amemory[-1] = self.xywh.copy()

        self.prediction = new_track.tlwh
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if state == "Virtual":
            self.state = TrackState.Virtual
        
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score


    ''' THIS FUNCTIONS IS FOR UPDATING THE TRACKLET INFORMATION FOR ALREADY EXISTING TRACKS '''
    def update_mamba(self, new_track, frame_id, update_feature = False, state = None, score = None):
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
        self._tlwh = new_tlwh
        # self.prediction = self.mamba_prediction.update(self.prediction)
        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_amemory[-1] = self.xywh.copy()

        ############################ BOT SORT ###################


        self.prediction = new_tlwh
      
        if state == "Virtual":
            self.state = TrackState.Virtual
            self.score = score
            self.is_activated = False
            self.is_virtual = True
            # print(" state of the object is : ", self.state)

            if update_feature:
                self.update_features(new_track.curr_feat, alpha = 0.85)
            
        else:
            self.state = TrackState.Tracked  
            self.score = new_track.score
            self.is_activated = True

            if update_feature:
                self.update_features(new_track.curr_feat)
        
     



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


    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] = ret[:2] + ret[2:] / 2
        # ret[2:] += ret[:2]
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


class MambaTrackerDiffMOT(object):
    def __init__(self, args, padding_window, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.virtual_stracks = []
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


        self.virtual_track_buffer = args.virtual_track_buffer

         # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.keypoint_matching_threshold  = 0.15
        # self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])


        # if args.with_reid:
        self.embedder = EmbeddingComputer(args.dataset_name, test_dataset = self.args.test, grid_off = True)
        self.alpha_fixed_emb = 0.95
            # self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        #######################################################################
        BaseTrack.reset_id()

    def terminate_virtual_tracks(self):
        current_virtual_tracks = [t for t in self.virtual_stracks if t.state == TrackState.Virtual]
        for track in current_virtual_tracks:
            if self.frame_id - track.last_seen > self.virtual_track_buffer:
                self.virtual_stracks.remove(track)
                self.removed_stracks.append(track)
        
    def update(self, output_results, img_info, img_size, img, tag):
        self.frame_id += 1
       
        activated_stracks = [] ## Already existing tracklets are stored in this
        refind_stracks = []  ## Tracklets that are lost and then found again are stored in this
        lost_stracks = []    ## Lost tracklets are stored in this
        removed_stracks = []    ## tracklet IDS that are completely gone are stored in this
        virtual_stracks = []    ### tracklets which are not matched in STAGE 1 and can be occluded are stored in this

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
  
        '''Extract embeddings '''
     
            ############### THIS IS FROM BOTSORT DEFAULT CODE ########################
            # features_keep = self.encoder.inference(img, dets)
        dets_embs = np.ones((dets.shape[0], 1))

        ##################3 THIS ONE IS FROM DEEP OC SORT code ####################################
        if dets.shape[0] != 0:
            dets_embs = self.embedder.compute_embedding(img, dets, tag)
        # print("dets are : ", dets.shape)
        trust = (scores_keep - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
        dets_alpha = af + (1 - af) * (1 - trust)
            # print(" features keep is : \n", features_keep)
            # np.savetxt("botsort.txt", features_keep)
        # print(" detections before the class thing : \n", dets)
        if len(dets) > 0:
            '''Detections'''
            ###  DETECTIONS ARE GETTING CONVERTED TO TOP LEFT WIDTH HEIGHT FORMAT
            # if self.args.with_reid:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), score = s, padding_window = self.padding_window,  temp_feat = f, feat_history= 30) for
                        (tlbr, s, f) in zip(dets, scores_keep, dets_embs)]
            # print(" detections are : \n", detections)
            # detections_second = [STrack(STrack.tlwh, s) for (tlwh, s) in zip(dets, scores_keep)]
        else:
            detections = []


     
        
        # exit(0)
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if (track.is_activated):
                tracked_stracks.append(track)
                
            elif track.is_virtual:
                virtual_stracks.append(track)
            else:
                unconfirmed.append(track)
                
            # if track.is_virtual:
            #     virtual_stracks.appeend(track)
            #     tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        # strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        new_tracklets = []
        new_tracklets.extend(self.lost_stracks)
        new_tracklets.extend(self.virtual_stracks)
        strack_pool = joint_stracks(tracked_stracks, new_tracklets)
        
        # Predict the current location withdets_embs KF
        
        STrack.predict_mamba(strack_pool, img_info)

        trk_embs = [st.emb for st in strack_pool]
        trk_embs = np.array(trk_embs)
        # print(" track embeddings ashape is : ", trk_embs)
        emb_cost = None if (trk_embs.shape[0] == 0 or dets_embs.shape[0] == 0) else trk_embs @ dets_embs.T

        
        #  Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)      

        ious_dists,  keypoint_dists, potential_multiple_matches = matching.iou_distance(strack_pool, detections, img, association = "first_association", buffer_size = 0.3)
        # confidence_dists = matching.calculate_confidence_cost_matrix(strack_pool, detections)
        # print("confidence distances are : \n",confidence_dists)
        # print("confidence dist is : ", confidence_dists)
        iou_matrix =  1 -ious_dists

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > 0.1).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                if emb_cost is None:
                    emb_cost = 0
                w_assoc_emb = self.args.w_assoc_emb
                aw_param = self.args.aw_param

                w_matrix = matching.compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param)
                emb_cost *= w_matrix

                final_cost = -(iou_matrix + emb_cost)
                matched_indices = matching.linear_assignment2(final_cost)
        else:
            matched_indices = np.empty(shape=(0, 2))


        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 1]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(strack_pool):
            if t not in matched_indices[:, 0]:
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < 0.1:
                unmatched_detections.append(m[1])
                unmatched_trackers.append(m[0])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        u_track = np.array(unmatched_trackers)
        u_detection = np.array(unmatched_detections)
        
        
        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # matched, u_track, u_det = associations.associate(dets, strack_pool, features_keep, self.args.match_thresh, self.args.w_ass_emb, aw_param = 0.5)
       
        ''' THESE ARE THE MATCHES AFTER 1st ROUND OF ASSOCIATION'''
        # print("Matched tracks at first step of association include : \n")
        for itracked, idet in matches:
            track = strack_pool[itracked]
            # print(track)
            det = detections[idet]   
            alp = dets_alpha[idet]
            if track.state == TrackState.Tracked:
                # print(" detections are : {}".format(detections[idet].tlwh))
                # track.add_detection(detections[idet].tlbr, detections[idet].score)
                track.add_detection(detections[idet].tlwh, detections[idet].score)
                track.update_mamba(detections[idet], self.frame_id)
                track.update_features(det.emb, alp)
                activated_stracks.append(track)
            else:
                track.re_activate_mamba(det, self.frame_id, new_id=False)
                track.update_features(det.emb, alp)
                refind_stracks.append(track)


        predictions_for_virtual_detections = []
        
        if len(potential_multiple_matches) > 0:
            for i_untracked in u_track:
                # if  strack_pool[i_untracked] in potential_multiple_matches[:, 0]:
                #     predictions_for_virtual_detections.append(strack_pool[i_untracked])
                for entry in potential_multiple_matches:
                    if strack_pool[i_untracked] in entry[0]:
                        predictions_for_virtual_detections.append([strack_pool[i_untracked], entry[1], entry[2]])
                        
            print(" potential matches were : ", potential_multiple_matches)
            print(" predictions with virtual detections : ", predictions_for_virtual_detections)
                    

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
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.padding_window, feat_history= 30) for
                          (tlbr, s) in zip(dets_second, scores_second)]
            # detections_second = [STrack(STrack.tlwh, s) for (tlwh, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        
        ''' 2nd LEVEL OF ASSOCIATIONS !!!!!!!!!!!!'''
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists, keypoint_dists, _ = matching.iou_distance(r_tracked_stracks, detections_second, img, association = "second_associations", buffer_size = 0.4)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        if len(u_track) > 0:
            print("unmatched tracks after 2nd association : ", [r_tracked_stracks[idx] for idx in u_track])

        
        # filtered_virtual_detections = [entry for entry in predictions_for_virtual_detections if entry[0] not in matches]
        
        
        # if len(filtered_virtual_detections) > 0:
        #     print(" filtered virtual detections are : ", filtered_virtual_detections)
      
        filtered_virtual_detections  = []   
        filtered_u_track = []
        if len(predictions_for_virtual_detections) > 0:
            for i_untracked in u_track:
                # print(" predictions for virtual detections  list is : ", predictions_for_virtual_detections[:, 0])
                if r_tracked_stracks[i_untracked] not in [entry[0] for entry in predictions_for_virtual_detections]:
                # if r_tracked_stracks[i_untracked] not in predictions_for_virtual_detections[:, 0]:
                    filtered_u_track.append(i_untracked)
               
                    # if r_tracked_stracks[i_untracked] not in predictions_for_virtual_detections[:, 0]:
                    #     # filtered_u_track.append(r_stracked[i_untracked])
                    #     filtered_u_track.append(i_untracked)
        
            for entry in predictions_for_virtual_detections:
                if entry[0] in [r_tracked_stracks[i_untracked] for i_untracked in u_track]:
                    filtered_virtual_detections.append(entry)
            # if r_tracked_stracks[i_untracked] not in predictions_for_virtual_detections[:, 0]:
            #     filtered_u_track.append(i_untracked)
            u_track = filtered_u_track


        # if len(u_track) > 0:
        #     print("unmatches tracklets after second association are : ")
        #     for i_untracked in u_track:
        #         print(r_tracked_stracks[i_untracked], "    ", r_tracked_stracks[i_untracked].state)

        if len(filtered_u_track) > 0:
            print("filtered untracked files after 2nd association is : ", [r_tracked_stracks[idx] for idx in u_track])
        
        if len(matches) > 0:
            print("matched tracks in 2nd association : \n")
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            print(track)
            det = detections_second[idet]

        
            if track.state == TrackState.Tracked:
                # track.add_detection(detections_second[idet].tlbr, detections_second[idet].score)
                track.add_detection(detections_second[idet].tlwh, detections_second[idet].score)
                track.update_mamba(det, self.frame_id)
                activated_stracks.append(track)

            else:
                track.re_activate_mamba(det, self.frame_id, new_id=False)
                refind_stracks.append(track)     
        ''' STARTING WITH VIRTUAL DETECTION TASKS'''

        # if len(u_track) > 0:
            # print("unmatched predictions are : ", u_track)
            # print("multiple potential predictions are :", potential_multiple_matches)
        # print(" matches are : ", matches)
        # matched_predictions = {pred_idx for pred_idx, _ in matches}
        
    
        if len(filtered_virtual_detections) > 0:
            
            virtual_detections_list = []
            virtual_detections_scores = []
            virtual_detections_tracks = []
            for index, pred_idx in enumerate(filtered_virtual_detections):
                # print(" pred idx is : ", pred_idx)
                virtual_detections_list.append(filtered_virtual_detections[index][1])
                virtual_detections_scores.append(filtered_virtual_detections[index][2])  # Create virtual detection with same bounding box as prediction
                virtual_detections_tracks.append(filtered_virtual_detections[index][0])
                print(" virtual detections tracks is : ", filtered_virtual_detections[index][0])
         
            virtual_detections = [STrack(STrack.tlbr_to_tlwh(tlbr), score = s, padding_window = self.padding_window, feat_history= 30) for
                        (tlbr, s) in zip(virtual_detections_list, virtual_detections_scores)]
            
            
            virtual_tracks_for_matching = [track for track in virtual_detections_tracks if ((track.state == TrackState.Virtual) or (track.state == TrackState.Tracked))]
            dists, _, _ = matching.iou_distance(virtual_tracks_for_matching, virtual_detections, img, association = "half_first_association")
            
            
            matches_virtual, u_unconfirmed_virtual, u_detection_virtual = matching.linear_assignment(dists, thresh=0.7)
            print(" matched virtual detections are : ", matches_virtual)
            # ''' Virtual Level Associations '''
            for itracked, idet in matches_virtual:
                track = virtual_tracks_for_matching[itracked]
                det = virtual_detections[idet]

                if track.state == TrackState.Tracked:
                    track.mark_virtual()
                elif track.state == TrackState.Virtual:
                    track.update_frame_id()
                    
                score = virtual_detections[idet].score
                if track.virtual_frames == 1:
                    score = 0.9 * virtual_detections[idet].score
                elif track.virtual_frames == 2:
                    score = 0.85 * virtual_detections[idet].score
                elif track.virtual_frames == 3:
                    score = 0.85 * virtual_detections[idet].score
                elif track.virtual_frames == 4:
                    score = 0.80 * virtual_detections[idet].score
                elif track.virtual_frames == 5:
                    score = 0.75 * virtual_detections[idet].score
                    
                print("{} virtual frames are : {}".format(track,  track.virtual_frames))
                # print()

               
                track.add_detection(virtual_detections[idet].tlwh, score)
                # if track.state == TrackState.Lost:
                #     track.re_activate_mamba(det, self.frame_id, new_id=False, state = "Virtual")
                # track.add_detection(virtual_detections[idet].tlwh, virtual_detections[idet].score)
                track.update_mamba(det, self.frame_id, state = "Virtual", score = score)
                virtual_stracks.append(track)
                    
                    
                    


            ## Handle unmatched virtual tracklets with threshold check
            # for it in u_unconfirmed_virtual:
            #     track = u_unconfirmed_virtual[it]
            #     if track.state == TrackState.Virtual:
            #         track.virtual_frames += 1
            #         if track.virtual_frames > self.virtual_track_buffer:  # Exceeded frame threshold for virtual state
            #             # Mark tracklet as lost or remove from list
            #             track.mark_removed()
            #             removed_stracks.append(track)
            #         else:
            #             # If not yet at threshold, retain in virtual state
            #             virtual_stracks.append(track)
            #     else:
            #         # If not in virtual state, mark as lost
            #         track.mark_removed()
            #         removed_stracks.append(track)
         

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # if len(u_track) > 0:
        #     # print("unmatches tracklets after Virtual  association are : ")
        #     for i_untracked in u_track:
        #         print(r_tracked_stracks[i_untracked], "    ", r_tracked_stracks[i_untracked].state)
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''

        detections = [detections[i] for i in u_detection]
        # dists, keypoint_dists,  _ = matching.iou_distance(unconfirmed, detections, img)
        dists, keypoint_dists,  _ = matching.iou_distance(unconfirmed, detections, img)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        # if not self.args.mot20:
        #     ious_dists = matching.fuse_score(ious_dists, detections)
  
        # if self.args.with_reid:
        #     emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
        #     raw_emb_dists = emb_dists.copy()
        #     emb_dists[emb_dists > self.appearance_thresh] = 1.0
        #     emb_dists[ious_dists_mask] = 1.0
        #     dists = np.minimum(ious_dists, emb_dists)
        # else:
        #     dists = ious_dists


        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)


        for itracked, idet in matches:
            # unconfirmed[itracked].update(detections[idet], self.frame_id)
            # print(" frame id is :", self.frame_id)
            alp = dets_alpha[idet]
            unconfirmed[itracked].update_mamba(detections[idet], self.frame_id)
            unconfirmed[itracked].update_features(detections[idet].emb, alp)

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
        
        ''' REMOVING THE LOST TRACKS IF THEY HAVE EXCEEDED THERE LIFETIME OF BEING LOST'''
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        
        for track in self.virtual_stracks:
            if self.frame_id - track.end_frame > self.virtual_track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
     

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.virtual_stracks = joint_stracks(self.tracked_stracks, virtual_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        
        self.virtual_stracks = sub_stracks(self.virtual_stracks, self.tracked_stracks)
        self.virtual_stracks.extend(virtual_stracks)
        self.virtual_stracks = sub_stracks(self.virtual_stracks, self.removed_stracks)
        
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        self.tracked_stracks = joint_stracks(self.tracked_stracks, virtual_stracks)
        
        # self.terminate_virtual_tracks()

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
    pdist , keypoint_dist,  _= matching.iou_distance(stracksa, stracksb, None)
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
