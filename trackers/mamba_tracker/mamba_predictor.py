# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
import torch
from models_mamba import FullModelMambaBBox, BBoxLSTMModel
import gc
from mambaAttention import FullModelMambaBBoxAttention
"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        #mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')
        
        
        

class MambaPredictor(object):
    # def __init__(self, model_type, dataset_name, model_path):
    def __init__(self, args_new):
    
        self.input_size = 4
        self.input_size = 4  # Bounding box has 4 coordinates: [x, y, width, height]
        self.hidden_size = 64 ## This one is used for LSTM NEtwork which I tried
        self.output_size = 4  # Output also has 4 coordinates
        self.num_layers = 4       ## For LSTM
        self.embedding_dim = 128 ## For Mamba
        self.num_heads = 4
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model_type = "attention-mamba"
        
        self.model_path = None
        self.model = None
        self.best_model_name = None
        self.num_blocks = None
        # print(" Are we reaching here before going into update? ")
        # Dynamic Model parameters
        if args_new is not None:
            self.model_path = args_new.model_path
            self.num_blocks = args_new.num_blocks ## For Mamba
            
            
        
            if self.model_type == "bi-mamba" or self.model_type == "vanilla-mamba":
                self.model = FullModelMambaBBox(self.input_size, self.embedding_dim, self.num_blocks, self.output_size, mamba_type =  self.model_type).to(self.device)
            # exit(0)
            if self.model_type == "attention-mamba":
                self.model = FullModelMambaBBoxAttention(self.input_size, self.embedding_dim, self.num_blocks, self.output_size, num_heads= self.num_heads , mamba_type =  self.model_type).to(self.device)
        
            else:
                self.model = BBoxLSTMModel(self.input_size, self.hidden_size, self.output_size, self.num_layers).to(self.device)


        # self.best_model_name  = "best_model_bbox_{}_variable_{}.pth".format(self.dataset_name, self.model_type)
        
        # self.best_model_name = "running_models/best_model_bbox_sportsmot_publish_variable_bi-mamba_15_October.pth"
        # self.best_model_name = "running_models/best_model_bbox_dancetrack_variable_bi-mamba_14_October.pth"
        # self.best_model_name = "running_models/best_model_bbox_sportsmot_publish_variable_vanilla-mamba_20_October.pth"
        # self.best_model_name = "running_models/best_model_bbox_dancetrack_variable_attention-mamba_29_October.pth"
        # self.best_model_name = "best_models/best_model_bbox_dancetrack_variable_attention-mamba_30_October.pth"                #### This one is the best predictor for now
        # self.best_model_name = "best_model_bbox_sportsmot_publish_variable_attention-mamba_01_November.pth"
        # self.best_model_name = "best_model_bbox_dancetrack_variable_attention-mamba_02_November.pth"                        ##### This one is the best predictor for noww
        # self.best_model_name = model_path
        # self.best_model_name = self.model_path
        
        ## Ablation Study stuff: 
        # self.best_model_name = "best_model_bbox_dancetrack_variable_attention-mamba_max_window_5_num_blocks_2_16_December.pth"
        # self.best_model_name = "best_model_bbox_dancetrack_variable_attention-mamba_max_window_5_num_blocks_3_16_December.pth"
        # self.best_model_name = "best_model_bbox_dancetrack_variable_attention-mamba_max_window_7_num_blocks_3_17_December.pth"
        # self.best_model_name = "best_model_bbox_dancetrack_variable_attention-mamba_max_window_7_num_blocks_4_17_December.pth"
        # self.best_model_name  = "best_model_bbox_dancetrack_variable_attention-mamba_false_augment_max_window_12_15_December.pth"
        
        # self.best_model_name = "/home/viplab/Research_DK/Mamba-MOT/best_model_bbox_dancetrack_variable_attention-mamba_22_November.pth"   ### This is for DRONE ( Sorry for the wrong name)
        # print("\nmodel to be loaded is : ", self.best_model_name)
        if args_new is not None:
            self.best_model_name = args_new.model_path
            # print(" model getting used is : ", self.best_model_name)
            # print("dataset getting used is  : ", args_new.dataset_name)
            # print("number of blocks : ", args_new.num_blocks)
            self.model.load_state_dict(torch.load(self.best_model_name))  # Load the best model


    def initiate(self, measurement):
        
        
        ######### DONT REALLY KNOW IF IT SHOULD BE CONFIDENCE. Just Assuming and doing.
        confidence = [0.0]
        # return np.r_[measurement, confidence]
        return measurement
    
       
    def denormalize_bbox(self, bboxes, img_size):
        """
        Denormalize bounding boxes to original scale. This converts it back to MOT Format as well. 
        """
        # bbox[:, 0] *= image_width  # Denormalize center_x
        # bbox[:, 1] *= image_height  # Denormalize center_y
        # bbox[:, 2] *= image_width  # Denormalize width
        # bbox[:, 3] *= image_height  # Denormalize height
        
        # # Convert back from (center_x, center_y, width, height) to (left, top, width, height)
        # bbox[:, 0] -= bbox[:, 2] / 2  # left = center_x - width/2
        # bbox[:, 1] -= bbox[:, 3] / 2  # top = center_y - height/2
        
        # return bbox
        # assert bboxes.shape[1:] == (4,), "Input bounding boxes should be in (center_x, center_y, width, height) format."
        # print("The image size used in DENORMALIZATION is: ", img_size)
        # Apply denormalization for each prediction
        img_h, img_w = img_size[0].item(), img_size[1].item()          ### SETTING IMAGE SIZE LIKE THIS IS NOT A MISTAKE
        bboxes[:, 0] *= img_w   # Denormalize center_x
        bboxes[:, 1] *= img_h  # Denormalize center_y
        bboxes[:, 2] *= img_w   # Denormalize width
        bboxes[:, 3] *= img_h  # Denormalize height

        # Convert from (center_x, center_y, width, height) to (left, top, width, height)
        bboxes[:, 0] -= bboxes[:, 2] / 2  # left = center_x - width / 2
        bboxes[:, 1] -= bboxes[:, 3] / 2  # top = center_y - height / 2
        
        return bboxes
        
        
    def normalize_yolo_bbox(self, bboxes, image_size):
        
        
        # print(" image size is : ", image_size)
        #### This function first converts the data from xywh format to YOLO Format, then nomralize it to process the model
        img_h, img_w = image_size[0], image_size[1]  # Image dimensions (height, width)

        # print("image height and width used for normalization is : {} , {}".format(img_h, img_w))
    # Convert xywh to center-based YOLO format by adjusting the (x, y) to be at the center
        bboxes[..., 0] = bboxes[..., 0] + bboxes[..., 2] / 2  # Update x to center x
        bboxes[..., 1] = bboxes[..., 1] + bboxes[..., 3] / 2  # Update y to center y

        # Normalize the bounding boxes
        bboxes[..., 0] /= img_w  # Normalize center x
        bboxes[..., 1] /= img_h  # Normalize center y
        bboxes[..., 2] /= img_w  # Normalize width
        bboxes[..., 3] /= img_h  # Normalize height

        return bboxes
    
    
    def multi_predict(self, tracklet_detections, img_size):
        
        # print("model name is : ", self.best_model_name)
        # print("model type is : ", self.model_type)
        # exit(0)
        
        # bounding_boxes = tracklet_detection[:4].copy()
        # print(" tracklet shape in the start is  : ", tracklet_detections.shape)
        # tracklet_detection = tracklet_detection.unsqueeze(0)
        
        # print(" tracklets before prediction : \n\n", tracklet_detections)
        tracklet_detections = self.normalize_yolo_bbox(tracklet_detections, img_size)
        # print(" tracklet before prediction is : ", tracklet_detections)
        
        tracklet_detections = torch.from_numpy(tracklet_detections).float().to(self.device)
        # print("tracklet before prediction shape is : ", tracklet_detections.shape)
        
        
        # print("tracklet detection is : ", tracklet_detections)

        # print(" shape of tracklet detections is : ", tracklet_detections.shape)
        # print(" tracklet after yolo conversion is :", tracklet_detections)
        
        ### NOW, These tracklers could be of different length, (NOT HANDLING THAT RIGHT NOW)
        #### There can be many tracklets together. Need to deal with that. Lets see
        with torch.no_grad():
            predictions = self.model(tracklet_detections)
            # print(" predictions type if : ", type(predictions))
            predicted_bbox  = predictions.cpu().numpy()
            
            # print(" predicted_bbox is : ", predicted_bbox)
            # print("image size is : ", img_size)
            predicted_bbox = self.denormalize_bbox(predictions.cpu().numpy(), img_size)            
            
            # exit(0)
            # print(" predicted bounding shape box is :", predicted_bbox.shape)
        
        
        # gc.collect()
        # torch.cuda.empty_cache()
        
        return predicted_bbox
        
        
        
        