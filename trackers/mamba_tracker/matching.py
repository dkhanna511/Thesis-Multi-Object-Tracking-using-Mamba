import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from trackers.byte_tracker import kalman_filter
import time
import torch
from ultralytics import YOLO

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
from matplotlib.patches import Polygon as MplPolygon

model_keypoint = YOLO("yolo11x-pose.pt")


def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious

def pose_estimate_cosine_similarity(prediction_poses, btlbrs):

    pred_vector =  np.array([pred.flatten() for pred in ])





def to_tlbr(bboxes):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
    bboxes_list = []
    for box in bboxes:
        box[2:] += box[:2]
        bboxes_list.append(box)
    # bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]  
    # bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]  

    return bboxes_list


def calculate_avg_iou(past_atracks, past_btracks, match_indices, col):

    past_iou_list = []
    for idx in match_indices:
        total_iou  = 0
        count = 0
        print(" for idx : {}".format(idx))
        for t in range(len(past_atracks[idx])):
            past_predictions = past_atracks[idx][t]
            past_predictions = to_tlbr([past_predictions])
            past_detections = to_tlbr([past_btracks[col][t][0]])
            print("past predictions are : ", past_predictions)
            print(" past detections are : ", past_detections)
            # # Calculate IoUs for the past frame.
            past_ious = ious(past_predictions, past_detections)
            total_iou +=past_ious
            count +=1

        past_iou_list.append(np.mean(total_iou/count))

    return np.array(past_iou_list)

def calculate_avg_iou_for_past(past_atracks, past_btracks, match_indices, col):
    """
    Calculate the average IoU over past tracklets for the given matches.
    
    :param past_atracks: List of predictions from t-6 to t-1
    :param past_btracks: List of detections from t-6 to t-1
    :param match_indices: List of predictions matched with a single detection
    :param col: Detection column index

    :rtype: float (Average IoU)
    """
    total_iou = 0
    count = 0
    # print("column is : ", col)
    past_atrack_np = np.array(past_atracks)
    
    past_iou_list = []
    # print(" past atracks sha[e is :]", np.shape(past_atracks))
    for t in range(past_atrack_np.shape[1]):  # Iterate over past frames (t-6 to t-1)
        # print(" length of past_atracks is : ", len((past_atrack_np[:, t])))
        # print("pat atrack is : ", past_atracks[t])
        # print("tracklet with matched index is : ", past_atrack_np[:, t])
        past_predictions = [past_atracks[idx][t] for idx in match_indices]
        # print("past predctions are : ", past_predictions)
        # # break
        past_predictions = to_tlbr(past_predictions)
        print("past prediction is : ", past_predictions)
        past_detections = to_tlbr([past_btracks[col][t][0]])
    
        # past_detection = past_btracks
        print("past detection is : ", past_detections)

        
        # # Calculate IoUs for the past frame.
        past_ious = ious(past_predictions, past_detections)
        print(" past iou is : ", past_ious)
        total_iou += np.mean(past_ious, axis = 1)  # Average IoU for this frame
        
        count += 1
    avrg_iou  = total_iou / count if count > 0 else 0

    print("avrg_iou is :", avrg_iou)
    return avrg_iou



def height_iou_np(atlbrs, btlbrs):
    """
    Calculate Height-based IoU (HIoU) for multiple predictions and detections in TLBR format.

    :param preds: NumPy array of shape (N, 4) with (y_top, x_left, y_bottom, x_right) for predictions.
    :param dets: NumPy array of shape (M, 4) with (y_top, x_left, y_bottom, x_right) for detections.

    :return: NumPy array of shape (N, M) containing HIoU values.
    """
    h_ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if h_ious.size == 0:
        return h_ious
    
    preds =  np.array(atlbrs)
    dets = np.array(btlbrs)
    # Extract the top and bottom coordinates
    y_top_preds = preds[:, 0][:, np.newaxis]  # Shape: (N, 1)
    y_bottom_preds = preds[:, 2][:, np.newaxis]  # Shape: (N, 1)

    y_top_dets = dets[:, 0][np.newaxis, :]  # Shape: (1, M)
    y_bottom_dets = dets[:, 2][np.newaxis, :]  # Shape: (1, M)

    # Calculate intersection heights: max(0, min(y_bottom) - max(y_top))
    intersection = np.maximum(0, np.minimum(y_bottom_preds, y_bottom_dets) -
                                  np.maximum(y_top_preds, y_top_dets))

    # Calculate union heights: max(y_bottom) - min(y_top)
    union = np.maximum(y_bottom_preds, y_bottom_dets) - \
            np.minimum(y_top_preds, y_top_dets)

    # Calculate HIoU: intersection / union, avoiding division by zero
    h_ious = np.divide(intersection, union, where=(union != 0), 
                      out=np.zeros_like(intersection, dtype=float))

    return h_ious



def extract_patches(image, boxes, box_type='xywh'):
    """Extract patches of bounding boxes from an image.
    
    Args:
        image (ndarray): The input image from which to extract patches.
        boxes (list of tuples): List of bounding boxes. Each box can be:
                                - 'xywh': (x, y, width, height)
                                - 'xyxy': (x1, y1, x2, y2)
        box_type (str): Type of box format, either 'xywh' or 'xyxy'.
    
    Returns:
        list of ndarray: List of image patches corresponding to the boxes.
    """
    patches = []

    for box in boxes:
        if box_type == 'xywh':
            x, y, w, h = box
            x2, y2 = x + w, y + h
        elif box_type == 'xyxy':
            x, y, x2, y2 = box
        else:
            raise ValueError("box_type must be either 'xywh' or 'xyxy'")
        

        # Ensure the coordinates are within image bounds
        x, y = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        # print(" x : {}, y: {}, x2 : {}, y2: {}".format(x, y, x2, y2))
        # Extract the patch
        patch = image[int(y):int(y2), int(x):int(x2)]
        patches.append(patch)

    return patches







def get_most_descriptive_keypoints(keypoints):
    """
    Returns the keypoint set with the fewest [0, 0] coordinates.

    Args:
        keypoints (Tensor): A PyTorch tensor of shape (n, 17, 2), where n is the number of keypoint sets.

    Returns:
        Tensor: The most descriptive keypoint set (17, 2).
    """
    # if keypoints.dim() != 3 or keypoints.size(2) != 2:
    #     raise ValueError("Input tensor must have shape (n, 17, 2).")

    # Count the number of [0, 0] entries in each keypoint set
    zero_counts = torch.sum((keypoints == 0).all(dim=2), dim=1)

    # Find the index of the set with the minimum number of [0, 0] entries
    best_index = torch.argmin(zero_counts)
    main_keypoints = keypoints[best_index]


    # Return the keypoint set with the fewest [0, 0] entries
    return keypoints[best_index]


def get_bboxes_from_joints(joints):

    left_body = [6, 10]
    right_body = [5, 9]
    lower_body = [11, 12, 15, 16]
    lower_body_joints = (joints[11], joints[12], joints[15], joints[16])
    joints_array = np.array(lower_body_joints)

    lefty_body_keypoints = joints[left_body]
    right_body_keypoints = joints[right_body]
    # print("\nleft keypoints are :", lefty_body_keypoints)
    # Calculate the bounding box coordinates
    left_x_min = torch.min(lefty_body_keypoints[:, 0])
    left_y_min = torch.min(lefty_body_keypoints[:, 1])
    left_x_max = torch.max(lefty_body_keypoints[:, 0])
    left_y_max = torch.max(lefty_body_keypoints[:, 1])
    
    right_x_min = torch.min(right_body_keypoints[:, 0])
    right_y_min = torch.min(right_body_keypoints[:, 1])
    right_x_max = torch.max(right_body_keypoints[:, 0])
    right_y_max = torch.max(right_body_keypoints[:, 1])


        # Calculate the bounding box
    lower_x_min = np.min(joints_array[:, 0])
    lower_x_max = np.max(joints_array[:, 0])
    lower_y_min = np.min(joints_array[:, 1])
    lower_y_max = np.max(joints_array[:, 1])

    
    return (torch.tensor([left_x_min, left_y_min, left_x_max, left_y_max]), torch.tensor([right_x_min, right_y_min, right_x_max, right_y_max]), torch.tensor([lower_x_min, lower_x_max, lower_y_min, lower_y_max]))


def filter_keypoints(keypoints):
    """Filter out [0, 0] keypoints from the array."""
    return np.array([point for point in keypoints if not np.all(point == [0, 0])])

def create_polygon_from_keypoints(keypoints):
    """Create a valid polygon from filtered keypoints."""
    filtered_points = filter_keypoints(keypoints)
    if len(filtered_points) < 3:
        raise ValueError("Not enough valid keypoints to form a polygon.")
    # print("filtered points are  ", filtered_points)
    # Use MultiPoint to ensure a valid convex hull
    # print(" type of filtered points is : ", type(filtered_points))
    polygon = MultiPoint(filtered_points).convex_hull

    # Apply buffer(0) to fix potential topology issues
    return polygon.buffer(0)


def calculate_polygon_iou_matrix(predictions, detections):
    """
    Calculate the IoU matrix between two sets of polygons.
    
    Args:
        predictions (list of Polygon): List of prediction polygons.
        detections (list of Polygon): List of detection polygons.
        
    Returns:
        np.ndarray: IoU matrix of shape (len(predictions), len(detections)).
    """
    # Initialize an empty matrix to store IoUs
    iou_matrix = np.zeros((len(predictions), len(detections)))

    # Calculate IoU for each pair of prediction and detection
    for i, pred in enumerate(predictions):
        for j, det in enumerate(detections):
            intersection_area = pred.intersection(det).area
            union_area = pred.union(det).area
            iou_matrix[i, j] = intersection_area / union_area if union_area > 0 else 0.0

    return iou_matrix


def get_keypoints(atlbrs, btlbrs, image):
    
    keypoints = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if keypoints.size == 0:
        # return [], [], [], [], [], []
        return [], []
        # return  Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])

    prediction_patches = extract_patches(image, atlbrs)
    # print("here")
    detection_patches = extract_patches(image, btlbrs)
    # print(" prediction patches length is : ", len(prediction_patches))
    # print(" detection patches length is : ", len(detection_patches))

    new_bounding_boxes = []
    pred_boxes_left = []
    pred_boxes_right = []
    pred_boxes_bottom = []
    pred_polygon_list = []
    count = 0
    for pred in prediction_patches:
        count +=1
        if pred.shape[0] == 0 or pred.shape[1] ==0:
            # continue
            # pred_boxes_left.append([0, 0, 0, 0])
            # pred_boxes_right.append([0, 0, 0, 0])
            # pred_boxes_bottom.append([0, 0, 0, 0])
            pred_polygon_list.append(Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))
            continue
        # print(" pred shape is :", pred.shap
        pred_keypoint = model_keypoint(pred, verbose = False)[0]
        # print("\n\npred keypoint is :", pred_keypoint)
        bounding_boxes = []
        # pred_keypoint.show()
        if len(pred_keypoint) > 0:
            pred_keypoint = pred_keypoint.keypoints.xy.cpu()
            # print("predicted keypoint is ", pred_keypoint)
            prediction_keypoints= get_most_descriptive_keypoints(pred_keypoint)
            pred_polygon = create_polygon_from_keypoints(prediction_keypoints)
            # print("most descriptive keypoint is :", prediction_keypoints)
            bounding_boxes = get_bboxes_from_joints(prediction_keypoints)
            # pred_boxes_left.append(bounding_boxes[0])
            # pred_boxes_right.append(bounding_boxes[1])
            # pred_boxes_bottom.append(bounding_boxes[1])
            pred_polygon_list.append(pred_polygon)
           
        # print('bounding boxes are : ', bounding_boxes)
        else:
            # print(" im coming here")
            # break
            # pred_boxes_left.append([0, 0, 0, 0])
            # pred_boxes_right.append([0, 0, 0, 0])
            # pred_boxes_bottom.append([0, 0, 0, 0])
            pred_polygon_list.append(Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))
    # print(" count is : ", count)

    # print('pred keypoint list :', pred_keypoint_list)
    # det_keypoint_list = []
    det_boxes_left = []
    det_boxes_right = []
    det_boxes_bottom = []
    det_polygon_list = []
    for det in detection_patches:
        if det.shape[0] == 0 or det.shape[1] == 0:
            # det_boxes_left.append([0, 0, 0, 0])
            # det_boxes_right.append([0, 0, 0, 0])
            # det_boxes_bottom.append([0, 0, 0, 0])
            det_polygon_list.append(Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))
            continue
        
        det_keypoint = model_keypoint(det)[0]
        if len(det_keypoint) >0:
            det_keypoint = det_keypoint.keypoints.xy.cpu()
            # print("det_keypoint  length is ", len(pred_keypoint))

            detection_keypoints= get_most_descriptive_keypoints(det_keypoint)
            # bounding_boxes = get_bboxes_from_joints(detection_keypoints)
            det_polygon = create_polygon_from_keypoints(detection_keypoints)
            # print(" det polygon is " , det_polygon)
            # det_boxes_left.append(bounding_boxes[0])
            # det_boxes_right.append(bounding_boxes[1])
            # det_boxes_bottom.append(bounding_boxes[1])
            det_polygon_list.append(det_polygon)
        else:
            # print(" else here")

            # det_boxes_left.append([0, 0, 0, 0])
            # det_boxes_right.append([0, 0, 0, 0])
            # det_boxes_bottom.append([0, 0, 0, 0])
            det_polygon_list.append( Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))
            # break

    # pred_keypoint_list = np.array(pred_keypoint_list)
    # det_keypoint_list = np.array(det_keypoint_list)

    # print("length of pred_keypoint list is :", len(pred_keypoint_list))

    # print("length of det_keypoint list is :", len(det_keypoint_list))


    # print("pred keypoints list length is :", len(pred_keypoint_list))
    del prediction_patches, detection_patches
    # return pred_boxes_left, pred_boxes_right, pred_boxes_bottom ,det_boxes_left, det_boxes_right, det_boxes_bottom
    return pred_polygon_list, det_polygon_list
    
def iou_distance(atracks, btracks, img, association = None):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    # print(" a tracks are : \n", atracks)
    # print(" btracks are : \n", btracks)

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
        
    else:
        # print(" btracks are : ", btracks)
        # atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        atlbrs = [track.tlbr for track in atracks]
        # print("past predictions are : ", [track.prediction for track in atracks])
        # print("past detections are : ", [track.tracklet for track in btracks])
        # print(" a track top left bottom right is :", atlbrs)
        # print(" b track top left bottom right :", btlbrs)
        # print("atlbr length is :", len(atlbrs))
        # print("btlbr length is :", len(btlbrs))
         # print("atlbr is " , atlbrs), 
        # pred_polygons, det_polygons  = get_keypoints(atlbrs, btlbrs, img)
         
        # pred_boxes_left, pred_boxes_right, pred_boxes_bottom, det_boxes_left, det_boxes_right, det_boxes_bottom = get_keypoints(atlbrs, btlbrs, img)
        # left_ious = ious(pred_boxes_left, det_boxes_left)
        # right_ious = ious(pred_boxes_right, det_boxes_right)
        # bottom_ious = ious(pred_boxes_bottom, det_boxes_bottom)
        # if len(pred_polygons) > 0 and len(det_polygons) > 0:
        #     polygon_ious = calculate_polygon_iou_matrix(pred_polygons, det_polygons)
        # else:
        #     polygon_ious  = []
    # print("atlbr is : ", atlbrs)
    # print("btlbr is : ", btlbrs)
    _ious = ious(atlbrs, btlbrs)
    # cost_matrix = 1 - _ious
    # return cost_matrix
    h_ious = height_iou_np(atlbrs, btlbrs)
   

    # if (len(pred_boxes_left) ==0 or len(det_boxes_right) == 0):
    #     cost_matrix = 1 - _ious
    #     return cost_matrix
    # elif (len(pred_boxes_left) == len(atlbrs)) and (len(pred_boxes_right) == len(btlbrs)):
    #     cost_matrix =  1 - np.multiply(_ious, left_ious, right_ious)
    #     return cost_matrix
    # if len(left_ious) !=0 and len(right_ious) !=0 and len(bottom_ious) and association == "first_association":
    #     cost_matrix = 1 - (0.7 * _ious + 0.1 *left_ious + 0.1 * right_ious + 0.1 * bottom_ious)
    # cost_matrix =  1 - (0.8 *_ious + 0.2 * h_ious)
    cost_matrix = 1 - _ious
    # return cost_matrix, []
    # if len(polygon_ious) > 0:
    #     cost_matrix = 1 - (0.8 * _ious + 0.2 * polygon_ious)
    #     return cost_matrix, None
    return cost_matrix,  []
     # Store columns (detections) that have multiple matching predictions.
    columns_with_multiple_matches = []
    # counts = []
    if association == "first_association":
        multiple_matched_detections = []
        for col in range(cost_matrix.shape[1]):
            # similar_bbboxes = cost_matrix[:, col] <0.2
            # count = sum(1 for x in cost_matrix[:, col] if x < 0.2)
            # [counts.append(index) for index, x in enumerate(cost_matrix[:, col]) if x < 0.2]  
            match_indices = [row for row, value in enumerate(cost_matrix[:, col]) if value < 0.15]
            if len(match_indices) > 1:
                print(" match indices are : ", match_indices)
                print("cost matrix before : \n", cost_matrix)
                multiple_matched_detections.append(btracks[col])
            #     columns_with_multiple_matches.append((col, match_indices))

            #     predictions = [atlbrs[idx] for idx in match_indices]
            #     detections = btlbrs[col]
            #     print("predictions length : ", predictions)
            #     print("detections are : ", detections)
            #     # exit(0)
            #     pred_boxes_left, pred_boxes_right, pred_boxes_bottom, det_boxes_left, det_boxes_right, det_boxes_bottom = get_keypoints(predictions, [detections], img)
            #     print(" pred boxes left aee : ", pred_boxes_left)
            #     print("pred boxes right are : ", pred_boxes_right)
            #     print("detect boxes left are :", det_boxes_left)
            #     print("detect_lboxes right are : ", det_boxes_right)
            #     left_ious = ious(pred_boxes_left, det_boxes_left)
            #     right_ious = ious(pred_boxes_right, det_boxes_right)
            #     bottom_ious = ious(pred_boxes_bottom, det_boxes_bottom)
                
            #     print("\nleft ious are : \n",  left_ious)
            #     print("\nright ious are : \n", right_ious)
            #     print("\nmain iou is : \n", _ious)
            #     new_left_cost = 1- left_ious
            #     new_right_cost = 1 - right_ious
            #     new_bottom_cost = 1 - bottom_ious
                for index, row in enumerate(match_indices):
                    print("overlapping tracklet include : \n", atlbrs[row])

                    # cost_matrix[row, col] = 0.7 * cost_matrix[row, col]  + 0.1 * new_left_cost[index] + 0.1 * new_right_cost[index] + 0.1 * new_bottom_cost[index]
                    cost_matrix[row, :] = 1.0
                print("cost matrix after : \n", cost_matrix)
            

        return cost_matrix, multiple_matched_detections
    return cost_matrix, []
    # # Process the columns with multiple matches.
    # # print('columns with multiple matches : ', columns_with_multiple_matches)
    # if len(columns_with_multiple_matches) > 0:
    #     a_prediction = [track.tracklet_predictions for track in atracks]
    #     print("past predictions are : ", len(a_prediction))
    #     print([track.tracklet_predictions for track in atracks])
    #     print("\n\n\n\n")
    #     b_detection = [track.tracklet for track in atracks]
    #     print("past detections are : \n", b_detection)
    #     print([track.tracklet for track in btracks])
    #     # print(" length of b_detection : ", len(b_detection))
    #     print("cost matrix right now is :\n", cost_matrix)
    # # exit(0)

    # if len(columns_with_multiple_matches) > 0:
    #     for col, match_indices in columns_with_multiple_matches:
            # atracks = atlbrs


    
    #     for col, match_indices in columns_with_multiple_matches:
    #         print(f"Multiple matches for detection {col} with predictions {match_indices}")
    #         past_atracks = [track.tracklet_predictions for track in atracks]
    #         past_btracks = [track.tracklet for track in atracks]
    #         print("current prediction is : ", atlbrs)
    #         print("current detection is :", btlbrs)
    #         # Calculate average IoU using past tracklets.
    #         # print(" past detectionc are : ", past_btracks)
    #         avg_iou = calculate_avg_iou(past_atracks, past_btracks, match_indices, col)
    #         # print(" average iou is : ", avg_iou)
    #         # Update the cost matrix using the recalculated average IoU.
    #         new_cost = 1 - avg_iou
    #         print("new cost is : ", new_cost)
    #         for index, row in enumerate(match_indices):
    #             cost_matrix[row, col] = new_cost[index]
    #             # print(f"Updated cost for prediction {row} and detection {col}: {new_cost}")

    #     print("update cost matrix is : \n", cost_matrix)
    #     exit(0)




    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    # print(" shape of iou sim")
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost