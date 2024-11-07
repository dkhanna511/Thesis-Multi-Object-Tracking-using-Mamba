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
from sklearn.metrics.pairwise import cosine_similarity

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


def to_tlbr(bboxes):
    bboxes_list = []
    
    for box in bboxes:
        new_box = box.copy()  # Copy each individual box
        new_box[2:] = [new_box[0] + new_box[2], new_box[1] + new_box[3]]
        bboxes_list.append(new_box)
    
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
        return keypoints
        # return  Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])

    prediction_patches = extract_patches(image, atlbrs)
    # print("here")
    detection_patches = extract_patches(image, btlbrs)
    # print(" prediction patches length is : ", len(prediction_patches))
    # print(" detection patches length is : ", len(detection_patches))

    
    pred_polygon_list = []
    pred_keypoint_list = []
    count = 0
    for pred in prediction_patches:
        count +=1
        if pred.shape[0] == 0 or pred.shape[1] ==0:
            # continue
            # pred_boxes_left.append([0, 0, 0, 0])
            # pred_boxes_right.append([0, 0, 0, 0])
            # pred_boxes_bottom.append([0, 0, 0, 0])
            prediction_keypoints = np.zeros((17, 2))
            pred_keypoint_list.append(prediction_keypoints)
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
            pred_keypoint_list.append(prediction_keypoints)
            pred_polygon = create_polygon_from_keypoints(prediction_keypoints)
            # print("most descriptive keypoint is :", prediction_keypoints)
            bounding_boxes = get_bboxes_from_joints(prediction_keypoints)
            # pred_boxes_left.append(bounding_boxes[0])
            # pred_boxes_right.append(bounding_boxes[1])
            # pred_boxes_bottom.append(bounding_boxes[1])
            # pred_polygon_list.append(pred_polygon)
           
        # print('bounding boxes are : ', bounding_boxes)
        else:
            # print(" im coming here")
            # break
            prediction_keypoints = np.zeros((17, 2))
            pred_keypoint_list.append(prediction_keypoints)
            # pred_boxes_left.append([0, 0, 0, 0])
            # pred_boxes_right.append([0, 0, 0, 0])
            # pred_boxes_bottom.append([0, 0, 0, 0])
            pred_polygon_list.append(Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))
    # print(" count is : ", count)

    # print('pred keypoint list :', pred_keypoint_list)
    # det_keypoint_list = []

    det_keypoint_list = []
    det_polygon_list = []
    for det in detection_patches:
        if det.shape[0] == 0 or det.shape[1] == 0:
            # det_boxes_left.append([0, 0, 0, 0])
            # det_boxes_right.append([0, 0, 0, 0])
            # det_boxes_bottom.append([0, 0, 0, 0])
            detection_keypoints = np.zeros((17, 2))
            det_keypoint_list.append(detection_keypoints)
            det_polygon_list.append(Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))
            continue
        
        det_keypoint = model_keypoint(det)[0]
        if len(det_keypoint) >0:
            det_keypoint = det_keypoint.keypoints.xy.cpu()
            # print("det_keypoint  length is ", len(pred_keypoint))

            detection_keypoints= get_most_descriptive_keypoints(det_keypoint)
            det_keypoint_list.append(detection_keypoints)
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
            
            detection_keypoints = np.zeros((17, 2))
            det_keypoint_list.append(detection_keypoints)
            det_polygon_list.append( Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))
            # break

    # pred_keypoint_list = np.array(pred_keypoint_list)
    # det_keypoint_list = np.array(det_keypoint_list)

    # print("length of pred_keypoint list is :", len(pred_keypoint_list))

    # print("length of det_keypoint list is :", len(det_keypoint_list))


    # print("pred keypoints list length is :", len(pred_keypoint_list))
    del prediction_patches, detection_patches
    # return pred_boxes_left, pred_boxes_right, pred_boxes_bottom ,det_boxes_left, det_boxes_right, det_boxes_bottom
    # return pred_polygon_list, det_polygon_list
    return np.array(pred_keypoint_list), np.array(det_keypoint_list)
    


def get_keypoint_cosines(atlbrs, btlbrs, img):

     
    sim_matrix = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if sim_matrix.size == 0:
        # return [], [], [], [], [], []
        return sim_matrix
    
    pred_keypoints, det_keypoints = get_keypoints(atlbrs, btlbrs, img)
    
    pred_vectors = np.array([pred.flatten() for pred in pred_keypoints])
    det_vectors = np.array([det.flatten() for det in det_keypoints])

    similarity_matrix = cosine_similarity(pred_vectors, det_vectors)

    # print("similarity matrix shape is :", similarity_matrix.shape)
    return similarity_matrix

def expand_boxes(boxes, buffer_size):
    modified_boxes = []

    for det in boxes:
        x_top, y_top, x_bottom, y_bottom = det
        
        # Calculate width and height
        width = x_bottom - x_top
        height = y_bottom - y_top
        # print("width is : " , width)
        # print("height is :", height)
        # Calculate buffer adjustments
        x_top_new = x_top - (buffer_size/2) * width
        y_top_new = y_top - (buffer_size)/2 * height
        x_bottom_new = x_bottom + (buffer_size)/2 * width
        y_bottom_new = y_bottom + (buffer_size)/2 * height
        
        
        # new_width = x_bottom_new - x_top_new
        # new_height = y_bottom_new - y_top_new
        # buffer_scale = (new_width - width) / width
        # print("buffer scale is : ", buffer_scale)
        
        # Append the modified detection
        modified_boxes.append([x_top_new, y_top_new, x_bottom_new, y_bottom_new])

    return np.array(modified_boxes)


def iou_distance(atracks, btracks, img, association = None, buffer_size = 0.0):
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
        a_tracklet = [track.tracklet for track in atracks]
        b_score = [track.score for track in btracks]
        # a_tracklet_new = a_tracklet.copy()
        # if association == "half_first_association":
            # print(" atlbrs are : ", atlbrs)
            # print("btlbrs  : ", btlbrs)
        # print(" buffer size is : ", buffer_size)
        atlbrs_buffered = expand_boxes(atlbrs, buffer_size)
        btlbrs_buffered = expand_boxes(btlbrs, buffer_size)
        
        # atlbrs = buffer_size * np.array(atlbrs)
        # btlbrs = buffer_size * np.array(btlbrs)
    
    
        # print("\n\na tracklet inside IOU Cost matching is :\n\n", a_tracklet)
        # confidence_cost_matrix = calculate_confidence_cost_matrix(a_tracklet, b_score)
        # print(" \ncost matrix confidence is : \n", confidence_cost_matrix)

        # color = (255, 0, 0)
        # if img is not None:
        #     for box in atlbrs:
        #         cv2.rectangle(img, (int(box[0]), int(box[1])),  (int(box[2]), int(box[3])), color, 2)
        #     color = (0, 0, 0)
        #     for box in btlbrs:
        #         cv2.rectangle(img, (int(box[0]), int(box[1])),  (int(box[2]), int(box[3])), color, 2)
        
        #     cv2.imshow('frame', img )
        #     cv2.waitKey(2)
           
            # cv2.destroyAllWindows()
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
    # pred_keypoints, det_keypoints = get_keypoints(atlbrs, btlbrs, img)
    
    _ious =  ious(atlbrs_buffered, btlbrs_buffered)
    
    h_ious = height_iou_np(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    # print(" \n\niou cost matrix is : \n", cost_matrix)
    # return cost_matrix, [], []

    # if association == "first_association":
    #     sim_matrix = get_keypoint_cosines(atlbrs, btlbrs, img)
    #     keypoint_cost_matrix = 1 -  sim_matrix

    #     return cost_matrix,  keypoint_cost_matrix,  []

    # else:
    #     return cost_matrix, [], [] 
    # print(" similarity matrix is : \n", sim_matrix.shape)
        
    

    # if (len(pred_boxes_left) ==0 or len(det_boxes_right) == 0):
    #     cost_matrix = 1 - _ious
    #     return cost_matrix
    # elif (len(pred_boxes_left) == len(atlbrs)) and (len(pred_boxes_right) == len(btlbrs)):
    #     cost_matrix =  1 - np.multiply(_ious, left_ious, right_ious)
    #     return cost_matrix
    # if len(left_ious) !=0 and len(right_ious) !=0 and len(bottom_ious) and association == "first_association":
    #     cost_matrix = 1 - (0.7 * _ious + 0.1 *left_ious + 0.1 * right_ious + 0.1 * bottom_ious)
    # cost_matrix =  1 - (0.8 *_ious + 0.2 * h_ious)
    # return cost_matrix, []
    # if len(polygon_ious) > 0:
    #     cost_matrix = 1 - (0.8 * _ious + 0.2 * polygon_ious)
    #     return cost_matrix, None
     # Store columns (detections) that have multiple matching predictions.
     
    if association == "first_association":
        multiple_matched_predictions = []
        multiple_matched_predictions_tlbrs = []
        for col in range(cost_matrix.shape[1]):
            # similar_bbboxes = cost_matrix[:, col] <0.2
            # count = sum(1 for x in cost_matrix[:, col] if x < 0.2)
            # [counts.append(index) for index, x in enumerate(cost_matrix[:, col]) if x < 0.2]  
            match_indices = [row for row, value in enumerate(cost_matrix[:, col]) if value < 0.15]
            if len(match_indices) > 1:
                # print(" match indices are : ", match_indices)
                # print("cost matrix before : \n", cost_matrix)
                # multiple_matched_predictions.append([match_indices, btlbrs[col], b_score[col]])
                multiple_matched_predictions.append([[atracks[idx] for idx in match_indices], btlbrs[col], b_score[col]])
                # multiple_matched_predictions_tlbrs.append(btlbrs[col])
            

        return cost_matrix, [],  multiple_matched_predictions
    return cost_matrix, [], []
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


def compute_aw_new_metric(emb_cost, w_association_emb, max_diff=0.5):
    w_emb = np.full_like(emb_cost, w_association_emb)
    w_emb_bonus = np.full_like(emb_cost, 0)

    # Needs two columns at least to make sense to boost
    if emb_cost.shape[1] >= 2:
        # Across all rows
        for idx in range(emb_cost.shape[0]):
            inds = np.argsort(-emb_cost[idx])
            # Row weight is difference between top / second top
            row_weight = min(emb_cost[idx, inds[0]] - emb_cost[idx, inds[1]], max_diff)
            # Add to row
            w_emb_bonus[idx] += row_weight / 2

    if emb_cost.shape[0] >= 2:
        for idj in range(emb_cost.shape[1]):
            inds = np.argsort(-emb_cost[:, idj])
            col_weight = min(emb_cost[inds[0], idj] - emb_cost[inds[1], idj], max_diff)
            w_emb_bonus[:, idj] += col_weight / 2

    return w_emb + w_emb_bonus



def bbox_center(bbox):
    """Compute the center of a bounding box."""
    x1, y1, x2, y2 = bbox  # (x_min, y_min, x_max, y_max)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return np.array([cx, cy])

def bbox_size(bbox):
    """Compute the width and height of a bounding box."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return np.array([width, height])

def calculate_displacement(bbox1, bbox2):
    
    """Compute the Euclidean distance between the centers of two bounding boxes."""
    # if type == "history":
    bbox1 = np.asarray(bbox1).copy()
    bbox1[2:] += bbox1[:2]
        
    bbox2 = np.asarray(bbox2).copy()
    bbox2[2:] += bbox2[:2]
    # elif type == "current":
    #     bbox1 = np.asarray(bbox1).copy()
    #     bbox1[2:] += bbox1[:2]
        
    center1 = bbox_center(bbox1)
    center2 = bbox_center(bbox2)
    return np.linalg.norm(center1 - center2)

def calculate_size_change(bbox1, bbox2):
    """Compute the size change between two bounding boxes."""
    
    # if type == "history":
    #     # print(" bounding box before was : ", bbox1)
    bbox1 = np.asarray(bbox1).copy()
    bbox1[2:] += bbox1[:2]
    #     # print("bounding box after is : ", bbox1)
        
    bbox2 = np.asarray(bbox2).copy()
    bbox2[2:] += bbox2[:2]
    
    # elif type == "current":
    #     bbox1 = np.asarray(bbox1).copy()
    #     bbox1[2:] += bbox1[:2]
    
    
    size1 = bbox_size(bbox1)
    size2 = bbox_size(bbox2)
    return np.linalg.norm(size1 - size2)

def estimate_confidence(atracks, alpha=0.4, beta=0.6):
    """
    Estimate the confidence of the predicted bounding box.
    
    Args:
        bboxes (list of tuples): List of previous bounding boxes (x1, y1, x2, y2).
        predicted_bbox (tuple): The predicted bounding box.
        alpha (float): Weight for displacement-based confidence.
        beta (float): Weight for size consistency-based confidence.
    
    Returns:
        float: Estimated confidence score between 0 and 1.
    """
    
    
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)):
        atlbrs = atracks
        
        
    else:
        # print(" btracks are : ", btracks)
        # atlbrs = [track.tlbr for track in atracks]
        atlbrs = [track.tlwh for track in atracks]
        a_previous = [track.tracklet for track in atracks]
    
    if len(atlbrs) == 0:
        return []
    
    print(" atlbrs is : \n", atlbrs)
    
    print(" previous tracklet info is : ", a_previous)
    # bboxes_list = [tracklet[:-1] for tracklet in atlbrs]
    
    # bboxes_list = []
    # bboxes_list.append([[tracklet[:-1]] for tracklet in atlbrs]) 
    # bboxes_list = [np.array(tracklet[:-1]) for tracklet in atlbrs]

    # print(" bboxes list is :\n", bboxes_list)
    # predicted_bbox_list = [tracklet[-1] for tracklet in atlbrs]

    # predicted_bbox_list = []
    # predicted_bbox_list.append([[tracklet[-1]] for tracklet in atlbrs])
    
    # print(" predted bbox is : \n", predicted_bbox_list)
    confidence_list = []
    for bboxes, predicted_bbox in zip(a_previous, atlbrs):
        # Calculate average displacement and size change over the tracklet
        # bbox1 = bboxes[0][i+1]
        displacements = [calculate_displacement(bboxes[i][0], bboxes[i + 1][0]) 
                        for i in range(len(bboxes) - 1)]
        avg_displacement = np.mean(displacements)

        size_changes = [calculate_size_change(bboxes[i][0], bboxes[i + 1][0]) 
                        for i in range(len(bboxes) - 1)]
        avg_size_change = np.mean(size_changes)

        print(" average size change is :", avg_size_change)
        print("average displacement : ", avg_displacement)
        # Calculate displacement and size  change for the predicted bbox
        last_bbox = bboxes[-1][0]
        print("\n\nlast bounding box was : ", last_bbox)
        print("predictes box is : ", predicted_bbox)
        displacement = calculate_displacement(last_bbox, predicted_bbox)
        size_change = calculate_size_change(last_bbox, predicted_bbox)

        # Normalize displacement and size change (higher values reduce confidence)
        displacement_conf = np.exp(-displacement / (avg_displacement + 1e-5))
        size_change_conf = np.exp(-size_change / (avg_size_change + 1e-5))
        print(" size_change_conf is :", size_change_conf)
        print("displacement_conf : ", displacement_conf)

        # Combine the two confidence scores
        confidence = alpha * displacement_conf + beta * size_change_conf
        # return np.clip(confidence, 0.0, 1.0)
        print(" confidence is : ", confidence)
        confidence_list.append(np.clip(confidence, 0.0, 1.0))
        
    return np.array(confidence_list)






def linear_assignment2(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


# Updated confidence function
def calculate_prediction_confidence(tracklet):
    
    tracklets_tlwh = [bbox[0] for bbox in tracklet]

    tracklets_tlbr = to_tlbr(tracklets_tlwh)
    

    initial_area = (tracklets_tlbr[0][2] - tracklets_tlbr[0][0]) * (tracklets_tlbr[0][3] - tracklets_tlbr[0][1])
    areas = [(box[2] - box[0]) * (box[3] - box[1]) / initial_area for box in tracklets_tlbr]
    aspect_ratios = [(box[2] - box[0]) / (box[3] - box[1]) for box in tracklets_tlbr]
    # centers = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in tracklets_tlbr]

   # Normalize the aspect ratios
    aspect_ratios = np.clip(aspect_ratios, 0.1, None)  # Avoid division by zero in future calculations
    aspect_ratios = (aspect_ratios - np.min(aspect_ratios)) / (np.max(aspect_ratios) - np.min(aspect_ratios))

    # Calculate center movements
    centers = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in tracklets_tlbr]
    center_movements = [np.linalg.norm(np.array(centers[i]) - np.array(centers[i - 1])) for i in range(1, len(centers))]
    
    # Calculate deviations
    area_deviation = np.std(areas)
    aspect_ratio_deviation = np.std(aspect_ratios)
    center_movement_deviation = np.std(center_movements)

    # Weighting factors can be adjusted here
    confidence = 1 - (0.2 * area_deviation + 0.3 * aspect_ratio_deviation + 0.5 * center_movement_deviation)
    return max(0, confidence)

def calculate_tracklet_confidence(tracklet):
    # Extract box bbo
    tracklets_tlwh = [bbox[0] for bbox in tracklet]
    # print(" tracklet tlwh is : ", tracklets_tlwh)
    # print(" tracklet tlwh  is : ", tracklets_tlwh)
    # return []
    tracklets_tlbr = to_tlbr(tracklets_tlwh)
    # print(" tracklet tlbr  is : ", tracklets_tlbr)
    # return []
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in tracklets_tlbr]  # Box areas
    # print(" areas are : ", areas)
    aspect_ratios = [(box[2] - box[0]) / (box[3] - box[1]) for box in tracklets_tlbr]  # Aspect ratios
    # print(" aspect ratio are : ", aspect_ratios)
    
    centers = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in tracklets_tlbr]  # Box centers
    
    # Calculate variances
    area_variance = np.var(areas)
    # print(" area variance is :", area_variance)
    aspect_ratio_variance = np.var(aspect_ratios)
    center_movements = [np.linalg.norm(np.array(centers[i]) - np.array(centers[i-1])) for i in range(1, len(centers))]
    center_movement_variance = np.var(center_movements)

    # Define weights for each metric
    area_weight, aspect_ratio_weight, movement_weight = 0.3, 0.3, 0.4
    
    # Compute normalized confidence score
    confidence = 1 - (area_weight * area_variance + aspect_ratio_weight * aspect_ratio_variance + movement_weight * center_movement_variance)
    return max(0, confidence)  # Ensure confidence is at least 0



def calculate_confidence_cost_matrix(a_tracklets, b_score):
    
    a_tracklets_new = a_tracklets.copy()
    # return []
    # print("tracklets inside dconfidence thingy is : ", a_tracklets)
    # print(" detection score  is : ", b_score)
    # Step 1: Calculate confidence for each prediction tracklet
    a_score = [calculate_prediction_confidence(yoyo) for yoyo in a_tracklets_new]
    # print(" a score is : ", a_score)
    # Step 2: Create an empty cost matrix with dimensions len(predictions) x len(detections)
    # return []
    cost_matrix_confidence = np.zeros((len(a_score), len(b_score)))
    
    # Step 2: Calculate cost for each prediction-detection pair
    for i, pred_conf in enumerate(a_score):
        for j, det_conf in enumerate(b_score):
            # Define cost as a function of both prediction and detection confidence
            # combined_confidence = (pred_conf + det_conf) / 2
            # print(" detection confidence is : ", det_conf)
            combined_confidence = abs(pred_conf - det_conf)
            cost_matrix_confidence[i, j] =  combined_confidence  # Lower cost for higher confidence

    # Step 4: (Optional) Normalize cost matrix if needed
    cost_matrix_confidence = np.clip(cost_matrix_confidence, 0, 1)
    # print(" cost matrix confidence is : ", cost_matrix_confidence)
    return cost_matrix_confidence