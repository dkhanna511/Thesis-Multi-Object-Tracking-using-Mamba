def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
        box1 (array-like): [x_min, y_min, x_max, y_max] format
        box2 (array-like): [x_min, y_min, x_max, y_max] format
    
    Returns:
        float: IoU value
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute the area of intersection rectangle
    inter_area = max(0, inter_x_max - inter_x_min + 1) * max(0, inter_y_max - inter_y_min + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    box2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of the prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def calculate_predicted_bboxes(previous_bboxes, offset_deltas):
    """
    Calculate the predicted bounding boxes using the offsets.
    
    Parameters:
        previous_bboxes (np.array): Numpy array of shape (num_objects, 4) containing the bounding boxes from t-th frame.
        offset_deltas (np.array): Numpy array of shape (num_objects, 4) containing the offset differences.
        
    Returns:
        np.array: Numpy array of shape (num_objects, 4) containing the predicted bounding boxes for the (t+1)-th frame.
    """
    # Calculate predicted bounding boxes by adding offsets to previous bounding boxes
    predicted_bboxes = previous_bboxes + offset_deltas
    return predicted_bboxes

def iou_matching(predicted_bboxes, actual_bboxes):
    """
    Match predicted bounding boxes with actual bounding boxes based on IoU.
    
    Parameters:
        predicted_bboxes (np.array): Numpy array of shape (num_predictions, 4)
        actual_bboxes (np.array): Numpy array of shape (num_actuals, 4)
    
    Returns:
        list of tuples: Each tuple contains (index of predicted bbox, index of actual bbox, IoU score)
    """
    num_predictions = predicted_bboxes.shape[0]
    num_actuals = actual_bboxes.shape[0]

    iou_matrix = np.zeros((num_predictions, num_actuals))

    # Calculate IoU for every pair of predicted and actual bounding boxes
    for i in range(num_predictions):
        for j in range(num_actuals):
            iou_matrix[i, j] = calculate_iou(predicted_bboxes[i], actual_bboxes[j])

    # Find the best matches using the Hungarian algorithm or any preferred matching method
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Negative because the algorithm minimizes cost

    # Create list of matched indices with IoU scores
    matches = [(i, j, iou_matrix[i, j]) for i, j in zip(row_ind, col_ind)]

    return matches