import torch

def convert_yolo_to_iou_format(boxes):
    boxes[:, 0] = boxes[:, 0] - boxes[:,2]/2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3]/2
    boxes[:, 2] = boxes[:, 0]  + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    
    return torch.Tensor(boxes)


def CIOU_Loss_GPT(pred_boxes, target_boxes):
    
    pred_boxes_iou = pred_boxes.clone()  # Avoid in-place modifications that may lead to errors

    pred_boxes_iou[:, 0] = pred_boxes_iou[:, 0] - pred_boxes_iou[:,2]/2
    pred_boxes_iou[:, 1] = pred_boxes_iou[:, 1] - pred_boxes_iou[:, 3]/2
    pred_boxes_iou[:, 2] = pred_boxes_iou[:, 0]  + pred_boxes_iou[:, 2]
    pred_boxes_iou[:, 3] = pred_boxes_iou[:, 1] + pred_boxes_iou[:, 3]
    
    target_boxes_iou = target_boxes.clone()
    target_boxes_iou[:, 0] = target_boxes_iou[:, 0] - target_boxes_iou[:,2]/2
    target_boxes_iou[:, 1] = target_boxes_iou[:, 1] - target_boxes_iou[:, 3]/2
    target_boxes_iou[:, 2] = target_boxes_iou[:, 0]  + target_boxes_iou[:, 2]
    target_boxes_iou[:, 3] = target_boxes_iou[:, 1] + target_boxes_iou[:, 3]
    

    # Extract the coordinates of the predicted and target boxes
    x1_p, y1_p, x2_p, y2_p = pred_boxes_iou[:, 0], pred_boxes_iou[:, 1], pred_boxes_iou[:, 2], pred_boxes_iou[:, 3]
    x1_t, y1_t, x2_t, y2_t = target_boxes_iou[:, 0], target_boxes_iou[:, 1], target_boxes_iou[:, 2], target_boxes_iou[:, 3]
    
    # Compute the area of the predicted and target boxes
    area_p = (x2_p - x1_p) * (y2_p - y1_p)
    area_t = (x2_t - x1_t) * (y2_t - y1_t)
    
    # Compute the intersection coordinates
    inter_x1 = torch.max(x1_p, x1_t)
    inter_y1 = torch.max(y1_p, y1_t)
    inter_x2 = torch.min(x2_p, x2_t)
    inter_y2 = torch.min(y2_p, y2_t)
    
    # Compute the intersection area
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    
    # Compute the union area
    union_area = area_p + area_t - inter_area
    
    # Compute the IoU
    iou = inter_area / union_area
    
    # Compute the center coordinates of the boxes
    center_pred_x = (x1_p + x2_p) / 2
    center_pred_y = (y1_p + y2_p) / 2
    center_target_x = (x1_t + x2_t) / 2
    center_target_y = (y1_t + y2_t) / 2
    
    # Compute the distance between the centers of the predicted and target boxes
    center_distance = (center_pred_x - center_target_x).pow(2) + (center_pred_y - center_target_y).pow(2)
    
    # Compute the diagonal distance of the smallest enclosing box
    enclose_x1 = torch.min(x1_p, x1_t)
    enclose_y1 = torch.min(y1_p, y1_t)
    enclose_x2 = torch.max(x2_p, x2_t)
    enclose_y2 = torch.max(y2_p, y2_t)
    diagonal_distance = (enclose_x2 - enclose_x1).pow(2) + (enclose_y2 - enclose_y1).pow(2)
    
    # Compute the aspect ratio term
    pred_w = x2_p - x1_p
    pred_h = y2_p - y1_p
    target_w = x2_t - x1_t
    target_h = y2_t - y1_t
    aspect_ratio = (4 / (torch.pi ** 2)) * ((torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2)
    
    # Compute the weight term for aspect ratio
    alpha = aspect_ratio / (1 - iou + aspect_ratio + 1e-7)
    
    # Compute the final CIoU loss
    ciou_loss = 1 - iou + (center_distance / diagonal_distance) + alpha * aspect_ratio
    
    return ciou_loss.mean()


def CIOU_Loss_Perplexity(pred_boxes, target_boxes, eps=1e-7):
    # Extract coordinates
    
    
    pred_boxes_iou = pred_boxes.clone()
    pred_boxes_iou[:, 0] = pred_boxes_iou[:, 0] - pred_boxes_iou[:,2]/2
    pred_boxes_iou[:, 1] = pred_boxes_iou[:, 1] - pred_boxes_iou[:, 3]/2
    pred_boxes_iou[:, 2] = pred_boxes_iou[:, 0]  + pred_boxes_iou[:, 2]
    pred_boxes_iou[:, 3] = pred_boxes_iou[:, 1] + pred_boxes_iou[:, 3]
    
    target_boxes_iou = target_boxes.clone()
    target_boxes_iou[:, 0] = target_boxes_iou[:, 0] - target_boxes_iou[:,2]/2
    target_boxes_iou[:, 1] = target_boxes_iou[:, 1] - target_boxes_iou[:, 3]/2
    target_boxes_iou[:, 2] = target_boxes_iou[:, 0]  + target_boxes_iou[:, 2]
    target_boxes_iou[:, 3] = target_boxes_iou[:, 1] + target_boxes_iou[:, 3]
    

    
    
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes_iou.unbind(-1)
    target_x1, target_y1, target_x2, target_y2 = target_boxes_iou.unbind(-1)

    # Calculate area of predicted and target boxes
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    # print("predited area :", pred_area)
    # print("target area is : ", target_area)



    # Intersection area
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union area
    union_area = pred_area + target_area - inter_area + eps


    # print("intersection area : ", inter_area)
    # print("union area : ", union_area)
    # IoU
    iou = inter_area / union_area
    # print(" iou is :", iou)

    # Diagonal length of the smallest enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    enclose_diagonal = torch.sqrt(
        (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    )

    # print(" enclosed diagnal is : ", enclose_diagonal)

    # Center distance
    center_x1 = (pred_x1 + pred_x2) / 2
    center_y1 = (pred_y1 + pred_y2) / 2
    center_x2 = (target_x1 + target_x2) / 2
    center_y2 = (target_y1 + target_y2) / 2
    center_distance = torch.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)

    # print(" center distance is : ", center_distance)


    # Calculate v and alpha
    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan((pred_x2 - (pred_x1- eps)) / (pred_y2 - (pred_y1 - eps))) -
        torch.atan((target_x2 - target_x1) / (target_y2 - target_y1)), 2
    )

    # print(" pred x2 - pred x1 : ", pred_x2 - pred_x1)
    # print("pred_y2 - pred y1", pred_y2 - pred_y1)
    temp_1 = torch.atan((pred_x2 - pred_x1) / (pred_y2 - pred_y1))
    temp_2 = torch.atan((target_x2 - target_x1) / (target_y2 - target_y1))



    # print("temp1 is : ", temp_1)
    # print("temp2 is : ", temp_2)


    alpha = v / (1 - iou + v + eps)

    # print(" V is : ", v)
    # print("alpha is : ", alpha)

    # CIoU loss
    ciou = iou - (center_distance ** 2) / (enclose_diagonal ** 2 + eps) - alpha * v

    # print("CIOU is : ", ciou)
    return 1 - ciou.mean()





# Define GIoU loss function
def GIOU_Loss(pred_boxes, target_boxes):
    """
    Calculate the Generalized Intersection over Union (GIoU) loss between predicted and target bounding boxes.

    Args:
    - pred_boxes (Tensor): Predicted bounding boxes of shape (N, 4) where N is the batch size.
                           Each box is defined as [x1, y1, x2, y2].
    - target_boxes (Tensor): Target bounding boxes of shape (N, 4).

    Returns:
    - loss (Tensor): Scalar GIoU loss.
    """
    ## Convert YOLO Format to MOT Format for IOU Calculation
    
    # pred_boxes = convert_yolo_to_iou_format(pred_boxes)
    # target_boxes = convert_yolo_to_iou_format(target_boxes)
    pred_boxes_iou = pred_boxes.clone()  # Avoid in-place modifications that may lead to errors

    pred_boxes_iou[:, 0] = pred_boxes_iou[:, 0] - pred_boxes_iou[:,2]/2
    pred_boxes_iou[:, 1] = pred_boxes_iou[:, 1] - pred_boxes_iou[:, 3]/2
    pred_boxes_iou[:, 2] = pred_boxes_iou[:, 0]  + pred_boxes_iou[:, 2]
    pred_boxes_iou[:, 3] = pred_boxes_iou[:, 1] + pred_boxes_iou[:, 3]
    
    target_boxes_iou = target_boxes.clone()
    target_boxes_iou[:, 0] = target_boxes_iou[:, 0] - target_boxes_iou[:,2]/2
    target_boxes_iou[:, 1] = target_boxes_iou[:, 1] - target_boxes_iou[:, 3]/2
    target_boxes_iou[:, 2] = target_boxes_iou[:, 0]  + target_boxes_iou[:, 2]
    target_boxes_iou[:, 3] = target_boxes_iou[:, 1] + target_boxes_iou[:, 3]
    
    
    
    # Compute the area of the predicted and target boxes
    pred_area = (pred_boxes_iou[:, 2] - pred_boxes_iou[:, 0]) * (pred_boxes_iou[:, 3] - pred_boxes_iou[:, 1])
    target_area = (target_boxes_iou[:, 2] - target_boxes_iou[:, 0]) * (target_boxes_iou[:, 3] - target_boxes_iou[:, 1])

    # Intersection
    inter_x1 = torch.max(pred_boxes_iou[:, 0], target_boxes_iou[:, 0])
    inter_y1 = torch.max(pred_boxes_iou[:, 1], target_boxes_iou[:, 1])
    inter_x2 = torch.min(pred_boxes_iou[:, 2], target_boxes_iou[:, 2])
    inter_y2 = torch.min(pred_boxes_iou[:, 3], target_boxes_iou[:, 3])
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / union_area

    # Enclosing box
    enc_x1 = torch.min(pred_boxes_iou[:, 0], target_boxes_iou[:, 0])
    enc_y1 = torch.min(pred_boxes_iou[:, 1], target_boxes_iou[:, 1])
    enc_x2 = torch.max(pred_boxes_iou[:, 2], target_boxes_iou[:, 2])
    enc_y2 = torch.max(pred_boxes_iou[:, 3], target_boxes_iou[:, 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    # GIoU
    giou = iou - (enc_area - union_area) / (enc_area + 1e-7)

    # Loss is 1 - GIoU (since we want to minimize it)
    loss = 1 - giou.mean()
    return loss