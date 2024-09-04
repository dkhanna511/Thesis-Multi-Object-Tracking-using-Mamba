import torch

def convert_yolo_to_iou_format(boxes):
    boxes[:, 0] = boxes[:, 0] - boxes[:,2]/2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3]/2
    boxes[:, 2] = boxes[:, 0]  + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    
    return torch.Tensor(boxes)

# Define GIoU loss function
def giou_loss(pred_boxes, target_boxes):
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