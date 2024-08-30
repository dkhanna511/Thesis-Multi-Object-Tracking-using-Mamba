import torch


def giou_loss(pred_boxes, target_boxes):
    """
    Compute the Generalized Intersection over Union (GIoU) loss between predicted and target bounding boxes.
    
    Parameters:
    - pred_boxes: Tensor of shape (N, 4) representing the predicted bounding boxes [x1, y1, x2, y2].
    - target_boxes: Tensor of shape (N, 4) representing the ground truth bounding boxes [x1, y1, x2, y2].
    
    Returns:
    - loss: Scalar GIoU loss.
    """

    # Compute intersection
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Compute union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / union_area.clamp(min=1e-6)

    # Compute the smallest enclosing box
    enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1).clamp(min=1e-6)

    # GIoU
    giou = iou - (enc_area - union_area) / enc_area.clamp(min=1e-6)

    # GIoU loss
    loss = 1 - giou

    return loss.mean()