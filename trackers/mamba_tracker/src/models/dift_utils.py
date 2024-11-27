import numpy as np
import torch
import gc
from PIL import Image
import cv2
import argparse
from pathlib import Path
import os
import torchvision
from torch.nn import functional as F
import time

def restrict_neighborhood(args, h, w):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * args.size_mask_neighborhood + 1):
                for q in range(2 * args.size_mask_neighborhood + 1):
                    if i - args.size_mask_neighborhood + p < 0 or i - args.size_mask_neighborhood + p >= h:
                        continue
                    if j - args.size_mask_neighborhood + q < 0 or j - args.size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - args.size_mask_neighborhood + p, j - args.size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)


def label_propagation(args, model, frame_tar, list_frame_feats, list_segs, mask_neighborhood=None):
    """
    propagate segs of frames in list_frames to frame_tar
    """
    gc.collect()
    torch.cuda.empty_cache()

    ## we only need to extract feature of the target frame
    feat_tar, h, w = extract_feature(args, model, frame_tar, return_h_w=True)

    gc.collect()
    torch.cuda.empty_cache()

    return_feat_tar = feat_tar.T # dim x h*w

    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / args.temperature) # nmb_context x h*w (tar: query) x h*w (source: keys)

    if args.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(args, h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=args.topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    gc.collect()
    torch.cuda.empty_cache()

    # list_segs = [s.cuda() for s in list_segs]
    # segs = torch.cat(list_segs)
    # nmb_context, C, h, w = segs.shape
    # segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    # seg_tar = torch.mm(segs, aff)
    # seg_tar = seg_tar.reshape(1, C, h, w)
    return None, return_feat_tar, mask_neighborhood
    # return seg_tar, return_feat_tar, mask_neighborhood


def get_horizontal_split_patches(self, image, bbox, tag, idx, viz=False):
    crop_size = (128, 384)    ### HARD Coded, Taking this from the FastReID Model arguments

    
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        h, w = image.shape[2:]

    bbox = np.array(bbox)
    bbox = bbox.astype(np.int_)
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
        # Faulty Patch Correction
        bbox[0] = np.clip(bbox[0], 0, None)
        bbox[1] = np.clip(bbox[1], 0, None)
        bbox[2] = np.clip(bbox[2], 0, image.shape[1])
        bbox[3] = np.clip(bbox[3], 0, image.shape[0])

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    ### TODO - Write a generalized split logic
    split_boxes = [
        [x1, y1, x1 + w, y1 + h / 3],
        [x1, y1 + h / 3, x1 + w, y1 + (2 / 3) * h],
        [x1, y1 + (2 / 3) * h, x1 + w, y1 + h],
    ]

    split_boxes = np.array(split_boxes, dtype="int")
    patches = []
    # breakpoint()
    for ix, patch_coords in enumerate(split_boxes):
        if isinstance(image, np.ndarray):
            im1 = image[patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2], :]

            if viz:  ## TODO - change it from torch tensor to numpy array
                dirs = "./viz/{}/{}".format(tag.split(":")[0], tag.split(":")[1])
                Path(dirs).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    os.path.join(dirs, "{}_{}.png".format(idx, ix)),
                    im1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255,
                )
            patch = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            patch = cv2.resize(patch, crop_size, interpolation=cv2.INTER_LINEAR)
            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            patch = patch.unsqueeze(0)
            # print("test ", patch.shape)
            patches.append(patch)
        else:
            im1 = image[:, :, patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2]]
            patch = torchvision.transforms.functional.resize(im1, (256, 128))
            patches.append(patch)

    patches = torch.cat(patches, dim=0)

    # print("Patches shape ", patches.shape)
    # patches = np.array(patches)
    # print("ALL SPLIT PATCHES SHAPE - ", patches.shape)

    return patches




def extract_feature(args, model, frame, bbox, img, return_h_w=False):
    crop_size = (128, 384)    ### HARD Coded, Taking this from the FastReID Model arguments
    normalize = False         ### HARD Coded, Taking this from the FastReID Model arguments
    grid_off = True             ### HARD Coded, Taking this from the FastReID Model arguments
    if grid_off:
        # Basic embeddings
        h, w = img.shape[:2]
        results = np.round(bbox).astype(np.int32)
        results[:, 0] = results[:, 0].clip(0, w)
        results[:, 1] = results[:, 1].clip(0, h)
        results[:, 2] = results[:, 2].clip(0, w)
        results[:, 3] = results[:, 3].clip(0, h)

        crops = []
        for p in results:
            crop = img[p[1] : p[3], p[0] : p[2]]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            if normalize:
                crop /= 255
                crop -= np.array((0.485, 0.456, 0.406))
                crop /= np.array((0.229, 0.224, 0.225))
            crop = torch.as_tensor(crop.transpose(2, 0, 1))
            # crop = torch.tensor(crop.transpose(2, 0, 1)) 

            crop = crop.unsqueeze(0)
            crops.append(crop)
    else:
        # Grid patch embeddings
        for idx, box in enumerate(bbox):
            crop = get_horizontal_split_patches(img, box, idx)
            crops.append(crop)
    crops = torch.cat(crops, dim=0)
    # print(" crops shape is : ", crops.shape)
    max_batch = 1024
    embs = []
    start_time_feature = time.time()
    for idx in range(0, len(crops)):
        # batch_crops = crops[idx: idx+max_batch]
        batch_crops = crops[idx]
        # print(" batch crops shape is : ", batch_crops.shape)
        # batch_crops = batch_crops.cuda()
        frame1, ori_h, ori_w = read_frame(batch_crops)
        frame1 = frame1.cuda()
        with torch.no_grad():
            unet_ft = model.forward(frame1,
                                    t=args.t,
                                    up_ft_index=args.up_ft_index,
                                    ensemble_size=args.ensemble_size).squeeze() # c, h, w
            dim, h, w = unet_ft.shape
            unet_ft = torch.permute(unet_ft, (1, 2, 0)) # h,w,c
            unet_ft = unet_ft.view(h * w, dim) # hw,c
        
        # embs.extend(unet_ft)
        embs.append(unet_ft)
    # print(" time taken for JUST the model.forward part : {}".format(time.time() - start_time_feature))
    embs = torch.stack(embs)
    # embs = torch.cat(embs, dim=0)
    embs = embs.reshape(len(embs), embs.shape[1] * embs.shape[2])
    
    embs = F.normalize(embs, dim=-1).cpu().numpy()
    # print(" embeddings (diffusion) shape in the middle? : ", embs.shape)
    # embs = torch.nn.functional.normalize(embs, embs.shape[-1]).cpu().numpy()
    
    # embs = torch.nn.functional.normalize(embs, embs.shape[-1]).cpu().numpy()
    
    if return_h_w:
        return embs, h, w
    return embs


def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)



def read_frame(img, scale_size=[480]):
    """
    read a single frame & preprocess
    """
    # img = cv2.imread(fra)
    _, ori_h, ori_w = img.shape

    img = np.transpose(img, (1, 2, 0))
    img = img.numpy()
    # print(" type  of image is : ", type(img))
    # print(" img shape is : ", img.shape)
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 32) * 32)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 32) * 32)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w





def color_normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

