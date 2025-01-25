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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pickle

class DiffusionEmbeddingComputer:
    def __init__(self, args,   test_dataset, grid_off, max_batch=4):
        self.model = None
        self.dataset = args.dataset
        self.test_dataset = test_dataset
        self.crop_size = (128, 384)
        self.up_ft_index = args.up_ft_index
        self.t = args.t
        dir_name_embeddings = "./cache/diffusion_embeddings_t_{}_up_ft_{}".format(self.t, self.up_ft_index)

        # os.makedirs("cache/embeddings_diffusion_t_21/", exist_ok=True)
        os.makedirs(dir_name_embeddings, exist_ok=True)
        self.cache_path = dir_name_embeddings+ "/{}_embeddings.pkl"
        # self.cache_path = os.path.join(config.reid_dir, "{}_embedding.pkl")
        self.cache = {}
        self.cache_name = ""
        self.grid_off = grid_off
        self.max_batch = max_batch
        # print(" dataset in embedded computer is : ",dataset)
        # print("Test dataset flag : ", test_dataset)
        # Only used for the general ReID model (not FastReID)
        self.normalize = False


    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)


    def restrict_neighborhood(self, args, h, w):
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


    def extract_bboxes_features(self, features, img, feature_dims, detections):
        image_h, image_w, _ = img.shape
        feat_h, feat_w = feature_dims
        # print(" features shape is : ", features.shape)
        # print( " image_h. image_w : {}, {}".format(image_h, image_w))
        # print( " feat_h. feat_w : {}, {}".format(feat_h, feat_w))
        ##@ Extract features per bounding box
        features_per_box = []

        scale_h = feat_h/image_h
        scale_w =feat_w/image_w
        scale_h = feat_h / image_h
        scale_w = feat_w / image_w
        scale = min(scale_h, scale_w)
        # print(" detections are : ", detections)
        for box in detections:
            
            assert not torch.isnan(features).any(), "NaN found in features"
            
            x_min, y_min, x_max, y_max = box
            # print(" x min , y_min, x_max, y_max are : {}, {}, {}, {}".format(x_min, y_min, x_max, y_max))
            # Scale bounding box coordinates
            x_min_fm = int(x_min * scale)
            y_min_fm = int(y_min * scale)
            x_max_fm = int(x_max * scale)
            y_max_fm = int(y_max * scale)
            # print(" reshaped boundin boxes are : {},  {}, {}, {}".format(x_min_fm, y_min_fm, x_max_fm, y_max_fm))

            # Ensure coordinates are within bounds
            x_min_fm = max(0, min(x_min_fm, feat_w - 1))
            y_min_fm = max(0, min(y_min_fm, feat_h - 1))
            x_max_fm = max(0, min(x_max_fm, feat_w - 1))
            y_max_fm = max(0, min(y_max_fm, feat_h - 1))
            # print(" reshaped boundin boxes are : {},  {}, {}, {}".format(x_min_fm, y_min_fm, x_max_fm, y_max_fm))
            # Extract the feature map slice for the bounding box
            # print("features shape is : ", features)
            # print(" x min is : {}, x max is : {}".format(x_min_fm, x_max_fm))
            
            width = x_max_fm - x_min_fm + 1
            height = y_max_fm - y_min_fm + 1
            # print(f"Computed width: {width}, height: {height}")
        
            feature_slice = features[y_min_fm:y_min_fm + height, x_min_fm:x_min_fm+width, :]
            assert not torch.isnan(feature_slice).any(), "NaN found in feature_slice"
            # processed_feature_maps = []
            # mean_feature_map = feature_slice.mean(dim = 2)
            # processed_feature_maps.append(mean_feature_map.data.cpu().numpy())

            # for i in range(len(processed_feature_maps)):
            #     print("coming here?")
            #     # ax = fig.add_subplot(5, 4, i + 1)
            #     # ax.imshow(processed_feature_maps[i])
            #     # plt.show()
            #     plt.imshow(processed_feature_maps[i])
            #     plt.show()
        
            
            # Pooling (average pooling as an example)
            # print( " feature slice shape is : ", feature_slice.shape)
            pooled_feature = feature_slice.mean(axis=(0, 1))  # Average pooling over height and width
            features_per_box.append(pooled_feature)
            assert not torch.isnan(pooled_feature).any(), "NaN found in features_per_box"

        # features_per_box = np.array(features_per_box)  # Shape: [num_boxes, 640]
        features_per_box = torch.stack(features_per_box)


        return features_per_box

    def label_propagation(self, args, model, frame_tar, frame_source , detections, img, track_embeds, tag, mask_neighborhood=None):
        """
        propagate segs of frames in list_frames to frame_tar
        """
        if self.cache_name != tag.split(":")[0]:
            self.load_cache(tag.split(":")[0])

        if tag in self.cache:
            embs = self.cache[tag]
            if embs.shape[0] != detections.shape[0]:
                raise RuntimeError(
                    "ERROR: The number of cached embeddings don't match the "
                    "number of detections.\nWas the detector model changed? Delete cache if so."
                )
            return embs

        ## we only need to extract feature of the target frame
        start_time = time.time()
        feat_tar, h, w = self.extract_feature(args, model, frame_tar, return_h_w=True)
        # print(" feat_tar shape is : ", feat_tar.shape)
        # print(" time taken for the extraction step is :", time.time() - start_time)
        # print(" heigtht and width after extraction step is : {},  {}".format(h, w))
        # gc.collect()
        # torch.cuda.empty_cache()
        processed_feature_maps = []  # List to store processed feature maps

        # for feature_map in feat_tar[:, :, ]:
            # feature_map = feat_tar.squeeze(0)
            # mean_feature_map = torch.sum(feature_map, 2) / feature_map.shape[2]
        mean_feature_map = feat_tar.mean(dim = 2)
        processed_feature_maps.append(mean_feature_map.data.cpu().numpy())

    #     print(" shape of processed feature map is : ", len(processed_feature_maps))
    #    # Plot the feature maps
    #     # fig = plt.figure(figsize=(20, 3))
        # for i in range(len(processed_feature_maps)):
        #     print("coming here?")
            # ax = fig.add_subplot(5, 4, i + 1)
            # ax.imshow(processed_feature_maps[i])
            # plt.show()
            # plt.imshow(processed_feature_maps[i])
            # plt.show()
    #         # ax.axis(&quot;off&quot;)
            # ax.set_title(layer_names[i].split('(')[0], fontsize=30)

        # ncontext = len(frame_source)
        # print(" track embeddin g is : ", frame_source)
        # print(" ncontext shape is :", ncontext)
        # feat_sources = torch.stack(frame_source)     # nmb_context x dim x h*w
        
        feat_tar = F.normalize(feat_tar, dim=2)
        # print(" features shape here in the beginning is : ", feat_tar.shape)
        assert not torch.isnan(feat_tar).any(), "NaN found in features target"

        # return_feat_tar = feat_tar.T # dim x h*w
        # feat_sources = F.normalize(feat_sources, dim=1, p=2)
        # print(" target features shape is : ", return_feat_tar.shape)
        # print(" source features shape is : ", feat_sources.shape)
        # feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
        
        feature_dims = (h,w)
        
        curr_bboxes_features = self.extract_bboxes_features(feat_tar, img, feature_dims, detections)
       
        
        # print("current bboxes_features : ", curr_bboxes_features)
        #### This measures the affinities(similarities) between the target frame and the source frames.
        # torch.bmm performs match matrix multiplication for efficient similarity computation
        # aff = torch.exp(torch.bmm(feat_tar, feat_sources) / args.temperature) # nmb_context x h*w (tar: query) x h*w (source: keys)
        # print(" shape of affinitity is : ", aff.shape)
        # if args.size_mask_neighborhood > 0:
        #     if mask_neighborhood is None:
        #         mask_neighborhood = self.restrict_neighborhood(h, w)
        #         mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        #     aff *= mask_neighborhood  ## if the mask neighborhood is defined, restrict the affinity computation in that neighbourhood


        # aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)



        # tk_val, _ = torch.topk(aff, dim=0, k=args.topk) ### Identifying the top "K" Affinities [ Default K = 5]
        # tk_val_min, _ = torch.min(tk_val, dim=0) ### Determining the lowest of the Top K Affinities.
        # aff[aff < tk_val_min] = 0                ### Defining affinity values lower than top_k_val_min to 0 ( Filter out the lower K affinity values)

        # aff = aff / torch.sum(aff, keepdim=True, axis=0)
        # # print("Shape of affinity after processing ; ", aff.shape)
        # # gc.collect()
        # torch.cuda.empty_cache()

        # list_segs = [s.cuda() for s in list_segs]
        
        # # print(" list segsis : ", len(list_segs))
        # segs = torch.cat(list_segs)
        # nmb_context, C, h, w = segs.shape
        # print(" segmentation shape is : ", segs.shape)
        # segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
        
        # # unique_values = torch.unique(segs)
        # unique_values, indices, counts = torch.unique(segs, return_inverse=True, return_counts=True)
        # # print(" unique values in segmentation mask is : ", unique_values[:150])
        # # print("counts of unique values : ", counts  )
        # seg_tar = torch.mm(segs, aff)

        # seg_tar = seg_tar.reshape(1, C, h, w)
        # return None, None, None
        curr_bboxes_features = curr_bboxes_features.detach().cpu().numpy()
        self.cache[tag] = curr_bboxes_features
        self.dump_cache(tag, tag.split(":")[0])

        return curr_bboxes_features
        # return numpy_array = tensor.detach().cpu().numpy()
        # return seg_tar, return_feat_tar, mask_neighborhood


    def extract_feature(self, args, model, frame, return_h_w=False):
    
        
       with torch.no_grad():
        unet_ft = model.forward(frame,
                                t=args.t,
                                up_ft_index=args.up_ft_index,
                                ensemble_size=args.ensemble_size).squeeze() # c, h, w
        dim, h, w = unet_ft.shape
        # print("dimension,  height and width of the UNET feature set is : {}, {} ,{}".format(dim, h, w) )
        unet_ft = torch.permute(unet_ft, (1, 2, 0)) # h,w,c
        # unet_ft = unet_ft.view(h * w, dim) # hw,c
        if return_h_w:
            return unet_ft, h, w
        return unet_ft


    def to_one_hot(self, y_tensor, n_dims=None):
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



    def read_frame(self, img, scale_size=[480]):
        """

        read a single frame & preprocess
        """
        # img = cv2.imread(img_batch)
        ori_h, ori_w, _ = img.shape
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
        img = self.color_normalize(img)
        # print(" image shape in read frame is : ", img.shape)
        return img, ori_h, ori_w



    def dump_cache(self, tag, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        
        # if self.cache_name:
            # with open(self.cache_path.format(self.cache_name), "wb") as fp:
            #     pickle.dump(self.cache, fp)
        existing_cache = {}
    
        # Load existing cache if it exists
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                existing_cache = pickle.load(fp)
        
        # Update existing cache with new data
        existing_cache[tag] = self.cache[tag]
        
        # Write updated cache back to file
        with open(cache_path, "wb") as fp:
            pickle.dump(existing_cache, fp)

        self.cache.clear()
        existing_cache.clear()

    def color_normalize(self, x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
        return x 

