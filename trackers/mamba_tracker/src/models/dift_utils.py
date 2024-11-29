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


import pickle

class DiffusionEmbeddingComputer:
    def __init__(self, dataset, test_dataset, grid_off, max_batch=4):
        self.model = None
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.crop_size = (128, 384)
        os.makedirs("cache/embeddings_diffusion/", exist_ok=True)
        self.cache_path = "./cache/embeddings_diffusion/{}_embeddings.pkl"
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


    def extract_feature(self, args, model, frame, bbox, img, tag,  return_h_w=False):
    
        if self.cache_name != tag.split(":")[0]:
            self.load_cache(tag.split(":")[0])

        if tag in self.cache:
            embs = self.cache[tag]
            if embs.shape[0] != bbox.shape[0]:
                raise RuntimeError(
                    "ERROR: The number of cached embeddings don't match the "
                    "number of detections.\nWas the detector model changed? Delete cache if so."
                )
            return embs
         
        
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
                crop = self.get_horizontal_split_patches(img, box, idx)
                crops.append(crop)
        crops = torch.cat(crops, dim=0)
        # print(" crops shape is : ", crops.shape)
        max_batch = 2
        embs = []
        
        start_time_feature = time.time()
        for idx in range(0, len(crops), max_batch):
            batch_crops = crops[idx: idx+max_batch]
            # print(" batch crops length is : ", len(batch_crops))    

            # batch_crops = crops[idx]
            # print(" batch crops shape is : ", batch_crops.shape)
            # batch_crops = batch_crops.cuda()
            frame1, (orig_sizes) = self.read_frame(batch_crops)
            # print(" frame 1 shape is : ", frame1.shape)
            frame1 = frame1.cuda()
            with torch.no_grad():
                unet_ft = model.forward(frame1,
                                        t=args.t,
                                        up_ft_index=args.up_ft_index,
                                        ensemble_size=args.ensemble_size).squeeze() # c, h, w
                # print(" shape of unet is : ", unet_ft.shape)
                # unet_ft.s
                try:
                    batches, dim, h, w = unet_ft.shape
                except:
                    batches = 1
                    dim, h, w = unet_ft.shape
                    unet_ft = unet_ft.unsqueeze(0)
                # print(" unet shape outsid is : ", unet_ft.shape)
                # batches = 1
                # unet_ft = torch.permute(unet_ft, (1, 2, 0)) # h,w,c
                unet_ft = torch.permute(unet_ft, (0, 2, 3, 1))
                # unet_ft = unet_ft.view(h * w, dim) # hw,c
                unet_ft = unet_ft.view(batches, -1, dim)
                # print(" shape of unet is : ", unet_ft.shape)

                # gc.collect()
                # torch.cuda.memory_summary(device=None, abbreviated=False)
                torch.cuda.empty_cache()
            
            for batch_idx in range(batches):
                # top_k_embs, _ = torch.topk(unet_ft[batch_idx], k = 48, dim = -1)
                unet_ft_reshaped = unet_ft[batch_idx].view(-1, dim).cpu().numpy()
                
                n_components = 256  # Choose the number of components to keep
                pca = PCA(n_components=n_components)
                unet_ft_pca = pca.fit_transform(unet_ft_reshaped)

                # Reshape back to original batch shape
                unet_ft_pca = unet_ft_pca.reshape(-1, n_components)
                embs.append(torch.from_numpy(unet_ft_pca))
            del frame1, unet_ft, batch_crops
            # embs.extend(unet_ft)
            # embs.append(unet_ft)
        # print(" time taken for JUST the model.forward part : {}".format(time.time() - start_time_feature))
        embs = torch.stack(embs)
        # embs = torch.cat(embs, dim=0)
        embs = embs.reshape(len(embs), embs.shape[1] * embs.shape[2])
        print(" embedding shape is : ", embs.shape)
        embs = F.normalize(embs, dim=-1).cpu().numpy()
        # print(" embeddings (diffusion) shape in the middle? : ", embs.shape)
        # embs = torch.nn.functional.normalize(embs, embs.shape[-1]).cpu().numpy()
        self.cache[tag] = embs
        self.dump_cache(tag, tag.split(":")[0])
        


        # embs = torch.nn.functional.normalize(embs, embs.shape[-1]).cpu().numpy()
        # gc.collect()

        if return_h_w:
            return embs, h, w
        return embs


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



    def read_frame(self, img_batch, scale_size=[480]):
        """
        read a single frame & preprocess
        """
        # img = cv2.imread(fra)
        # _, ori_h, ori_w = img.shape
        B, C, ori_h, ori_w = img_batch.shape
        ori_sizes = []
        processed_imgs = []
        for i in range(B):
            img = img_batch[i]  # [C, H, W]
            ori_sizes.append((ori_h, ori_w))  # Save original dimensions
            
            # Convert from [C, H, W] to [H, W, C] and numpy format
            img = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            
            # Determine new height and width based on scale_size
            if len(scale_size) == 1:
                if ori_h > ori_w:
                    tw = scale_size[0]
                    th = (tw * ori_h) / ori_w
                    th = int((th // 32) * 32)
                else:
                    th = scale_size[0]
                    tw = (th * ori_w) / ori_h
                    tw = int((tw // 32) * 32)
            else:
                th, tw = scale_size
            
            # Resize the image
            img = cv2.resize(img, (tw, th))
            
            # Normalize and preprocess
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
            img = img[:, :, ::-1]  # Convert RGB to BGR
            img = np.transpose(img.copy(), (2, 0, 1))  # Back to [C, H, W]
            img = torch.from_numpy(img).float()  # Convert to torch tensor
            img = self.color_normalize(img)  # Apply normalization
            
            processed_imgs.append(img)
        # Stack processed images into a batch
        preprocessed_batch = torch.stack(processed_imgs)  # [B, C, th, tw]
        # print(" preprocessed batch is : ", preprocessed_batch.shape)
        return preprocessed_batch, ori_sizes



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

