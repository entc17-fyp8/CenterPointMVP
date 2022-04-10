import imp
from turtle import forward
from nusc_image_projection import read_file, to_batch_tensor, to_tensor, projectionV2, reverse_view_points, get_obj
from tqdm import tqdm 
import argparse 
import numpy as np 
import torch 
import cv2 
import os 

import cv2

from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
import time

from visualize import visualize_points

CUDA_DEVICE = 'cuda:0'

class MVP:
    def __init__(self, 
        args,
        config_file='c2_config/nuImages_CenterNet2_DLA_640_8x.yaml', 
        opts_list=['MODEL.WEIGHTS', '/workspace/Checkpoints/centernet2/centernet2_checkpoint.pth']
    
    ):
        self.H=900
        self.W=1600
        
        self.centernet2_predictor = self.__init_detector(args, config_file, opts_list)
        
    def __init_detector(self,args, config_file, opts_list):
        from CenterNet2.train_net import setup2
        from detectron2.engine import DefaultPredictor

        cfg = setup2(args, config_file, opts_list )
        predictor = DefaultPredictor(cfg)
        return predictor 
    
    
    def preprocess_image(self, original_image):
        '''
        Inputs:
            original_image : a cv2 image object (in BGR format)
        
        Returns:
            inputs_dict: {"image": image, "height": height, "width": width}
        '''
        # original_image = cv2.imread(path)
        with torch.no_grad():    
            # whether the model expects BGR inputs or RGB
            if self.centernet2_predictor.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
                
            # Augmentations # TODO - How to do this at test time??
            image = self.centernet2_predictor.aug.get_transform(original_image).apply_image(original_image)
            
            # To Tensor
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            # Get image height & width
            height, width = original_image.shape[:2]
            inputs_dict = {"image": image, "height": height, "width": width}
            
            return inputs_dict  # [infos, 6x{"image": image, "height": height, "width": width} ]
    
    def do_inference(self,
                lidar_points,
                image_data,
                all_cams_from_lidar_tms,
                all_cams_intrinsic
                ):
        '''
        Args:
            lidar_points            : # (N, 5)
            image_data              :  [ {"image": image, "height": height, "width": width} ]
            all_cams_from_lidar_tms :  transforms to all cameras from lidar  [ <np.float32: 4, 4> ]
            all_cams_intrinsic      :  camera intrinsics of all cameras      [ <float> [3, 3] ]
            
        Returns:
            points : 
        '''    
        # lidar_points : # (N, 5) : [x,y,z,0,0]
        lidar_points = lidar_points.reshape([-1, 5])[:, :4] # (N, 5) => (N, 4)   # Here 4 is 'num_features'  
        # lidar_points : # (N, 4) : [x,y,z,0]

        
        virtual_points, real_points, real_points_indices = self.__generate_virtual_points(
            lidar_points,
            all_cams_from_lidar_tms, 
            all_cams_intrinsic, 
            
            image_data  # [6x{"image": image, "height": height, "width": width}]            
            )

        # 'virtual_points':  #shape:(_, 14) - [x,y,z,             time? , one_hot_labels_for_10_classes]
        # 'real_points':     #shape:(_, 15) - [x,y,z,reflectance, time? , one_hot_labels_for_10_classes]
        
        # visualize_points(lidar_points)
        # visualize_points(real_points)
        # visualize_points(virtual_points)

        point_features = self.__concat_point_features(
            original_points=lidar_points,
            real_points= real_points,
            virtual_points=virtual_points
        )

        torch.cuda.empty_cache() # if you get OOM error 

        return point_features  # (_, 15)  [x,y,z, time? , one_hot_labels_for_10_classes_or_all_1s, type_encoding]
        
            
    def __generate_virtual_points(self,
            lidar_points,
            all_cams_from_lidar, # transforms from all cameras to lidar
            all_cams_intrinsic, # camera intrinsics of all cameras
            
            image_data,  # [6x{"image": image, "height": height, "width": width}]
            
            num_camera=6
        ):
        '''
        Generate Virtual Points
        
        Returns:
            all_virtual_points          :  virtual_points           #(_, 14)
            all_real_points             :  foreground_real_points   #(_, 15)
            foreground_real_point_mask_ :  foreground_indices
        '''
        with torch.no_grad():    
            ####
            # Get Masks and Labels on Images
            ####

            # One host labels for 10 output classes of CenterNet2 trained on NuImages
            # classes : car, truck, trailer, bus, construction_vehicle, bicycle, motorcycle, pedestrian, traffic_cone, barrier
            one_hot_labels = [] 
            for i in range(10):
                one_hot_label = torch.zeros(10, device=CUDA_DEVICE, dtype=torch.float32)
                one_hot_label[i] = 1
                one_hot_labels.append(one_hot_label)
            one_hot_labels = torch.stack(one_hot_labels, dim=0) 

            masks = [] 
            labels = [] 
            camera_ids = torch.arange(6, dtype=torch.float32, device=CUDA_DEVICE).reshape(6, 1, 1)

            # Run CenterNet2 on images
            result = self.centernet2_predictor.model(image_data)

            # For each camera image
            for camera_id in range(num_camera):
                # Run postprocessing on output of CenterNet2
                pred_label, score, pred_mask = self.__postprocess(result[camera_id])
                camera_id = torch.tensor(camera_id, dtype=torch.float32, device=CUDA_DEVICE).reshape(1,1).repeat(pred_mask.shape[0], 1)
                pred_mask = torch.cat([pred_mask, camera_id], dim=1)
                transformed_labels = one_hot_labels.gather(0, pred_label.reshape(-1, 1).repeat(1, 10))
                transformed_labels = torch.cat([transformed_labels, score.unsqueeze(-1)], dim=1)

                masks.append(pred_mask)
                labels.append(transformed_labels)
            
            masks = torch.cat(masks, dim=0)
            labels = torch.cat(labels, dim=0)
            
            ###
            # Project Lidar Points into 2D
            ###
            # Get (camera_x, camera_y, depth) for all 3D points
            projected_points_in_2d = projectionV2(to_tensor(lidar_points), to_batch_tensor(all_cams_from_lidar), to_batch_tensor(all_cams_intrinsic)) ## (6,N,4) 
            camera_ids = torch.arange(6, dtype=torch.float32, device=CUDA_DEVICE).reshape(6, 1, 1).repeat(1, projected_points_in_2d.shape[1], 1)      ## (6,N,1) 
            projected_points_in_2d = torch.cat([projected_points_in_2d, camera_ids], dim=-1)        # (6,N,5) 

            # Match Lidar Point Clouds with masks and get the virtual points in 3D
            if len(masks) == 0:
                res = None
            else:
                res  = self.__add_virtual_mask(
                        masks, labels, projected_points_in_2d, 
                        to_tensor(lidar_points), 
                        intrinsics=to_batch_tensor(all_cams_intrinsic), 
                        transforms=to_batch_tensor(all_cams_from_lidar) 
                    )
            
            # Return the virtual points and real points
            if res is not None:
                virtual_points, foreground_real_points, foreground_indices = res 
                return virtual_points.detach().cpu().numpy(), foreground_real_points.detach().cpu().numpy(), foreground_indices.detach().cpu().numpy()
                # return virtual_points.cpu().numpy(), foreground_real_points.cpu().numpy(), foreground_indices.cpu().numpy()
            else:
                virtual_points = np.zeros([0, 14])
                foreground_real_points = np.zeros([0, 15])
                foreground_indices = np.zeros(0)
                return virtual_points, foreground_real_points, foreground_indices

        
    
    def __postprocess(self, res):
        result = res['instances']
        labels = result.pred_classes
        scores = result.scores 
        masks = result.pred_masks.reshape(scores.shape[0], 1600*900) 
        boxes = result.pred_boxes.tensor

        # remove empty mask and their scores / labels 
        empty_mask = masks.sum(dim=1) == 0

        labels = labels[~empty_mask]
        scores = scores[~empty_mask]
        masks = masks[~empty_mask]
        boxes = boxes[~empty_mask]
        masks = masks.reshape(-1, 900, 1600).permute(0, 2, 1).reshape(-1, 1600*900)
        return labels, scores, masks


    def __is_within_mask(self,points_xyc, masks):
        '''
        Check whether LiDAR points projected into 2D lie within any of the given masks
        '''
        seg_mask = masks[:, :-1].reshape(-1, self.W, self.H)
        camera_id = masks[:, -1]
        points_xyc = points_xyc.long()
        valid = seg_mask[:, points_xyc[:, 0], points_xyc[:, 1]] * (camera_id[:, None] == points_xyc[:, -1][None])
        return valid.transpose(1, 0) 

    def __add_virtual_mask(self,masks, labels, points_2d, raw_points, num_virtual=50, dist_thresh=3000, num_camera=6, intrinsics=None, transforms=None):
        '''

        Args:
            masks           :   
            labels          :   
            points          :   # (6,N,5) - [[ [x, y, z, valid_indicator, camera id ] ]]
            raw_points      :   
            num_virtual     :   
            dist_thresh     :   
            num_camera      :   
            intrinsics      :   
            transforms      :   
        
        Returns:
            all_virtual_points          :  virtual_points           #shape:(_, 14)
            all_real_points             :  foreground_real_points   #shape:(_, 15)
            foreground_real_point_mask_ :  foreground_indices

        
        '''
        with torch.no_grad():    
            # (6,N,5) => (6*N, 5) => (6*N, 3)
            points_xyc = points_2d.reshape(-1, 5)[:, [0, 1, 4]] # [ [x, y, camera id ] ]

            valid = self.__is_within_mask(points_xyc, masks)
            valid = valid * points_2d.reshape(-1, 5)[:, 3:4]

            # remove camera id from masks 
            camera_ids = masks[:, -1]
            masks = masks[:, :-1]

            box_to_label_mapping = torch.argmax(valid.float(), dim=1).reshape(-1, 1).repeat(1, 11)
            point_labels = labels.gather(0, box_to_label_mapping)
            point_labels *= (valid.sum(dim=1, keepdim=True) > 0 )  

            foreground_real_point_mask = (valid.sum(dim=1, keepdim=True) > 0 ).reshape(num_camera, -1).sum(dim=0).bool()

            offsets = [] 
            for mask in masks:
                indices = mask.reshape(self.W, self.H).nonzero()
                selected_indices = torch.randperm(len(indices), device=masks.device)[:num_virtual]
                if len(selected_indices) < num_virtual:
                        selected_indices = torch.cat([selected_indices, selected_indices[
                            selected_indices.new_zeros(num_virtual-len(selected_indices))]])

                offset = indices[selected_indices]
                offsets.append(offset)
            
            offsets = torch.stack(offsets, dim=0)
            virtual_point_instance_ids = torch.arange(1, 1+masks.shape[0], 
                dtype=torch.float32, device=CUDA_DEVICE).reshape(masks.shape[0], 1, 1).repeat(1, num_virtual, 1)

            virtual_points = torch.cat([offsets, virtual_point_instance_ids], dim=-1).reshape(-1, 3)
            virtual_point_camera_ids = camera_ids.reshape(-1, 1, 1).repeat(1, num_virtual, 1).reshape(-1, 1)

            valid_mask = valid.sum(dim=1)>0
            real_point_instance_ids = (torch.argmax(valid.float(), dim=1) + 1)[valid_mask]
            real_points = torch.cat([points_xyc[:, :2][valid_mask], real_point_instance_ids[..., None]], dim=-1)

            # avoid matching across instances 
            real_points[:, -1] *= 1e4 
            virtual_points[:, -1] *= 1e4 

            if len(real_points) == 0:
                return None 
            
            dist = torch.norm(virtual_points.unsqueeze(1) - real_points.unsqueeze(0), dim=-1) 
            nearest_dist, nearest_indices = torch.min(dist, dim=1) 
            mask = nearest_dist < dist_thresh 

            indices = valid_mask.nonzero(as_tuple=False).reshape(-1)

            nearest_indices = indices[nearest_indices[mask]]
            virtual_points = virtual_points[mask]
            virtual_point_camera_ids = virtual_point_camera_ids[mask]
            all_virtual_points = [] 
            all_real_points = [] 
            all_point_labels = []

            for i in range(num_camera):
                camera_mask = (virtual_point_camera_ids == i).squeeze()
                per_camera_virtual_points = virtual_points[camera_mask]
                per_camera_indices = nearest_indices[camera_mask]
                per_camera_virtual_points_depth = points_2d.reshape(-1, 5)[per_camera_indices, 2].reshape(1, -1)

                per_camera_virtual_points = per_camera_virtual_points[:, :2] # remove instance id 
                per_camera_virtual_points_padded = torch.cat(
                        [per_camera_virtual_points.transpose(1, 0).float(), 
                        torch.ones((1, len(per_camera_virtual_points)), device=per_camera_indices.device, dtype=torch.float32)],
                        dim=0
                    )
                per_camera_virtual_points_3d = reverse_view_points(per_camera_virtual_points_padded, per_camera_virtual_points_depth, intrinsics[i])

                per_camera_virtual_points_3d[:3] = torch.matmul(
                        torch.inverse(transforms[i]), #Be careful - avoid singularity in transforms[i]
                        torch.cat([
                                per_camera_virtual_points_3d[:3, :], 
                                torch.ones(1, per_camera_virtual_points_3d.shape[1], dtype=torch.float32, device=per_camera_indices.device)
                            ], dim=0)
                    )[:3]

                all_virtual_points.append(per_camera_virtual_points_3d.transpose(1, 0))
                all_real_points.append(raw_points.reshape(1, -1, 4).repeat(num_camera, 1, 1).reshape(-1,4)[per_camera_indices][:, :3])
                all_point_labels.append(point_labels[per_camera_indices])

            all_virtual_points = torch.cat(all_virtual_points, dim=0)
            all_real_points = torch.cat(all_real_points, dim=0)
            all_point_labels = torch.cat(all_point_labels, dim=0)

            all_virtual_points = torch.cat([all_virtual_points, all_point_labels], dim=1)

            real_point_labels = point_labels.reshape(num_camera, raw_points.shape[0], -1)
            real_point_labels  = torch.max(real_point_labels, dim=0)[0]

            all_real_points = torch.cat([raw_points[foreground_real_point_mask.bool()], real_point_labels[foreground_real_point_mask.bool()]], dim=1)

            return all_virtual_points, all_real_points, foreground_real_point_mask.bool().nonzero(as_tuple=False).reshape(-1)


    def __concat_point_features(self,
                              original_points, # (N, 4) 
                              real_points,
                              virtual_points,
                              num_point_feature = 4 
                              ):
        # original_lidar_points_arr : [x,y,z, time? ] # (N, 4) 
        original_lidar_points_arr = original_points # (N, 4) 
        original_lidar_points_arr = np.concatenate([original_lidar_points_arr, np.ones([original_lidar_points_arr.shape[0], 15-num_point_feature])], axis=1) # padding with 11 columns of 1 # (_,4) => (_,15) 
            # original_lidar_points_arr : [x,y,z, time?,  1,1,1,1,1,1,1,1,1,1,1] # (N, 15) 

        # real_points_arr : [x,y,z,reflectance, time? , one_hot_labels_for_10_classes] #shape: (_, 15)
        # 1 - First remove reflectance as other virtual points don't have this value   # index 3 is omitted 
        real_points_arr     = real_points[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] #shape: (_, 15)=>(_,14)
            # real_points_arr : [x,y,z, time? , one_hot_labels_for_10_classes] #shape: (_, 14)
        # 2 - padding with a column of 0s   # (_,14)=>(_,15)
        real_points_arr     = np.concatenate([real_points_arr,         np.zeros([real_points_arr.shape[0], 1])]     , axis=1) 
            # real_points_arr : [x,y,z, time? , one_hot_labels_for_10_classes, 0] #shape: (_, 15)

        # virtual_points_arr :[x,y,z, time? , one_hot_labels_for_10_classes] # (_, 14) 
        virtual_points_arr  = virtual_points # (_, 14) 
        # 3 - padding with a column of -1s  
        virtual_points_arr  = np.concatenate([virtual_points_arr, -1 * np.ones([virtual_points_arr.shape[0], 1])]   , axis=1) # (_,14)=>(_,15)
            # virtual_points_arr :[x,y,z, time? , one_hot_labels_for_10_classes, -1] # (_, 15) 

        # 4 - Concat all
            ## original_lidar_points_arr : [x,y,z, time?,  1,1,1,1,1,1,1,1,1,1,1]              # (N, 15) 
            ## real_points_arr           : [x,y,z, time? , one_hot_labels_for_10_classes, 0]   # (_, 15)
            ## virtual_points_arr        : [x,y,z, time? , one_hot_labels_for_10_classes, -1]  # (_, 15) 
        point_features = np.concatenate([original_lidar_points_arr, real_points_arr, virtual_points_arr], axis=0).astype(np.float32) # (_, 15)
        
        return point_features # (_, 15)  [x,y,z, time? , one_hot_labels_for_10_classes_or_all_1s, type_encoding]
        
