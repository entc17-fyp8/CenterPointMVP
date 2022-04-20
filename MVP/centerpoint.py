
import rospy
import ros_numpy
import numpy as np

import torch
import time 

from sensor_msgs.msg import PointCloud2, PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion

# from det3d import __version__, torchie
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator

DEVICE ="cuda:0"

def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)

def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices  


def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(0, 0.4, label_preds_, scores_)
    truck_indices =                get_annotations_indices(1, 0.4, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(2, 0.4, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(3, 0.3, label_preds_, scores_)
    trailer_indices =              get_annotations_indices(4, 0.4, label_preds_, scores_)
    barrier_indices =              get_annotations_indices(5, 0.4, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(6, 0.15, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(7, 0.15, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(8, 0.1, label_preds_, scores_)
    traffic_cone_indices =         get_annotations_indices(9, 0.1, label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations


class CenterPointForwardModel:
    def __init__(self, config_path, model_path, modality='MVP'):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None

        self.modality = modality
        if self.modality == 'MVP':
            self.num_features = 16
        else:
            self.num_features = 5

        self.__init_from_config()
        
    def __init_from_config(self):
        
        config_path = self.config_path
        cfg = Config.fromfile(self.config_path)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = DEVICE
        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        self.net.load_state_dict(torch.load(self.model_path)["state_dict"])
        self.net = self.net.to(self.device).eval()

        self.range = cfg.voxel_generator.range
        self.voxel_size = cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = cfg.voxel_generator.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[1],
        )

    def run_model(self, point_features):
        '''
        Run CenterPoint with Virtual Points

        Args:
            point_features # (_,5) if lidar-only;  or (_, 15) if MVP 

        Returns:
            return scores
            boxes_lidar
            types
        '''
        
        if self.modality == 'MVP': # Using virtual points
            # point_features has shape: # (_, 15)  
                # [x,y,z, time? , one_hot_labels_for_10_classes_or_all_1s, type_encoding]
            # self.points should be (_,16)
            # Add a column of 0s to the end.   # For time
            self.points = np.concatenate([  
                    point_features[:, [0, 1, 2, ]],                
                    point_features[:,[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
                    np.zeros([point_features.shape[0], 1])
                ],
                axis=1
            )
        else: # Using only LiDAR points
            # point_features has shape : (_,5)             
            self.points = point_features.reshape([-1, self.num_features]) # num_features = 5
            # self.points also should be (_,5)
            self.points[:, 4] = 0 # timestamp value set to 0s

        
        tt_1_b = time.time()
        # Generate voxels from points (Using GPU with numba)
        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)
        print("\tTime Cost: Voxelization \t \t:\t", time.time() - tt_1_b)
        
        # Move tensors to GPU
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)
        
        self.inputs = dict(
            # points = voxels,
            voxels = voxels,
            num_points = num_points,
            num_voxels = num_voxels,
            coordinates = coords,
            shape = [grid_size]
        )

        # Waits for all kernels in all streams on a CUDA device to complete.
        torch.cuda.synchronize()
        tt_2 = time.time()

        # Run NN forward pass
        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)[0]
    
        
        torch.cuda.synchronize()
        print("\tTime Cost: CenterPoint Forward Pass \t:\t", time.time() - tt_2)

        outputs = remove_low_score_nu(outputs, 0.45)

        boxes_lidar = outputs["box3d_lidar"].detach().cpu().numpy()
        # print("  predict boxes:", boxes_lidar.shape)

        scores = outputs["scores"].detach().cpu().numpy()
        types = outputs["label_preds"].detach().cpu().numpy()

        boxes_lidar[:, -1] = -boxes_lidar[:, -1] - np.pi / 2

        return scores, boxes_lidar, types

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''
    '''
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg
