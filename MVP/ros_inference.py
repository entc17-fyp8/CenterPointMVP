import imp
from unittest.mock import DEFAULT
from tqdm import tqdm 
import argparse 
import numpy as np 
import torch 
import cv2 
import os 

import rospy
import ros_numpy
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import message_filters
import tf

from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
import time

from centerpoint import CenterPointForwardModel, yaw2quaternion
from mvp_model import MVP

DEFAULT_CONFIG_FILE_PATHS = {
    'CenterPoint':
        {
            'LIDAR':
                {
                    'PP':'/workspace/CenterPoint/configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py',
                    'VN':'/workspace/CenterPoint/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py'
                },
            'MVP':
                {
                    'PP':'/workspace/CenterPoint/configs/mvp/nusc_centerpoint_pp_fix_bn_z_scale_virtual.py'
                }
        },
    'CenterNet2':'c2_config/nuImages_CenterNet2_DLA_640_8x.yaml'
}

DEFAULT_WEIGHT_FILE_PATHS = {
    'CenterPoint':
        {
            'LIDAR':
                {
                    'PP':'/workspace/Checkpoints/nusc_02_pp/latest.pth',
                    'VN':'/workspace/Checkpoints/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/epoch_20.pth'
                },
            'MVP':
                {
                    'PP':'/workspace/Checkpoints/centerpoint_mvp/nusc_centerpoint_pp_fix_bn_z_scale_virtual/epoch_20.pth'
                }
        },
    'CenterNet2':'/workspace/Checkpoints/centernet2/centernet2_checkpoint.pth'
}

class ROSNode:

    frame_names = {
        'CAM_FRONT_LEFT'    : 'cam_front_left',
        'CAM_FRONT'         : 'cam_front',
        'CAM_FRONT_RIGHT'   : 'cam_front_right',
        'CAM_BACK_LEFT'     : 'cam_back_left',
        'CAM_BACK'          : 'cam_back',
        'CAM_BACK_RIGHT'    : 'cam_back_right',
        'LIDAR_TOP'         : 'lidar',
    }

    CAM_CHANS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    def __init__(self, 
                    modality, # MVP | LIDAR
                    mvp_model,
                    centerpoint_model:CenterPointForwardModel
                ):
        rospy.init_node('mvp_node')
        rospy.loginfo('mvp_node started')

        self.modality = modality

        self.mvp_model = mvp_model
        self.centerpoint_model = centerpoint_model
    
        self.cv_bridge = CvBridge()
        self.tf_listener = tf.TransformListener()

        # Subscribes
        if self.modality=='MVP': # Using both LiDAR and Cameras
            self.img_subscribers = {
                'CAM_FRONT_LEFT'    : message_filters.Subscriber('/camera/front_left/rgb_image',    Image),
                'CAM_FRONT'         : message_filters.Subscriber('/camera/front/rgb_image',         Image),
                'CAM_FRONT_RIGHT'   : message_filters.Subscriber('/camera/front_right/rgb_image',   Image),
                'CAM_BACK_LEFT'     : message_filters.Subscriber('/camera/back_left/rgb_image',     Image),
                'CAM_BACK'          : message_filters.Subscriber('/camera/back/rgb_image',          Image),
                'CAM_BACK_RIGHT'    : message_filters.Subscriber('/camera/back_right/rgb_image',    Image),
            }
            self.camera_info_subscribers = {
                'CAM_FRONT_LEFT'    : message_filters.Subscriber('/camera/front_left/camera_info',  CameraInfo),
                'CAM_FRONT'         : message_filters.Subscriber('/camera/front/camera_info',       CameraInfo),
                'CAM_FRONT_RIGHT'   : message_filters.Subscriber('/camera/front_right/camera_info', CameraInfo),
                'CAM_BACK_LEFT'     : message_filters.Subscriber('/camera/back_left/camera_info',   CameraInfo),
                'CAM_BACK'          : message_filters.Subscriber('/camera/back/camera_info',        CameraInfo),
                'CAM_BACK_RIGHT'    : message_filters.Subscriber('/camera/back_right/camera_info',  CameraInfo),
            }
            self.pc_subscribers={
                'LIDAR_TOP' : message_filters.Subscriber('/lidar/top', PointCloud2, queue_size=1, buff_size=2**24),
            }
            subscribers = []
            for channel in self.CAM_CHANS:
                subscribers.append(self.img_subscribers[channel])
            for channel in self.CAM_CHANS:
                subscribers.append(self.camera_info_subscribers[channel])
            subscribers.append(self.pc_subscribers['LIDAR_TOP'])
            
            self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(subscribers,queue_size=2,slop=0.7,allow_headerless=True)
            # self.time_synchronizer = message_filters.TimeSynchronizer(subscribers,queue_size=10)
            self.time_synchronizer.registerCallback(self.data_recieved_callback)

        else: # LiDAR only
            self.img_subscribers = None
            self.camera_info_subscribers = None
            self.pc_subscribers={
                'LIDAR_TOP' : rospy.Subscriber("/lidar/top", PointCloud2, self.data_recieved_callback, queue_size=1, buff_size=2**24)
            }

        # Publisher for detected bounding boxes
        self.pub_arr_bbox = rospy.Publisher("bboxes_detected", BoundingBoxArray, queue_size=1)

        rospy.spin()
        
    def __get_xyz_points(self, cloud_array, remove_nans=True, dtype=np.float):
        '''
        '''
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
            cloud_array = cloud_array[mask]

        points = np.zeros(cloud_array.shape + (5,), dtype=dtype) # (N,5)
        points[...,0] = cloud_array['x']
        points[...,1] = cloud_array['y']
        points[...,2] = cloud_array['z']
        points[...,3] = cloud_array['intensity']
        return points

    def data_recieved_callback(self,*data):
        tt_0 = time.time()
        print('callback fn..')
        
        if self.modality=='MVP': # Using both LiDAR and Cameras
            img_msgs, camera_info_msgs, pc_msg = data[0:6] , data[6:12], data[12]

            # convert sensor_msgs/Image to tensors
            rgb_image_data = []
            for rgb_image_msg in img_msgs:
                rgb_image_cv2 = self.cv_bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
                rgb_img_tensor = self.mvp_model.preprocess_image(rgb_image_cv2)
                rgb_image_data.append(rgb_img_tensor)
                
            # LiDAR to Camera transforms
            all_cams_from_lidar_tms = []
            for cam_channel in self.CAM_CHANS:
                try:
                    (trans,rot) = self.tf_listener.lookupTransform(self.frame_names[cam_channel], self.frame_names['LIDAR_TOP'], rospy.Time(0))
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logerr("ROS TF Exception")
                    return
                tm = transform_matrix(trans, Quaternion(rot), inverse=False)
                all_cams_from_lidar_tms.append(tm)
            
            # Intrinsics
            all_cams_intrinsic =[]
            for camera_info_msg in camera_info_msgs:
                cameraMatrix = np.array(camera_info_msg.K).reshape((3, 3))
                all_cams_intrinsic.append(cameraMatrix)

            # LiDAR Points
            original_pc_arr = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
            original_pc_arr = self.__get_xyz_points(original_pc_arr, True)

            # Generate Virtual Points using CenterNet2
            tt_00p = time.time()
            virtual_pc = self.mvp_model.do_inference(
                lidar_points=original_pc_arr,
                image_data=rgb_image_data,
                all_cams_from_lidar_tms=all_cams_from_lidar_tms,
                all_cams_intrinsic=all_cams_intrinsic
            )
            print("\tTime Cost: Generating Virtual Points \t:\t", time.time() - tt_00p)
            
            # Run CenterPoint model
            scores, dt_box_lidar, types = self.centerpoint_model.run_model(virtual_pc)

        else: # Using only LiDAR
            pc_msg = data[0]
            img_msgs, camera_info_msgs, = None, None

            # Convert LiDAR Points to np arrays
            original_pc_arr = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
            original_pc_arr = self.__get_xyz_points(original_pc_arr, True)

            # Run CenterPoint model
            scores, dt_box_lidar, types = self.centerpoint_model.run_model(original_pc_arr)
        
    
        # Publish to ROS
        self.__publish_detection_results(scores, dt_box_lidar, types, pc_msg.header.stamp)

        print("\tTotal Time Cost: Inside Callback \t:\t", time.time() - tt_0)


    def __publish_detection_results(self, scores, dt_box_lidar, types, original_timestamp):
        arr_bbox = BoundingBoxArray()
        if scores.size != 0:
            print("Detected %d objects.."%(scores.size))
            for i in range(scores.size):
                bbox = BoundingBox()
                bbox.header.frame_id = 'lidar'
                bbox.header.stamp = original_timestamp
                q = yaw2quaternion(float(dt_box_lidar[i][8]))
                bbox.pose.orientation.x = q[1]
                bbox.pose.orientation.y = q[2]
                bbox.pose.orientation.z = q[3]
                bbox.pose.orientation.w = q[0]           
                bbox.pose.position.x = float(dt_box_lidar[i][0])
                bbox.pose.position.y = float(dt_box_lidar[i][1])
                bbox.pose.position.z = float(dt_box_lidar[i][2])
                bbox.dimensions.x = float(dt_box_lidar[i][4])
                bbox.dimensions.y = float(dt_box_lidar[i][3])
                bbox.dimensions.z = float(dt_box_lidar[i][5])
                bbox.velocity.x = float(dt_box_lidar[i][6])
                bbox.velocity.y = float(dt_box_lidar[i][7])
                bbox.velocity.z = float(0)
                bbox.value = scores[i]
                bbox.label = int(types[i])
                arr_bbox.boxes.append(bbox)
        
        arr_bbox.header.frame_id = 'lidar'
        arr_bbox.header.stamp = original_timestamp
        if len(arr_bbox.boxes) is not 0:
            self.pub_arr_bbox.publish(arr_bbox)
            # print('Published to ROS')
            arr_bbox.boxes = []
        else:
            arr_bbox.boxes = []
            self.pub_arr_bbox.publish(arr_bbox)

            

if __name__ == '__main__':
    ##
    # Read Arguments
    ##
    import argparse
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('modality', choices=['mvp', 'lidar'],default='mvp', help='select which sensor modality to use') # MVP | LIDAR
    parser.add_argument('backbone', choices=['pp', 'vn'], default='pp', help='select which backbone to use (PointPillars or VoxelNet)') # PointPillars | VoxelNet

    args = parser.parse_args()
    
    print(args)

    modality = args.modality.upper() # MVP | LIDAR
    backbone = args.backbone.upper() # PP | VN

    
    ##
    # Init models
    ##

    # MVP
    if modality=='MVP':
        mvp_model = MVP(
            args,
            config_file = DEFAULT_CONFIG_FILE_PATHS['CenterNet2'], 
            opts_list = ['MODEL.WEIGHTS', DEFAULT_WEIGHT_FILE_PATHS['CenterNet2']]
        )
    else:
        mvp_model = None 

    # CenterPoint 
    centerpoint_model = CenterPointForwardModel(
        config_path = DEFAULT_CONFIG_FILE_PATHS['CenterPoint'][modality][backbone], 
        model_path  = DEFAULT_WEIGHT_FILE_PATHS['CenterPoint'][modality][backbone],
        modality =  modality
    )

    ##
    # Run ROS node 
    ##
    try:
        ROSNode(modality, mvp_model, centerpoint_model)
    except rospy.ROSInterruptException:
        pass
