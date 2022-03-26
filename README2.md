# CenterPoint - Center-based 3D Object Detection and Tracking

## Clone Repo

    git clone <URL>
    git submodule update --init --recursive



## Data Preparation

    # Train set

    python tools/create_data.py nuscenes_data_prep --root_path="/workspace/CenterPoint/data/nuScenes" --version="v1.0-trainval" --nsweeps=10

    # Test set
    python tools/create_data.py nuscenes_data_prep --root_path="/workspace/CenterPoint/data/nuScenes/v1.0-test" --version="v1.0-test" --nsweeps=10


## ROS

    python tools/ros_inference_single_sweep.py 

### Testing - with Optimal Config file

    cd CenterPoint

    python tools/dist_test.py configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py  --work_dir ../Checkpoints/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z  --checkpoint ../Checkpoints/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/epoch_20.pth --speed_test 

## To get the video

    cd CenterPoint/tools
    mkdir demo
    python demo_only_300_iters.py