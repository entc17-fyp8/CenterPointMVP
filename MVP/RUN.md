# Run CenterPoint-MVP

### Data Preparation - for MVP

Refer to https://github.com/tianweiy/MVP for more details.

To save time and space, before running the above command, you can also remove lines below https://github.com/tianweiy/CenterPoint/blob/master/tools/create_data.py#L13 to avoid generating gt database. After that, remember to set https://github.com/tianweiy/CenterPoint/blob/master/configs/mvp/nusc_centerpoint_pp_fix_bn_z_scale_virtual.py#L135 to None. The improvements of gt sampling on nuscenes is marginal (<0.5nds). 

    python3 tools/create_data.py nuscenes_data_prep --root_path="/workspace/CenterPoint/data/nuScenes" --version="v1.0-trainval"  --nsweeps=10 --virtual True 




### Generate Virtual Points

    python3 virtual_gen.py --info_path /workspace/CenterPoint/data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl  MODEL.WEIGHTS /workspace/Checkpoints/centernet2/centernet2_checkpoint.pth 

### Run ROS Inference Code

    python3 ros_inference.py  MODEL.WEIGHTS /workspace/Checkpoints/centernet2/centernet2_checkpoint.pth 

or simply

    python3 ros_inference.py