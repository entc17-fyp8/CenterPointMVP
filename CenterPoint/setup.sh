# Deformable Convolution (Optional and only works with old torch versions e.g. 1.1)
# (we don't use dcn in the most recent version)
cd det3d/ops/dcn 
python3 setup.py build_ext --inplace

# Rotated NMS 
cd .. && cd  iou3d_nms
python3 setup.py build_ext --inplace
