
We use the default detectron2 demo script. 

To run inference on an image folder using our pre-trained model, run

    python3 projects/CenterNet2/demo/demo.py --config-file projects/CenterNet2/configs/nuImages_CenterNet2_DLA_640_8x.yaml --input path/to/image/ --opts MODEL.WEIGHTS models/CenterNet2_R50_1x.pth