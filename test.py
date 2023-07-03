import os

# Single gpu
os.system("CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/test.py faster-rcnn_r50_afpn_1x_coco.py ./weight/afpn_weight.pth")
