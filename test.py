import os

# Single gpu
os.system("CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/test.py faster-rcnn_r50_afpn_1x_coco.py ./weight/afpn_weight.pth")

# Single gpu
# os.system("CUDA_VISIBLE_DEVICES=0 python ./mmyolo/tools/test.py yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py ./weight/yolov5_n_afpn.pth")
