import os

# Multiple gpus
os.system("CUDA_VISIBLE_DEVICES=0,1 ./mmdetection/tools/dist_train.sh faster-rcnn_r50_afpn_1x_coco.py 2 --work-dir ./weight/")

# Single gpu
# os.system("CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/train.py faster-rcnn_r50_afpn_1x_coco.py --work-dir ./weight/")

# Multiple gpus
# os.system("CUDA_VISIBLE_DEVICES=0,1 ./mmyolo/tools/dist_train.sh yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py 2 --work-dir ./weight/")

# Single gpu
# os.system("CUDA_VISIBLE_DEVICES=0 python./mmyolo/tools/train.sh yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py --work-dir ./weight/")
