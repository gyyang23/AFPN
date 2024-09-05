
AFPN: Asymptotic Feature Pyramid Network for Object Detection ([arXiv](https://arxiv.org/abs/2306.15988))
---------------------
By Guoyu Yang, Jie Lei, Zhikuan Zhu, Siyu Cheng, Zunlei Feng, Ronghua Liang

This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

Environment
----------------
```
mmengine==0.7.3
mmcv==2.0.0
mmdet==3.0.0
mmyolo==0.5.0
```

Install
-------------
Please refer to [mmdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation.

Dataset
----------
```
AFPN
├── mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
├── faster-rcnn_r50_afpn_1x_coco.py
├── train.py
├── test.py
```

Training
--------------
Single gpu for train:
```shell
CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/train.py faster-rcnn_r50_afpn_1x_coco.py --work-dir ./weight/
```

Multiple gpus for train:
```shell
CUDA_VISIBLE_DEVICES=0,1 ./mmdetection/tools/dist_train.sh faster-rcnn_r50_afpn_1x_coco.py 2 --work-dir ./weight/
```

Train in pycharm: If you want to train in pycharm, you can run it in train.py.

see more details at [mmdetection](https://github.com/open-mmlab/mmdetection).

Testing
-----------
```shell
CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/test.py faster-rcnn_r50_afpn_1x_coco.py <CHECKPOINT_FILE>
```

For example,
```shell
CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/test.py faster-rcnn_r50_afpn_1x_coco.py ./weight/afpn_weight.pth
```

Test in pycharm: If you want to test in pycharm, you can run it in test.py.

see more details at [mmdetection](https://github.com/open-mmlab/mmdetection).

Results on MS COCO val2017
---------
|      Detector        |  Backbone  | Image size | GFLOPs | Params (M) |  AP  | AP<sub>0.5</sub> | AP<sub>0.75</sub> |   Weight   |
|----------------------|------------|------------|--------|------------|------|------------------|-------------------|------------|
| Faster R-CNN + FPN   | ResNet-50  | 640 x 640  |  91.3  |    41.8    | 37.4 |       57.3       |       40.3        |    None    |
| Faster R-CNN + AFPN  | ResNet-50  | 640 x 640  |  89.7  |    49.8    | 39.0 |       57.6       |       42.0        | [Link](https://drive.google.com/file/d/1P7diJ01i88LSBrJ83ELAExjyY4xQULpu/view?usp=sharing)   |
|YOLOv5-n + YOLOv5PAFPN| CSPDarknet | 640 x 640  |  2.26  |    1.87    | 28.0 |       45.9       |       29.4        | [Link](https://drive.google.com/file/d/19n7_jYsvelEzyHAqen6jRE1zAm0CcPvV/view?usp=sharing)   |
|YOLOv5-n + YOLOv5AFPN | CSPDarknet | 640 x 640  |  2.18  |    1.67    | 29.1 |       45.8       |       30.7        | [Link](https://drive.google.com/file/d/197HsYpAgZ5Tt02UnjgIXgNJL8zyvssir/view?usp=sharing)   |


Citations
------------
If you find AFPN useful in your research, please consider citing:
```
@article{yang2023afpn,
  title={AFPN: Asymptotic Feature Pyramid Network for Object Detection},
  author={Yang, Guoyu and Lei, Jie and Zhu, Zhikuan and Cheng, Siyu and Feng, Zunlei and Liang, Ronghua},
  journal={arXiv preprint arXiv:2306.15988},
  year={2023}
}
```
or
```
@article{yang2024asymptotic,
  title={Asymptotic Feature Pyramid Network for Labeling Pixels and Regions},
  author={Yang, Guoyu and Lei, Jie and Tian, Hao and Feng, Zunlei and Liang, Ronghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```
