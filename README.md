[MaskRCNN](https://arxiv.org/abs/1703.06870) implementation using PyTorch

### Install

* `pip install mmcv-full==1.5.2`
* `pip install mmdet==2.25.0`

### Train

* `bash ./main.sh ./nets/exp01.py $ --train` for training, `$` is number of GPUs

### Results

|  Detector  | Backbone | Neck | LR Schedule | Box mAP | Mask mAP | Config |
|:----------:|:--------:|:----:|:-----------:|--------:|---------:|-------:|
| Mask R-CNN |  Swin-T  | FPN  |     1x      |    42.8 |     39.4 |  exp01 |
| Mask R-CNN |  Swin-T  | FPN  |     1x      |    42.8 |     39.4 |  exp02 |
| Mask R-CNN |  Swin-T  | FPN  |     3x      |       - |        - |  exp03 |
| Mask R-CNN |  Swin-T  | FPN  |     3x      |       - |        - |  exp04 |

### TODO

* [x] [exp01](./nets/exp01.py), default [Mask R-CNN](https://arxiv.org/abs/1703.06870)
* [x] [exp02](./nets/exp02.py), added [PolyLoss](https://arxiv.org/abs/2204.12511)
* [x] [exp03](./nets/exp03.py), added [RandomAugment](https://arxiv.org/abs/1909.13719)
* [x] [exp04](./nets/exp04.py), added [MOSAIC](https://arxiv.org/abs/2004.10934)
  |  [MixUp](https://arxiv.org/abs/1710.09412) | [CopyPaste](https://arxiv.org/abs/2012.07177)

### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/open-mmlab/mmdetection