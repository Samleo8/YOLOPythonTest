# YOLO Test (Python)

## Intro

This is an attempt to test a Python version of [@pjreddie](https://github.com/pjreddie/)'s [YOLO](https://pjreddie.com/darknet/yolo/), based on the code at [PyImageSearch](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv). The code has been modified for more

The repo contains 2 main python files: `yolo_image.py` and `yolo_video.py`
They are for running the YOLO algorithm on single image and video (webcam by default) respectively, using Python.

The `cfg` folder contains the YOLO 2 weights and configs, as well as their tiny versions.

## Usage

### Video
```
python3 yolo_video.py
	[--input|-i <video_file> (default: webcam)] [--output|-o <video_file>] [--cfg|-g <cfg_folder>]
	[--confidence|-c <min_confidence> (default: 0.5)] [--threshold|-t <threshold_nms> (default: 0.3)]
	[--fast|-f] [-s|--scaleRatio <video_scale_down_factor_for_faster_detection>]
	[--yoloVersion|-y (2 | 3 | 3-tiny) (default: 2)]
```

### Image (Single)
```
python3 yolo_image.py
	[--input|-i <image_file>] [--output|-o <video_file>] [--cfg|-g <cfg_folder>]
	[--confidence|-c <min_confidence> (default: 0.5)] [--threshold|-t <threshold_nms> (default: 0.3)]
	[--yoloVersion|-y (2 | 3 | 3-tiny) (default: 3)]
```

## Dependencies/Setup
- OpenCV: 3.4.4

- dlib

- numpy

- argparse


### Hardware Used
- ASUS Laptop (i7-8550, quad core), Ubuntu 18.10.
- Nvidia GeForce MX150 GPU (2GB), CUDA 10.1, with CUDNN

## Interesting Observations
1) In terms of speed, [YOLOv2](https://pjreddie.com/darknet/yolov2/) is almost 3 times faster than [YOLOv3](https://pjreddie.com/darknet/yolo/) (~0.15 seconds/frame vs ~0.6 seconds/frame). The difference in accuracy (at least in my testing) isn't that great compared to YOLOv2, so use that if you cannot take the slow speed.

Of course, the YOLO2/3-tiny versions are the fastest, but also super inaccurate.

2)  above, it's uncertain if the latest versions of OpenCV or dlib have GPU support now, but I actually suspect that the answer is YES.

Running YOLO2 on darknet's repo (supposed to use CUDA/GPU) gave me images processed in around 0.15 seconds. When I ran it with no GPU support, it was almost 10s. This python code gave images processed in about 0.2 seconds, so I think that GPU is supported now. (Not sure tho)
