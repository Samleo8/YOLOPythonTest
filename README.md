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

### Make video slower
As the output file from running `yolo_video.py` is very fast, I made a python script to also slow down the output:

```
python3 make_video_slower.py
	[--input|-i <path_to_input_video>]
	[--output|-o <path_to_output_video>]
	[--fps|-f <_FPS_no> (default 3.0)]
```

### Extract Frames from Video
To extract a bunch of frames from a video, I have made a simple python script `extract_frames_from_video.py`:

```python3 extract_frames_from_video.py
	[--input|-i <video_file>]
	[--output|-o <path_to_folder_to_save_batch_images>]
	[--startFrame|-s <frame_no> (default: 0)]
	[--endFrame|-e <frame_no> (default: end of video)]
	[--displayFrames|-d]
```

## Dependencies/Setup
- OpenCV: 3.4.4

- dlib

- numpy

- argparse


## Testing against Oxford Town Centre dataset
I tested YOLO versions 2 and 3 (and their tiny versions) against the Oxford Town Center dataset (truncated):

```
Confidence threshold: 0.5
Non-max Suppression threshold: 0.3

YOLO VERSION						v2		v3		v2(tiny)						v3(tiny)

Avg. Processing Time/Frame [sec]	0.25	0.65	0.05							0.05

Avg. Accuracy (mAP)					~20%	~90%	~15% (many false detections)	<10% (many false detections, worse than v2 tiny)
```

The actual results can be found in `videos/pedestrians_yolo<version>.mp4`

### Hardware Used
- ASUS Zenbook Laptop (i7-8550, quad core), Ubuntu 18.10
- Nvidia GeForce MX150 GPU (2GB memory), CUDA 10.1, CUDNN 7.5.0


## General Observations
1) In terms of speed, [YOLOv2](https://pjreddie.com/darknet/yolov2/) is almost 3 times faster than [YOLOv3](https://pjreddie.com/darknet/yolo/) (~0.15 seconds/frame vs ~0.6 seconds/frame). The difference in accuracy, however, is better than YOLOv2 (see videos/pedestrains_yolo2 -vs- 3).

Of course, the YOLO2/3-tiny versions are the fastest, but also super inaccurate.

2) From the comments on PyImageSearch, it's uncertain if the latest versions of OpenCV or dlib have GPU support now, but I actually suspect that the answer is YES.

Running YOLO2 on darknet's repo (supposed to use CUDA/GPU) gave me images processed in around 0.15 seconds. When I ran it with no GPU support, it was almost 10s. This python code gave images processed in about 0.2 seconds, so I think that GPU is supported now. (Not sure tho)
