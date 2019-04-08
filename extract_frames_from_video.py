'''
USAGE

python3 extract_frames_from_video.py
	[--input|-i <video_file>] [--output|-o <path_to_folder_to_save_batch_images>]
	[--startFrame|-s <frame_no> (default: 0)] [--endFrame|-e <frame_no> (default: end of video)]
'''

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", default="images/extractedFrames",
	help="path to folder where program saves batch images")
ap.add_argument("-s", "--startFrame", type=int, default=0,
	help="minimum probability to filter weak detections")
ap.add_argument("-e", "--endFrame", type=int, default=-1,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

startFrame = max(0,args["startFrame"])

video_path = args["input"]

vs = cv2.VideoCapture(video_path)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] Could not determine # of frames in video")
	print("[INFO] No approx. completion time can be provided")
	total = -1

frameN = 0

while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	cv2.imwrite(args["output"]+"/"+str(frameN)+".jpg", frame)
	print("[INFO] Extracted frame {} of {}".format(frameN,total))

	frameN = frameN+1
