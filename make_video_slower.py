'''
USAGE

python3 make_video_slower.py
	[--input|-i <path_to_input_video>]
	[--output|-o <path_to_output_video>]
	[--fps|-f <_FPS_no> (default 3.0)]
'''

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", default=None, help="path to output video")
ap.add_argument("-f", "--fps", type=float, default=3.0, help="FPS")
args = vars(ap.parse_args())

writer = None

video_path = args["input"]

output_path = args["output"]
if output_path == None:
    output_path = "slower_"+video_path

fps = args["fps"]

vs = cv2.VideoCapture(video_path)

W, H = None, None

while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]), True)

	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] Cleaning up...")
writer.release()
vs.release()
