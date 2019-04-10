 '''
USAGE

python3 extract_frames_from_video.py
	[--input|-i <video_file>] [--output|-o <path_to_folder_to_save_batch_images>]
	[--startFrame|-s <frame_no> (default: 0)] [--endFrame|-e <frame_no> (default: end of video)]
	[--displayFrames|-d]
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
ap.add_argument("-d", "--displayFrames", action="store_true",
	help="display frames being extracted")
args = vars(ap.parse_args())

startFrame = max(0,args["startFrame"])
endFrame = args["endFrame"]

video_path = args["input"]

vs = cv2.VideoCapture(video_path)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))

	if endFrame == -1:
		endFrame = total

	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[ERROR] Could not determine # of frames in video")
	print("[WARNING] No approx. completion time can be provided")
	total = -1

# Set to start from startFrame
vs.set(cv2.CAP_PROP_POS_FRAMES,startFrame)
frameN = startFrame

while True:
	(grabbed, frame) = vs.read()

	if not grabbed or frameN == endFrame+1:
		print("[INFO] Complete: Frames {} to {} saved in {}".format(startFrame, endFrame, args["output"]))
		break

	cv2.imwrite(args["output"]+"/"+str(frameN)+".jpg", frame)
	print("[INFO] Extracted frame {} of {}".format(frameN,total))

	if args["displayFrames"]:
		cv2.putText(frame, "Frame: "+str(frameN), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
		cv2.imshow("Extracted Frame",frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q') or key == 27: #esc or 'Q'
		print("[WARNING] User exited early")
		break

	frameN = frameN+1
