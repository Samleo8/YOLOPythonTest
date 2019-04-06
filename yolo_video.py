'''
USAGE

python3 yolo_video.py
	[--input|-i <video_file> (default: webcam)] [--output|-o <video_file>] [--cfg|-g <cfg_folder>]
	[--confidence|-c <min_confidence> (default: 0.5)] [--threshold|-t <threshold_nms> (default: 0.3)]
	[--fast|-f] [-s|--scaleRatio <video_scale_down_factor_for_faster_detection>]
	[--yoloVersion|-y (2 | 3 | 3-tiny) (default: 2)]
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
ap.add_argument("-i", "--input", default=0,
	help="path to input video")
ap.add_argument("-o", "--output", default="output",
	help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
ap.add_argument("-f", "--fast", action='store_true',
	help="faster detection?")
ap.add_argument("-s","--scaleRatio", default="4", help="Scale down by factor for fasterDetect")
ap.add_argument("-g","--cfg", default="cfg", help="Folder containing cfg and weight files")
ap.add_argument("-y","--yoloVersion", default="2", help="yoloVersion: 2 | 3 | 3-tiny")
args = vars(ap.parse_args())

# global variables
fasterDetect = args["fast"]
scaleRatio = 4

# load the COCO class labels our YOLO model was trained on
yolo_cfg_dir = args["cfg"]
labelsPath = os.path.sep.join([yolo_cfg_dir, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([yolo_cfg_dir, "yolov"+args["yoloVersion"]+".weights"])
configPath = os.path.sep.join([yolo_cfg_dir, "yolov"+args["yoloVersion"]+".cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO config, weights files from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#net = cv2.dnn.readNet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
video_path = args["input"]
if video_path == 0 or video_path == "0" or video_path == "webcam": video_path = 0

vs = cv2.VideoCapture(video_path)
writer = None
(W, H) = (None, None)

if video_path:
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
else:
	total = -1

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, _frame) = vs.read()

	if fasterDetect:
		frame = cv2.resize(_frame, (0, 0), fx=1/scaleRatio, fy=1/scaleRatio)
	else:
		frame = _frame.copy()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			if fasterDetect:
				# Scale back up face locations since the frame we detected in was scaled to 1/4 size
				x *= scaleRatio
				y *= scaleRatio
				w *= scaleRatio
				h *= scaleRatio

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(_frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(_frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"]+".avi", fourcc, 15.0,
			(_frame.shape[1], _frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))
	else:
		elap = (end - start)
		print("[INFO] single frame took {:.4f} seconds".format(elap))

	# write the output frame to disk
	writer.write(_frame)

	if video_path == 0:
		cv2.putText(_frame, "Q/Esc to Quit", (W-130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
		cv2.imshow("YOLO Detection",_frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q') or key == 27 or grabbed==False: #esc or 'Q'
		break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
