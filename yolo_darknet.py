'''
Test for YOLO using yolo library to best optimize the YOLO code for python
Built using CUDA, CUDNN, OPENCV and OPENMP
'''
from pydarknet import Detector, Image
import cv2

#print("Loading detector")
net = Detector(bytes("cfg/yolov2.cfg", encoding="utf-8"), bytes("cfg/yolov2.weights", encoding="utf-8"), 0, bytes("data/coco.data",encoding="utf-8"))
print("Detector initialized.")

imPath = "images/extractedFrames/5.jpg"

print("Loading "+imPath+" ...")
img = cv2.imread(imPath)
img_darknet = Image(img)
#print("Image loaded")

results = net.detect(img_darknet)
print("Detected!")

for cat, score, bounds in results:
    x, y, w, h = bounds
    cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
    cv2.putText(img,str(cat.decode("utf-8")),(int(x - w/2),int(y - h/2 + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0))

imOutPath = "images/predictions_python2.jpg"
cv2.imshow("Detection", img)
cv2.imwrite(imOutPath,img)

print("Prediction written to "+imOutPath)

cv2.waitKey(0)
