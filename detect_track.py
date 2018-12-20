import numpy as np 
import cv2
import sys
from random import randint
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import imutils
import time
classNames = { 0: 'background',
	1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
	5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
	10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
	14: 'motorbike', 15: 'person', 16: 'pottedplant',
17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

def ssdDetection(net, frame, threshold):
	frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
	blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
	net.setInput(blob)
	detections = net.forward()

	cols = frame_resized.shape[1] 
	rows = frame_resized.shape[0]

	bboxes = []
	labels = []
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2] #Confidence of prediction 
		if confidence > threshold: # Filter prediction 
			class_id = int(detections[0, 0, i, 1]) # Class label

			# Object location 
			xLeftBottom = int(detections[0, 0, i, 3] * cols) 
			yLeftBottom = int(detections[0, 0, i, 4] * rows)
			xRightTop   = int(detections[0, 0, i, 5] * cols)
			yRightTop   = int(detections[0, 0, i, 6] * rows)
			
			# Factor for scale to original size of frame
			heightFactor = frame.shape[0]/300.0  
			widthFactor = frame.shape[1]/300.0 
			# Scale object detection to frame
			xLeftBottom = int(widthFactor * xLeftBottom) 
			yLeftBottom = int(heightFactor * yLeftBottom)
			xRightTop   = int(widthFactor * xRightTop)
			yRightTop   = int(heightFactor * yRightTop)
			# Draw location of object  
			# cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
			# 			  (0, 255, 0))
			

			# Draw label and confidence of prediction in frame resized
			if class_id in classNames:
				bboxes.append((xLeftBottom, yLeftBottom, xRightTop, yRightTop))
				label = classNames[class_id] + ": " + str(confidence)
				labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

				yLeftBottom = max(yLeftBottom, labelSize[1])
				cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
									 (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
									 (255, 255, 255), cv2.FILLED)
				cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

				print(label) #print class and confidence
				labels.append(label)
	return bboxes, labels

def personTracking(frame,bounding_box, tracker):
	ok, bounding_box = tracker.update(frame)
	print bounding_box
	if ok:
		p1 = (int(bounding_box[0]), int (bounding_box[1]))
		p2 = (int(bounding_box[0]+ bounding_box[2]), int(bounding_box[1]+ bounding_box[3]))
		cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
	else:
		# print ("not ok")
		cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
		cv2.putText(frame, "GOTURN Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
	# cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
	return ok, frame

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
 
def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
	tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
	tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
	tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
	tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
	tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
	tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
	tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
	tracker = cv2.TrackerCSRT_create()
  else:
	tracker = None
	print('Incorrect tracker name')
	print('Available trackers are:')
	for t in trackerTypes:
	  print(t)
	 
  return tracker

def multiTracking(frame, bboxes, labels, multiTracker):
	for bbox in bboxes:
		multiTracker.add(tracker, frame, bbox)

	while cap.isOpened():
		success, frame = cap.read()
		if not success:
			break
   		success, boxes = multiTracker.update(frame)
   		if not success:
   			break
 
		for i, newbox in enumerate(boxes):
			p1 = (int(newbox[0]), int(newbox[1]))
			p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
			cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
			cv2.putText(frame, labels[i], (int(newbox[0]), int(newbox[1])),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
 
  # show frame
  		cv2.namedWindow("MultiTracker", cv2.WINDOW_NORMAL)
		cv2.imshow('MultiTracker', frame)
   
 
  # quit on ESC button
		if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
			break

## main

print("[INFO] loading model...")
prototxt_path = './MobileNetSSD_deploy.prototxt'
caffemodel_path = './MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

ct = CentroidTracker()
(H, W) = (None, None)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
	# read the next frame from the video stream and resize it
	frame = vs.read()
	frame_resized = imutils.resize(frame, width=300)

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	heightFactor = H/300.0  
	widthFactor = W/300.0 

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

	# blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
	# 	(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		if detections[0, 0, i, 2] > 0.5 and int(detections[0, 0, i, 1]) == 15:
			# print detections[0, 0, i, 1]
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			# (startX, startY, endX, endY) = (int(startX *widthFactor), startY*int(heightFactor), endX*int(widthFactor), endY*int(heightFactor))
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "Person {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# show the output frame
	cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()