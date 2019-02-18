#! /usr/bin/env python

import numpy as np 
import cv2
import sys
from pyimagesearch.centroidtracker import CentroidTracker
import time
import rospy
from std_msgs.msg import String 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

classNames = { 0: 'background',
	1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
	5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
	10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
	14: 'motorbike', 15: 'person', 16: 'pottedplant',
17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

class pedestrianDetecTrack():
	def __init__(self):
		rospy.init_node("pedestrian_detection_and_tracking", anonymous = True)
		self.bridge = CvBridge()

		self.image_pub = rospy.Publisher("/pedestrian_detection", Image, queue_size=1)
		self.image_sub = rospy.Subscriber("/camera/color/image_rect_color", Image, self.callback,queue_size=1)
		# self.image_sub = rospy.Subscriber("/gta/frame", Image, self.callback,queue_size=1)

		self.prototxt_path = './MobileNetSSD_deploy.prototxt'
		self.caffemodel_path = './MobileNetSSD_deploy.caffemodel'
		print("[INFO] loading model...")
		self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)
		self.ct = CentroidTracker()

		self.cv_image = 0
		self.ros_image = Image()
		self.msg_header = 0

	def callback(self, data):
		try:
			self.msg_header = data.header
			self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print (e)

	def pub_image(self, image):
		try:
			ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
			ros_image.header=self.msg_header

			self.image_pub.publish(ros_image)
		except CvBridgeError as e:
			print (e)


	def run(self):
		(H, W) = (None, None)
		rate = rospy.Rate(10)
		time.sleep(1.0)

		while not rospy.is_shutdown():
			# read the next frame from the video stream and resize it
			frame = self.cv_image

			frame_resized = cv2.resize(frame,(300,300))

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
			self.net.setInput(blob)
			detections = self.net.forward()
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
			objects = self.ct.update(rects)

			# loop over the tracked objects
			for (objectID, centroid) in objects.items():
				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "Person {}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
			self.pub_image(frame)
			rate.sleep()

if __name__ == '__main__':
	dt = pedestrianDetecTrack()
	try:
		dt.run()
	except rospy.ROSInterruptException: 
		pass
