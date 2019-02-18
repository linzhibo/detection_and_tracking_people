import math
from geometry_msgs.msg import Point
import numpy as np 
import cv2

Kd = 2
color_names = ['Red', 'Green', 'Yellow']

def ifNotInf(A):
	return A[0] != float('inf') and A[0] != -float('inf') and A[1] != float('inf') and A[1] != -float('inf')

def distanceTwoPoints(point1, point2):
	return math.sqrt((point1.x - point2.x)**2 +  (point1.y - point2.y)**2)

def PIDControl(target, current):
	return Kd*(target-current)

def twistAfterPID(twist, robot_odom):
	twist.linear.x = twist.linear.x + PIDControl(twist.linear.x,  robot_odom.twist.twist.linear.x)
	twist.angular.z = twist.angular.z + PIDControl(twist.angular.z, robot_odom.twist.twist.angular.z)
	return twist

def robotCoorToRealWorldCoor(pose, r_point):
	w_point = Point()
	theta = pose.theta
	w_point.x = r_point.x * math.cos(theta) - r_point.y * math.sin(theta) + pose.x
	w_point.y = r_point.x * math.sin(theta) + r_point.y * math.cos(theta) + pose.y
	w_point.z = 0
	return w_point

def realWorldCoorToRobotCoor(pose, w_point):
	r_point = Point()
	theta = pose.theta
	diff_x = w_point.x - pose.x
	diff_y = w_point.y - pose.y
	r_point.x =  diff_x*math.cos(theta) + diff_y*math.sin(theta)
	r_point.y = -diff_y*math.sin(theta) + diff_y*math.cos(theta)
	return r_point

def getTrafficLightColor(img):
	'''
	img is format brg

	for HSV details see:
	https://en.wikipedia.org/wiki/HSL_and_HSV#/media/File:HSV-RGB-comparison.svg
	'''
	hsv= cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),cv2.COLOR_RGB2HSV)
	h,s,v = cv2.split(hsv)
	#red intensity
	red_lowerb_1 = (0,100,100)
	red_lowerb_2 = (160,100,100)
	red_upperb_1 = (20,255,255)
	red_upperb_2 = (180,255,255)

	red_mask1 = cv2.inRange(hsv, red_lowerb_1,red_upperb_1)
	red_mask2 = cv2.inRange(hsv, red_lowerb_2,red_upperb_2)
	red_mask = cv2.bitwise_or(red_mask1, red_mask2)

	r = np.mean(red_mask)

	#yellow intensity
	yellow_lowerb = (20,100,100)
	yellow_upperb = (40,255,255)

	yellow_mask = cv2.inRange(hsv, yellow_lowerb, yellow_upperb)
	y = np.mean(yellow_mask)

	#green intensity
	green_lowerb = (50,100,100)
	green_upperb = (70,255,255)

	green_mask = cv2.inRange(hsv, green_lowerb, green_upperb)
	g = np.mean(green_mask)

	return color_names[np.argmax([r,g,y])]