import numpy as np;
import argparse;
import cv2;
import time;


parser = argparse.ArgumentParser(); #creates argument object

parser.add_argument('-v', '--video', help = 'Path to the video file'); # specifies argument

args = vars(parser.parse_args()); #gets the arguments entered in commandline in form of dict

color_lower = np.array([150 , 50 , 0] , dtype = 'uint8'); #sets lower threshold for blue color

color_upper = np.array([255, 150 , 100] , dtype = 'uint8'); #sets upper threshold for blue color

#camera = cv2.VideoCapture(args['video']);# reads video file specified by user on command line



if not args.get('video' , False):
	camera = cv2.VideoCapture(0);# reads video file specified by user on command line
else:	
	camera = cv2.VideoCapture(args['video']);# reads video file specified by user on command line	

while True:

			#(grabbed , frame) = camera.read(); #reads each frame from video.
	
	(grabbed , frame) = camera.read();
	if not grabbed:
		break
	
	#threshold based on color
	color_threshold = cv2.inRange(frame , color_lower , color_upper); #cv2.inRange() returns a binary image with pixels that fall with the color range set to white(255) and the rest set to black(0)
	
	blur = cv2.GaussianBlur(color_threshold , (3, 3) , 0); # to reduce noise and increase detection of the tracking of our object

	(cnts , _)  = cv2.findContours(blur.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE); #cv2.findContours is used to get the contours corresponding to the object.
													#Returns a list of contours each corresponding to detected object color boundaries
													# copy of image is made as this function is destructive to numpy arrays

	if len(cnts) > 0:
		cnt = sorted(cnts , key = cv2.contourArea , reverse = True)[0] # sorts the list of contours from largest to smallest and returns the largest based on area(cv2.contourArea) which is assumed 										      #	to be the object interest
		
		bbox = cv2.minAreaRect(cnt);#cv2.minAreaRect() computes the mininimum bounding box around the contour,
		rects = np.int0(cv2.boxPoints(bbox)) # BoxPoints()converts these bounding box to list of points # 
							#Note without the 0 in **np.int0** the error TypeError: only size-1 arrays can be converted Python scalars

		cv2.drawContours(frame , [rects] , -1 , (0, 255, 0) , 2);# Draw a bounding around object. frame is image to draw box , -1 draws all contours, (0,255,0) color of bounding box, 2 is line thickness
		
	cv2.imshow('Tracking In Progress'  , frame) ;
	cv2.imshow('Threshold' , color_threshold);

	time.sleep(0.0025);
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break;

camera.release();

cv2.destroyAllWindows()
	
		
	#if args.get(args[0] , False):
		#(grabbed , frame) = cv2.VideoCapture(0);
	#else:
		#(grabbed , frame) = cv2.VideoCapture(args['video']);

	
	


	
