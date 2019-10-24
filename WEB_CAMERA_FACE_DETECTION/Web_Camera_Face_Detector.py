import cv2
import argparse
from Detector import Detector;



parser = argparse.ArgumentParser();

parser.add_argument('-f' , '--face', required = True , help ='Path to the Cascade XML File');
parser.add_argument('-v', '--video', help ='Optional path to video');

args = vars(parser.parse_args()); #gets and stores all commandline commands in a dict format

if not args.get('video' , False): #Check if path to video was inputted by the user
	camera = cv2.VideoCapture(0); # read the video feed from webcam

else:
	camera = cv2.VideoCapture(args['video']); # Get video supplied by user.

faceObj = Detector(args['face']); #Creates the face detector object and initializes its path to user supplied path

while True:
	(grabbed  , frame) = camera.read(); #reads each frame of the video, grabbed is a boolean indicating if frame was read or not
	
	if args.get('video') and not grabbed: # If video is read from a file and frame was not grabbed then video is over and end
		break
	## remember to resize each frame for fast processing
	#width = 300;
	#height = 300;

	#frame = cv2.resize(frame , (width , height), interpolation = cv2.INTER_AREA);
	gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY); # image to gray
	
	rects = faceObj.detect(gray, scaleFactor = 1.2 , minNeighbors =  4, minSize = 30);
	
	frameCopy = frame.copy(); # Make copy just in case we need the original images for some preprocessing

	for (x,y,w,h) in rects:
		cv2.rectangle(frameCopy, (x,y) , (x+w , y+h) , (0,255,0), 2);

	cv2.imshow('FACE', frameCopy);
	if cv2.waitKey(1) & 0xFF == 'q':
		break

camera.release();
cv2.detroyAllWindows();
	



