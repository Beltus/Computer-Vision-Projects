
# Eye and Face Tracking
import argparse;
from Eye_Tracking_Class import eyeTracker; # import local class eyeTracker from Eye_Tracking_Class.py file
from Detector import Detector; # Import Detector class from Detector.py file
import cv2; 
import imutils ;

parser = argparse.ArgumentParser();
parser.add_argument('-f' , '--face' , required = True , type = str  , help = 'path to face cascade');
parser.add_argument('-e' , '--eye' , required = True , type = str  , help = 'path to eye cascade');
parser.add_argument('-v' , '--video' ,  type = str  , help = 'path to video file');

args = vars(parser.parse_args()); # get the entered command arguments

camera = cv2.VideoCapture(args['video']); # Get video from entered file path in command line

tracker = eyeTracker(args['face'] , args['eye']) # initialize eye tracker object

faceTracker = Detector(args['face']); # object for the face detector.

if not args.get('video' , False):
	camera = cv2.VideoCapture(0); # Read feed from webcam

while True:
	(grabbed  , frame) = camera.read(); # Converts video to series of frames
	if args.get(args['video']) and not grabbed: # video is ended
		break;

	#frame = imutils.resize(frame , width = 400) 
	gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY);

	faceRects = faceTracker.detect(gray , scaleFactor = 1.15 , minNeighbors = 10 , minSize = 3) #  detects face in video

	eyeRects = tracker.track(gray); # detects the eyes and returns coordinates of the bounding boxes around the eye
	
	for faceRect in faceRects: # for each detected face draw rectangle around it.
		faceRect = cv2.rectangle(frame , (faceRect[0] , faceRect[1] ) , (faceRect[0] + faceRect[2] , faceRect[1] + faceRect[2])  , (255, 0 , 0) , 4)

	for eyeRect in eyeRects: # For each detected eye draw a bounding box around
		eyes = cv2.rectangle(frame , (eyeRect[0], eyeRect[1]) , (eyeRect[2] , eyeRect[3]) , (0 , 255 , 0) , 2); # 
	
		cv2.imshow('Tracking',  eyes) ; 

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break;
	
camera.release();
cv2.destroyAllWindows();
	
#command to run the program	
#python Eye_Tracking_Video.py --face /home/beltus/image/frontalFace10/haarcascade_frontalface_default.xml --eye /home/beltus/image/frontalEyes35x16XML/haarcascade_eye.xml
