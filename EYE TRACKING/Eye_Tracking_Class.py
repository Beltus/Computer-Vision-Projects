import cv2;

class eyeTracker:

	def __init__(self , eyeCascadePath , faceCascadePath):
		self.eyeCascadePath = eyeCascadePath;
		self.faceCascadePath = faceCascadePath;
		self.eyeClassifier = cv2.CascadeClassifier(eyeCascadePath);
		self.faceClassifier = cv2.CascadeClassifier(self.faceCascadePath); 


	def track(self , image ):
		
		faceRects = self.faceClassifier.detectMultiScale(image , scaleFactor = 1.1 , minNeighbors = 3 , minSize = (30,30)); #returns list of turple each with 4 elements,the x,y cordinate detected face and width, height #of the face	
		rects = [];
		
		for (fx , fy , fw, fh) in faceRects:

			faceROI = image[fx:fx + fw , fy: fy + fh]; # save the image region containing face

			rects.append((fx , fy , fx + fw , fy + fh)); # save the coordinates of the bounding box around face

			eyeRects = self.eyeClassifier.detectMultiScale(faceROI , scaleFactor = 1.3 , minNeighbors = 25, minSize = (15,15))
			
			for (ex , ey , ew ,eh) in eyeRects:
				
				rects.append((ex + fx , ey + fy , ex + fx + ew , ey + fy + eh)); # cordinates of the eye bounding box in the image
		return rects
				
