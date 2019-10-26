import cv2;

# Face Detector Class
class Detector:
    
    # Method to create an instance of the Face Detector Class.
    
    def __init__ (self, faceCascadePath):
        
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath); # Initialize instances to CascadeClassifier
    
    #def get_classifier_path(self):
        #return self.path;
    
    #def set_classifier_path(self, new_path):
       # self.path = new_path;   
    
    def detect(self, image, scaleFactor = 1.03, minNeighbors = 4, minSize = 30):
        # Method encapsulating the detectmultiscale method of CascadeClassifier as it has same input parameter
        rects = self.faceCascade.detectMultiScale(image , scaleFactor , minNeighbors , minSize); # Detection method
                
        return rects; # Returns a list of turple. Each turple has 4 elements.( starting x,y coordinate of detected face and width and height of the detected face.
        
        
