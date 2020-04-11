import cv2
import numpy as np
#import time

def main():
    
    ##Set color Threshold
    #color_lower = np.array([0 , 50 , 160] , dtype = 'uint8'); #sets lower threshold for blue color
    color_lower = np.array([150 , 50 , 0] , dtype = 'uint8'); #sets lower threshold for blue color


    #color_upper = np.array([100, 150 , 255] , dtype = 'uint8'); #sets upper threshold for blue color
    color_upper = np.array([255, 150 , 100] , dtype = 'uint8'); #sets upper threshold for blue color
    
    #capture the video from webcam
    capture = cv2.VideoCapture('camera.avi')
    
    
    #get the frame width and height and convert to integers
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width//2 ,frame_height//2))
                          
                          
    while True:

        ret, frame = capture.read()

        if not ret:
            break
        
        
        #resize frame to half its width and height
        resize_frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA) #resize image to width of 100 and height of 100 respectively
        
        #threshold based on color
        color_threshold = cv2.inRange(resize_frame , color_lower , color_upper); #cv2.inRange() returns a binary image with pixels that fall with the color range set to white(255) and the rest set to black(0)

        blur = cv2.GaussianBlur(color_threshold , (3, 3) , 0); # to reduce noise and increase detection of the tracking of our object

        (cnts , _)  = cv2.findContours(blur.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE); #cv2.findContours is used to get the contours corresponding to the object.
        #Returns a list of contours each corresponding to detected object color boundaries
        # copy of image is made as this function is destructive to numpy arrays

        if len(cnts) > 0:
            cnt = sorted(cnts , key = cv2.contourArea , reverse = True)[0] # sorts the list of contours from largest to smallest and returns the largest based on area(cv2.contourArea) which is assumed 										      #	to be the object interest

            bbox = cv2.minAreaRect(cnt);#cv2.minAreaRect() computes the mininimum bounding box around the contour,
            rects = np.int0(cv2.boxPoints(bbox)) # BoxPoints()converts these bounding box to list of points # 
                #Note without the 0 in **np.int0** the error TypeError: only size-1 arrays can be converted Python scalars

            cv2.drawContours(resize_frame , [rects] , -1 , (0, 255, 0) , 2);# Draw a bounding around object. frame is image to draw box , -1 draws all contours, (0,255,0) color of bounding box, 2 is line thickness
        
        # Write the frame into the file 'output.avi'
        out.write(resize_frame)
        
	#display the video with object being tracked
        cv2.imshow('Tracking In Progress'  , resize_frame) ;
	
	#display the binary thresholded image of the object
        cv2.imshow('Threshold' , color_threshold);

        #time.sleep(0.0025);

        key = cv2.waitKey(30) & 0xff

        if key == 27:
            break;

    capture.release();
    out.release();
    cv2.destroyAllWindows()


# Run the program
if __name__ == '__main__':
    main()
