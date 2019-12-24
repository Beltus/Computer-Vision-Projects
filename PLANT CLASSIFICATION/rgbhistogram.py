# We use the RGB Histogram for Flower classification because the colors of the flowers can be used as a feature to distinguish the different Flower species

#import opencv

# 
import cv2

#Defining a class to extract features from the flower images. Features will then be used as input to model to classify the image

class RGBHistogram:

#initialize class constructor

	def __init__(self , bins):
		self.bins = bins;

#method to extract the features from image

	def describe(self, image , mask = None):
		# input image: image from which the RGB color histogram is to be extracted
		# mask: optional input which contains only the relevant image portion that makes up the flowerand and doesnt include the background

		hist = cv2.calcHist([image] , [0,1,2] , mask, self.bins, [0 , 256 , 0 , 256 , 0 , 256]) # cv2.calcHist([images] , [channel_dim] , mask , histSize , range)
										 #channel_dim = channel to calculate hist over, mask = calculate hist of particular section of image,
								#histSize = number of bins to be used-256 means all pixel intensities are used and plotted, range - range of intensities we want to measure
		

		hist = cv2.normalize(hist, hist) #normalize the histogram features 

		return hist.flatten() # return the extracted features as a feature vector
