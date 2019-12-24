#command to run the application
#python classify.py --image /home/beltus/image/Computer_Vision/Plant_Classification/dataset/images --mask /home/beltus/image/Computer_Vision/Plant_Classification/dataset/masks


from rgbhistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder # encodes class labels associated with each flower species
from sklearn.ensemble import RandomForestClassifier #classifier to be used for classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import cv2
import glob

#initialize the Argument parser object
parser = argparse.ArgumentParser()

#create command line arguments
parser.add_argument('-i' , '--image' , required = True,  help = 'Path to image dataset') #points to image dataset directory
parser.add_argument('-m'  , '--mask' , required = True , help = 'Path to image mask') 

#returns the argument values in the form of a dictionary. {'image' : 'path/image/g.jpg'}
args = vars(parser.parse_args())

#get sorted image  and mask paths
imagePaths = sorted(glob.glob(args['image'] + '/*.png')); # glob gets paths of images and its saved as a sorted list
maskPaths = sorted(glob.glob(args['mask'] + '/*.png'))

# initialize the data matrix
data = []
target  = [] ## initialize the list of class labels(species of flowers)

#initialize our image RGB color histogram descriptor with 8bins per channel 
desc = RGBHistogram([8,8,8])

#loop through all the paths of image and mask
for (imagePath, maskPath) in zip(imagePaths , maskPaths):

	image = cv2.imread(imagePath);
	mask = cv2.imread(maskPath);
	
	#convert mask to grayscale
	mask = cv2.cvtColor(mask , cv2.COLOR_BGR2GRAY);

	#Extract features from the image region specified by the mask using the RGB Histogram descriptor
	features = desc.describe(image , mask); # returns an  8 x 8 x 8 = 512 dimensional feature vector xterizing the image
 
	
	# append features to data list
	data.append(features)
	#Flower class name is extracted from the image's path e.g 'image_daisy_007' , return daisy
	target.append(imagePath.split('_')[-2]) #appends flower class to the target list

#get unique flower names-class names from the target list 
targetSpecies = np.unique(target) 

#Initialize the encoder
encoder = LabelEncoder()

print(len(data))
#transform Flower species names from string to intergers corresponding to the unique data points or species
targets = encoder.fit_transform(target)

#Spit the Data and target to Train and test split/ Use 30% of the data for testing
#(trainData , testData , trainTarget , testTarget) = train_test_split(data , target , test_size  = 0.7 , train_size = 0.3, random_state = 42) # random_state is used so as to be able to reproduce the 																		#ressults later

(trainData, testData, trainTarget, testTarget) = train_test_split(data, targets, test_size = 0.3, random_state = 42)

#initialize the RandomForest classifier with 25 decision trees
model = RandomForestClassifier(n_estimators = 25 , random_state = 84) ;#random_state is used so as to be able to reproduce the ressults later

# training the model
model.fit(trainData , trainTarget)

#predict Flower species
predict = model.predict(testData)

#Classification_report compares the species predictions to the true targets and prints out the accurary report for individual labels and also overall performance of the model
print(classification_report(testTarget , predict , target_names = targetSpecies))


######## FURTHER EXAMINATION OF THE MODEL OUTPUT AND PERFORMANCE

#randomly pick 10 image paths one by one from the data and test with model
for i in np.random.choice(np.arange(0 , len(imagePaths)), 10): ##np.arange(0 , n) returns an array of integers from 0 to n 
								##np.random.choice(array , n) takes array of inputs and outputs an array containing n number of random elements in the array
	image = cv2.imread(imagePaths[i])
	mask = cv2.imread(maskPaths[i])
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	
	#extract features from the image
	features = desc.describe(image  , mask)

	#Predict Flower species
	predict = model.predict([features])
	
	# Gets the Names of the Flower Species using the LabelEncoder.inverse_transform() fucntion
	flower = encoder.inverse_transform(predict)[0]

	# Print Results
	#print(imagePath)
	print('To the Best of Knowledge, this is %s', (flower.upper()))

	cv2.imshow('Image' , image)
	cv2.waitKey(0)







