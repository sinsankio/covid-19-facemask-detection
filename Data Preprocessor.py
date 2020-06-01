import cv2 as openCV, os
import numpy as np
from keras.utils import np_utils

root = "dataset"
directories = os.listdir(root)
labels = [i for i in range(len(directories))]

labelDict = dict(zip(directories, labels))

imgData = []
categories = []

for directory in directories:
    dirPath = os.path.join(root, directory)
    imgNames = os.listdir(dirPath)

    for name in imgNames:
        imagePath = os.path.join(dirPath, name)
        image = openCV.imread(imagePath)

        try:
            grayImage = openCV.cvtColor(image, openCV.COLOR_BGR2GRAY)
            resizedImage = openCV.resize(grayImage, (100, 100))
            imgData.append(resizedImage)
            categories.append(labelDict[directory])
        except Exception:
            print("Image Not Available ( PATH = " + imagePath + " )")

#converting pixel ranges of every image in to 0 and 1
imgData = np.array(imgData) / 255.0

#since CNN needs a 4-dimensional array of images we should convert imgData into 4-dimensionaly
imgData = np.reshape(imgData, (imgData.shape[0], 100, 100, 1)) #last arg = 1 representing that image is a gray one.

categories = np.array(categories)

#The last layer of CNN contains of 2 neurons representing the status of 'with mask' and 'without mask'
#Therefore we should convert labels in to categorical labels.
categoricalLabels = np_utils.to_categorical(categories)

np.save("image-data", imgData)
np.save("labels", categoricalLabels)

print("Done!")