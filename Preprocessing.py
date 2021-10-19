import numpy as np
import cv2
import os
import random 
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

DATASET_DIRECTORY = r"E:\P R O D I G Y\A C A D E M I C\C O D E\DEEP LEARNING\Face Mask Detection\dataset"
PREPROCESSED_DIRECTORY = r"E:\P R O D I G Y\A C A D E M I C\C O D E\DEEP LEARNING\Face Mask Detection\preprocessed"
CATEGORIES = ['with_mask', 'without_mask']

IMG_HEIGHT = 224
IMG_WIDTH = 224

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATASET_DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        # img_arr = cv2.imread(img_path)
        # img_arr = cv2.resize(img_arr,(IMG_HEIGHT,IMG_WIDTH))
        img_arr = load_img(img_path, target_size=(IMG_HEIGHT,IMG_WIDTH))         #For some reason, CV2 imread and resize is throwing an error at some particular images in the dataset
        img_arr = img_to_array(img_arr)
        img_arr = preprocess_input(img_arr)                                      #this is for MobileNet_v2

        data.append(img_arr)
        labels.append(category)

#Converting the labels into 0 or 1 
binarizer = LabelBinarizer()
labels = binarizer.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)
print(len(data))
# Spliting Train Data and Test Data
(trainX, testX , trainY, testY)  = train_test_split(data, labels, test_size=0.20 , stratify=labels, random_state=83)

# pickle.dump(trainX , open(os.path.join(PREPROCESSED_DIRECTORY, 'trainX.pkl'),'wb'))
# pickle.dump(testX , open(os.path.join(PREPROCESSED_DIRECTORY, 'testX.pkl'),'wb'))
# pickle.dump(trainY , open(os.path.join(PREPROCESSED_DIRECTORY, 'trainY.pkl'),'wb'))
# pickle.dump(testY , open(os.path.join(PREPROCESSED_DIRECTORY, 'testY.pkl'),'wb'))