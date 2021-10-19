import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_HEIGHT = 224
IMG_WIDTH = 224

folder = 'basic_testing'
samples = []
results = []

model = tf.keras.models.load_model('face_mask_detector.model')

for img in os.listdir(folder):
    img_path = os.path.join(folder, img)
    img_arr = load_img(img_path, target_size=(IMG_HEIGHT,IMG_WIDTH))        
    img_arr = img_to_array(img_arr)
    img_arr = preprocess_input(img_arr) 
    samples.append(img_arr)

X = np.array(samples)

predictions = model.predict(X)
result_arr =np.argmax(predictions, axis = 1)
print(result_arr)

for val in result_arr:
    if (val == 0):
        results.append('with_mask')
    else :
        results.append('without_mask')

print(results)