from os import name
import os
import numpy as np
import pickle
import Preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.backend import shape
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.optimizers import adam_v2

###############################################################################
#Preprocessing Stuff
PREPROCESSED_DIRECTORY = r"E:\P R O D I G Y\A C A D E M I C\C O D E\DEEP LEARNING\Face Mask Detection\preprocessed"
trainX = pickle.load(open(os.path.join(PREPROCESSED_DIRECTORY, 'trainX.pkl'),'rb'))
trainY = pickle.load(open(os.path.join(PREPROCESSED_DIRECTORY, 'trainY.pkl'),'rb'))
testX  = pickle.load(open(os.path.join(PREPROCESSED_DIRECTORY, 'testX.pkl'),'rb'))
testY  = pickle.load(open(os.path.join(PREPROCESSED_DIRECTORY, 'testY.pkl'),'rb'))

#initial Learning rate, Epochs and Batch Size
INIT_LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

#Data Augmentation
aug = ImageDataGenerator(
    rotation_range= 20,
    zoom_range= 0.15,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    shear_range= 0.15,
    horizontal_flip= True,
    fill_mode="nearest"
)

baseModel = MobileNetV2(weights="imagenet", include_top=False , input_tensor=Input(shape=(224,224,3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128,activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2,activation="softmax")(headModel)

model = Model(inputs=baseModel.input , outputs=headModel)

#loop over all layers in the base model and freeze them so they will not be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

#compiling the model
print("[INFO] Compiling Model......")
opt = adam_v2.Adam(lr=INIT_LR , decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy" , optimizer=opt, metrics=["accuracy"])

#training the head of the model
print("[INFO] Training the Head of the Model......")
with tf.device('/gpu:0'):
    Head = model.fit(
        aug.flow(trainX,trainY, batch_size=BATCH_SIZE),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        validation_data= (testX,testY),
        validation_steps=len(testX) // BATCH_SIZE,
        epochs= EPOCHS
    )

#making predictions on the testing set
print("[INFO] Evaluating Network.....")
predictions = model.predict(testX, batch_size= BATCH_SIZE)

#for each image in the testing set, we need to find the index of the label with corresponding largest predicted probability
predictions = np.argmax(predictions , axis = 1)

#A nicely formatted classification report
print(classification_report(testY.argmax(axis=1),predictions, target_names=Preprocessing.binarizer.classes_))

#saving the model to disk
print("[INFO] Saving the Model.....")
model.save("face_mask_detector.model",save_format="h5")

#plotting the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N),Head.history["loss"] , label="train_loss")
plt.plot(np.arange(0,N),Head.history["val_loss"] , label="val_loss")
plt.plot(np.arange(0,N),Head.history["accuracy"] , label="train_acc")
plt.plot(np.arange(0,N),Head.history["val_accuracy"] , label="val_acc")
plt.title("training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("accuracy_Graph.png")