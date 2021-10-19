from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import urllib.request
import cv2
import numpy as np
import imutils

def detect_and_predict_mask(frame, faceNet, maskNet):
    #grab the dimensions of the frame and constructing a blob
    (height,width)  = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,1.0,(224,224), (104.0,177.0,123.0))

    #passing the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    #initialize our list of faces , their corresponding locations, and the list of predictions from the face 
    faces = []
    locs = []
    preds = []

    #loop over the detections
    for i in range(0,detections.shape[2]):
        #extract the probability associated with the detection
        confidence = detections[0,0,i,2]

        #filter out weak detections by ensuring the confidence is greater than the minimum confidence threshold
        if confidence>0.5:
            #computer the (x,y) coordinates fo the bounding box for the object
            box = detections[0,0,i,3:7] * np.array([width, height, width, height])
            (startX , startY, endX, endY) = box.astype("int")

            #ensure the bounding box fall withing the dimensions of the frame
            (startX , startY) = (max(0,startX),max(0,startY))
            (endX , endY) = (min(width - 1 , endX),min(height - 1 , endY))

            #extract the face ROI , convert it from BGR to RGB channel ordering and resizing it to 224x224 and preprocessing
            face = frame[startY:endY , startX:endX]
            face = cv2.cvtColor(face , cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224,224))
            face = img_to_array(face)
            face = preprocess_input(face)

            #add the face and bouding boxes to their respective lists
            faces.append(face)
            locs.append((startX,startY,endX,endY))

    #only make a prediction if at least one face was detected
    if(len(faces) > 0):
        #for faster inference we'll make a batch predictions on all faces at the same time rather than one-by-one predictions in the above for loop
        faces = np.array(faces , dtype="float32")
        preds = maskNet.predict(faces , batch_size = 32)

    #return a 2-tuple of the face locations and their corresponding locations
    return (locs,preds)

def run():
    prototxtPath = r"E:\P R O D I G Y\A C A D E M I C\C O D E\DEEP LEARNING\Face Mask Detection\INDIAN\Face-Mask-Detection\face_detector\deploy.prototxt"
    weightsPath = r"E:\P R O D I G Y\A C A D E M I C\C O D E\DEEP LEARNING\Face Mask Detection\INDIAN\Face-Mask-Detection\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    #Loading my model
    maskNet = load_model("face_mask_detector.model")


    #initialize the video Stream
    print("[INFO] Initializing Video Stream")
    url='http://192.168.50.13/cam-hi.jpg'       #URL to stream from ESP32-CAM
    

    #looping over the frames from the video stream
    while True:
        imgResp=urllib.request.urlopen(url)
        imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img=cv2.imdecode(imgNp,-1)

        img = imutils.resize(img,width=400)

        #detect faces in the fram and determine if they are wearing a face mask or not
        (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)

        #loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs,preds):
            #unpack the bounding box and predictions
            (startX , startY, endX , endY) = box
            (mask , withoutMask) = pred

            #determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask  else "No Mask"
            color = (0,255,0) if label == "Mask" else (0,0,255)

            #pass this to the arduino
            label_binary = label
            
            #include the probability in the label
            label = "{} : {:.2f}%".format(label, max(mask, withoutMask) * 100)

            #display the label and bounding box rectangle on the output frame
            cv2.putText(img,label , (startX , startY-10), cv2.FONT_HERSHEY_SIMPLEX , 0.45, color , 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        #show the output Frame
        cv2.imshow("Frame", img)
        key = cv2.waitKey(1) & 0xFF

        #if we press Q, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            return label_binary

    #cleanUP
    cv2.destroyAllWindows()

def stop():
    exit()

# def getVideofromWebCam():
#     cap = cv2.VideoCapture()
#     cap.open(0, cv2.CAP_DSHOW)
#     # print(cap)
#     while(True):
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#         cv2.imshow('Input', frame)

#         c = cv2.waitKey(1)
#         if c == 'q':
#             break

# getVideofromWebCam()
# cv2.destroyAllWindows()


run()
