import application
import serial


arduino1 = serial.Serial(port='COM3', baudrate=9600, timeout=1.02) # Taking an instance of the Arduino Serial Port at 9600 baudrate

while True:
    isNear =  arduino1.readline()
    isNear = isNear.decode("utf-8")
    isNear = isNear[:len(isNear)-2]

    if(isNear == "YES"):
        print("[INFO]SOMEONE IS IN FRONT OF THE DOOR")
        isMask = application.run() # Running the Mask Detection Neural Network Model 
        print(isMask)
        if(isMask == "No Mask"):
            arduino1.write(bytes("0", 'utf-8')) # The Gate will close and the LED will turn OFF
            exit()
        else:
            arduino1.write(bytes("1", 'utf-8')) # The Gate will Open and the LED will turn ON
            exit()
    else:
        print("[WARNING]NO ONE IS IN FRONT OF THE GATE")
        application.stop()
