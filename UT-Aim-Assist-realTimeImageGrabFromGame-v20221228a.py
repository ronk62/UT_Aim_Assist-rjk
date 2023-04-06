

'''
ref:



https://github.com/niconielsen32/NeuralNetworks/blob/main/MobileNet.ipynb

https://www.youtube.com/watch?v=oBqQWuhMTsY


https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API


'''

# change log
#
# date          who         what
# 12/28/2022    Ron King    - initial testing with UT game captured frames; using keras and tensorflow with GPU
#                             to increase FPS
#

import ctypes
import time
import win32com.client as comclt
import win32api, win32con, win32gui, win32ui
import keyboard
import numpy as np
import pyautogui
import math
from PIL import ImageGrab, Image, ImageDraw
#import cv2 as cv
from tensorflow import keras
from keras.preprocessing import image
from keras.applications import imagenet_utils
#from sklearn.metrics import confusion_matrix
#from IPython.display import Image


def getClassLabel(class_id, classes):
    for key,value in classes.items():
        if class_id == key:
            return value


### old cv method
#cvNet = cv.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")



SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def click():
    # win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
    time.sleep(0.2)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)

def capture_window():
    # 800x600 windowed mode
    image =  np.array(ImageGrab.grab(bbox=(0,40,800,600)))
    return image

# directx scan codes http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
wsh= comclt.Dispatch("WScript.Shell")
wsh.AppActivate("Unreal Tournament") # select another application
time.sleep(3)

w, h = 10, 10
shape = [(0, 0), (w, h)]

last_time = time.time()

while True:
    # creating new Image object
    image = capture_window()
    
    #img = cv.imread("example.jpg")             # static image load for testing
    #img = cv.imread("example-bluePlayer.PNG")  # static image load for testing
    rows = image.shape[0]
    cols = image.shape[1]
    
    ### old cv method
    #cvNet.setInput(cv.dnn.blobFromImage(image, size=(800, 600), swapRB=True, crop=False))
    #cvOut = cvNet.forward()


    ### old cv method
    # for detection in cvOut[0,0,:,:]:
    #     score = float(detection[2])
    #     if score > 0.3:
    #         class_id = detection[1]
    #         class_label = getClassLabel(class_id,COCO_labels)
    #         left = detection[3] * cols
    #         top = detection[4] * rows
    #         right = detection[5] * cols
    #         bottom = detection[6] * rows
    #         # print("")
    #         cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
    #         cv.putText(image,class_label ,(int(left),int(top)+25),cv.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),3,cv.LINE_AA)
    #         if type(class_label) == str:
    #             print(str(str(class_id) + " " + str(detection[2])  + " " + class_label))

    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    

    cv.imshow('window',cv.cvtColor(image, cv.COLOR_BGR2RGB))
    if cv.waitKey(25) & 0xFF == ord('q'):
         cv.destroyAllWindows()
         break

