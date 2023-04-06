

'''
ref:

https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API




https://www.youtube.com/watch?v=BFDkTTeBgCQ&list=WL&index=2&t=17s

https://www.fypsolutions.com/opencv-python/ssdlite-mobilenet-object-detection-with-opencv-dnn/


https://github.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects


https://cocodataset.org/#home


'''

# change log
#
# date          who         what
# 12/27/2022    Ron King    - initial testing with UT game captured frames; started with file, "tf_cv_dnn_primer-20211030b.py"

import ctypes
import time
import win32com.client as comclt
import win32api, win32con, win32gui, win32ui
import keyboard
import numpy as np
import pyautogui
import math
from PIL import ImageGrab, Image, ImageDraw
import cv2 as cv


def getClassLabel(class_id, classes):
    for key,value in classes.items():
        if class_id == key:
            return value


COCO_labels = { 0: 'background',
    1: '"person"', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter',
    15: 'zebra', 16: 'bird', 17: 'cat', 18: 'dog',19: 'horse',20: 'sheep',21: 'cow',22: 'elephant',
    23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella',
    31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
    36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 
    67: 'dining table',70: 'toilet', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator',84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }


### both of the following work, but using the fully-qualified path is ugly, so keepin' in short
#cvNet = cv.dnn.readNetFromTensorflow("C:/Users/ronk6/OneDrive/Documents/PythonProjects/winLocalPython/UT_Aim_Assist-rjk/frozen_inference_graph.pb", "C:/Users/ronk6/OneDrive/Documents/PythonProjects/winLocalPython/UT_Aim_Assist-rjk/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
cvNet = cv.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")



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
    hwnd = win32gui.FindWindow(None, "Unreal Tournament")
    ## next two lines for testing
    #print("hwnd = ", hwnd)
    #time.sleep(12)
    window_rect = win32gui.GetWindowRect(hwnd)
    temp_rect = (window_rect[0]+8,window_rect[1]+25, window_rect[2]-8,window_rect[3]-8)
    image = np.array(ImageGrab.grab(temp_rect))
    # size = (int(image.size[0]/6),int(image.size[1]/6))
    # image = image.resize(size, Image.ANTIALIAS)
    # image.show()
    # print("stop")
    return image,temp_rect

# directx scan codes http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
wsh= comclt.Dispatch("WScript.Shell")
wsh.AppActivate("Unreal Tournament") # select another application
time.sleep(3)

w, h = 10, 10
shape = [(0, 0), (w, h)]

while True:
    # creating new Image object
    #img = Image.new("RGB", (w, h))         vestigial from HALO Aim Assist roots - remove later
    image, temp_rect = capture_window()
    #image = np.array(image)                perform this in the funtion

    #img = cv.imread("example.jpg")             # static image load for testing
    #img = cv.imread("example-bluePlayer.PNG")  # static image load for testing
    rows = image.shape[0]
    cols = image.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(image, size=(800, 640), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    ## testing
    #image.show()
    #print("sleeping for 20 seconds")
    #time.sleep(20)

    ## vestigial from HALO Aim Assist roots - remove later
    #mid_x = int(image.size[0] / 2)
    #mid_y = int(image.size[1] / 2)



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

    cv.imshow('image', image)
    cv.waitKey(1)

