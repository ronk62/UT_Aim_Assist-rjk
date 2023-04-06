

'''
ref:

https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

https://www.youtube.com/watch?v=BFDkTTeBgCQ&list=WL&index=2&t=17s

https://www.fypsolutions.com/opencv-python/ssdlite-mobilenet-object-detection-with-opencv-dnn/


https://github.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects


https://cocodataset.org/#home


# 3/26/2023 ref. https://keras.io/api/applications/#usage-examples-for-image-classification-models

'''

# change log
#
# date          who         what
# 12/27/2022    Ron King    - initial testing with UT game captured frames; started with file, "tf_cv_dnn_primer-20211030b.py"
#  3/26/2023    Ron King    - updated v20221227 version with 'tf.keras.applications.MobileNetV2'

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

import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
import keras.utils as image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.keras.applications.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)


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


'''
COCO_labels = { 0: 'background',
    1: 'person' }
'''

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
    # game in 640x400 windowed mode
    image =  np.array(ImageGrab.grab(bbox=(0,0,672,672)))
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
    img = capture_window()

    # img = np.resize(img,(224,224,3))
    img = img[::3, ::3]
    
    rows = img.shape[0]
    cols = img.shape[1]
    ## for testing
    print("rows", rows)
    print("cols", cols)

    # img_path = 'example-bluePlayer.PNG'
    # img_path = 'example-livingRoom.jpg'
    # img = image.load_img(img_path, target_size=(224, 224))
    ### img_data = np.random.random(size=(224, 224, 3))
    ### img = tf.keras.utils.array_to_img(img_data)
    x = image.img_to_array(img)
    ## for testing
    print(x.dtype)
    print(x.shape)
    x = np.expand_dims(x, axis=0)
    ## for testing
    print(x.dtype)
    print(x.shape)
    x = preprocess_input(x)
    preds = model.predict(x)
    ##for testing
    #print("preds:", preds)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    #print('Predicted:', decode_predictions(preds, top=3)[0])
    print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

    #print('loop took {} seconds'.format(time.time()-last_time))
    print('FPS = ', 1/(time.time() - last_time))
    last_time = time.time()



'''
    # for detection in cvOut[0,0,:,:]:
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            class_id = detection[1]
            class_label = getClassLabel(class_id,COCO_labels)
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            # print("")
            cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            cv.putText(image,class_label ,(int(left),int(top)+25),cv.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),3,cv.LINE_AA)
            if type(class_label) == str:
                print(str(str(class_id) + " " + str(detection[2])  + " " + class_label))

    # print('loop took {} seconds'.format(time.time()-last_time))
    print('FPS = ', 1/(time.time()-last_time))
    last_time = time.time()
    

    cv.imshow('window',cv.cvtColor(image, cv.COLOR_BGR2RGB))
    if cv.waitKey(25) & 0xFF == ord('q'):
         cv.destroyAllWindows()
         break
'''
