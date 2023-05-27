# no she-bang in Windows

# ref https://www.youtube.com/watch?v=2yQqg_mXuPQ&t=654s
# initial setup 4/9/2023

import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
from PIL import ImageGrab, Image, ImageDraw
from Keyboard_And_Mouse_Controls import *
# import threading

# x_is_pressed = 0

# def keyboardInput(name):
#     while (True):
#         global x_is_pressed  # Declare x_is_pressed as global to force use of global 'x_is_pressed' in this function/thread
#         x_is_pressed = keyboard.is_pressed(45)
#         # print("x_is_pressed = ", x_is_pressed)
#         # time.sleep(0.5)
#         time.sleep(0.05)

np.random.seed(123)

class Detector:
    def __init__(self):
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

        # Colors list
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

        print(len(self.classesList), len(self.colorList))

    def downloadModel(self, modelURL):

        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]

        ## for testing/debugging
        # print(fileName)
        # print(self.modelName)

        self.cacheDir = "./pretrained_models"

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=fileName, origin=modelURL, cache_dir=self.cacheDir, 
                 cache_subdir="checkpoints", extract=True)
    
    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))

        print("Model " + self.modelName + " loaded successfully")
    
    def createBoundigBox(self, image, threshold = 0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        ## perf testing, next 1 line
        detStartTime = time.time()
        detections = self.model(inputTensor)
        ## perf testing, next 3 line3
        detEndTime = time.time()
        detFPS = 1/((detEndTime + 0.000000000001) - detStartTime)
        print("detFPS = ", detFPS)

        bboxs = []
        classIndexes = []
        classScores = []
        targetList = []

        bboxs_all = detections['detection_boxes'][0].numpy()
        classIndexes_all = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores_all = detections['detection_scores'][0].numpy()

        for i in range(len(bboxs_all)):
            if classIndexes_all[i] == 1:
                bboxs.append(bboxs_all[i])
                classIndexes.append(classIndexes_all[i])
                classScores.append(classScores_all[i])
        
        if bboxs == []:
            return image, targetList

        imH, imW, imC = image.shape

        # bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
        #                                        iou_threshold=threshold, score_threshold=threshold)
        
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
                                               iou_threshold=0.1, score_threshold=0.4)
        
        ## for testing
        # print(bboxIdx)

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classesList[classIndex]
                classColor = self.colorList[classIndex]

                displayText = '{}: {}%'.format(classLabelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)

                bbox_height = (ymax - ymin)
                bbox_ycenter = ymin + bbox_height/2
                
                bbox_width = (xmax - xmin)
                bbox_xcenter = xmin + bbox_width/2

                if bbox_height > 1.3 * bbox_width:
                    if bbox_ycenter > 300 and bbox_ycenter < 600:
                        if bbox_xcenter > 650 and bbox_ycenter < 950:
                            xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                            cv2.putText(image, displayText, (xmin, ymin -10 ), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                            target = (int(ymin + 0.25 * bbox_height), int(bbox_xcenter))

                            targetList.append(target)

        return image, targetList

    
    
    def predictImage(self, imagePath, threshold = 0.5):
        image = cv2.imread(imagePath)

        bboxImage, targetList = self.createBoundigBox(image, threshold)

        cv2.imwrite(self.modelName + ".jpg", bboxImage)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def predictVideo(self, videoPath, threshold = 0.5):
        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened() == False):
            print("Error opening file...")
            return
        
        (success, image) = cap.read()

        startTime = 0
        
        while success:
            currentTime = time.time()

            fps = 1/(currentTime - startTime)
            startTime = currentTime

            bboxImage, targetList = self.createBoundigBox(image, threshold)

            cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
            cv2.imshow("Result", bboxImage)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            (success, image) = cap.read()
        cv2.destroyAllWindows()
    

    def capture_window(self):
        # UT game in 1278 x 686 windowed mode
        image =  np.array(ImageGrab.grab(bbox=(0,0,1600,900)))
        return image


    def predictImgCap(self, threshold = 0.5):
        image = self.capture_window()

        startTime = 0
        
        while True:
            currentTime = time.time()

            # fix colors
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            fps = 1/(currentTime - startTime)
            startTime = currentTime

            bboxImage, targetList = self.createBoundigBox(image, threshold)

            # aim and shoot at targets
            for target in range(len(targetList)):
                ## next line for testing
                print("target = ", targetList[target])

                if keyboard.is_pressed(45):
                    # # move mouse to point at target
                    # AimMouse(targetList[target])

                    # move mouse to point at target
                    AimMouseAlt(targetList[target])

                    # fire at target 3 times
                    click()
                    click()
                    click()

            cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
            cv2.imshow("Result", bboxImage)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            image = self.capture_window()
        cv2.destroyAllWindows()
    

