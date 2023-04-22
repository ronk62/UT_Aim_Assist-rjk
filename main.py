# no she-bang in Windows

# ref https://www.youtube.com/watch?v=2yQqg_mXuPQ&t=654s
# initial setup 4/9/2023

from UT_Aim_Assist import *

##############
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##############


### this first model works fine so far on my machine with static images, vid file, and screen_cap (~8 FPS)
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"

### WORKING BEST SO FAR WITH Unreal Tourn and screen_cap
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

### this model is sketchy, crashes VS Code with OOM unless you bounce VS Code before running
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"

### can't properly load or run this model - throws errors
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz"

classFile = "coco.names"
# imagePath = "test/example-livingRoom.jpg"
imagePath = "example-bluePlayer.PNG"
videoPath = "street.mp4"

threshold = 0.38


'''

NEXT UP...work next on

-- thresholding
-- filtering by class ("person" only)
-- eliminatimg bboxes that have more red than blue
--- (blue team is the ememy, so disqualify targets that are red)
-- elimiate bboxes that are horizontal rectangles since those are likely downed targets

'''

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
# detector.predictImage(imagePath, threshold)
# detector.predictVideo(videoPath, threshold)
detector.predictImgCap(threshold)