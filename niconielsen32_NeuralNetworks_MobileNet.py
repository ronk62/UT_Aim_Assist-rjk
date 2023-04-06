# converted orig niconielsen32 .ipynb file to .py
# (ref https://github.com/niconielsen32/NeuralNetworks/blob/main/MobileNet.ipynb)


import numpy as np
import tensorflow as tf
from tensorflow import keras
# fixed orig line below
# https://stackoverflow.com/questions/72383347/how-to-fix-it-attributeerror-module-keras-preprocessing-image-has-no-attribu
import keras.utils as image
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
from IPython.display import Image
import time


###############

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

###############

## Directory to dataset in drive
# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)

###############

model = tf.keras.applications.MobileNet()

###############

model.summary()

###############

def prepare_image(file):
    #img_path = '/content/gdrive/MyDrive/images/'       # not using gdrive; using logcal images instead
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

###############

Image(filename='example-livingRoom.jpg', width=640,height=480) 

###############

preprocessed_image = prepare_image('example-livingRoom.jpg')

last_time = time.time()

while True:
    last_time_pred1 = time.time()
    predictions = model.predict(preprocessed_image)
    print('FPS_pred1 = ', 1/(time.time()-last_time_pred1 + 0.000000000000001))
    # print("predictions1", predictions)
    ###############

    last_time_results1 = time.time()
    results = imagenet_utils.decode_predictions(predictions)
    print('FPS_results1 = ', 1/(time.time()-last_time_results1 + 0.000000000000001))
    # results
    # print("results", results)

    ###############

    Image(filename='example-bluePlayer.PNG', width=393,height=311) 

    ###############

    preprocessed_image = prepare_image('example-bluePlayer.PNG')

    last_time_pred2 = time.time()
    predictions = model.predict(preprocessed_image)
    print('FPS_pred2 = ', 1/(time.time()-last_time_pred2 + 0.000000000000001))
    # print("predictions2", predictions)

    # last_time_results2 = time.time()
    # results = imagenet_utils.decode_predictions(predictions)
    # print('FPS_load = ', 1/(time.time()-last_time_results2 + 0.000000000000001))
    ## results
    ## print("results", results)

    ###############

