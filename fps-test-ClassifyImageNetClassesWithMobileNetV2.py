# ref. https://keras.io/api/applications/#usage-examples-for-image-classification-models

import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
import keras.utils as image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import time

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

last_time = time.time()

while True:
    # img_path = 'example-bluePlayer.PNG'
    img_path = 'example-livingRoom.jpg'
    last_time_load = time.time()
    img = image.load_img(img_path, target_size=(224, 224))
    print('FPS_load = ', 1/(time.time()-last_time_load))
    # last_time_img_to_array = time.time()
    x = image.img_to_array(img)
    # print('FPS_img_to_array = ', 1/(time.time()-last_time_img_to_array))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('FPS_load_preprocess = ', 1/(time.time()-last_time))
    last_time_predict = time.time()
    preds = model.predict(x)
    print('FPS_predict = ', 1/(time.time()-last_time_predict))
    ##for testing
    #print("preds:", preds)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    #print('Predicted:', decode_predictions(preds, top=3)[0])
    print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

    #print('loop took {} seconds'.format(time.time()-last_time))
    print('FPS_total = ', 1/(time.time()-last_time))
    last_time = time.time()
