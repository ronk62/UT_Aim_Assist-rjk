# ref. https://keras.io/api/applications/#usage-examples-for-image-classification-models

import tensorflow as tf
import keras.utils as image
import numpy as np
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


##### MobileNet (V1) ####
print("Using MobileNet (V1)")
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
model = MobileNet(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

'''
##

##### MobileNetV2 ####
print("Using MobileNetV2")
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

model = MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

##

##### MobileNetV3Large ####
print("Using MobileNetV3Large")
from keras.applications.mobilenet_v3 import MobileNetV3Large
from keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
model = MobileNetV3Large(
    input_shape=None,
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
)


##

##### MobileNetV3Small ####
print("Using MobileNetV3Small")
from keras.applications.mobilenet_v3 import MobileNetV3Small
from keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
model = MobileNetV3Small(
    input_shape=None,
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
)


##

##### MobileNetV3Small minimalistic ####
print("Using MobileNetV3 minimalistic")
from keras.applications.mobilenet_v3 import MobileNetV3Small
from keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
model = MobileNetV3Small(
    input_shape=None,
    alpha=1.0,
    minimalistic=True,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
)

'''

last_time = time.time()

while True:
    # img_path = 'example-bluePlayer.PNG'
    img_path = 'example-livingRoom.jpg'
    last_time_load = time.time()
    img = image.load_img(img_path, target_size=(224, 224))
    print('FPS_load = ', 1/(time.time()-last_time_load + 0.000000000000001))
    # last_time_img_to_array = time.time()
    x = image.img_to_array(img)
    # print('FPS_img_to_array = ', 1/(time.time()-last_time_img_to_array + 0.000000000000001))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('FPS_load_preprocess = ', 1/(time.time()-last_time + 0.000000000000001))
    last_time_predict = time.time()
    ### see https://stackoverflow.com/questions/60837962/confusion-about-keras-model-call-vs-call-vs-predict-methods#:~:text=The%20difference%20between%20call(),()%20contrary%20to%20predict()%20.
    ## Ron K, 4/1/2023 note: I see small performance gain (3 FPS) using model() vs model.predict()
    # preds = model.predict(x)        # best for batches of data
    preds = model(x).numpy()      # best for small data sample
    print('FPS_predict = ', 1/(time.time()-last_time_predict + 0.000000000000001))
    #for testing
    # print("preds:", preds)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

    #print('loop took {} seconds'.format(time.time()-last_time))
    print('FPS_total = ', 1/(time.time()-last_time + 0.000000000000001))

    last_time = time.time()
