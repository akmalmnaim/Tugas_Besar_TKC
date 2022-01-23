from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Concatenate
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time



class FeatureExtractor8192:
    def __init__(self):
        vgg_net = VGG19(include_top=False, input_tensor=Input(shape=(128, 128, 3)))
        
        headModel2 = vgg_net.output
        headModel2 = GlobalAveragePooling2D()(headModel2)
        headModel2 = Flatten(name="flatten2")(headModel2)
        
        #headModel2 = Dropout(0.2)(headModel2)
        headModel2 = Dense(4096, activation='relu')(headModel2)
        self.model = Model(inputs=vgg_net.input, outputs=headModel2)
        

    def extract(self, img):
        img = img.resize((128,128)).convert("RGB")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
       