import os
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Concatenate
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from PIL import Image
from keras.applications.efficientnet import EfficientNetB7
from keras.applications.vgg19 import VGG19, preprocess_input
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from feature_extractor4096 import FeatureExtractor4096
from feature_extractor4096gap import FeatureExtractor4096gap
from feature_extractor8192 import FeatureExtractor8192
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import h5py

app = Flask(__name__)

def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)



class FeatureExtractor:
    def __init__(self):
        vgg_net = VGG19(include_top=False, input_tensor=Input(shape=(128, 128, 3)))
        
        headModel2 = vgg_net.output
        headModel2 = GlobalAveragePooling2D()(headModel2)
        headModel2 = Flatten(name="flatten2")(headModel2)
        
        #headModel2 = Dropout(0.2)(headModel2)
        headModel2 = Dense(4096, activation='relu')(headModel2)
        self.model = Model(inputs=vgg_net.input, outputs=headModel2)
        

    def extract_features(self, img_path):
        im = Image.open(img_path)
        X = preprocess(im,(128,128))
        X = reshape([X])
        
        feat = self.model.predict(X)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat

fe = FeatureExtractor()

features = []
img_paths = []
h5f = h5py.File('static/featureh5/featureCNN-vgg19-p.h5','r')

feats = h5f['feats'][:]
# print(feats)

imgNames = h5f['names'][:]
h5f.close()

@app.route("/", methods = ["GET", "POST"])


def index():
    


    if request.method == "POST":
        file = request.files["file"]
        file.save(os.path.join('static', 'temp.jpg'))
        open_file = Image.open(file)
        #run search
      
        queryVec = fe.extract_features(open_file)
        value = np.dot(queryVec, feats.T)
        rank_ID = np.argsort(value)[::-1]
        rank_score = value[rank_ID]
        maxres = 30
        #imlist = [str(imgNames[index]).split("'")[1].split("_")[0] for i,index in enumerate(rank_ID[0:maxres])]
        imlist = [str(imgNames[index]).split("'")[1] for i,index in enumerate(rank_ID[0:maxres])]
        imagelists = [image for image in imlist]
        return render_template("index1.html", query_path=open_file, imagelists =imagelists) 
    else:    
        return render_template("index1.html")


if __name__ == '__main__':
    app.run()

