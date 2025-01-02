import keras
#from keras.utils import np_utils
import numpy as np

import os
import cv2
import random
from glob import glob
import keras
from tensorflow.keras.layers import Input, Convolution2D, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt




import cv2
import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog


clas1 = [item[10:-1] for item in sorted(glob("./dataset/*/"))]


from keras.preprocessing import image                  
from tqdm import tqdm

    
# Note: modified these two functions, so that we can later also read the inception tensors which 
# have a different format 
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    print(img_path)
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_paths, width, height)]
    return np.vstack(list_of_tensors)

#vilization_and_show()


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 


filename = filedialog.askopenfilename(title='open')

img = cv2.imread(filename )
cv2.imshow("Input Image", img)
bins                   = 8


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)




lower_green = np.array([25,0,20])
upper_green = np.array([100,255,255])
mask = cv2.inRange(hsv_img, lower_green, upper_green)
result = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Resulted", result)

      

from tensorflow.keras.models import load_model
model = load_model('trained_model.h5')


test_tensors = paths_to_tensor(filename)/255
pred=model.predict(test_tensors)
print(np.argmax(pred))
print('Given Currancy is  Predicted as: '+str(clas1[np.argmax(pred)]))
