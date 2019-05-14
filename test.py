BATCH_SIZE = 4
weights_path = "results/cities_vgg16-12-0.229.hdf5"

import keras
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.layers import Dense, Activation, Flatten, Input
import os
from PIL import Image
import re
import pandas as pd

dict_num_cities = {0: 'chicago',
                   1: 'london',
                   2: 'losangeles',
                   3: 'melbourne',
                   4: 'miami',
                   5: 'newyork',
                   6: 'sanfrancisco',
                   7: 'singapore',
                   8: 'sydney',
                   9: 'toronto'}
                   
dict_cities_num = {'chicago'     :0,
                   'london'      :1,
                   'losangeles'  :2,
                   'melbourne'   :3,
                   'miami'       :4,
                   'newyork'     :5,
                   'sanfrancisco':6,
                   'singapore'   :7,
                   'sydney'      :8,
                   'toronto'     :9 }

def list_files_in_dir(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            p=os.path.join(root,file)
            p=os.path.abspath(p)
            paths.append(p)
    return paths
    
def get_city(paths, single_input = True):
    if single_input:
        searchObj_step1 = re.search("(test|train|val)/.+/", paths).group()
        searchObj_step2 = re.search("/.+/", searchObj_step1).group()
        searchObj_step3 = re.sub("/", "", searchObj_step2)
        searchObj_step4 = dict_cities_num[searchObj_step3]
    else:
        searchObj_step1 = [re.search("(test|train|val)/.+/", path).group() for path in paths]
        searchObj_step2 = [re.search("/.+/", path).group() for path in searchObj_step1]
        searchObj_step3 = [re.sub("/", "", path) for path in searchObj_step2]
        searchObj_step4 = [dict_cities_num[path] for path in searchObj_step3]
    return(searchObj_step4)

def matrix_image_loading(path):
    img = Image.open(path)
    img = np.array(img)
    return(img)
    
def batch_generator(items, batch_size):
    return_batch = []
    cnt = 0
    for item in items:
        if cnt != batch_size:
            return_batch.append(item)           
        else:
            yield return_batch
            return_batch = []
            return_batch.append(item)
        cnt = cnt%batch_size + 1
    if cnt != 0:
        yield return_batch  

def data_generator(files):
    while True:  # so that Keras can loop through this as long as it wants
        for batch in batch_generator(files, BATCH_SIZE):
            batch_images = []
            batch_labels = []
            for img in batch:
                matrix = matrix_image_loading(img)
                batch_images.append(matrix)
                batch_labels.append(get_city(img))
            batch_imgs = np.stack(batch_images, axis=0)
            # convert targets into 2D tensor [batch_size, num_classes]
            batch_targets = keras.utils.np_utils.to_categorical(batch_labels, 10)
            yield batch_imgs, batch_targets
        
def vgg16():
    # load pre-trained model graph, don't add final layer
    model = keras.applications.VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.Flatten()(model.output)
    # add new dense layer for our labels
    new_output = keras.layers.Dense(512, activation='relu')(new_output)
    new_output = keras.layers.Dense(512, activation='relu')(new_output)
    new_output = keras.layers.Dense(512, activation='relu')(new_output)
    new_output = keras.layers.Dense(10, activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model
    
images = np.array(list_files_in_dir("InstaCities1M_processed"))

val_idx = np.array([x.split("/")[5] == "val" for x in images])
train_paths = images[np.invert(val_idx)]
val_paths = images[val_idx]
    
model = vgg16()

model.load_weights(weights_path)

val = data_generator(val_paths)
val_proba = model.predict_generator(val, steps=12500, verbose=1)

val_y = get_city(val_paths, single_input = False)

val_proba_2 = np.stack(val_proba, axis=0)
val_pred = val_proba_2.argmax(axis=-1)
true_val_y = val_y

val_accuracy = np.mean(val_pred==true_val_y)
print("\nVal accuracy: {} %".format(val_accuracy*100))

results_val=pd.DataFrame({"Filename":val_paths,
                      "True_class":true_val_y,
                      "Predictions":val_pred,
                           "0":val_proba_2[:,0],
                           "1":val_proba_2[:,1],
                           "2":val_proba_2[:,2],
                           "3":val_proba_2[:,3],
                           "4":val_proba_2[:,4],
                           "5":val_proba_2[:,5],
                           "6":val_proba_2[:,6],
                           "7":val_proba_2[:,7],
                           "8":val_proba_2[:,8],
                           "9":val_proba_2[:,9]})
                           
results_val.to_csv("results/cities_vgg16_modified.csv", index=False)