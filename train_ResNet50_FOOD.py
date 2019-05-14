# batch generator
BATCH_SIZE = 64
N_CLASSES = 10
model_filename = "results_ResNet50_FOOD_3/doof_ResNet50-{epoch:02d}-{val_acc:.3f}.hdf5"
last_finished_epoch = 0

import pickle
import sys
import os
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras 
from keras.callbacks import ModelCheckpoint
from PIL import Image
from keras import backend as K
import tensorflow as tf
from random import shuffle, seed

print("imports successful")

def list_files_in_dir(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            p=os.path.join(root,file)
            p=os.path.abspath(p)
            paths.append(p)
    return paths
    
instacities_paths = pickle.load( open("experiments/instafood_paths.p", "rb" ) )
train_paths = [x.split("lewyd/")[1] for x in instacities_paths[0]]    
print("train_paths:",train_paths[0:10])
    
images = list_files_in_dir("food_processed")
images = [x.split("lewyd/")[1] for x in images]
images = np.array(images)

val_idx = np.array([x.split("/")[2] == "val" for x in images])
val_paths = images[val_idx]

print("train_paths:", len(train_paths))
print("val_paths:", len(val_paths))
print(val_paths[0:10])
dict_num_cities = {0: 'applepie',
                   1: 'burger',
                   2: 'donuts',
                   3: 'frenchfries',
                   4: 'hotdog',
                   5: 'macandcheese',
                   6: 'pancake',
                   7: 'pizza',
                   8: 'spaghetti',
                   9: 'steak'}
                   
dict_cities_num = {'applepie'     :0,
                   'burger'      :1,
                   'donuts'  :2,
                   'frenchfries'   :3,
                   'hotdog'       :4,
                   'macandcheese'     :5,
                   'pancake':6,
                   'pizza'   :7,
                   'spaghetti'      :8,
                   'steak'     :9 }
                   
def get_city(paths, single_input = True):
    #print(paths)
    if single_input:
        searchObj_step1 = re.search("/.+/(test|train|val)", paths).group()
        #print(searchObj_step1)
        searchObj_step2 = re.search("/.+/", searchObj_step1).group()
        #print(searchObj_step2)
        searchObj_step3 = re.sub("/", "", searchObj_step2)
        #print(searchObj_step3)
        searchObj_step4 = dict_cities_num[searchObj_step3]
    else:
        searchObj_step1 = [re.search("/.+/(test|train|val)", path).group() for path in paths]
        searchObj_step2 = [re.search("/.+/", path).group() for path in searchObj_step1]
        searchObj_step3 = [re.sub("/", "", path) for path in searchObj_step2]
        searchObj_step4 = [dict_cities_num[path] for path in searchObj_step3]
    return(searchObj_step4)
    
print(train_paths[0:10])
print(get_city(train_paths[0:10], single_input = False))

print(train_paths[0])
print(get_city(train_paths[0], single_input = True))


def matrix_image_loading(path):
    img = Image.open(path)
    img = np.array(img)
    return(img)
    
#img = matrix_image_loading(train_paths[0])

#plt.imsave(arr=img, fname="test.jpg")

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
                #print(img)
                #print(get_city(img))
                batch_labels.append(get_city(img))
            batch_imgs = np.stack(batch_images, axis=0)
            # convert targets into 2D tensor [batch_size, num_classes]
            batch_targets = keras.utils.np_utils.to_categorical(batch_labels, N_CLASSES)
            yield batch_imgs, batch_targets
            
def reset_tf_session():
    K.clear_session()
    tf.reset_default_graph()
    s = K.get_session()
    return s
s = reset_tf_session()

EXPECTED_DIM = (224, 224, 3)
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
import numpy as np
from keras.layers import Dense, Activation, Flatten, Input, Dropout

#ResNet152(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

def vgg16():
    # load pre-trained model graph, don't add final layer
    model = ResNet50(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.Flatten()(model.output)
    # add new dense layer for our labels
    #new_output = keras.layers.Dense(4096, activation='relu')(new_output)
    #new_output = keras.layers.Dropout(0.5)(new_output)
    #new_output = keras.layers.Dense(4096, activation='relu')(new_output)
    #new_output = keras.layers.Dropout(0.5)(new_output)
    new_output = keras.layers.Dense(N_CLASSES, activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model
    
model = vgg16()
#model.load_weights("results_vgg_original_experiment_food_dropout5_trainable_layers_change/food_vgg16-21-0.469.hdf5")
# set all layers trainable by default
#for layer in model.layers:
#    layer.trainable = True
    
# fix deep layers (fine-tuning only last 5)
#for layer in model.layers[:-5]:
#    layer.trainable = False
   
print(model.summary())
   
# compile new model
model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=1e-5),  # we can take big lr here because we fixed first layers
    metrics=['accuracy']  # report accuracy during training ### categorical_accuracy
)

#seed(1234)
shuffle(train_paths)
shuffle(val_paths)

print(train_paths[0])

checkpointer = ModelCheckpoint(filepath=model_filename, verbose=1, save_best_only=True)

model.fit_generator(
    data_generator(train_paths), 
    steps_per_epoch=len(train_paths) // BATCH_SIZE // 4,
    epochs=8 * 4,
    validation_data=data_generator(val_paths), 
    validation_steps= len(val_paths) // BATCH_SIZE,
    callbacks=[checkpointer],
    verbose=1,
    initial_epoch=last_finished_epoch or 0, 
)

history = model.history.history
# summarize history for accuracy
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("results_ResNet50_FOOD_3/results_vgg_original_experiment_FOOD.jpg")