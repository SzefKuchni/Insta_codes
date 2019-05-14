# batch generator
BATCH_SIZE = 64
N_CLASSES = 10
last_finished_epoch = 0

folder_name = "results_vgg_original_experiment_cities_dropout_fine_tune_4_varied_LR"

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
    
instacities_paths = pickle.load( open("experiments/instacities_paths.p", "rb" ) )
train_paths = [x.split("lewyd/")[1] for x in instacities_paths[0]]    
print("train_paths:",train_paths[0:10])
    
images = np.array(list_files_in_dir("InstaCities1M_processed"))
val_idx = np.array([x.split("/")[5] == "val" for x in images])
val_paths = images[val_idx]

print("train_paths:", len(train_paths))
print("val_paths:", len(val_paths))

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
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.layers import Dense, Activation, Flatten, Input, Dropout

def vgg16():
    # load pre-trained model graph, don't add final layer
    model = keras.applications.VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.Flatten()(model.output)
    # add new dense layer for our labels
    new_output = keras.layers.Dense(4096, activation='relu')(new_output)
    new_output = keras.layers.Dropout(0.5)(new_output)
    new_output = keras.layers.Dense(4096, activation='relu')(new_output)
    new_output = keras.layers.Dropout(0.5)(new_output)
    new_output = keras.layers.Dense(N_CLASSES, activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model
    
model = vgg16()

#model.load_weights("results_vgg_original_experiment_cities_dropout/cities_vgg16-48-0.201.hdf5")
#set all layers trainable by default

### Trainin problem specific layers on extracted features
### Learning rate stage 1: 1e-4
print("### stage 1: Problem specific, LR 1e-4 ###")

for layer in model.layers:
    layer.trainable = True
    
# fix deep layers (fine-tuning only last 5)
for layer in model.layers[:-3]:
    layer.trainable = False
   
print(model.summary())
   
#seed(1234)
shuffle(train_paths)
shuffle(val_paths)
   
# compile new model
model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=1e-4),  # we can take big lr here because we fixed first layers
    metrics=['accuracy']  # report accuracy during training ### categorical_accuracy
)


model_filename = folder_name+"/phase1_best.hdf5"
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
pickle.dump(history, open(folder_name+"/history_phase1.p", "wb"))
# summarize history for accuracy
plt.clf()
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(folder_name+"/results_vgg_original_experiment_cities_LR_1e-4.jpg")

### Learning rate stage 2: 1e-5
print("### stage 2: Problem specific, LR 1e-5 ###")
model.load_weights(folder_name+"/phase1_best.hdf5")

model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=1e-5),  # we can take big lr here because we fixed first layers
    metrics=['accuracy']  # report accuracy during training ### categorical_accuracy
)


model_filename = folder_name+"/phase2_best.hdf5"
checkpointer = ModelCheckpoint(filepath=model_filename, verbose=1, save_best_only=True)

model.fit_generator(
    data_generator(train_paths), 
    steps_per_epoch=len(train_paths) // BATCH_SIZE // 4,
    epochs=8 * 4,
    validation_data=data_generator(val_paths), 
    validation_steps= len(val_paths) // BATCH_SIZE,
    callbacks=[checkpointer],
    verbose=1,
    initial_epoch=0, 
)

history = model.history.history
pickle.dump(history, open(folder_name+"/history_phase2.p", "wb"))
# summarize history for accuracy
plt.clf()
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(folder_name+"/results_vgg_original_experiment_cities_LR_1e-5.jpg")

### Learning rate stage 3: 1e-6
print("### stage 3: Problem specific, LR 1e-6 ###")
model.load_weights(folder_name+"/phase2_best.hdf5")

model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=1e-6),  # we can take big lr here because we fixed first layers
    metrics=['accuracy']  # report accuracy during training ### categorical_accuracy
)


model_filename = folder_name+"/phase3_best.hdf5"
checkpointer = ModelCheckpoint(filepath=model_filename, verbose=1, save_best_only=True)

model.fit_generator(
    data_generator(train_paths), 
    steps_per_epoch=len(train_paths) // BATCH_SIZE // 4,
    epochs=8 * 4,
    validation_data=data_generator(val_paths), 
    validation_steps= len(val_paths) // BATCH_SIZE,
    callbacks=[checkpointer],
    verbose=1,
    initial_epoch=0, 
)


history = model.history.history
pickle.dump(history, open(folder_name+"/history_phase3.p", "wb"))
# summarize history for accuracy
plt.clf()
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(folder_name+"/results_vgg_original_experiment_cities_LR_1e-6.jpg")

### Fine Tunning all layers with decreasing learning rate
### Learning rate stage 4: 1e-5
print("### stage 4: Fine_tune, LR 1e-5 ###")
for layer in model.layers:
    layer.trainable = True
    
print(model.summary())

model.load_weights(folder_name+"/phase3_best.hdf5")

model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=1e-5),  # we can take big lr here because we fixed first layers
    metrics=['accuracy']  # report accuracy during training ### categorical_accuracy
)


model_filename = folder_name+"/phase4_best.hdf5"
checkpointer = ModelCheckpoint(filepath=model_filename, verbose=1, save_best_only=True)

model.fit_generator(
    data_generator(train_paths), 
    steps_per_epoch=len(train_paths) // BATCH_SIZE // 4,
    epochs=8 * 4,
    validation_data=data_generator(val_paths), 
    validation_steps= len(val_paths) // BATCH_SIZE,
    callbacks=[checkpointer],
    verbose=1,
    initial_epoch=0, 
)


history = model.history.history
pickle.dump(history, open(folder_name+"/history_phase4.p", "wb"))
# summarize history for accuracy
plt.clf()
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(folder_name+"/results_vgg_original_experiment_cities_fine_tune_LR_1e-5.jpg")

### Learning rate stage 5: 1e-6
print("### stage 5: Fine tune, LR 1e-6 ###")
for layer in model.layers:
    layer.trainable = True

model.load_weights(folder_name+"/phase4_best.hdf5")

model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=1e-6),  # we can take big lr here because we fixed first layers
    metrics=['accuracy']  # report accuracy during training ### categorical_accuracy
)


model_filename = folder_name+"/phase5_best.hdf5"
checkpointer = ModelCheckpoint(filepath=model_filename, verbose=1, save_best_only=True)

model.fit_generator(
    data_generator(train_paths), 
    steps_per_epoch=len(train_paths) // BATCH_SIZE // 4,
    epochs=8 * 4,
    validation_data=data_generator(val_paths), 
    validation_steps= len(val_paths) // BATCH_SIZE,
    callbacks=[checkpointer],
    verbose=1,
    initial_epoch=0, 
)


history = model.history.history
pickle.dump(history, open(folder_name+"/history_phase5.p", "wb"))
# summarize history for accuracy
plt.clf()
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(folder_name+"/results_vgg_original_experiment_cities_fine_tune_LR_1e-6.jpg")