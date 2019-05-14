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
import sys
import pickle

BATCH_SIZE = 150

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
        #print(paths)
        searchObj_step1 = re.search("(test|train|val)/.+/", paths).group()
        #print(searchObj_step1)
        searchObj_step2 = re.search("/.+/", searchObj_step1).group()
        #print(searchObj_step2)
        searchObj_step3 = re.sub("/", "", searchObj_step2)
        #print(searchObj_step3)
        searchObj_step4 = dict_cities_num[searchObj_step3]
        #print(searchObj_step4)
    else:
        searchObj_step1 = [re.search("(test|train|val)/.+/", path).group() for path in paths]
        #print(searchObj_step1)
        searchObj_step2 = [re.search("/.+/", path).group() for path in searchObj_step1]
        #print(searchObj_step2)
        searchObj_step3 = [re.sub("/", "", path) for path in searchObj_step2]
        #print(searchObj_step3)
        searchObj_step4 = [dict_cities_num[path] for path in searchObj_step3]
        #print(searchObj_step4)
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
                #print(img)
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
    new_output = keras.layers.Dense(4096, activation='relu')(new_output)
    new_output = keras.layers.Dropout(0.5)(new_output)
    new_output = keras.layers.Dense(4096, activation='relu')(new_output)
    new_output = keras.layers.Dropout(0.5)(new_output)
    new_output = keras.layers.Dense(10, activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model

def main(experiment_data, model_path):
    print("experiment_data:",experiment_data)
    print("model_path:",model_path)
    
    exp_result_path = model_path.split("/")[0]
    print("exp_result_path:",exp_result_path)
    
    model_name = model_path.split("/")[1].split(".h")[0]
    print(model_name)
    if not os.path.exists(exp_result_path+"/"+model_name):
        os.makedirs(exp_result_path+"/"+model_name)
    
    data = pickle.load( open( experiment_data+".p", "rb" ) )
    
    #train_data = data[0]
    random_test_data = data[1]["train_path_random_test"]
    random_test_assignment = data[1]["random_set"]
    #print("random_test_data[0:10]")
    #print(random_test_data[0:10])

    
    model = vgg16()
    model.load_weights(model_path)
    
    temp = data_generator(random_test_data)
    random_test_data_proba = model.predict_generator(temp, steps=200, verbose=1)
    
    random_test_data_y = get_city(random_test_data, single_input = False)
    
    random_test_data_proba_2 = np.stack(random_test_data_proba, axis=0)
    random_test_data_pred = random_test_data_proba_2.argmax(axis=-1)
    
    random_test_data_accuracy = np.mean(random_test_data_pred==random_test_data_y)
    print("\nRandom_test_data_accuracy accuracy: {} %".format(random_test_data_accuracy*100))
    
    results_random_test_data=pd.DataFrame({"Filename":random_test_data,
                              "True_class":random_test_data_y,
                              "Predictions":random_test_data_pred,
                              "random_test_assignment":random_test_assignment,
                                   "0":random_test_data_proba_2[:,0],
                                   "1":random_test_data_proba_2[:,1],
                                   "2":random_test_data_proba_2[:,2],
                                   "3":random_test_data_proba_2[:,3],
                                   "4":random_test_data_proba_2[:,4],
                                   "5":random_test_data_proba_2[:,5],
                                   "6":random_test_data_proba_2[:,6],
                                   "7":random_test_data_proba_2[:,7],
                                   "8":random_test_data_proba_2[:,8],
                                   "9":random_test_data_proba_2[:,9]})
                                   
    results_random_test_data.to_csv(exp_result_path+"/"+model_name+"/results_random_test_data.csv", index=False)
    
    images = np.array(list_files_in_dir("InstaCities1M_test_processd"))
    # print(images.shape)
    # there are 21449 images
    
    temp = data_generator(images)
    test_data_proba = model.predict_generator(temp, steps=143, verbose=1)
    
    test_data_y = get_city(images, single_input = False)
    
    test_data_proba_2 = np.stack(test_data_proba, axis=0)
    test_data_pred = test_data_proba_2.argmax(axis=-1)
    
    test_data_accuracy = np.mean(test_data_pred==test_data_y)
    print("\nTest_data_accuracy accuracy: {} %".format(test_data_accuracy*100))
    
    results_random_test_data=pd.DataFrame({"Filename":images,
                              "True_class":test_data_y,
                              "Predictions":test_data_pred,
                                   "0":test_data_proba_2[:,0],
                                   "1":test_data_proba_2[:,1],
                                   "2":test_data_proba_2[:,2],
                                   "3":test_data_proba_2[:,3],
                                   "4":test_data_proba_2[:,4],
                                   "5":test_data_proba_2[:,5],
                                   "6":test_data_proba_2[:,6],
                                   "7":test_data_proba_2[:,7],
                                   "8":test_data_proba_2[:,8],
                                   "9":test_data_proba_2[:,9]})
                                   
    results_random_test_data.to_csv(exp_result_path+"/"+model_name+"/results_test_data.csv", index=False)
    

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    