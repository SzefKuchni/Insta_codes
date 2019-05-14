import os
import numpy as np
import random
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def list_files_in_dir(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            p=os.path.join(root,file)
            p=os.path.abspath(p)
            paths.append(p)
    return paths
    
def create_set_assignment_df(paths, sets, obs_in_class):
    train_paths_reduced, train_path_random_test = train_test_split(paths, test_size = sets*obs_in_class*10, stratify = [x.split("/")[5] for x in train_paths])
    
    print("train_paths_reduced:", len(train_paths_reduced))
    print("train_path_random_test:", len(train_path_random_test))
    
    print(np.unique([x.split("/")[5] for x in train_path_random_test], return_counts=True))
    
    train_path_random_test = sorted(train_path_random_test)
    idx = list(range(0,sets))*obs_in_class
    random.shuffle(idx)
    
    random_set_assignment = pd.DataFrame({"train_path_random_test":train_path_random_test,
                                          "random_set":idx*10,
                                          "city":[x.split("/")[5] for x in train_path_random_test]})
    
    instacities_paths = [train_paths_reduced, random_set_assignment]
    
    return(instacities_paths)
    
#train_paths = list_files_in_dir("InstaCities1M_processed/img/train/")
train_paths = list_files_in_dir("food_processed/")

train_paths2 = np.array(train_paths)
train_idx = np.array([x.split("/")[5] == "train" for x in train_paths])
train_paths = train_paths2[train_idx]

paths = create_set_assignment_df(train_paths, 10, 300)

#pickle.dump(paths, open("experiments/instacities_paths.p", "wb" ))
pickle.dump(paths, open("experiments/instafood_paths.p", "wb" ))