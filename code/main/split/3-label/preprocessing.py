import sys
import os
import numpy as np
import pandas as pd
import imageio
from os import listdir
import skimage.transform
import pickle
from sklearn.preprocessing import MultiLabelBinarizer


if __name__ == '__main__':
    data_entry = 'dataset/Data_Entry_2017.csv' 
    bbox_list = 'dataset/BBox_List_2017.csv' 
    image_folder = 'dataset/images/' 
    train_list_txt = 'dataset/train_list.txt' 
    val_list_txt = 'dataset/val_list.txt' 
    data_folder = 'preprocessed_data/' 
    data_frame = pd.read_csv(data_entry)

    # load data
    with open(train_list_txt, "r") as f:
        # delete the \n from lines using strip() method
        train_list = [i.strip() for i in f.readlines()]
    with open(val_list_txt, "r") as f:
        val_list = [ i.strip() for i in f.readlines()]


    # training images transformation
    print("training example:",len(train_list))
    train_set = []
    for i in range(len(train_list)):
        img_path = os.path.join(image_folder,train_list[i])
        img = imageio.imread(img_path)
        #  some image has shape (1024,1024,4) in training set
        if img.shape != (1024,1024): 
            img = img[:,:,0]
        img_resized = skimage.transform.resize(img,(256,256), mode = 'reflect') 
        train_set.append((np.array(img_resized)).reshape(256,256,1))
    train_set = np.array(train_set)
    np.save(os.path.join(data_folder,"train_features.npy"), train_set)

    # transform validation images
    print("validation example:",len(val_list))
    valid_set = []
    for i in range(len(val_list)):
        img_path = os.path.join(image_folder,val_list[i])
        img = imageio.imread(img_path)
        if img.shape != (1024,1024):
            img = img[:,:,0]
        img_resized = skimage.transform.resize(img,(256,256),mode = 'reflect')
        valid_set.append((np.array(img_resized)).reshape(256,256,1))
    valid_set = np.array(valid_set)
    np.save(os.path.join(data_folder,"valid_features.npy"), valid_set)


    # process label
    print("label preprocessing")

    train_y = []
    for t_id in train_list:
        t_labels = data_frame.loc[data_frame["Image Index"]==t_id,"Finding Labels"]
        t_labels= t_labels.tolist()[0].split("|")
        train_y.append(t_labels)

    valid_y = []
    for v_id in val_list:
        v_labels = data_frame.loc[data_frame["Image Index"]==v_id,"Finding Labels"]
        v_labels = v_labels.tolist()[0].split("|")
        valid_y.append(v_labels)

    encoder = MultiLabelBinarizer()
    encoder.fit(train_y+valid_y)
    label_classes = list(encoder.classes_)
    print("classes", label_classes)

    '''=====================================================================================
    classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 
                'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 
                'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
     ======================================================================================'''
    train_onehot_label = encoder.transform(train_y)
    valid_onehot_label = encoder.transform(valid_y)
    ''' =====
    # delete out "No Finding" column and keep 14 classes
    # train_onehot_label = np.delete(train_onehot_label, [10],1) # delete out "No Finding" column
    #valid_onehot_label = np.delete(valid_onehot_label, [10],1) # delete out "No Finding" column
     ====== '''
    # delete out "No Finding" column and keep only 8 classes
    train_onehot_label = np.delete(train_onehot_label, [2,3,5,6,7,10,12],1) 
    valid_onehot_label = np.delete(valid_onehot_label, [2,3,5,6,7,10,12],1) 
    print("length train_onehot_label",len(train_onehot_label))
    print("length valid_onehot_label",len(valid_onehot_label))

    np.save(os.path.join(data_folder,"train_onehot_label.npy"),train_onehot_label)
    np.save(os.path.join(data_folder,"valid_onehot_label.npy"),valid_onehot_label)
