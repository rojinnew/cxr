# -*- coding: utf-8 -*-
import csv
import os
import random

if __name__ == '__main__':
    f_box = open('BBox_List_2017.csv')
    lines_box = f_box.read().splitlines()
    del lines_box[0]
    bb_list = [] 
    for i in range(len(lines_box)):
        split = lines_box[i].split(',')
        file_name = split[0]
        bb_list.append(file_name)
    f_box.close()
    # cerate id of patient in bbox.csv  
    box_patient_ids = []
    bbox_dic = {}
    for i in range(len(bb_list)):
        if bb_list[i][:8] not in box_patient_ids:
            box_patient_ids.append(bb_list[i][:8])  
            bbox_dic[bb_list[i][:8]] = []
            bbox_dic[bb_list[i][:8]].append(bb_list[i])
        else:
            bbox_dic[bb_list[i][:8]].append(bb_list[i])

    # Generate label index csv file
    f_de = open( 'Data_Entry_2017.csv')
    dataset_type = ['train_val', 'test']
    lines_de = f_de.read().splitlines()
    del lines_de[0]
    image_name_list = [] 
    for i in range(len(lines_de)):
        split = lines_de[i].split(',')
        file_name = split[0]
        image_name_list.append(file_name)
     
    f_de.close()
    # group the same patients together to guaranty no patient overlap between splits
    patients ={}
    for i in range(len(image_name_list)):
        if image_name_list[i][:8] not in patients.keys():
            patients[image_name_list[i][:8]]=[]
        patients[image_name_list[i][:8]].append(image_name_list[i])
    
    test_list = []
    for i in range(len(box_patient_ids)):
      if(box_patient_ids[i] in patients.keys()):
        test_list += patients[box_patient_ids[i]]
        for x in bbox_dic[box_patient_ids[i]]:
            if x not in patients[box_patient_ids[i]]:
                test_list += [x] 
        del patients[box_patient_ids[i]] 
      else:
        test_list += bbox_dic[box_patient_ids[i]] 
    patients_keys = list(patients.keys())      
    random.shuffle(patients_keys)
    # split them into train and val
    train_val_list = []
    train_num = 0
    l1 = set(image_name_list)
    l2 = set(bb_list)
    length = len ( l1.union(l2))
    for i in patients_keys:
        if train_num <(length*8)/10:
            train_val_list += patients[i]
            train_num += len(patients[i])
        else:
            test_list += patients[i]
    print("patinets_keys",len(patients_keys))
    print("train_val_list length", len(train_val_list))
    print("test_list length", len(test_list))
    # sort the list
    train_val_list.sort()
    test_list.sort()

    with open('train_val_list.txt', 'w+') as wf_train_val:
        with open('test_list.txt', 'w+') as wf_test:
            for data in train_val_list[:-1]:
                wf_train_val.write(data+'\n')
            wf_train_val.write(train_val_list[-1])
            for data in test_list[:-1]:
                wf_test.write(data+'\n')
            wf_test.write(test_list[-1])
    wf_train_val.close()
    wf_test.close()
    
