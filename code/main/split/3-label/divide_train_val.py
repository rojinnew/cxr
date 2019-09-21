# This code is inspired by https://github.com/TRKuan/cxr8/blob/master/label_gen.py

# -*- coding: utf-8 -*-
import csv
import os
import random
dataset_path = './dataset'

if __name__ == '__main__':
    training_list = []
    validation_list = []
    with open('dataset/train_val_list.txt') as f:
        with open('dataset/train_list.txt', 'w+') as wf_t:
            with open('dataset/val_list.txt', 'w+') as wf_v:
                image_name_list = f.read().split('\n')
                # group the same patients together to guarantee no patient overlap between splits
                patients = []
                last = ''
                for i in range(len(image_name_list)):
                    if last == image_name_list[i][:8]:
                        patients[-1].append(image_name_list[i])
                    elif image_name_list[i] :
                        patients.append([image_name_list[i]])
                    last = image_name_list[i][:8]
                random.shuffle(patients)
                training_list = []
                validation_list = []
                train_count = 0
                for i in range(len(patients)):
                    if train_count < (len(image_name_list*7)/8):
                        training_list += patients[i]
                        train_count += len(patients[i])
                    else:
                        validation_list += patients[i]
                training_list.sort()
                validation_list.sort()
                
                for data in training_list[:-1]:
                    if data:
                        wf_t.write(data+'\n')
                wf_t.write(training_list[-1])
                for data in validation_list[:-1]:
                    wf_v.write(data+'\n')
                wf_v.write(validation_list[-1])

    print('training and validation list generated')
