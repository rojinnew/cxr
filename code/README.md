Georgia Institute of Technology
Project for CS 7643 class

# Project Title
### Prerequisites
Following Packages need to be installed for running the code.

torch: 1.0.1
torchvision: 0.2.2
numpy: 1.15.4
sklearn: 0.20.1 
matplotlib: 3.0.2
cv2: 3.4.2
imageio: 2.4.1

You can run the code on GPU. 


- Unzip the folder and go inside main folder 
# Preprocessing 

STEP 1-1:
 
- Go inside "split/1-process" directory.
- Put "sample_labels.csv" downloaded from Kaggle in "1-process" folder. 
- Put "BBox_List_2017.csv" and "Data_Entry_2017.csv" from original dataset into "1-process" folder that contains "proc.pay"
- Run "proc.py by" following command: python3 proc.py
- Two files called "new_BBox_List_2017.csv" and "new_Data_Entry_2017.csv" are created in "1-process" folder.

STEP 1-2: 

- Copy "new_BBox_List_2017.csv" and "new_Entry_2017.csv" from "1-process" folder to "2-split" and rename them as "BBox_List_2017" and "Data_Entry_2017.csv". 
- Go inside "split/2-split directory".
- Run "split.py" by this command: python3 split.py 
- Two files called  "train_val_list.txt" and "test_list.txt" are created. 

STEP 1-3: 

- Go inside "split/3-label" directory
- Create an empty folder called dataset inside "3-label"
- Copy downloaded images into "images" folder
- Copy the "BBox_List_2017" and "Data_Entry_2017.csv" and "train_val_list.txt" and "test_list.txt" to "3-label/dataset" folder.
- Run divide_train_val.py by running python3 divide_train_val.py
- Two new files called  "train_list.txt" and "val_list.txt" are created inside dataset folder. 
- create an empty folder called "preprocessed_data" inside "3-label" folder.
- run python3 preprocessing.py
- Four files called "train_features.npy", "train_onehot_label.npy", "valid_features.npy", "valid_onehot_label.npy" will be generated inside "preprocessed_data" folder. 



# Training
STEP 2-1: Training and Validation using Supervised Approaches

- Move "preprocessed_data" folder inside "code_submission" folder
- You can modify train.py inside code_submission folder by adjusting number of epoch (e.g. 10), batch size (e.g. 16), model (options: resnet50, vgg16, densenet121), loss (options: BCE, WBCE, WBCE2, LSEP), learning rate (e.g. 0.0002), weight_decay (e.g. 0.0005), and drop rate (e.g. 0.3).
- Run train.py by executing: python3 train.py
- A summary of metrics value will be printed at the end of each epoch. 
- The model from each epoch is saved in the model folder.
- The Loss history, training and validation accuracy and AUROC trend are plotted and saved in plots folder. 

STEP 2-2: Training and Validation using Semi-Supervised Approaches

- Run mcl.py by executing: python3 mcl.py


# Localization 

- Put the trained model in "best" folder
- Run the detection.py: python3 detection.py
- Find heatmaps in "heatmaps" folder
- Find processed image with bounding box in "boundresults" folder

For different implementation we customized and adopted some part of existing codes as follows:

- Part of preprocessing and data augmentation implementation is inspired by  ChexNet code [1].
- The LSEP loss implementation is inspired by the  multilabel classification of bird sounds [2].
- Grad-CAM implementation [3] is customized for  our model and dataset.
- The implementation of loss function for weakly labeled data is inspired by MCL code [4] and it is customized for multi-class and the multi-label dataset.
-The code for generating plots is adopted from HW1 coding - CS 7643 [5].

References:

[1] https://github.com/thtang/CheXNet-with-localization
[2] https://github.com/Mipanox/Bird_cocktail
[3] https://github.com/meliketoy/gradcam.pytorch
[4] https://github.com/GT-RIPL/L2C
[5] https://www.cc.gatech.edu/classes/AY2019/cs7643_spring/hw1/





