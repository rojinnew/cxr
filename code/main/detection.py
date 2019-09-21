import csv
import cv2
import sys
import os
import torch
import pickle
import numpy as np
import skimage.transform
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import skimage
from skimage.io import *
from skimage.transform import *
import scipy
import scipy.ndimage as ndimage
from gradcam import *
from imageio import imread
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

use_gpu = torch.cuda.is_available()  
num_classes = 8
class_index = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
bbox_name =[]
bbox_info = []
image_dic = {}
with open("./split/3-label/dataset/BBox_List_2017.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    i = 0 
    for row in readCSV:
      if i != 0:
        if row[0] not in image_dic.keys(): 
            image_dic[row[0]] = {} 
            image_dic[row[0]][row[1]] = None 
            #image_dic[row[0]][row[1]] = [] 
        image_dic[row[0]][row[1]] = row 
        bbox_name.append(row[0])
        bbox_info.append(row)
      i = i + 1
test_txt_path = "./split/3-label/dataset/test_list.txt" 
image_folder_path = "./split/3-label/dataset/images"  
with open(test_txt_path, "r") as f:
    test_list = [i.strip() for i in f.readlines()]
test_data = []
os.mkdir("original")
os.mkdir("masks")
os.mkdir("heatmaps")
os.mkdir("boundresults")

for i in range(len(test_list)):
    image_path = os.path.join(image_folder_path, test_list[i])
    image = imread(image_path)
    if image.shape != (1024,1024):
        image = image[:,:,0]
    image_resized = skimage.transform.resize(image,(256,256), mode='reflect')
    test_data.append((np.array(image_resized)).reshape(256,256,1))
test_data = np.array(test_data)
#print("length of test data",len(test_list))

# from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou 



class CXRData(Dataset):
    def __init__(self, input = test_data, transform=None):
        self.x = np.uint8(test_data*255)
        self.transform = transform
    def __getitem__(self, index):
        current_x = np.tile(self.x[index],3)
        image = self.transform(current_x)
        return image
    def __len__(self):
        return len(self.x)

class densenet121(nn.Module):
    def __init__(self, out_size):
        super(densenet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        #print("self.densenet121", self.densenet121)
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(num_features, out_size))#,nn.Sigmoid())
    def forward(self, x):
        x = self.densenet121(x)
        return x

model = densenet121(num_classes)
if use_gpu == True:
    model = model.cuda()

model = torch.nn.DataParallel(model)
#model.load_state_dict(torch.load("model/dense121_BCE9_0.6729095625692398.pkl", map_location={'cuda:0': 'cpu'}))
model.load_state_dict(torch.load("best/dense121_WBCE24_0.6797693935647515.pkl", map_location={'cuda:0': 'cpu'}))
#model.load_state_dict(torch.load("model/dense121_maskFalse_BCE_lr0.0002_wd0.0_drop_rate0.3_0.7092140641429037.pkl", map_location={'cuda:0': 'cpu'}))
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
test_dataset = CXRData(input = test_data,transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)
                                        ]))

print("======= creating heatmap starts=======")
model.eval()

heatmap_output = []
image_id = []
output_class = []


if use_gpu == True:
    gcam = GradCAM(model=model, cuda=True)
else:
    gcam = GradCAM(model=model, cuda=False)
all_iou_value = []
for index in range(len(test_dataset)):
  fname = test_list[index]
  func = nn.Sigmoid()
  test_output = func(model(test_dataset[index].unsqueeze(0)).detach())
  activate_classes = (test_output>0.5)[0]
  #optimal_thresholds = [0.024621395, 0.005373399, 0.012895545, 0.017944088, 0.09455918, 0.03133899, 0.0011880096, 0.022412254]
  #activate_classes = (test_output.numpy() >np.asarray(optimal_thresholds))[0]
  activate_classes = np.where(activate_classes ==True)[0]
  #activate_classes = range(0, len(class_index)) 
  if fname in bbox_name and len(activate_classes)!=0:
    print("fname", fname)
    print("activate_classes",activate_classes)
    if use_gpu == True:
        input_image = Variable((test_dataset[index]).unsqueeze(0).cuda(), requires_grad=True)
    else:
        input_image = Variable((test_dataset[index]).unsqueeze(0), requires_grad=True)
    probs = gcam.forward(input_image)
    #activate_classes = np.where((probs > thresholds)[0]==True)[0] # get the activated class
    for activate_class in activate_classes:
        gcam.backward(idx=activate_class)
        #output = gcam.generate(target_layer="module.densenet121.features.denseblock4.denselayer16.conv.2")
        output = gcam.generate(target_layer="module.densenet121.features.denseblock4.denselayer16.conv2")
        if np.sum(np.isnan(output)) > 0:
            print("fxxx nan")
        heatmap_output.append(output)
        image_id.append(index)
        output_class.append(activate_class)

        k = int(activate_class)
        original = input_image.detach().numpy()
        original = np.transpose(original, (0,2,3,1))[0]
        original = original * std + mean
        original = np.uint8(original * 255.0)
        gcam.save("./heatmaps/"+ fname[:-4] +"_"+class_index[activate_class]+".png", output, original)
        print("heatmap output done")

        mask = np.uint8(output * 255.0)
        cv2.imwrite("./masks/"+fname[:-4] +"_"+class_index[activate_class]+".png", mask)
        mask_image = cv2.imread("./masks/"+fname[:-4] +"_"+class_index[activate_class]+".png")
        #original_image = original
        cv2.imwrite("./original/"+fname[:-4]+"_"+class_index[k]+".png",original ) 
        original_image = cv2.imread("./original/"+fname[:-4]+"_"+class_index[k]+".png") 
        #print("original_image", original_image[1,:])
        #ret, threshed_image = cv2.threshold(cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY), 125, 255, cv2.THRESH_BINARY)
        ret, threshed_image = cv2.threshold(cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((1,1), np.uint8)
        closing = cv2.morphologyEx(threshed_image, cv2.MORPH_CLOSE, kernel, iterations=20)
        #======
        #CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and 
        #leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points
        #CV_RETR_EXTERNAL retrieves only the extreme outer contours. 
        #It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
        #======
        _, contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        heatmap = cv2.imread("./heatmaps/"+fname[:-4]+"_"+class_index[k]+".png") 
        # Predictions (RED)
        temp_iou = -1 
        for cnt in contours:
            area = cv2.contourArea(cnt)
            #if (area > 30**2):
            if (area > 20**2):
                # bounding box
                x2, y2, w2, h2 = cv2.boundingRect(cnt)
                cv2.rectangle(original_image, (x2,y2), (x2+w2, y2+h2), (0,0,255), 2)
                if class_index[k] in image_dic[fname]:  
                    row_i = image_dic[fname][class_index[k]]
                    fname = row_i[0]
                    x , y, w,h = int(float(row_i[2])/4-16),int(float(row_i[3])/4-16),int(float(row_i[4])/4), int(float(row_i[5])/4)
                    #cv2.rectangle(heatmap, (x,y), (x+w, y+h), (0,0,0), 2)
                    # truth
                    #cv2.rectangle(original_image, (x,y), (x+w, y+h), (255,0,0), 2)
                    cv2.rectangle(original_image, (x,y), (x+w, y+h), (0,255,255), 2)
                    boxA = [float(x) , float(y), float(x) + float(w),float(y) + float(h)]
                    boxB = [float(x2), float(y2),float(x2)+ float(w2), float(y2) + float(h2)]
                    new_iou = bb_intersection_over_union(boxA, boxB) 
                    if temp_iou < new_iou:
                        temp_iou = new_iou
                    cv2.imwrite("./boundresults/"+fname[:-4]+"_"+class_index[k]+".png",original_image) 
        if temp_iou !=-1:
            print("fname", fname)
            print("class_index[k]", class_index[k])
            print("temp_iou", temp_iou)
            all_iou_value.append(temp_iou)
        #cv2.imwrite("./boundresults/"+fname[:-4]+"_"+class_index[k]+".png",original_image) 
print(all_iou_value)
print("len(all_iou_value)",len(all_iou_value))
greater_than_thresh = []
for elem in all_iou_value:
    if(elem > 0.5):
        greater_than_thresh.append(1.0) 
    else:
        greater_than_thresh.append(0.0) 
#print("greater_than_thresh 0.5",greater_than_thresh)
length  = len(greater_than_thresh)
print("Accuracy for threshold = 0.5", sum(greater_than_thresh) / length)
greater_than_thresh2 = []
for elem in all_iou_value:
    if(elem > 0.25):
        greater_than_thresh2.append(1.0) 
    else:
        greater_than_thresh2.append(0) 
#print("greater_than_thresh 0.25",greater_than_thresh2)
length  = len(greater_than_thresh2)
print("Accuracy for threshold = 0.25", sum(greater_than_thresh2) / length)
