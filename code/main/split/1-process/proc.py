import csv
import pandas as pd

sample = 'sample_labels.csv'
image_list = []
sample_reader = csv.reader(open(sample, newline='\n') ,delimiter=',') 
i = 0
for row in sample_reader:
    if(i !=0):
        image_list.append(row[0])
    i = i +1
print("length", len(image_list))

entry_reader = csv.reader(open('Data_Entry_2017.csv', newline='\n') ,delimiter=',') 
entryWriter = csv.writer( open ('new_Data_Entry_2017.csv', 'w',newline='' ), delimiter=',') 
i = 0
for row in entry_reader:
    if(i !=0):
       x =0
       if(row[0] in image_list):
            entryWriter.writerow(row)
    else:
       entryWriter.writerow(row)
       print("row",row)  
    i = i +1
i =  0
bb_reader = csv.reader(open('BBox_List_2017.csv', newline='\n') ,delimiter=',') 
bbWriter = csv.writer( open ('new_BBox_List_2017.csv', 'w',newline='' ), delimiter=',') 
for row in bb_reader:
    if(i !=0):
       if(row[0] in image_list):
            bbWriter.writerow(row)
    else:
       bbWriter.writerow(row)
    i = i +1
