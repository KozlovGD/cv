#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[4]:


import numpy as np
import cv2
import csv

def read_task_file(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_val = int(row['row'])
            col_val = int(row['col'])
            length_val = int(row['len'])
        return row_val, col_val, length_val

def crop_image(image, row, col, length):
    image_normalized = image.astype(np.float32) / 255.0
    image_averaged = np.mean(image_normalized, axis=2)
    patch = image_averaged[row:row+length, col:col+length]
    return patch

image = cv2.imread('image.png')

row, col, length = read_task_file('task.csv')

patch = crop_image(image, row, col, length)

np.save('seminar01_crop.npy', patch, allow_pickle=False)


# In[ ]:




