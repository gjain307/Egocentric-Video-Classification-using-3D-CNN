import os
#import preprocess
import itertools
import numpy as np

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling2D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.utils import shuffle
import cv2
#import imageio
import os
import numpy as np
import sys
from glob import glob

def load_images_from_folder(folder, get_count=False):
    images = []
    count = 0
    num=0
    list1=os.listdir(folder)
    list1=sorted(list1,comp)
    length1=len(list1)/16
    for filename in list1:
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img, (112, 112))
            if(num%length1==0 and count<16):
            	images.append(img)
            	if get_count==True:
                	count += 1
        num+=1
    if get_count==True:
        return images, count
    return images
def comp(a,b):
	a_num=0
	b_num=0
	for i in range(len(a)):
		if(a[i]=='_'):
			a_num=i+1
			break
	for i in range(len(b)):
		if(b[i]=='_'):
			b_num=i+1
			break
	if(int(a[a_num:-4])>=int(b[b_num:-4])):
		return 1
	else:
		return -1
	#return int(a[a_num:-4])<=int(b[b_num:-4])
	
def get_input(test_factor=0.7):
	path1="/home/jyoti/govardhan1/classification/ego_new"
	path2="/home/jyoti/govardhan1/classification/non-ego_new"
	output_train=[]
	labels_train=[]
	output_test=[]
	labels_test=[]
	mincount=120000
	directories=os.listdir(path1)
	directories=directories
	num=0
	for direc in directories:
		folder_path=path1+"/"+direc
		images,count=load_images_from_folder(folder=folder_path,get_count=True)
		if(num<=test_factor*len(directories)):
			output_train.append(images)
			print ("video processed in "+folder_path)
			labels_train.append(0)
		else:
			output_test.append(images)
			print ("video processed in "+folder_path)
			labels_test.append(0)
		num+=1
	directories=os.listdir(path2)
	num=0
	directories=directories
	for direc in directories:
		folder_path=path2+"/"+direc
		images,count=load_images_from_folder(folder=folder_path,get_count=True)
		if(num<=test_factor*len(directories)):
			output_train.append(images)
			print ("video processed in "+folder_path)
			labels_train.append(1)
		else:
			output_test.append(images)
			print ("video processed in "+folder_path)
			labels_test.append(1)
		num+=1
	output_train,labels_train=shuffle(output_train,labels_train,random_state=2)
	output_test,labels_test=shuffle(output_test,labels_test,random_state=2)
	output_train = np.array(output_train)
	output_test=np.array(output_test)
	print ("Output shape: {}".format(output_train.shape))
	print ("Labels shape: {}".format(len(labels_train)))
	output_train = output_train.astype('float32')
	output_train -= np.mean(output_train)
	output_train /= np.max(output_train)

	output_test = output_test.astype('float32')
	output_test -= np.mean(output_test)
	output_test /= np.max(output_test)
	# Save processed videos to disk
	#np.save("videos_train.npy", output_train)
	#np.save("labels_train.npy",labels_train)
	#np.save('videos_test.npy',output_test)
	#np.save('labels_test.npy',labels_test)
	return output_train, labels_train,output_test,labels_test

