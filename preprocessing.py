import numpy
import os
import os.path
from PIL import Image
from numpy import *

# input image dimensions
img_rows, img_cols = 64,64

#%%
#  data
path1_class1a = '/home/govardhan/Goverdhan_Program/classification/egocentric'    #path of folder of class1 positive images

path1_class2a = '/home/govardhan/Goverdhan_Program/classification/non-egocentric'

#path1_class3a = '/home/govardhan/Goverdhan_Program/auto_encoder/classification/fuse_2'

#path1_class4a = '/home/govardhan/Goverdhan_Program/auto_encoder/classification/fuse_3'







Save_class1 = '/home/govardhan/Goverdhan_Program/classification/egocentric_grey' # Path to folder to save image of class2
Save_class2 = '/home/govardhan/Goverdhan_Program/classification/non_egocentric_grey' # Path to folder to save image of class3
#Save_class3 = '/home/govardhan/Goverdhan_Program/auto_encoder/classification/fuse_2new'
#Save_class4 = '/home/govardhan/Goverdhan_Program/auto_encoder/classification/fuse_3new'
#Save_class5 = '/home/govardhan/Goverdhan_Program/ultrasound/class_5_grey' # Path to folder to save image of class5

List_Class1a =  os.listdir(path1_class1a)

List_Class2a =  os.listdir(path1_class2a)

#List_Class3a =  os.listdir(path1_class3a)

#List_Class4a =  os.listdir(path1_class4a)

#List_Class5a =  os.listdir(path1_class5a)

##path2_ear_processed ='/home/ranjeet/Ranjeet_Python/Image_Paper_code/dataset/ear_processed'  #path of folder to save ear images
##path1_fkp = '/home/ranjeet/Ranjeet_Python/Image_Paper_code/dataset/fkp'    #path of folder of fkp images
##path2_fkp_processed ='/home/ranjeet/Ranjeet_Python/Image_Paper_code/dataset/fkp_processed'  #path of folder to save fkp images
##path1_iris = '/home/ranjeet/Ranjeet_Python/Image_Paper_code/dataset/iris'    #path of folder of iris images
##path2_iris_processed ='/home/ranjeet/Ranjeet_Python/Image_Paper_code/dataset/iris_processed'  #path of folder to save iris images
##list_ear = os.listdir(path1_ear)
##list_fkp = os.listdir(path1_fkp)
##list_iris = os.listdir(path1_iris)
##ear_samples = size(list_ear)
##fkp_samples = size(list_fkp)
##iris_samples = size(list_iris)
##num_samples = ear_samples + fkp_samples + iris_samples
##print num_samples

# save the processed Class1 Image
for file in List_Class1a:
    im = Image.open(path1_class1a + '/' + file)  
    #img = im.resize((img_rows,img_cols))
    gray = im.convert('L')       
    gray.save(Save_class1 +'/' +  file, "JPEG")


# save the processed Class2 Image
for file in List_Class2a:
    im = Image.open(path1_class2a + '/' + file)  
    #img = im.resize((img_rows,img_cols))
    gray = im.convert('L')       
    gray.save(Save_class2 +'/' +  file, "JPEG")


# save the processed Class3 Image
#for file in List_Class3a:
    #im = Image.open(path1_class3a + '/' + file)  
    #img = im.resize((img_rows,img_cols))
    #gray = img.convert('L')       
    #img.save(Save_class3 +'/' +  file, "JPEG")
    
#for file in List_Class4a:
    #im = Image.open(path1_class4a + '/' + file)  
    #img = im.resize((img_rows,img_cols))
    #gray = img.convert('L')       
    #img.save(Save_class4 +'/' +  file, "JPEG")






# save the processed Class5 Image
#for file in List_Class5a:
    #im = Image.open(path1_class5a + '/' + file)  
    #img = im.resize((img_rows,img_cols))
    #gray = img.convert('L')       
    #gray.save(Save_class5 +'/' +  file, "bmp")
