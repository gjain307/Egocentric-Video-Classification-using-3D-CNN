from keras.layers import Input, Dense
from keras.models import Model
import os
import os.path
import numpy 
import theano
from PIL import Image
from numpy import * 
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import tensorflow as tf
# this is the size of our encoded representations
encoding_dim = 512 # 32 floats  the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(3072,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
#encoded = Dense(8192, activation='relu')(encoded)
#encoded = Dense(4096, activation='relu')(encoded)
encoded = Dense(1024, activation='relu')(encoded)
encoded = Dense(512, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(512, activation='relu')(encoded)
decoded = Dense(1024, activation='relu')(decoded)
decoded = Dense(3072, activation='relu')(decoded)
#decoded = Dense(8192, activation='relu')(decoded)

#decoded = Dense(49152, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer1= autoencoder.layers[-3]
o1=decoder_layer1(encoded_input)

#decoder_layer2= autoencoder.layers[-4]
#o2=decoder_layer2(o1)

#decoder_layer3= autoencoder.layers[-3]
#o3=decoder_layer3(o2)

decoder_layer4= autoencoder.layers[-2]
o4=decoder_layer4(o1)

decoder_layer5= autoencoder.layers[-1]
o5=decoder_layer5(o4)

# create the decoder model
decoder = Model(encoded_input,o5)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# feeding our own biometric data
path_iris= '/home/govardhan/Goverdhan_Program/auto_encoder/output_pizza2'
path_ear= '/home/govardhan/Goverdhan_Program/auto_encoder/input_pizza2'
#path_face= '/home/biometric/Auto/Autoencoder_data/Face'
#path_fingerprint= '/home/biometric/Auto/Autoencoder_data/Fingerprint'
#path_iris= '/home/biometric/Auto/Autoencoder_data/Iris'
#path_fkp= '/home/biometric/Auto/Auto/fkp'
#path_palm= '/home/biometric/Auto/Auto/palm'
list_ear = os.listdir(path_ear)
list_iris= os.listdir(path_iris)
#list_fingerprint = os.listdir(path_fingerprint)
#list_iris = os.listdir(path_iris)
#list_face = os.listdir(path_face)
#list_knuckle = os.listdir(path_fkp)
#list_palm = os.listdir(path_palm)
ear_samples = size(list_ear)
iris_samples = size(list_iris)
#fkp_samples = size(list_knuckle)
#iris_samples = size(list_iris)
#face_samples = size(list_face)
#fingerprint_samples = size(list_fingerprint)
#palm_samples = size(list_palm)
num_samples = ear_samples #+ fkp_samples  + palm_samples 
#print num_samples

#Processing for ear images
ear_imlist = os.listdir(path_ear)
iris_imlist = os.listdir(path_iris)
ear_imlist.sort()
iris_imlist.sort()
#print ear_imlist
ear_im1 = array(Image.open(path_ear + '/'+ ear_imlist[0])) # open one ear image to get size
ear_m,ear_n = ear_im1.shape[0:2] # get the size of the ear images
ear_imnbr = len(ear_imlist) # get the number of ear images
iris_im1 = array(Image.open(path_iris + '/'+ iris_imlist[0])) # open one ear image to get size
iris_m,iris_n = iris_im1.shape[0:2] # get the size of the ear images
iris_imnbr = len(iris_imlist)

# create matrix to store all flattened ear images
ear_immatrix = array([array(Image.open(path_ear + '/' + ear_im2)).flatten()
              for ear_im2 in ear_imlist],'object')
iris_immatrix = array([array(Image.open(path_iris + '/' + iris_im2)).flatten()
              for iris_im2 in iris_imlist],'object')

#Processing for fkp images
#fkp_imlist = os.listdir(path_fkp)
#print fkp_imlist
#fkp_im1 = array(Image.open('/home/biometric/Auto/Auto/fkp' + '/'+ fkp_imlist[0])) # open one fkp image to get size
#fkp_m,fkp_n = fkp_im1.shape[0:2] # get the size of the fkp images
#fkp_imnbr = len(fkp_imlist) # get the number of fkp images

# create matrix to store all flattened fkp images
#fkp_immatrix = array([array(Image.open('/home/biometric/Auto/Auto/fkp' + '/' + fkp_im2)).flatten()
              #for fkp_im2 in fkp_imlist],'object')

#Processing for palm images
#palm_imlist = os.listdir(path_palm)
#print palm_imlist
#palm_im1 = array(Image.open('/home/biometric/Auto/Auto/palm' + '/'+ palm_imlist[0])) # open one fkp image to get size
#palm_m,palm_n = palm_im1.shape[0:2] # get the size of the palm images
#palm_imnbr = len(palm_imlist) # get the number of palm images

# create matrix to store all flattened fkp images
#palm_immatrix = array([array(Image.open('/home/biometric/Auto/Auto/palm' + '/' + palm_im2)).flatten()
              #for palm_im2 in palm_imlist],'object')

# Combining each matrix
#immatrix = numpy.concatenate((ear_immatrix, fkp_immatrix), axis=0)
#immatrix = numpy.concatenate((immatrix, palm_immatrix), axis=0)
label=numpy.ones((num_samples,),dtype = int)
immatrix1=ear_immatrix
immatrix2=iris_immatrix


label=immatrix2
#label[102:201]=1
#label[202:]=2

data,Label = shuffle(immatrix1,label, random_state=2)
train_data = [data,Label]

(X, y) = (train_data[0],train_data[1])
# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


x_train = X_train.astype('float32') / 255.
x_test = X_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), numpy.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), numpy.prod(x_test.shape[1:])))
y_train = y_train.astype('float32') / 255.
y_test = y_test.astype('float32') / 255.
y_train = y_train.reshape((len(y_train), numpy.prod(y_train.shape[1:])))
y_test = y_test.reshape((len(y_test), numpy.prod(y_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)
print(y_train.shape)
print(y_test.shape)
autoencoder.fit(x_train, y_train,
                epochs=3,
                batch_size=300,
                shuffle=True,
                validation_data=(x_test, y_test))
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
print("Accuracy=",1-np.mean(abs(y_test-decoded_imgs)),'\n')
import matplotlib.pyplot as plt
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(32, 32, 3))
   #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
   # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
