import os
#import preprocess
import test
import itertools
import numpy as np

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling2D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD,Adam
from keras.utils import np_utils

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define custom generator


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes

    # Returns
        A binary matrix representation of the input.
    '''
    print(y)
    print(nb_classes)
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

"""def generate_vid_samples(con_vid_arr):
    while 1:
        save_path = os.path.join(con_vid_arr,os.path.basename(con_vid_arr))
        data_save_path = save_path + '_data.npy'
        labels_save_path = save_path + '_labels.npy'

        all_data = np.load(data_save_path)
        all_labels = np.load(labels_save_path)
        
        for vid_sample, label in itertools.izip(all_data, all_labels):
        # create numpy arrays of input data
        # and labels, from each line in the file
            x = np.array(vid_sample)
            x = np.expand_dims(x, axis=0)
            y = np.array(label)
            y = np.expand_dims(y, axis=0)
        yield (x, y)"""


def def_model(model_dir):
    '''
    Modify the original C3D model:
    Drop the last three layers (alongwith dropouts), and add two new fc layers(along with dropout), 
    named differently, to train them on a different dataset.
    '''
    # Load the model from json file
    model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    # Load Model Architecture
    print("[Info] Reading model architecture...")
    model = model_from_json(open(model_json_filename, 'r').read())
    print("[Info] .. Done")

    # Remove last three fc layers
    model.layers.pop() # Remove fc8 layer
    model.layers.pop() # Remove dropout_2 layer
    model.layers.pop() # Remove fc7 layer
    model.layers.pop() # Remove dropout_1 layer
    model.layers.pop() # Remove fc6 layer
    #model.layers.pop() #Remove the flatten layer

    # Debug:
    # model.summary()

    ########## Add STN + three new fc layers ###########
    prev_output = model.layers[-1].output

    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]

    # Localization net for STN
    input_shape = prev_output
    """locnet = Sequential()
    locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
    locnet.add(Convolution2D(20, (5, 5)))
    locnet.add(MaxPooling2D(pool_size=(2,2)))
    locnet.add(Convolution2D(20, (5, 5)))

    locnet.add(Flatten())
    locnet.add(Dense(50))
    locnet.add(Activation('relu'))
    locnet.add(Dense(6, weights=weights))

    stl = SpatialTransformer(localization_net=locnet, 
        output_size=(64,64), input_shape=input_shape)(prev_output)"""
    
    #fc6_afew = Dense(4096, activation='relu', name='fc6_afew')(stl)
    #temp=Flatten(input_shape)
    fc6_afew = Dense(4096, activation='relu', name='fc6')(prev_output)
    dropout_1_afew = Dropout(.5)(fc6_afew)
    
    fc7_afew = Dense(4096, activation='relu', name='fc7')(dropout_1_afew)
    dropout_2_afew = Dropout(.5)(fc7_afew)
    
    fc8_afew = Dense(2, activation='softmax', name='fc8_afew')(prev_output)

    #fc=Dense(4096,activation='relu')(prev_output)
    #fc=Dropout(0.5)(fc)
    #fc=Dense(2048,activation='relu')(fc)
    #fc=Dropout(0.4)(fc)
    #fc=Dense(1000,activation='relu')(fc)
    #fc=Dropout(0.4)(fc)
    #fc=Dense(500,activation='relu')(fc)
    #fc=Dropout(0.4)(fc)
    #fc=Dense(100,activation='relu')(prev_output)
    #fc=Dropout(0.5,name='dropout2')(fc)
    #fc=Dense(50,activation='relu')(fc)
    #fc=Dropout(0.5,name='dropout3')(fc)  
    #predictions=Dense(12,activation='softmax')(prev_output)

    model_new = Model(model.input,fc8_afew)
    #model_new.summary()
    for layer in model.layers:
        layer.trainable = False

    # Load corresponding model weights
    print("[Info] Loading model weights...")
    model_new.load_weights(model_weight_filename, by_name=True)
    print("[Info] .. Done")

    return model_new

def train(model,train_X,train_Y,test_X,test_Y,batch_size):
	num_epochs=1000	
	max_acc=0
	for epoch in range(0,num_epochs):
		print ("Epoch is: %d\n" % epoch)
		print ("Number of batches: %d\n" % int(train_X.shape[0]/batch_size))
		num_batches=int(train_X.shape[0]/batch_size)
		for batch in range(num_batches):
			checkpointer = ModelCheckpoint(filepath="saved_models/model1.h5", verbose=1, save_best_only=True,save_weights_only=True,monitor='val_acc',mode='max')
			callbacks_list=[checkpointer]
			batch_train_X=train_X[batch*batch_size:min((batch+1)*batch_size,train_X.shape[0])]
			batch_train_Y=train_Y[batch*batch_size:min((batch+1)*batch_size,train_Y.shape[0])]
			(loss,accuracy)=model.train_on_batch(batch_train_X,batch_train_Y)
			print ('epoch_num: %d batch_num: %d loss: %f accuracy: %f\n' % (epoch,batch,loss,accuracy))
		acc=model.evaluate(test_X,test_Y,32)
		print("Epoch: %d loss: %f test_accuracy: %f \n" % (epoch,acc[0],acc[1]))
		if(acc[1]>max_acc):
			model.save_weights("saved_models/model1.h5")
			print("accuracy increased from %f to %f\n" % (max_acc,acc[1]))
			print("model weights saved")
			max_acc=acc[1]
		
def main():
    num_classes =2
    batch_size = 64
    num_epochs = 5000
    # train_dir = 'data/afew_2016_modified/Train'
    # val_dir = 'data/afew_2016_modified/Val'
    # val_dir = 'data/afew_frames/Val'

    # print "[Info] Loading training data"
    # train_X, train_Y = preprocess.vid_to_arr(train_dir)
    

    # print "[Info] Loading validation data"
    # val_X, val_Y = preprocess.vid_to_arr(val_dir)    
    train_X, train_Y,test_X,test_Y= test.get_input(0.8)
    
    #train_X,train_Y,test_X,test_Y=train_test_split(train_X,train_Y,test_size=0.2,random_state=2)
    #print(train_Y)
    # Load model
    model = def_model('models')
    # checkpointer = ModelCheckpoint(filepath="./saved_weights/fine_tuned_weights.hdf5", verbose=1, save_best_only=True)
    sgd=SGD(lr=0.0001,momentum=0.9,decay=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    model.summary()

    model_json = model.to_json()
    with open("saved_models/model1.json", "w") as json_file:
        json_file.write(model_json)
        print("json saved")

    
    train_Y = to_categorical(train_Y, num_classes)
    test_Y = to_categorical(test_Y, num_classes)
    # Train model
    print(train_X.shape)
    print(train_Y.shape)

    """checkpointer = ModelCheckpoint(filepath="saved_models/model1.h5", verbose=1, save_best_only=True,save_weights_only=True,monitor='val_acc',mode='max')
    callbacks_list=[checkpointer]

    hist = model.fit(train_X,train_Y,validation_data=(test_X,test_Y),batch_size=batch_size,nb_epoch=num_epochs,shuffle=True,verbose=1,callbacks=callbacks_list)
	"""
    train(model,train_X,train_Y,test_X,test_Y,32)
    # Evaluate the model
    out = score = model.evaluate(
        test_X,
        test_Y,
        batch_size=batch_size
        )

    mnames = model.metrics_names

    print ("[Info] Results")
    for mname, metric in itertools.izip(mnames, out):
        print ("\t{}: {}".format(mname, metric))

    # Train generator
    # train_gen = generate_vid_samples(train_dir)
    # val_gen = generate_vid_samples(val_dir)
    # steps_per_epoch = 173
    # validation_steps = 289

    # model.fit_generator(train_gen, steps_per_epoch, epochs=5, verbose=1, workers=10)
    # model.evaluate_generator(val_gen, validation_steps, workers=10)
"""def train(BATCH_SIZE):
    
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()                     #shape of X_train= (60000,28,28)

    #X_train = (X_train.astype(np.float32) - 127.5)/127.5                         #normalising image data
    #X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])         #shape of X_train= (60000,1,28,28)
    X_train=np.load('/home/inderpreet/DeepLearning/DCGan/Mat.npy')
    print(X_train.shape)                                        #array of noise z with rows=batch_size and 100 columns 
    for epoch in range(100):
        print "Epoch is:", epoch
        print "Number of batches:", int(X_train.shape[0]/BATCH_SIZE)
	a=int(X_train.shape[0]/BATCH_SIZE)
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):                   # to iterate over total number of batches
                                   # noise is arranged in range -1 to +1 for each image
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]             # getting batch of training images                 #is it giving whole batch of noise as input to gen. 
            
     
            
                 #row vector having 1 upto batch size and 0 upto batch size 
            
	    	d_loss = discriminator.train_on_batch(X, y)             #data is given to discriminator and it learns to distinguish 
            print("batch %d d_loss : %f" % (index, d_loss))         #between real and generated data and in doing so its loss is 
                                                      #computed
	    for i in range(BATCH_SIZE):                             #noise is generated for full batch of images 
                noise[i, :] = np.random.uniform(-1, 1, 100)
            
	    discriminator.trainable = False                         #We stopped the discriminator to train because we just did it
            generated_images1 = generator.predict(noise, verbose=0)
            
	    g_loss = discriminator_on_generator.train_on_batch(     #We train combined model including both gen. and dis. giving noise 
                noise, [1] * BATCH_SIZE)                            #as input and ground truth of 1 because we want them to be real and 
                                                                    #loss is computed based on dis. output and ground truth 
	    discriminator.trainable = True                          #training of dis. is again activated
            
	    print("batch %d g_loss : %f" % (index, g_loss))
            
	    if index % 10 == 0:                                     #saving weights of both dis. and gen.
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)
            #Gloss.append(g_loss)
	    #Gloss1=np.array(Gloss)
	    #Dloss1=np.array(Dloss)
	    #print(Gloss1)
	    #print(Dloss1)
	    #graph(a,Gloss1,Dloss1)
"""

if __name__ == '__main__':
    main()
