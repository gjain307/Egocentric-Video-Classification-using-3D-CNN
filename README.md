# Egocentric-Video-Classification-using-3D-CNN
Classification of egocentric and Non-egocentric videos by calculating optical flow using autoencoder
The input to the autoencoder is the single image which consist of two frames at t and t+1 time
embedded on a single image. The whole video dataset was normalized to 15 frames per
second(15fps) and each video is divided into clips of 4s (60) frames and then the optical flow
is computed between 2 consecutive frames which is the groundtooth optical flow which is
compared with the predicted optical flow and backpropagates the network to minimize the
loss between computed and predicted optical flow.

After training the autoencoder the predicted optical flow is given to the 3d convolutional
neural network which classifies the video as egocentric or not.

From the results we can absolutely say that the autoencoder can successfully learn optical
flow from the pair of images which can be shown in the above figure (c). Now this predicted
optical flow will be used to test whether the Video is egocentric or not using 3-d CNN
architecture which is shown in fig4. In 3d CNN network the input to the network is optical
flow which is derived from 4 seconds of clips. The results after training the model for about
10 hours (1Lakh optical flow images) shows the testing accuracy of 98.6%. The model is
trained to predict that if the video is egocentric or not. In this work cascading of both the
networks (autoencoder and 3d CNN network) will be done to check if there are some new
features learned by the network or not. Actually the novelty in this technique is that we are
trying to make the system fully automated that is the user has to just show the video and the
whole network (autoencoder+3d CNN) will determine whether the video is ego-centric or
not.
