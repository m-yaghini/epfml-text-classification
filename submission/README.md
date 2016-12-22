# epfml-text-classification
EPFL ML 2016 Project 2: Michael Meyer, Sylvain Beaud, Mohammad Yaghini

To run the files present in our project, you will need to install some requierments.

The main model is produced using Keras which is a high-level library for neural networks Python and it runs on top of either TensorFlow or Theano. 
In our case, we used it on the top of TensorFlow. We can simply install Keras by following these instructions : https://keras.io/#installation

You can execute the run.py file which produces the predicition based on the saved model "model.h5". This model is the result of the other file train_model.py which execute the following steps :
First, we clean the tweets using our cleaning functions and we load the pretrained embeddings produced with Fasttext. Then, the data are passed to the model which is a Convolutional Neural Network.

TODO : - More detailed instructions
       - Explain the role of each file in the folder
       - Briefly explain the final model maybe ??? 
