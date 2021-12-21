
# To get a feel for what kind of features our convnet has learned, 
# one fun thing to do is to visualize how an input gets transformed 
# as it goes through the convnet.

# Let's pick a random old or modern image from the training set, 
# and then generate a figure where each row is the output of a layer, 
# and each image in the row is a specific filter in that output feature map. 
# Rerun this cell to generate intermediate representations for a variety 
# of training images.

import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def convnet_visualization(model, training_dir, target_size=(224, 224)):
    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.

    successive_outputs = [layer.output for layer in model.layers[1:]]

    #visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
    
    training_img_list = [os.path.join(training_dir, training_subdir, item) 
                         for training_subdir in os.listdir(training_dir) 
                         for item in os.listdir(os.path.join(training_dir, training_subdir))]

    img_path = random.choice(training_img_list)    
    img = load_img(img_path, target_size=target_size)  # this is a PIL image

    x   = img_to_array(img)                           # Numpy array with shape (224, 224, 3)
    x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 224, 224, 3)

    # Rescale by 1/255
    x /= 255.0

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    # -----------------------------------------------------------------------
    # Now let's display our representations
    # -----------------------------------------------------------------------
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
      
      if len(feature_map.shape) == 4:
        
        #-------------------------------------------
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        #-------------------------------------------
        n_features = feature_map.shape[-1]  # number of features in the feature map
        size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
        
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
        
        #-------------------------------------------------
        # Postprocess the feature to be visually palatable
        #-------------------------------------------------
        for i in range(n_features):
            x  = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std ()
            x *=  64
            x += 128
            x  = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

        #-----------------
        # Display the grid
        #-----------------

        scale = 20. / n_features
        plt.figure( figsize=(scale * n_features, scale) )
        plt.title ( layer_name )
        plt.grid  ( False )
        plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 




def plot_training_curve(history, training=(None, None), validation=(None, None)):
    # Retrieve metric performance and loss

    train_perf = history.history[training[0]]
    val_perf = history.history[validation[0]]
    train_loss = history.history[training[1]]
    val_loss = history.history[validation[1]]

    epochs = range(len(train_perf)) # Get the number of epochs

    #--------------------------------------------------------
    # Plot training and validation performance per epoch
    #--------------------------------------------------------
    plt.plot(epochs, train_perf, label='training')
    plt.plot(epochs, val_perf, label='validation')
    plt.title('Training and validation perf')
    plt.legend()
    plt.figure()

    #--------------------------------------------------------
    # Plot training and validation loss per epoch
    #--------------------------------------------------------
    plt.plot(epochs, train_loss, label='training')
    plt.plot(epochs, val_loss, label='validation')
    plt.title('Training and validation loss')
    plt.legend()
    plt.figure()
    