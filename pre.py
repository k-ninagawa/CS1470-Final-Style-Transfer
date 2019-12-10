
import os
import numpy as np
import tensorflow as tf


'''
:param file_path: file_path of the data files
:param batch_sizes: batch size by which we divide the dataset
:return: A DirectoryIterator of tuples of (x, y) where x is a numpy array
containing a batch of images with shape (batch_size, *target_size, channels) and
 y is a numpy array of corresponding labels.
'''
def get_data(file_path, batch_sizes):
    #set up the ImageDataGenerator with rescaleing factor of 1/255 for normalization
    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255)
    #get the shuffled data with size (batch_size, 256 , 256, 3)

    '''
    We might need to change this to (batch_size, 252, 252, 3) to match up with the input for VGG19/VGG16
    '''
    data_set = train_data_generator.flow_from_directory(file_path, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=batch_sizes, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='jpg', follow_links=True, subset=None, interpolation='nearest')




    return data_set
