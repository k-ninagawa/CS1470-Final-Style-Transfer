
import os
from PIL import Image
import numpy as np
import tensorflow as tf
def get_imlist(path):
  """  Returns a list of filenames for
    all jpg images in a directory. """

  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
traindatagen = tf.keras.preprocessing.image.ImageDataGenerator()
data_set = traindatagen.flow_from_directory('./data', target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=2, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='jpg', follow_links=False, subset=None, interpolation='nearest')
print((data_set[0][0]))
print("============")
list = (get_imlist("./val_256"))

image_list = []
counter = 0
for i in list:
    counter += 1
    if (counter % 2 == 0):
        break
        print(counter)
    im = Image.open(i)
    np_im = np.array(im)
    if(np_im.size == 196608):
        image_list.append(np_im)
    print(np_im)



print(np.array(image_list).shape)
image_list = (np.array(image_list).astype('float32') /255 )
image_list = np.transpose(image_list,[0, 3, 1, 2])
print(image_list.shape)
