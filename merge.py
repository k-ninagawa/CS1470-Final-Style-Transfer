import tensorflow as tf
import numpy as np
import PIL.Image
import os
from imageio import imwrite
import sys

def get_images(path_to_img, path_to_img1, path_to_img2, height, width):
  height = int(height)
  width = int(width)
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)

  img1 = tf.io.read_file(path_to_img1)
  img1 = tf.image.decode_image(img1, channels=3)
  img1 = tf.image.convert_image_dtype(img1, tf.float32)

  shape1 = tf.cast(tf.shape(img1)[:-1], tf.float32)

  img2 = tf.io.read_file(path_to_img2)
  img2 = tf.image.decode_image(img2, channels=3)
  img2 = tf.image.convert_image_dtype(img2, tf.float32)

  shape2 = tf.cast(tf.shape(img2)[:-1], tf.float32)

  height_1 = min(shape[0], shape1[0],  shape2[0])

  width_1 = min(shape[1], shape1[1],  shape2[1])


  new_shape = tf.constant([height, width], dtype = 'int32')
  if(height == 0 and width == 0):
     new_shape = tf.constant([height_1, width_1], dtype = 'int32')
  img = tf.image.resize(img, (new_shape))
  img = img[tf.newaxis, :]
  img1 = tf.image.resize(img1, new_shape)
  img1 = img1[tf.newaxis, :]
  img2 = tf.image.resize(img2, new_shape)
  img2 = img2[tf.newaxis, :]

  return img, img1, img2

def gram_matrix(input_tensor):

  input_shape = tf.shape(input_tensor)
  reshaped = tf.reshape(input_tensor, [-1, input_shape[3]])
  result = tf.matmul(reshaped, reshaped, transpose_a=True)
  return result/tf.cast(input_shape[1]*input_shape[2], tf.float32)


class MergeModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers, content_layers_1, name_model):
    super(MergeModel, self).__init__()
    self.name_model = name_model
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    if(name_model == "vgg16"):
        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    if(name_model == "vgg19"):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    self.vgg = tf.keras.Model([vgg.input], outputs)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.content_layers_1 = content_layers_1
    self.num_style_layers = len(style_layers)
    self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

  def call(self, inputs):
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg16.preprocess_input(inputs)
    if(self.name_model == "vgg16"):
        preprocessed_input = tf.keras.applications.vgg16.preprocess_input(inputs)
    if(self.name_model == "vgg19"):
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs,content_outputs_1 = (outputs[:self.num_style_layers], outputs[self.num_style_layers:], outputs[:self.num_style_layers])

    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

    content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
    content_dict_1 = {content_name:value for content_name, value in zip(self.content_layers_1, content_outputs_1)}
    style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
    content_dict.update(content_dict_1)
    return (content_dict, style_dict)



def loss_function(outputs, style_targets, content_targets, num_style_layers, num_content_layers, content_targets_1):
    style_weight=1e-2
    content_weight=1e4
    style_outputs = outputs[1]
    content_outputs = outputs[0]
    style_loss = tf.reduce_sum([tf.reduce_mean(tf.square(style_outputs[name]-style_targets[name])) for name in style_outputs.keys()]) * style_weight / num_style_layers
    content_loss_1 = tf.reduce_sum([tf.reduce_mean(tf.square(content_outputs[name]-content_targets_1[name])) for name in content_outputs.keys()]) * 10000
    content_loss = tf.reduce_sum([tf.reduce_mean(tf.square(content_outputs[name]-content_targets[name])) for name in content_outputs.keys()]) * content_weight / num_content_layers
    loss = style_loss + content_loss + content_loss_1

    return loss

@tf.function()
def train(image, extractor, style_targets, content_targets, num_style_layers, num_content_layers, style_targets_1):
  opt = extractor.opt

  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = loss_function(outputs, style_targets, content_targets, num_style_layers, num_content_layers, style_targets_1)
    loss += tf.reduce_sum(tf.image.total_variation(image))
  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(tf.keras.backend.clip(image, 0 , 1))



def main():
    name = sys.argv[1]
    height = sys.argv[2]
    width = sys.argv[3]
    content_path = tf.keras.utils.get_file('ny2.jpg', 'https://live.staticflickr.com/7106/6976066570_40216d2823_b.jpg')
    style_path = tf.keras.utils.get_file('t1.jpg', 'https://fashion-basics.com/wp-content/uploads/2016/09/2016-9-tokyo-yakei-kode-date-001.jpg')
    content_path_1 = tf.keras.utils.get_file('t1.jpg', 'https://fashion-basics.com/wp-content/uploads/2016/09/2016-9-tokyo-yakei-kode-date-001.jpg')
    content_image, style_image, content_image_1 = get_images(content_path,style_path,content_path_1, height, width)
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1','block4_conv1','block5_conv1']
    content_layers_1 = ['block5_conv2']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    extractor = MergeModel(style_layers, content_layers, content_layers_1,name)
    style_targets = extractor(style_image)[1]
    content_targets_1 = extractor(content_image_1)[0]
    content_targets = extractor(content_image)[0]
    if not os.path.exists('./out_dir'):
        os.makedirs('./out_dir')
    image = tf.Variable(content_image)
    epochs = 30
    steps_per_epoch = 100
    for i in range(epochs):
      for j in range(steps_per_epoch):
        print(i,"and", j)
        train(image, extractor, style_targets, content_targets, num_style_layers, num_content_layers, content_targets_1)

      im = ((image.numpy())*255)[0]
      s = './out_dir'+'/'+str(i)+'.png'
      imwrite(s, im)

if __name__ == '__main__':
   main()
