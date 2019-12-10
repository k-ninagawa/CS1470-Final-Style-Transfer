import tensorflow as tf
import numpy as np
import PIL.Image
import os
from imageio import imwrite
import sys
from pre import get_data

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
    self.conv1 = tf.keras.layers.Conv2D(32, kernel_size = (9,9), strides = 1, padding = "SAME")
    self.batch_norm_1 = tf.keras.layers.BatchNormalization()
    self.relu_1 = tf.keras.layers.ReLU()

    self.conv2 = tf.keras.layers.Conv2D(64, kernel_size = (3,3), strides = 2, padding = "SAME")
    self.batch_norm_2 = tf.keras.layers.BatchNormalization()
    self.relu_2 = tf.keras.layers.ReLU()

    self.conv3 = tf.keras.layers.Conv2D(128, kernel_size = (3,3), strides = 2, padding = "SAME")
    self.batch_norm_3 = tf.keras.layers.BatchNormalization()
    self.relu_3 = tf.keras.layers.ReLU()

    self.res1 = ResidualBlock(128)
    self.res2 = ResidualBlock(128)
    self.res3 = ResidualBlock(128)
    self.res4 = ResidualBlock(128)
    self.res5 = ResidualBlock(128)

    self.conv4 = tf.keras.layers.Conv2DTranspose(64, kernel_size = (3,3), strides = 2, padding = "SAME")
    self.batch_norm_4 = tf.keras.layers.BatchNormalization()
    self.relu_4 = tf.keras.layers.ReLU()

    self.conv5 = tf.keras.layers.Conv2DTranspose(32, kernel_size = (3,3), strides = 2, padding = "SAME")
    self.batch_norm_5 = tf.keras.layers.BatchNormalization()
    self.relu_5 = tf.keras.layers.ReLU()

    self.conv6 = tf.keras.layers.Conv2DTranspose(3, kernel_size = (9,9), strides = 1, padding = "SAME")
    self.batch_norm_6 = tf.keras.layers.BatchNormalization()
    self.relu_6 = tf.keras.layers.ReLU()

    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    self.vgg = tf.keras.Model([vgg.input], outputs)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.content_layers_1 = content_layers_1
    self.num_style_layers = len(style_layers)
    self.opt = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.99, epsilon=1e-1)
  def get_image(self, x):

    conv1 = self.conv1(x)

    batch_norm_1 = self.batch_norm_1(conv1)

    relu_1 = self.relu_1(batch_norm_1)

    conv2 = self.conv2(relu_1)

    batch_norm_2 = self.batch_norm_2(conv2)

    relu_2 = self.relu_2(batch_norm_2)

    conv3 = self.conv3(relu_2)

    batch_norm_3 = self.batch_norm_3(conv3)

    relu_3 = self.relu_3(batch_norm_3)

    res_1 = self.res1.forward(relu_3)
    res_2 = self.res2.forward(res_1)
    res_3 = self.res3.forward(res_2)
    res_4 = self.res4.forward(res_3)
    res_5 = self.res5.forward(res_4)


    conv4 = self.conv4(res_5)
    batch_norm_4 = self.batch_norm_4(conv4)
    relu_4 = self.relu_4(batch_norm_4)

    conv5 = self.conv5(relu_4)
    batch_norm_5 = self.batch_norm_5(conv5)
    relu_5 = self.relu_5(batch_norm_5)


    conv6 = self.conv6(relu_5)
    batch_norm_6 = self.batch_norm_6(conv6)
    relu_6 = self.relu_6(batch_norm_6)
    result = tf.keras.backend.clip(relu_6, 0.1, 255)

    return (result ) / 255
  def call(self, inputs):

    conv1 = self.conv1(inputs)

    batch_norm_1 = self.batch_norm_1(conv1)

    relu_1 = self.relu_1(batch_norm_1)

    conv2 = self.conv2(relu_1)

    batch_norm_2 = self.batch_norm_2(conv2)

    relu_2 = self.relu_2(batch_norm_2)

    conv3 = self.conv3(relu_2)

    batch_norm_3 = self.batch_norm_3(conv3)

    relu_3 = self.relu_3(batch_norm_3)

    res_1 = self.res1.forward(relu_3)
    res_2 = self.res2.forward(res_1)
    res_3 = self.res3.forward(res_2)
    res_4 = self.res4.forward(res_3)
    res_5 = self.res5.forward(res_4)



    conv4 = self.conv4(res_5)
    batch_norm_4 = self.batch_norm_4(conv4)
    relu_4 = self.relu_4(batch_norm_4)

    conv5 = self.conv5(relu_4)
    batch_norm_5 = self.batch_norm_5(conv5)
    relu_5 = self.relu_5(batch_norm_5)


    conv6 = self.conv6(relu_5)
    batch_norm_6 = self.batch_norm_6(conv6)
    relu_6 = self.relu_6(batch_norm_6)



    result = tf.keras.backend.clip(relu_6, 0 , 255)
    inputs = (result ) / 255
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
  def style_content(self, inputs):

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
    style_weight=4.65e-2
    content_weight=60
    style_outputs = outputs[1]
    content_outputs = outputs[0]
    style_loss = tf.reduce_sum([tf.reduce_mean(tf.square(style_outputs[name]-style_targets[name])) for name in style_outputs.keys()]) * style_weight / num_style_layers
    content_loss_1 = tf.reduce_sum([tf.reduce_mean(tf.square(content_outputs[name]-content_targets_1[name])) for name in content_outputs.keys()]) * 0
    content_loss = tf.reduce_sum([tf.reduce_mean(tf.square(content_outputs[name]-content_targets[name])) for name in content_outputs.keys()]) * content_weight / num_content_layers
    loss = style_loss + content_loss

    return loss

@tf.function()
def train(image, extractor, style_targets, content_targets, num_style_layers, num_content_layers, style_targets_1):
  opt = extractor.opt

  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = loss_function(outputs, style_targets, content_targets, num_style_layers, num_content_layers, style_targets_1)
    loss += tf.reduce_sum(tf.image.total_variation(image)) * 1e-5
  grads = tape.gradient(loss, extractor.trainable_variables)
  opt.apply_gradients(zip(grads, extractor.trainable_variables))


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(channels, kernel_size = (3,3), strides = 1, padding = "same")

        self.norm1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(channels, kernel_size = (3,3), strides = 1, padding = "same")

        self.norm2 = tf.keras.layers.BatchNormalization()

    @tf.function()
    def forward(self, x):
        residual = x

        out = self.relu(self.norm1(self.conv1(x)))

        out = self.norm2(self.conv2(out))

        out = out + residual
        return out

def main():
    name = sys.argv[1]
    height = sys.argv[2]
    width = sys.argv[3]
    mode = sys.argv[4]

    content_path = tf.keras.utils.get_file('ken.jpg', 'https://scontent.fzty2-1.fna.fbcdn.net/v/t1.0-9/69497958_2478425912438909_3450352180421197824_n.jpg?_nc_cat=101&_nc_ohc=V4z59bUbz1oAQlwF5rMLHtqypJHMG_t17mpiHmFlL-AKOhXrEHJnVc9qw&_nc_ht=scontent.fzty2-1.fna&oh=29ebb194584872745009fbe4e856a904&oe=5E834777')
    style_path = tf.keras.utils.get_file('s.jpg', 'https://iheartintelligence.com/wp-content/uploads/2015/09/THE-STARRY-NIGHT.jpg')
    content_path_1 = tf.keras.utils.get_file('s.jpg', 'https://iheartintelligence.com/wp-content/uploads/2015/09/THE-STARRY-NIGHT.jpg')
    content_image, style_image, content_image_1 = get_images(content_path,style_path,content_path_1, height, width)

    dataset = get_data('./data_1', 4)



    content_layers = ['block2_conv2']
    style_layers = ['block1_conv1', 'block2_conv2', 'block3_conv3','block4_conv3']
    content_layers_1 = ['block5_conv2']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    extractor = MergeModel(style_layers, content_layers, content_layers_1,name)

    style_targets = extractor.style_content(style_image)[1]
    content_targets_pic = extractor.style_content(content_image)[0]




    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(extractor=extractor)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    if not os.path.exists('./out_dir_pic'):
        os.makedirs('./out_dir_pic')
    image = tf.constant(content_image)

    if (mode == "test"):
        checkpoint.restore(manager.latest_checkpoint)
        im = ((extractor.get_image(image).numpy())*255)[0]
        tensor = np.array(im, dtype=np.uint8)
        s = './out_dir_pic'+'/'+'test'+'.png'
        imwrite(s, im)
        return

    if(mode == "train"):
        epochs = 3
        steps_per_epoch = 8000
        for i in range(epochs):
          for j in range(steps_per_epoch):
            print(i,"and", j)
            content_targets_1 = extractor.style_content(tf.constant(dataset[j][0], dtype = 'float32'))[0]
            content_targets = extractor.style_content(tf.constant(dataset[j][0], dtype = 'float32'))[0]
            train(image, extractor, style_targets, content_targets, num_style_layers, num_content_layers, content_targets_1)
            train(image, extractor, style_targets, content_targets_pic, num_style_layers, num_content_layers, content_targets_1 );
            if(j % 400 == 0):
                im = ((extractor.get_image(image).numpy())*255)[0]
                tensor = np.array(im, dtype=np.uint8)
                s = './out_dir_pic'+'/'+str(j)+'.png'
                imwrite(s, im)
                manager.save()
                print("==saving progress===")
          im = ((extractor.get_image(image).numpy())*255)[0]
          tensor = np.array(im, dtype=np.uint8)
          s = './out_dir_pic'+'/'+str(i)+'.png'
          imwrite(s, im)
          manager.save()
          print("==saving progress===")

if __name__ == '__main__':
   main()
