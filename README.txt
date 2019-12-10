Here are some details for each file.


tar_to_jpg.py:  convert tar file into jpg files and put it in the working directly.

pre.py: get the files from the directly of the specified path and batch size. Used in with_data.py and with_data_pic.py

real_good.py:

Takes three arguments: 1) name of the pretrained model you want to use. Highly recommended to use vgg16
2) Height of an output image. 3) Width of the output image. For 2), 3) highly recommended to set 256 256 or 400 400. If the size is too big
or bigger than the original image, it might raise an error.

It will train on content image(given a url to the image) and the style image(given url to an image).
It will save the progress into the out_dir directly and the checkpoint will be saved.


with_data.py:

This is the extension from the real_good.py. It will train on the dataset and the style image from given url.

Instead of training on the specific content image, it will train on the images(batch size 4 as suggested in the paper) but with
the consistent style image.

with_data_pic.py:

Same as above. Only difference is that it trains between the dataset and the content target back and forth, so that we get
a stable image for the content image.


Merge.py:

This is an experimental code where it tries to merge the two images. It got a hint from the first model by extending the idea and
it also includes the content loss from the style image so that the generated image have the content from both style and content image but with
consistent style.


project.py:

Image style transfer code reimplemented in pytorch.
