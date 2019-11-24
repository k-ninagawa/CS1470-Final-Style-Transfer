import glob
import numpy as np
import os
import sys

from pylab import *
import numpy as np
import tensorflow as tf


# input

image_size = 50
max_read = 280
X = []
Y = []

import tarfile,os
import sys
import tarfile
tf = tarfile.open("val_256.tar")
tf.extractall()
