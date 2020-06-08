"""1. Getting Started with FCN Pre-trained Models
==============================================

This is a quick demo of using GluonCV FCN model on PASCAL VOC dataset.
Please follow the `installation guide <../../index.html#installation>`__
to install MXNet and GluonCV if not yet.
"""
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
import time
# using cpu
ctx = mx.cpu(0)


##############################################################################
# Prepare the image
# -----------------
#
# download the example image
# url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/1.jpg'
filename = 'example.jpg'
# gluoncv.utils.download(url, filename)

##############################################################################
# load the image
img = image.imread(filename)
img = image.fixed_crop(src=img, x0=0, y0=0, w =img.shape[1], h=img.shape[0], size=(224, 224))
from matplotlib import pyplot as plt
# plt.imshow(img.asnumpy())
# plt.show()

##############################################################################
# normalize the image using dataset mean
from gluoncv.data.transforms.presets.segmentation import test_transform
img = test_transform(img, ctx)

##############################################################################
# Load the pre-trained model and make prediction
# ----------------------------------------------
#
# get pre-trained model
model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)

##############################################################################
# make prediction using single scale
output = model.predict(img)

tic = time.perf_counter()
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
toc = time.perf_counter()
print("Proccesed in {}".format(toc-tic))  

##############################################################################
# Add color pallete for visualization
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'pascal_voc')
mask.save('output.png')

##############################################################################
# show the predicted mask
mmask = mpimg.imread('output.png')
plt.imshow(mmask)
plt.show()

##############################################################################
# More Examples
# -------------
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/4.jpg
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/4.png
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/5.jpg
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/5.png
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/6.jpg
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/6.png
#    :width: 45%