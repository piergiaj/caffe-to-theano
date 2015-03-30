import numpy as np
import matplotlib.cm as cm
from PIL import Image
import pylab
import theano
import theano.tensor as T
import cPickle, gzip
import scipy.io as io
from scipy.ndimage import imread
from scipy.misc import imresize

mean_image=np.load('ilsvrc_2012_mean.npy')
mean_image = np.swapaxes(mean_image, 0, 1)
mean_image = np.swapaxes(mean_image, 1, 2)

def get_img(img_name):
    img = imread(img_name)
    if img.shape[0] != img.shape[1]:
        if img.shape[0] < img.shape[1]:
            img = imresize(img, (256, int((256.0/img.shape[0])*img.shape[1])))
            c = img.shape[1]/2
            img = img[:,c-128:c+128,:]
        else:
            img = imresize(img, (int((256.0/img.shape[1])*img.shape[0]), 256))
            c = img.shape[0]/2
            img = img[c-128:c+128,:,:]
    img = imresize(img, (256, 256))
    if len(img.shape) == 2:
        img = np.asarray([img,img,img])
        img = np.swapaxes(img,0,1)
        img = np.swapaxes(img,1,2)
    if img.shape[2] == 4:
        img = img[:,:,0:3]
    img = img-mean_image
    img = imresize(img, (227, 227))
    img = np.asarray(img, dtype='float32')# / 255
    img = img.flatten()
    return img
