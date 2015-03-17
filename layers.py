import numpy as np
import theano
import theano.tensor as T

import downsample # theano 0.6 doesn't have a max_pool that supports strides. Use
# this updated version
from theano.tensor.nnet import conv
from pylearn2.expr.normalize import CrossChannelNormalization
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import google.protobuf

import caffe_pb2 as caffe

DTYPE = 'float32' # or 'float64'
rng = RandomStreams()

class Layer(object):
    '''
      The base class for layers.
       layer is a caffe.NetParameter, blobs is the numpy arrays
             (see caffe.proto for the structure of the object)
    '''

    def __init__(self, in_size, layer, blobs):
        self.name = layer.name
        self.top = layer.top
        self.bottom = layer.bottom
        self.in_size = in_size

    def get_output_size(self):
	'''
	  Gets the new size of the output
	'''
        return [self.in_size[0], self.out_size, int(np.floor((self.in_size[2] + self.pad[0]+self.pad[1] - self.filter_shape[0])/self.stride[0])+1), int(np.floor((self.in_size[3]+self.pad[2]+self.pad[3]-self.filter_shape[1])/self.stride[1])+1)]


    def get_theano_function(self):
	'''
	 Gets the theano implementation of the layer
	'''
        pass

    def save_layer(self):
        pass

    def load_layer(self, layer_name):
        pass

class Convolution(Layer):

    def __init__(self, in_size, layer, blobs):
        super(Convolution, self).__init__(in_size, layer, blobs)

	# save all the important parameters
        self.filter_shape = tuple([layer.convolution_param.kernel_size]*2)
        self.stride = tuple([layer.convolution_param.stride]*2) # subsample in theano
        self.pad = tuple([layer.convolution_param.pad]*4) # TODO: currently ignored...
        # TODO: pad (or pad_h and pad_w) [default 0]: specifies the number of pixels to (implicitly) add to each side of the input

        self.out_size = layer.convolution_param.num_output

	# this gets the full filter shape, as theano wants it
	# (num_filters, in_channels, rf_width, rf_height)
        self.theano_filter_shape = tuple([self.out_size, self.in_size[1]] + list(self.filter_shape))

        # get filters, or sets them to 0 if not given
        if len(blobs) >= 1:
            self.filters = blobs[0]
        else:
            self.filters = np.zeros(self.theano_filter_shape, dtype=DTYPE)

	# makes it a theano variable
        self.w = theano.shared(value=self.filters, name='w')

        # get biases, or sets to 0 if not given
        if len(blobs) >= 2:
            self.bias = blobs[1] # why squeeze?
        else:
            self.bias = np.zeros([1,self.out_size], dtype=DTYPE)
        self.b = theano.shared(value=self.bias, name='b')


    def get_theano_function(self, input):
	'''
	 Performs a convoltuion, with the strides (not padding yet)
	  then adds the bias
	'''
        conv_out = conv.conv2d(input, self.w, filter_shape=self.theano_filter_shape,
                               subsample=self.stride, image_shape=self.in_size)
        return conv_out + self.b

    def save_layer(self):
	'''
	  Saves the filters and bias as numpy files, easier and faster
	  for visualization
	'''
        np.save(open('layer_'+self.name+'_filters','w'), self.filters)
        np.save(open('layer_'+self.name+'_bias','w'), self.bias)

    def load_layer(self):
        self.filters = np.load(open('layer_'+self.name+'_filters','r'))
        self.w = theano.shared(value=self.filters, name='w')

        self.bias = np.load(open('layer_'+self.name+'_bias','r'))
        self.b = theano.shared(value=self.bias, name='b')
        
class ReLU(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(ReLU,self).__init__(in_size, layer, blobs)

    def get_theano_function(self, input):
        # this can be more complex, if negative_slope is inclded (but its not
	# in alexnet)
        return T.maximum(input, 0.0)

class LRN(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(LRN,self).__init__(in_size, layer, blobs)
        
        self.local_size = layer.lrn_param.local_size # the number of channels to sum over (for cross channel LRN) or the side length of the square region to sum over (for within channel LRN)
        self.alpha = layer.lrn_param.alpha # scaling param
        self.beta = layer.lrn_param.beta # exponent
        # TODO: can have a norm region as well (dealing with channels)

	# gets the lrn function from pylearn2 (which implements this in
	#  theano)
        self.lrn = CrossChannelNormalization(alpha=self.alpha, beta=self.beta, n=self.local_size)

    def get_theano_function(self, input):
	# performs the lrn, then converts it to the correct type
        return self.lrn(input).astype(DTYPE)


class Pooling(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(Pooling,self).__init__(in_size, layer, blobs)

        self.pool_type = layer.pooling_param.pool # TODO: not always max_pool
        self.pool_size = tuple([layer.pooling_param.kernel_size]*2)
        self.filter_shape = self.pool_size
        self.out_size = in_size[1]
        self.stride = tuple([layer.pooling_param.stride]*2)
        self.pad = tuple([layer.pooling_param.pad]*4)

    def get_theano_function(self, input):
	# uses the new version of max_pool including strides
        return downsample.max_pool_2d(input, self.pool_size, st=self.stride, padding=self.pad, ignore_border=True) 



class Dropout(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(Dropout,self).__init__(in_size, layer, blobs)

        self.dropout_ratio = layer.dropout_param.dropout_ratio

    def get_theano_function(self, input):
	# implements dropout via binomial distribution
        return (input*rng.binomial(input.shape, p=1-self.dropout_ratio, dtype=DTYPE))/(1-self.dropout_ratio)

class FullyConnected(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(FullyConnected,self).__init__(in_size, layer, blobs)
        if len(in_size) > 1:
            self.in_size = [in_size[0]*in_size[1]*in_size[2]*in_size[3]]

        self.pad = [0,0,0,0]
        self.stride = [1,1]
        
        self.filter_shape = [1,1]
        self.out_size = layer.inner_product_param.num_output

        print 'Insize',self.in_size,'outsize',self.out_size,'blobsize',len(blobs[0])
        if len(blobs) >= 1:
            self.filters = blobs[0].reshape(self.in_size[0], self.out_size, order='F')
        else:
            self.filters = np.zeros([self.in_size[0],self.out_size],dtype=DTYPE)

        self.w = theano.shared(value=self.filters, name='w')

        if len(blobs) >= 2:
            self.bias = blobs[1] # why squeeze?
        else:
            self.bias = np.zeros([1,self.out_size], dtype=DTYPE)

        self.b = theano.shared(value=self.bias, name='b')


    def get_output_size(self):
        return [self.out_size]

    def get_theano_function(self, input):
        input = T.flatten(input, outdim=2)
        return T.dot(input, self.w) + self.b
  
    def save_layer(self):
        np.save(open('layer_'+self.name+'_filters','w'), self.filters)
        np.save(open('layer_'+self.name+'_bias','w'), self.bias)

    def load_layer(self):
        self.filters = np.load(open('layer_'+self.name+'_filters','r'))
        self.w = theano.shared(value=self.filters, name='w')
        self.bias = np.load(open('layer_'+self.name+'_bias','r'))
        self.b = theano.shared(value=self.bias, name='b')
        

class Softmax(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(Softmax,self).__init__(in_size, layer, blobs)
        
        self.out_size = 10
        self.weights = np.zeros([in_size[0],self.out_size],dtype=DTYPE)
        self.w = theano.shared(value=self.weights, name='w')

    def get_theano_function(self, input):
	'''
	TODO: weights aren't given, so they need to be learned.
	'''
        return input
        #input = T.dot(input, self.w)
        #e = T.exp(input - input.max(axis=1).dimshuffle(0,'x'))
        #return e / e.sum(axis=1).dimshuffle(0,'x')
