import numpy as np
import theano
import theano.tensor as T

#from theano.tensor.signal 
import downsample
from theano.tensor.nnet import conv
from pylearn2.expr.normalize import CrossChannelNormalization
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import google.protobuf

import caffe_pb2 as caffe

DTYPE = 'float32' # or 'float64'
rng = RandomStreams()

class Layer(object):
    '''
    layer is a caffe.NetParameter
    '''

    def __init__(self, in_size, layer, blobs):
        self.name = layer.name
        self.top = layer.top
        self.bottom = layer.bottom
        self.in_size = in_size

    def get_output_size(self):
        return [self.in_size[0], self.out_size, int(np.floor((self.in_size[2] + self.pad[0]+self.pad[1] - self.filter_shape[0])/self.stride[0])+1), int(np.floor((self.in_size[3]+self.pad[2]+self.pad[3]-self.filter_shape[1])/self.stride[1])+1)]


    def get_theano_function(self):
        pass

    def save_layer(self, name='', p_only=False):
        pass

    def load_layer(self):
        pass

class Convolution(Layer):

    def __init__(self, in_size, layer, blobs):
        super(Convolution, self).__init__(in_size, layer, blobs)

        self.filter_shape = tuple([layer.convolution_param.kernel_size]*2)
        self.stride = tuple([layer.convolution_param.stride]*2) # subsample in theano
        self.pad = tuple([layer.convolution_param.pad]*4) # currently ignored...
        # TODO: pad (or pad_h and pad_w) [default 0]: specifies the number of pixels to (implicitly) add to each side of the input

        self.out_size = layer.convolution_param.num_output
        self.groups = layer.convolution_param.group

        self.theano_filter_shape = tuple([self.out_size, self.in_size[1]] + list(self.filter_shape))


        # get filters
        if len(blobs) >= 1:
            self.filters = blobs[0]
        else:
            self.filters = np.zeros(self.theano_filter_shape, dtype=DTYPE)

        if self.groups == 1:
            self.w = theano.shared(value=self.filters, name='w')
        else:
            w1 = self.filters[:self.out_size/2,:,:,:]
            w2 = self.filters[self.out_size/2:,:,:,:]
            self.w1 = theano.shared(value=w1, name='w1')
            self.w2 = theano.shared(value=w2, name='w2')
            self.theano_filter_shape = tuple([self.out_size, self.in_size[1]/2] + list(self.filter_shape))
            self.in_size[1] = self.in_size[1]/2

        self.theano_in_size = (self.in_size[0], self.in_size[1], 
                               self.in_size[2]+2*self.pad[0],
                               self.in_size[3]+2*self.pad[3])

        if self.pad[0] != 0:
            sz = self.theano_in_size
            if self.groups > 1:
                sz = (self.theano_in_size[0], self.theano_in_size[1]*2,
                      self.theano_in_size[2], self.theano_in_size[3])
            self.do_pad = np.zeros(sz).astype(DTYPE)
            self.dpad = theano.shared(self.do_pad, name='padding')

        # get biases
        if len(blobs) >= 2:
            self.bias = blobs[1].reshape((self.out_size,))
        else:
            self.bias = np.zeros((self.out_size,), dtype=DTYPE)
        self.b = theano.shared(value=self.bias, name='b')

    def get_theano_function(self, input):
        print self.name
        print 'in',self.in_size
        print 'filt',self.theano_filter_shape
        print 'bias', self.bias.shape
        conv_out = None
        if self.pad[0] != 0:
            tmp_new = T.zeros_like(self.dpad)
            input = T.set_subtensor(tmp_new[:,:,self.pad[0]:(-self.pad[0]), 
                                            self.pad[3]:(-self.pad[3])], input)
        if self.groups == 1:
            conv_out = conv.conv2d(input, self.w, 
                                   filter_shape=self.theano_filter_shape,
                                   subsample=self.stride, 
                                   image_shape=self.theano_in_size)
        else:
            in_1 = input[:,:input.shape[1]/2,:,:]
            in_2 = input[:,input.shape[1]/2:,:,:]
            fs = self.theano_filter_shape
            fs = (fs[0]/2,fs[1],fs[2],fs[3])
            conv_1 = conv.conv2d(in_1, self.w1, filter_shape=fs,
                                 subsample=self.stride, 
                                 image_shape=self.theano_in_size)
            conv_2 = conv.conv2d(in_2, self.w2, filter_shape=fs,
                                 subsample=self.stride, 
                                 image_shape=self.theano_in_size)
            conv_out = T.concatenate([conv_1, conv_2], axis=1)
        return conv_out + self.b.dimshuffle('x',0,'x','x')

    def save_layer(self, name='', p_only=False):
        if p_only:
            return
        if self.groups == 1:
            np.save(open('layer_'+self.name+name+'_filters','w'), self.w.get_value(borrow=True))
        else:
            np.save(open('layer_'+self.name+name+'_filters1','w'), self.w1.get_value(borrow=True))
            np.save(open('layer_'+self.name+name+'_filters2','w'), self.w2.get_value(borrow=True))
        np.save(open('layer_'+self.name+name+'_bias','w'), self.b.get_value(borrow=True))

    def load_layer(self,name=''):
        self.filters = np.load(open('layer_'+self.name+name+'_filters','r'))
        print 'name', self.name
        print 'filt size', self.filters.shape

        if self.groups == 1:
            self.w = theano.shared(value=self.filters, name='w')
        else:
            w1 = self.filters[:self.out_size/2,:,:,:]
            w2 = self.filters[self.out_size/2:,:,:,:]
            self.w1 = theano.shared(value=w1, name='w1')
            self.w2 = theano.shared(value=w2, name='w2')

        self.bias = np.load(open('layer_'+self.name+name+'_bias','r')).reshape((self.out_size,))
        self.b = theano.shared(value=self.bias, name='b')
        
class ReLU(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(ReLU,self).__init__(in_size, layer, blobs)

    def get_theano_function(self, input):
        # this can be more complex, if negative_slope is inclded
        return T.maximum(input, 0.0)

class LRN(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(LRN,self).__init__(in_size, layer, blobs)
        
        self.local_size = layer.lrn_param.local_size # the number of channels to sum over (for cross channel LRN) or the side length of the square region to sum over (for within channel LRN)
        self.alpha = layer.lrn_param.alpha # scaling param
        self.beta = layer.lrn_param.beta # exponent
        # todo: can have a norm region as well (dealing with channels)

        self.lrn = CrossChannelNormalization(alpha=self.alpha, beta=self.beta, n=self.local_size)

    def get_theano_function(self, input):
        return self.lrn(input).astype(DTYPE)


class Pooling(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(Pooling,self).__init__(in_size, layer, blobs)

        self.pool_type = layer.pooling_param.pool # TODO: use this
        self.pool_size = tuple([layer.pooling_param.kernel_size]*2)
        self.filter_shape = self.pool_size
        self.out_size = in_size[1]
        self.stride = tuple([layer.pooling_param.stride]*2)
        self.pad = tuple([layer.pooling_param.pad]*4)

    def get_theano_function(self, input):
        return downsample.max_pool_2d(input, self.pool_size, st=self.stride, padding=self.pad, ignore_border=True) 



class Dropout(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(Dropout,self).__init__(in_size, layer, blobs)

        self.dropout_ratio = layer.dropout_param.dropout_ratio

    def get_theano_function(self, input):
        return (input*rng.binomial(input.shape, p=1-self.dropout_ratio, dtype=DTYPE))/(1-self.dropout_ratio)

class FullyConnected(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(FullyConnected,self).__init__(in_size, layer, blobs)
        self.batch_size = in_size[0]
        if len(in_size) > 2:
            self.in_size = [in_size[0], in_size[1]*in_size[2]*in_size[3]]

        self.pad = [0,0,0,0]
        self.stride = [1,1]
        
        self.filter_shape = [1,1]
        self.out_size = layer.inner_product_param.num_output

        if len(blobs) >= 1:
            self.filters = blobs[0].reshape(self.in_size[1], self.out_size, order='F')
        else:
            self.filters = np.zeros([self.in_size[1],self.out_size],dtype=DTYPE)

        self.w = theano.shared(value=self.filters, name='w')

        if len(blobs) >= 2:
            self.bias = blobs[1]
        else:
            self.bias = np.zeros([1,self.out_size], dtype=DTYPE)
        self.bias = self.bias.reshape((1,1,1,self.out_size))

        self.b = theano.shared(value=self.bias, name='b')
        self.b = T.addbroadcast(self.b, 0)

    def get_output_size(self):
        return [self.batch_size, self.out_size]

    def get_theano_function(self, input):
        print 'fcl', self.name
        print 'in',self.in_size
        print 'out', self.get_output_size()
        
        input = input.flatten(2).reshape(self.in_size)
        input = T.dot(input, self.w) + self.b
        return input.flatten(2).reshape(self.get_output_size())
  
    def save_layer(self,name='', p_only=False):
        if p_only:
            return
        np.save(open('layer_'+self.name+name+'_filters','w'), self.w.get_value(borrow=True))
#        np.save(open('layer_'+self.name+name+'_bias','w'), self.b.get_value(borrow=True))

    def load_layer(self,name=''):
        self.filters = np.load(open('layer_'+self.name+name+'_filters','r'))
        self.w = theano.shared(value=self.filters, name='w')
        self.bias = np.load(open('layer_'+self.name+name+'_bias','r'))
        self.bias = self.bias.reshape((1,1,1,self.out_size))
        self.b = theano.shared(value=self.bias, name='b')
        self.b = T.addbroadcast(self.b, 2)

class Softmax(Layer):
    
    def __init__(self, in_size, layer, blobs):
        super(Softmax,self).__init__(in_size, layer, blobs)
        
#        self.out_size = 17
#        self.weights = np.zeros([in_size[1],self.out_size],dtype=DTYPE)
#        self.w = theano.shared(value=self.weights, name='w')
#        self.params = [self.w]

    def cost(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

#    def save_layer(self,name=''):
#        np.save(open('layer_'+self.name+name+'_filters','w'), self.w.get_value(borrow=True))

#    def load_layer(self,name=''):
#        try:
#            self.filters = np.load(open('layer_'+self.name+name+'_filters','r'))
#            self.w = theano.shared(value=self.filters, name='w')
#            self.params = [self.w]
#        except:
#            pass

    def get_theano_function(self, input):
        self.p_y_given_x = T.nnet.softmax(input)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        return self.y_pred
        
