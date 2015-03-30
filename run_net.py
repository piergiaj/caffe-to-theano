import sys
import time
import numpy as np
import google.protobuf
import caffe_pb2 as caffe
import layers
import theano.tensor as T
import cPickle
import gzip
import theano
from PIL import Image
import matplotlib.pyplot as plt
from preprocess_image import get_img
from vis_squares import vis_square
import pylab


DTYPE='float32' # or float64
theano.config.floatX = DTYPE
#theano.config.exception_verbosity='high'
#theano.config.optimizer='None'
batch_size = 1 # this can be changed, if you want a larger batchsize


def save_net(net, name, p_only=False):
    # save a network file as well as each layer (probably should just use net-param)
    for layer in net:
        layer.save_layer(name, p_only)


# load network structure
net_param = caffe.NetParameter()
print 'loading structure from', sys.argv[1]
google.protobuf.text_format.Merge(open(sys.argv[1],'rb').read(),net_param)

# go through each layer and recreated it in theano
network = []

in_size = [batch_size, net_param.input_dim[1], net_param.input_dim[2],
              net_param.input_dim[3]]
sm = None
for i in range(len(net_param.layer)):
    # data layers start are +1 (because data is the first layer...)
    layer = net_param.layer[i]
    print 'processing layer',i,layer.name
    print in_size

    if layer.type == 'Convolution' or layer.type == 4:
        cl = layers.Convolution(in_size, layer, [])
        in_size = cl.get_output_size()
        network.append(cl)

    elif layer.type == 'ReLU' or layer.type == 18:
        rl = layers.ReLU(in_size, layer, ())
        network.append(rl)

    elif layer.type == 'LRN' or layer.type == 15:
        norm = layers.LRN(in_size, layer, ())
        network.append(norm)

    elif layer.type == 'Pooling' or layer.type == 17:
        pool = layers.Pooling(in_size, layer, ())
        in_size = pool.get_output_size()
        network.append(pool)

    elif layer.type == 'InnerProduct' or layer.type == 14:
        fcl = layers.FullyConnected(in_size, layer, [])
        in_size = fcl.get_output_size()
        network.append(fcl)

    elif layer.type == 'Dropout' or layer.type == 6:
#        d = layers.Dropout(in_size, layer, ())
#        network.append(d)
        pass

    elif layer.type == 'Softmax' or layer.type == 20:
        sm = layers.Softmax(in_size, layer, [])
        network.append(sm)

for layer in network:
    layer.load_layer()


# build theano network
out_size = in_size
in_size = [batch_size, net_param.input_dim[1], net_param.input_dim[2], net_param.input_dim[3]]

index = T.lscalar()
x = T.matrix('x', dtype=DTYPE)
y = T.ivector('y')
inp = x.reshape((batch_size,int(in_size[1]),int(in_size[2]),int(in_size[3])))
print 'input shape should be', batch_size,int(in_size[1]),int(in_size[2]),int(in_size[3])
intermediate_steps = [inp]

for layer in network:
    print layer.name
    intermediate_steps.append(layer.get_theano_function(intermediate_steps[-1]))




print 'Processing image', sys.argv[2]
data = np.asarray([get_img(sys.argv[2])])

d = data.reshape((227, 227, 3))
img = Image.fromarray(np.uint8(d))
pylab.imshow(img)
pylab.show()


input_data = theano.shared(data)

if len(sys.argv) > 3 and (sys.argv[3] == 'show' or sys.argv[3] == 'save'):
    filters = network[0].filters
    vis_square(filters.transpose(0, 2, 3, 1))
    for i in range(len(intermediate_steps)-1):
        step = intermediate_steps[i+1]
        layer = network[i]
        f = theano.function([], step, givens={x:input_data})
        tmpd = f()
        if len(tmpd.shape) > 2:
            tmpd = tmpd.reshape((tmpd.shape[1],tmpd.shape[2],tmpd.shape[3]))
            vis_square(tmpd, name=layer.name, show=sys.argv[3])
        elif 'prob' not in layer.name:
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.plot(tmpd.flat)
            plt.subplot(2, 1, 2)
            plt.title('Layer '+layer.name)
            _ = plt.hist(tmpd.flat[tmpd.flat > 0], bins=100)
            plt.show() if sys.argv[3] == 'show' else plt.savefig('l_'+layer.name+'.png')
        

best_class = theano.function([], sm.y_pred, givens={
        x: input_data})
predictions = theano.function([], sm.p_y_given_x, givens={
        x: input_data})

print 'Best Class:', best_class()
p = predictions()[0]
pylab.plot(p)
pylab.show()
