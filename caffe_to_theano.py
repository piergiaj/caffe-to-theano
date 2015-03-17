import sys
import numpy as np
import google.protobuf
import caffe_pb2 as caffe
import layers
import theano.tensor as T

DTYPE='float32' # or float64
batch_size = 1 # this can be changed, if you want a larger batchsize

def get_blobs(blob,i):
    '''
      Converts the blobs to a numpy array of the right shape
    '''
    if i >= len(blob):
        return []
    else:
        blob = blob[i]
    blobs = []
    for b in blob.blobs:
        blobs.append(np.array(b.data, dtype=DTYPE).reshape(b.num,b.channels,b.height,b.width))
    return blobs


def save_net(net):
    '''
      Saves all the filters as numpy files (useful for quick visualization)
    '''
    for layer in net:
        layer.save_layer()

def load_net(net_name):
    # TODO: load the network file (net-param), then each layer
    pass


# load network structure
net_param = caffe.NetParameter()
print 'loading structure from', sys.argv[1]
google.protobuf.text_format.Merge(open(sys.argv[1],'rb').read(),net_param)

# load network filter data
net_data = caffe.NetParameter()
if len(sys.argv) >= 3:
    print 'loading data from', sys.argv[2]
    net_data.MergeFromString(open(sys.argv[2],'rb').read())

# go through each layer and recreated it in theano
network = []


in_size = [batch_size, net_param.input_dim[1], net_param.input_dim[2],
              net_param.input_dim[3]]

for i in range(len(net_param.layer)):
    # data layers start are +1 (because data is the first layer...)
    layer = net_param.layer[i]
    print 'processing layer',i,layer.name
    print in_size

    if layer.type == 'Convolution' or layer.type == 4:
        cl = layers.Convolution(in_size, layer, get_blobs(net_data.layers, i+1))
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
        fcl = layers.FullyConnected(in_size, layer, get_blobs(net_data.layers,i+1))
        in_size = fcl.get_output_size()
        network.append(fcl)

    elif layer.type == 'Dropout' or layer.type == 6:
        d = layers.Dropout(in_size, layer, ())
        network.append(d)

    elif layer.type == 'Softmax' or layer.type == 20:
        sm = layers.Softmax(in_size, layer, get_blobs(net_data.layers,i+1))
        network.append(sm)


# build theano network
out_size = in_size
in_size = [batch_size, net_param.input_dim[1], net_param.input_dim[2], net_param.input_dim[3]]

net = T.matrix('x', dtype=DTYPE)
net = net.reshape((batch_size,int(in_size[1]),int(in_size[2]),int(in_size[3])))

for layer in network:
    print net.dtype
    print layer.name
    net = layer.get_theano_function(net)

# save the network
save_net(network)
