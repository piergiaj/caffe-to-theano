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
import matplotlib.pyplot as plt

DTYPE='float32' # or float64
theano.config.floatX = DTYPE
#theano.config.exception_verbosity='high'
#theano.config.optimizer='None'
batch_size = 40 # this can be changed, if you want a larger batchsize


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

for i in range(len(net_param.layer)): # ignore the last layer
# this is trained for the imagenet
# though we could probably play with it later...
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
        if i+2 >= len(net_param.layer): # ignore last fully-connected
            continue
        else:
            fcl = layers.FullyConnected(in_size, layer, [])
            in_size = fcl.get_output_size()
            network.append(fcl)

    elif layer.type == 'Dropout' or layer.type == 6:
        d = layers.Dropout(in_size, layer, ())
        network.append(d)

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


f = gzip.open('flowers.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=DTYPE))
    shared_y = theano.shared(np.asarray(data_y, dtype=DTYPE))
    return shared_x, T.cast(shared_y, 'int32')


train_set_x, train_set_y = shared_dataset(train_set)

#test = theano.function([index], intermediate_steps[-1], givens={
#        x: train_set_x[index * batch_size:(index+1)*batch_size]})
#theano.printing.pydotprint(test, with_ids=True,
#                           var_with_name_simple=True, outfile='tmp1.png')
#tmp = test(0)
#print tmp.shape


valid_set_x, valid_set_y = shared_dataset(valid_set)
test_set_x, test_set_y = shared_dataset(test_set)

learning_rate = 0.03
n_epochs = 150

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_test_batches = test_set_x.get_value(borrow=True).shape[0]
n_train_batches /= batch_size
n_valid_batches /= batch_size
n_test_batches /= batch_size
print n_train_batches
print batch_size

params = network[-1].params
cost = network[-1].cost(y)

grads = T.grad(cost, params)

updates = []
for param_i, grad_i in zip(params, grads):
    updates.append((param_i, param_i - learning_rate * grad_i))

train_model = theano.function([index], cost, updates=updates,
                givens = {
                    x: train_set_x[index*batch_size:(index+1)*batch_size],
                    y: train_set_y[index*batch_size:(index+1)*batch_size]
                }, allow_input_downcast=True)

test_model = theano.function([index], network[-1].errors(y),
                givens={
                    x: test_set_x[index * batch_size: (index + 1) * batch_size],
                    y: test_set_y[index * batch_size: (index + 1) * batch_size]})

validate_model = theano.function([index], network[-1].errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})



print 'training'
patience = 10000 
patience_increase = 2
improvement_threshold = 0.995  
validation_frequency = min(n_train_batches, patience / 2)

best_params = None
best_validation_loss = np.inf
best_iter = 0
test_score = 0.
start_time = time.clock()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

        iter = (epoch - 1) * n_train_batches + minibatch_index
        cost_ij = train_model(minibatch_index)

        if (iter + 1) % validation_frequency == 0:
            validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                  (epoch, minibatch_index + 1, n_train_batches, \
                   this_validation_loss * 100.))
            if this_validation_loss < best_validation_loss:
                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter
                save_net(network, str(iter), True)


                test_losses = [test_model(i) for i in xrange(n_test_batches)]
                test_score = np.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of best '
                       'model %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.))

        if patience <= iter:
            done_looping = True
            break
