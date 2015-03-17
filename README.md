# caffe-to-theano
Convert a Caffe Model to a Theano Model. This currently works on AlexNet, but should work for any Caffe model that only includes layers that have been impemented.

# Usage
To run:

    python caffe_to_theano.py MODEL.prototxt MODEL.caffemodel
where MODEL.prototxt is the protocol buffer text file with the structure of the model,
for example, [this file] (https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt) for
Caffe's AlexNet. The MODEL.caffemodel is the binary file with all the weights, for example, [this file](http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel) for
Caffe's AlexNet. The full AlexNet model can be found [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet).

