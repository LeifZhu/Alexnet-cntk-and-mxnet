from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

import matplotlib.pyplot as plt
import math
import numpy as np
import os
import PIL
import sys
import logging
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

import cntk as C

if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-' \
                    '%(message)s')
data_path = "/home/leizhu/dataset/imagenet8/raw/"
image_height = 224
image_width = 224
num_channels = 3
num_classes = 8

import cntk.io.transforms as xforms
def create_reader(map_file, is_training):
    logging.info("reading %s" % (map_file))
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist." %map_file)

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.88671875, jitter_type='uniratio') # train uses jitter
        ]
    else:
        transforms += [
            xforms.crop(crop_type='center', side_ratio=0.88671875) # test has no jitter
        ]

    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
    ]

    # deserializer
    return C.io.MinibatchSource(
        C.io.ImageDeserializer(map_file, C.io.StreamDefs(
            features=C.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels=C.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=is_training,
        multithreaded_deserializer=True)

reader_train = create_reader(map_file=os.path.join(data_path, 'train_map.txt'),
                             is_training=True)
reader_test = create_reader(map_file=os.path.join(data_path, 'val_map.txt'),
                           is_training=False)

def LocalResponseNormalization(k, n, alpha, beta, name=''):
    x = C.placeholder(name='lrn_arg')
    x2 = C.square(x)
    # reshape to insert a fake singleton reduction dimension after the 3th axis (channel axis). Note Python axis order and BrainScript are reversed.
    x2s = C.reshape(x2, (1, C.InferredDimension), 0, 1)
    W = C.constant(alpha/(2*n+1), (1,2*n+1,1,1), name='W')
    # 3D convolution with a filter that has a non 1-size only in the 3rd axis, and does not reduce since the reduction dimension is fake and 1
    y = C.convolution (W, x2s)
    # reshape back to remove the fake singleton reduction dimension
    b = C.reshape(y, C.InferredDimension, 0, 2)
    den = C.exp(beta * C.log(k + b))
    apply_x = C.element_divide(x, den)
    return apply_x

# Create the network.
def create_alexnet(features, out_dims):
    with C.layers.default_options(activation=None, pad=True, bias=True):
        model = C.layers.Sequential([
            # we separate Convolution and ReLU to name the output for feature extraction (usually before ReLU)
            C.layers.Convolution2D((11,11), 96, init=C.initializer.normal(0.01), pad=False, strides=(4,4), name='conv1'),
            C.layers.Activation(activation=C.relu, name='relu1'),
            LocalResponseNormalization(1.0, 2, 0.0001, 0.75, name='norm1'),
            C.layers.MaxPooling((3,3), (2,2), name='pool1'),

            C.layers.Convolution2D((5,5), 192, init=C.initializer.normal(0.01), init_bias=0.1, name='conv2'),
            C.layers.Activation(activation=C.relu, name='relu2'),
            LocalResponseNormalization(1.0, 2, 0.0001, 0.75, name='norm2'),
            C.layers.MaxPooling((3,3), (2,2), name='pool2'),

            C.layers.Convolution2D((3,3), 384, init=C.initializer.normal(0.01), name='conv3'),
            C.layers.Activation(activation=C.relu, name='relu3'),
            C.layers.Convolution2D((3,3), 384, init=C.initializer.normal(0.01), init_bias=0.1, name='conv4'),
            C.layers.Activation(activation=C.relu, name='relu4'),
            C.layers.Convolution2D((3,3), 256, init=C.initializer.normal(0.01), init_bias=0.1, name='conv5'),
            C.layers.Activation(activation=C.relu, name='relu5'),
            C.layers.MaxPooling((3,3), (2,2), name='pool5'),

            C.layers.Dense(4096, init=C.initializer.normal(0.005), init_bias=0.1, name='fc6'),
            C.layers.Activation(activation=C.relu, name='relu6'),
            C.layers.Dropout(0.5, name='drop6'),
            C.layers.Dense(4096, init=C.initializer.normal(0.005), init_bias=0.1, name='fc7'),
            C.layers.Activation(activation=C.relu, name='relu7'),
            C.layers.Dropout(0.5, name='drop7'),
            C.layers.Dense(out_dims, init=C.initializer.normal(0.01), name='fc8')
            ])
    return model(features)


def train_loop(reader_train, reader_test, model_func, max_epochs):
    input_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))
    mean_removed_features = C.minus(input_var, C.constant(114), name='mean_removed_input')
    z = model_func(mean_removed_features, out_dims=num_classes)
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)

    epoch_size = 10400
    minibatch_size = 256

    lr_per_mb = [0.01] * 25 + [0.001] * 25 + [0.0001] * 25 + [0.00001] * 25 + [0.000001]
    lr_schedule = C.learning_parameter_schedule(lr_per_mb, minibatch_size=minibatch_size, epoch_size=epoch_size)
    mm_schedule = C.learners.momentum_schedule(0.9, minibatch_size=minibatch_size)
    l2_reg_weight = 0.0005  # CNTK L2 regularization is per sample, thus same as Caffe

    learner = C.learners.momentum_sgd(z.parameters,
                                      lr=lr_schedule,
                                      momentum=mm_schedule,
                                      minibatch_size=minibatch_size,
                                      unit_gain=False,
                                      l2_regularization_weight=l2_reg_weight)
    progress_printer = C.logging.ProgressPrinter(tag='Training',
                                                 num_epochs=max_epochs)
    trainer = C.Trainer(z, (ce, pe), [learner], [progress_printer])

    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }
    C.logging.log_number_of_parameters(z);
    print()

    for epoch in range(max_epochs):
        sample_count = 0
        while sample_count < epoch_size:
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count),
                                               input_map=input_map)
            trainer.train_minibatch(data)
            sample_count += data[label_var].num_samples
        trainer.summarize_training_progress()

train_loop(reader_train,
           reader_test,
           max_epochs=5,
           model_func=create_alexnet)