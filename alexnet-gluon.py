# Referrence
# [1] https://gluon.mxnet.io/chapter04_convolutional-neural-networks/deep-cnns-alexnet.html
# [2] https://github.com/apache/incubator-mxnet/tree/master/example/gluon
from __future__ import print_function
import logging, time, os
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader


mx.random.seed(1)
logging.basicConfig(level=logging.INFO)


ctx = mx.cpu()


def get_imagenet_transforms(data_shape=224, dtype='float32'):
    def train_transform(image, label):
        image, _ = mx.image.random_size_crop(image, (data_shape, data_shape), 0.08, (3/4., 4/3.))
        image = mx.nd.image.random_flip_left_right(image)
        image = mx.nd.image.to_tensor(image)
        image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return mx.nd.cast(image, dtype), label

    def val_transform(image, label):
        image = mx.image.resize_short(image, data_shape + 32)
        image, _ = mx.image.center_crop(image, (data_shape, data_shape))
        image = mx.nd.image.to_tensor(image)
        image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return mx.nd.cast(image, dtype), label
    return train_transform, val_transform


def get_imagenet_iterator(root, batch_size, num_workers, data_shape=224, dtype='float32'):
    """Dataset loader with preprocessing."""
    train_dir = os.path.join(root, 'train')
    train_transform, val_transform = get_imagenet_transforms(data_shape, dtype)
    logging.info("Loading image folder %s, this may take a bit long...", train_dir)
    train_dataset = ImageFolderDataset(train_dir, transform=train_transform)
    train_data = DataLoader(train_dataset, batch_size, shuffle=True,
                            last_batch='discard', num_workers=num_workers)
    val_dir = os.path.join(root, 'validation')
    if not os.path.isdir(os.path.expanduser(os.path.join(root, 'validation', 'n01440764'))):
        user_warning = 'Make sure validation images are stored in one subdir per category, a helper script is available at https://git.io/vNQv1'
        raise ValueError(user_warning)
    logging.info("Loading image folder %s, this may take a bit long...", val_dir)
    val_dataset = ImageFolderDataset(val_dir, transform=val_transform)
    val_data = DataLoader(val_dataset, batch_size, last_batch='keep', num_workers=num_workers)
    return train_data, val_data


train_data, val_data = get_imagenet_iterator(root='./imagenet8/raw/',
                                              batch_size=256,
                                              num_workers=4)

alex_net = gluon.nn.Sequential()
with alex_net.name_scope():
    #  First convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=96, kernel_size=11, strides=(4,4), activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
    #  Second convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=192, kernel_size=5, activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=(2,2)))
    # Third convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
    # Fourth convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
    # Fifth convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
    # Flatten and apply fullly connected layers
    alex_net.add(gluon.nn.Flatten())
    alex_net.add(gluon.nn.Dense(4096, activation="relu"))
    alex_net.add(gluon.nn.Dense(4096, activation="relu"))
    alex_net.add(gluon.nn.Dense(8))

alex_net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

trainer = gluon.Trainer(alex_net.collect_params(), 'sgd', {'learning_rate': .001})

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

epoch = 0
target_loss = 1.5
curr_loss = float("inf")
btime = time.time()
while curr_loss > target_loss:
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = alex_net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        curr_loss = nd.mean(loss).asscalar()
        logging.info("Epoch[%d] - batch[%d] - time passed: %f s  Loss: %s" % (epoch, i, time.time() - btime, curr_loss))
        if curr_loss <= target_loss:
            logging.info("Finish training: achieve target loss %f, cost %f s" % (target_loss, time.time() - btime))
            break
    epoch += 1