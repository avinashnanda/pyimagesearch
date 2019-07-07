# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K


class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False,
                        reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # The data parameter is simply the input to the residual module. The value K defines the number of filters that will
        # be learned by the final CONV in the bottleneck. The first two CONV layers will learn K / 4 filters, as per the He
        # et al. paper. The stride controls the stride of the convolution. We’ll use this parameter to help us reduce the
        # spatial dimensions of our volume without resorting to max pooling. We then have the chanDim parameter which defines
        # the axis which will perform batch normalization – this value is specified later in the build function based on
        # whether we are using “channels last” or “channels first” ordering. Not all residual modules will be responsible for
        # reducing the dimensions of our spatial volume – the red (i.e., “reduce”) boolean will control whether we are
        # reducing spatial dimensions (True) or not (False). We can then supply a regularization strength to all CONV layers
        # in the residual module via reg. The bnEps parameter controls the e responsible for avoiding “division by zero”
        # errors when normalizing inputs. In Keras, e defaults to 0:001; however, for our particular implementation,
        # we’ll allow this value to be reduced significantly. The bnMom controls the momentum for the moving average. This
        # value normally defaults to 0:99 inside Keras, but He et al. as well as Wei Wu recommend decreasing the value to 0:9.

        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data

        shortcut = data

        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                 momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act1)
        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                 momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
                       padding="same", use_bias=False,
                       kernel_regularizer=l2(reg))(act2)
        # the third block of the ResNet module is another set of 1x1
        # CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                 momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act3)
        # if we are to reduce the spatial size, apply a CONV layer to
        # the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride,
                              use_bias=False, kernel_regularizer=l2(reg))(act1)
        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])

        # return the addition as the output of the ResNet module
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters,
              reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):


