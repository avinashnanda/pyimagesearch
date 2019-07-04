# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K


class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
        # define a CONV => BN => RELU pattern
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)

        # return the block
        return x

    # The conv_module function is responsible for applying a convolution, followed by a batch normalization,
    # and then finally an activation.

    # The parameters to the method are detailed below:
    # 1 x: The input layer to the function.
    # 2 K: The number of filters our CONV layer is going to learn.
    # 3 kX and kY: The size of each of the K filters that will be learned.
    # 4 stride: The stride of the CONV layer.
    # 5 chanDim: The channel dimension, which is derived from either “channels last” or “channels first” ordering.
    # 6 padding: The type of padding to be applied to the CONV layer.

    # On Line 19 we create the convolutional layer. The actual parameters to Conv2D are identical to examples in
    # previous architectures such as AlexNet and VGGNet, but what changes here is how we supply the input to a given
    # layer. Since we are using a Model rather than a Sequential to define the network architecture, we cannot call
    # model.add as this would imply that the output from one layer follows sequentially into the next layer. Instead,
    # we supply the input layer in parenthesis at the end of the function call, which is called a Functional API.
    # Each layer instance in a Model is callable on a tensor and also returns a tensor. Therefore, we can supply the
    # inputs to a given layer by calling it as a function once the object is instantiated.

    # The output of the Conv2D layer is then passed into the BatchNormalization layer on Line. The output of
    # BatchNormalization then goes through a ReLU activation. If we were to construct a figure to help us visualize
    # the conv_module it would look like below:

    # First the convolution is applied, then a batch normalization, followed by an activation. Note that this module
    # did not perform any branching. That is going to change with the definition of the inception_module below:

    @staticmethod
    def inception_module(x, numK1x1, numK3x3, chanDim):
        # define two CONV modules, then concatenate across the
        # channel dimension
        conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1,
                                             (1, 1), chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3,
                                             (1, 1), chanDim)
        x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

        # return the block
        return x

    # Our Mininception module will perform two sets of convolutions – a 1 X 1 CONV and a 3 X 3 CONV. These two
    # convolutions will be performed in parallel and the resulting features concatenated across the channel
    # dimension. Lines 30 and 31 use the handy conv_module we just defined to learn numK1x1 filters (1 X 1). Lines 32
    # and 33 then apply conv_module again to learn numK3x3 filters (3 X 3). By using the conv_module function we are
    # able to reuse code and not have to bloat our MiniGoogLeNet class by inserting many blocks of CONV => BN => RELU
    # blocks – this stacking is taken care of concisely via conv_module. Notice how both the input to the 1 X 1 and 3
    # X 3 Conv2D class is x, the input to the layer. When using the Sequential class, this type of layer structure
    # was not possible. But by using the Model class, we can now have multiple layers accept the same input. Once we
    # have both conv_1x1 and conv_3x3, we concatenate them across the channel dimension.

    # The output of both convolutions is then concatenated. We are allowed to concatenate the layer outputs because the
    # output volume size for both convolutions is identical due to padding="same".

    @staticmethod
    def downsample_module(x, K, chanDim):
        # define the CONV module and POOL, then concatenate
        # across the channel dimensions
        conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2),
                                             chanDim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chanDim)

        # return the block
        return x
        # This method requires us to pass in an input x, the number of filters K our convolutional layer will learn,
        # along with the chanDim for batch normalization and channel concatenation. The first branch of the
        # downsample_module learns a set of K, 3 X 3 filters using a stride of 2 X 2, thereby decreasing the output
        # volume size. We apply max pooling on  (the second branch), again with window size of 3 X 3 and stride of 2
        # X 2 to reduce volume size. The conv_3x3 and pool outputs are then concatenated and returned to the calling
        # function. The downsample_module is responsible for reducing the spatial dimensions of our input volume. The
        # first branch learns a set of filters with 2 X 2 stride to reduce the output volume. The second branch also
        # reduces the spatial dimensions, this time by applying max pooling. The output of the downsample_module is
        # concatenated along the channel dimension.We can visualize the downsample_module in Figure 11.5. As the
        # figure demonstrates, a convolution and max pooling operation are applied to the same input and then
        # concatenated.

    @staticmethod
    def build(width, height, depth, classes):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # define the model input and first CONV module
        inputs = Input(shape=inputShape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

        # We define the build method to our network, as is standard for all other examples in this book. Our build
        # method accepts an input width, height, depth, and total number of classes that will be learned.initialize
        # our inputShape and chanDim assuming we are using “channels last” ordering. If we are instead using
        # “channels first” ordering,  update these variables, respectively. Let’s define the model Input along with
        # the first conv_module: define the model input and first CONV module
        inputs = Input(shape=inputShape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)
        # The call to Input initializes the architecture – all inputs to the network will start at this layer which
        # simply “holds” the input data (all networks need to have an input, after all). The first CONV => BN => RELU
        # is applied where we learn 96, 3 X 3 filters. From there, we stack two Inception modules followed by a
        # downsample module:
        # two Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim)
        x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 80, chanDim)

        # The first Inception module (Line 70) learns 32 filters for both the 1 X 1 and 3 X 3 CONV layers. When
        # concatenated, this module outputs a volume with K = 32+32 = 64 filters. The second Inception module (Line
        # 71) learns 32, 1 X 1 filters and 48, 3 X 3 filters. Again, when concatenated, we see that the output volume
        # size is K = 32+48 = 80. The downsample module reduces our input volume sizes but keeps the same number of
        # filters learned at 80. Next, let’s stack four Inception modules on top of each other before applying a
        # downsample,allowing GoogLeNet to learn deeper, richer features: four Inception modules followed by a
        # downsample module
        x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 96, chanDim)

        # Notice how in some layers we learn more 1X1 filters than 3X3 filters, while other Inception modules learn
        # more 3X3 filters than 1X1. This type of alternating pattern is done on purpose and was justified by Szegedy
        # et al. after running many experiments. When we implement the deeper variant of GoogLeNet later in this
        # chapter, we’ll also see this pattern as well. Continuing our implementation of Figure 11.2 by Zhang et al.,
        # we’ll now apply two more inception modules followed by a global pool and dropout: two Inception modules
        # followed by global POOL and dropout
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # The output volume size after  is 7 X 7 X 336. Applying an average pooling of 7 X 7 reduces the volume size
        # to 1 X 1 X 336 and thereby alleviates the need to apply many dense fully-connected layers – instead,
        # we simply average over the spatial outputs of the convolution. Dropout is applied with a probability of 50
        # percent on Line 85 to help reduce overfitting. Finally, we add in our softmax classifier based on the
        # number of classes we wish to learn: softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="googlenet")

        # return the constructed network architecture
        return model
        # The actual Model is then instantiated on Line 93 where we pass in the inputs, the layers (x,which includes
        # the built-in branching), and optionally a name for the network. The constructed architecture is returned to
        # the calling function.
