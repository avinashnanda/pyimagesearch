# import the necessary packages
from keras.utils import np_utils
import numpy as np
import h5py


class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None,
                 aug=None, binarize=True, classes=2):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]

        # 1. dbPath: The path to our HDF5 dataset that stores our images and corresponding class labels. 2.
        # batchSize: The size of mini-batches to yield when training our network. 3. preprocessors: The list of image
        # preprocessors we are going to apply (i.e., MeanPreprocessor, ImageToArrayPreprocessor, etc.). 4. aug:
        # Defaulting to None, we could also supply a Keras ImageDataGenerator to apply data augmentation directly
        # inside our HDF5DatasetGenerator. 5. binarize: Typically we will store class labels as single integers
        # inside our HDF5 dataset;however, as we know, if we are applying categorical cross-entropy or binary
        # cross-entropy as our loss function, we first need to binarize the labels as one-hot encoded vectors – this
        # switch indicates whether or not this binarization needs to take place (which defaults to True). 6. classes:
        # The number of unique class labels in our dataset. This value is required to accurately construct our
        # one-hot encoded vectors during the binarization phase.

        # These variables are stored on Lines 12-16 so we can access them from the rest of the class. Line 20 opens a
        # file pointer to our HDF5 dataset file Line 21 creates a convenience variable used to access the total
        # number of data points in the dataset. Next, we need to define a generator function, which as the name
        # suggests, is responsible for yielding batches of images and class labels to the Keras .fit_generator
        # function when training a network:

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            for i in np.arange(0, self.numImages, self.batchSize):
                # extract the images and labels from the HDF dataset
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                # Line 23 defines the generator function which can accept an optional argument, passes. Think of the
                # passes value as the total number of epochs – in most cases, we don’t want our generator to be
                # concerned with the total number of epochs; our training methodology (fixed number of epochs,
                # early stopping, etc.) should be responsible for that. However, in certain situations, it’s often
                # helpful to provide this information to the generator. On Line 29 we start looping over the number
                # of desired epochs – by default, this loop will run indefinitely until either: 1. Keras reaches
                # training termination criteria. 2. We explicitly stop the training process (i.e., ctrl + c). Line 31
                # starts looping over each batch of data points in the dataset. We extract the images and labels of
                # size batchSize from our HDF5 dataset on Lines 33 and 34.

                # check to see if the labels should be binarized
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)

                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # initialize the list of processed images
                    procImages = []

                    # loop over the images
                    for image in images:
                        # loop over the preprocessors and apply each
                        # to the image
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # update the list of processed images
                        procImages.append(image)

                    # update the images array to be the processed
                    # images
                    images = np.array(procImages)

                # Provided the preprocessors is not None (Line 42), we loop over each of the images in the batch and
                # apply each of the preprocessors by calling the preprocess method on the individual image. Doing
                # this enables us to chain together multiple image pre-processors. For example, our first
                # pre-processor may resize the image to a fixed size via our SimplePreprocessor class. From there we
                # may perform mean subtraction via the MeanPreprocessor. And after that, we’ll need to convert the
                # image to a Keras-compatible array using the ImageToArrayPreprocessor. At this point it should be
                # clear why we defined all of our pre-processing classes with a preprocess method – it allows us to
                # chain our pre-processors together inside the data generator. The preprocessed images are then
                # converted back to a NumPy array on Line 58. Provided we supplied an instance of aug,
                # an ImageDataGenerator class used for data augmentation, we’ll also want to apply data augmentation
                # to the images as well:

                # if the data augmenator exists, apply it
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images,
                                                          labels, batch_size=self.batchSize))
                # yield a tuple of images and labels
                yield (images, labels)

            # increment the total number of epochs
            epochs += 1

    def close(self):

        # close the database
        self.db.close()
