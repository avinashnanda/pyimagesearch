# import the necessary packages
import h5py
import os

from pytz import unicode


class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        # check to see if the output path exists, and if so, raise
        # an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already "
                             "exists and cannot be overwritten. Manually delete "
                             "the file before continuing.", outputPath)

        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims,
                                           dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],),
                                             dtype="int")

        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0


# The constructor to HDF5DatasetWriter accepts four parameters, two of which are optional. The dims parameter
# controls the dimension or shape of the data we will be storing in the dataset. Think of dims as the .shape of a
# NumPy array. If we were storing the (flattened) raw pixel intensities of the 28 * 28 = 784 MNIST dataset,
# then dims=(70000, 784) as there are 70,000 examples in MNIST, each with a dimensionality of 784. If we wanted to
# store the raw CIFAR-10 images, then we would set dims=(60000, 32, 32, 3) as there are 60,000 total images in the
# CIFAR-10 dataset, each represented by a 32*32*3 RGB image. In the context of transfer learning and feature
# extraction, we’ll be using the VGG16 architecture and taking the outputs after the final POOL layer. The output of
# the final POOL layer is 512*7*7 which, when flattened, yields a feature vector of length 25,088. Therefore,
# when using VGG16 for feature extraction, we’ll set dims=(N, 25088) where N is the total number of images in our
# dataset. The next parameter to the HDF5DatasetWriter constructor is the outputPath – this is the path to where our
# output HDF5 file will be stored on disk. The optional dataKey is the name of the dataset that will store the data
# our algorithm will learn from. We default this value to "images",since in most cases we’ll be storing raw images in
# HDF5 format. However, for this example, when we instantiate the HDF5DatasetWriter we’ll set dataKey="features" to
# indicate that we are storing features extracted from a CNN in the file. Finally, bufSize controls the size of our
# in-memory buffer, which we default to 1,000 feature vectors/images. Once we reach bufSize, we’ll flush the buffer
# to the HDF5 dataset.

def add(self, rows, labels):
    # add the rows and labels to the buffer
    self.buffer["data"].extend(rows)
    self.buffer["labels"].extend(labels)

    # check to see if the buffer needs to be flushed to disk
    if len(self.buffer["data"]) >= self.bufSize:
        self.flush()


# The add method requires two parameters: the rows that we’ll be adding to the dataset, along with their
# corresponding class labels. Both the rows and labels are added to their respective buffers on Lines 32 and 33. If
# the buffer fills up, we call the flush method to write the buffers to file and reset them.

def flush(self):
    # write the buffers to disk then reset the buffer
    i = self.idx + len(self.buffer["data"])
    self.data[self.idx:i] = self.buffer["data"]
    self.labels[self.idx:i] = self.buffer["labels"]
    self.idx = i
    self.buffer = {"data": [], "labels": []}

# If we think of our HDF5 dataset as a big NumPy array, then we need to keep track of the current index into the next
# available row where we can store data (without overwriting existing data) Line 41 determines the next available row
# in the matrix. Lines 42 and 43 then apply NumPy array slicing to store the data and labels in the buffers. Line 45
# then resets the buffers. We’ll also define a handy utility function named storeClassLabels which, if called,
# will store the raw string names of the class labels in a separate dataset:

def storeClassLabels(self, classLabels):
    # create a dataset to store the actual class label names,
    # then store the class labels
    dt = h5py.special_dtype(vlen=unicode)
    labelSet = self.db.create_dataset("label_names",(len(classLabels),), dtype=dt)
    labelSet[:] = classLabels

#storeClassLabels which, if called, will store the raw string names of the class labels in a separate dataset.


def close(self):
	# check to see if there are any other entries in the buffer
	# that need to be flushed to disk
	if len(self.buffer["data"]) > 0:
		self.flush()

	# close the dataset
	self.db.close()

#close will be used to write any data left in the buffers to HDF5 as well as close the dataset
