# import the necessary packages
from dogs_vs_cats.config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.AspectAwarePreprocessor import AspectAwarePreprocessor
from pyimagesearch.io.HDF5DatasetWriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# grab the paths to the images
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[1].split(".")[0] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
# perform stratified sampling from the training set to build the
# testing split from the training data
split = train_test_split(trainPaths, trainLabels,
                         test_size=config.NUM_TEST_IMAGES, stratify=trainLabels,
                         random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

# perform another stratified sampling, this time to build the
# validation data
split = train_test_split(trainPaths, trainLabels,
                         test_size=config.NUM_VAL_IMAGES, stratify=trainLabels,
                         random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)]

# initialize the image preprocessor and the lists of RGB channel
# averages
aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])
# On Line 38 we define a datasets list that includes our training, validation, and testing variables. Each entry in
# the list is a 4-tuple consisting of: 1. The name of the split (i.e., training, testing, or validation). 2. The
# respective image paths for the split. 3. The labels for the split. 4. The path to the output HDF5 file for the
# split. We then initialize our AspectAwarePreprocessor on Line 45 used to resize images to 256 X 256 pixels (keeping
# the aspect ratio of the image in mind) prior to being written to HDF5. We’ll also initialize three lists on Line 46
# – R, G, and B, used to store the average pixel intensities for each channel. Finally, we are ready to build our
# HDF5 datasets:

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)
    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
                                   widgets=widgets).start()

    # we start looping over each of the 4-tuple values in the datasets list. For each data split, we instantiate the
    # HDF5DatasetWriter on Line 52. Here the dimensions of the output dataset will be the (len(paths), 256, 256, 3),
    # implying there are len(paths) total images, each of them with a width of 256 pixels, a height of 256 pixels,
    # and 3 channels.then initialize our progress bar so we can easily monitor the process of the dataset generation.
    # Again, this code block (along with the rest of the progressbar function calls) is entirely optional,
    # so feel free to comment them out if you so wish. Next, let’s write each image in a given data split to the
    # writer:

    # loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load the image and process it
        image = cv2.imread(path)
        image = aap.preprocess(image)

        # if we are building the training dataset, then compute the
        # mean of each channel in the image, then update the
        # respective lists
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        # add the image and label # to the HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)

        # close the HDF5 writer
    pbar.finish()
    writer.close()


# construct a dictionary of averages, then serialize the means to a JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()

# $ python build_dogs_vs_cats.py
