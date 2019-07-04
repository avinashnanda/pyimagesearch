# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from dogs_vs_cats.config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import imagetoarraypreprocessor
from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import trainingmonitor
from pyimagesearch.io import hdf5datasetgenerator
from pyimagesearch.nn.conv import alexnet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")
# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = simplepreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = imagetoarraypreprocessor()

# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = alexnet.build(width=227, height=227, depth=3,
                      classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
    os.getpid())])
callbacks = [trainingmonitor(path)]
# train the network
model.fit_generator(trainGen.generator(),
                    steps_per_epoch=trainGen.numImages // 128,
                    validation_data=valGen.generator(),
                    validation_steps=valGen.numImages // 128,
                    epochs=75,
                    max_queue_size=128 * 2,
                    callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()