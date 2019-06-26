# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-o", "--output", required=True, help="path to output directory to store augmentation examples")
ap.add_argument("-p", "--prefix", type=str, default="image", help="output filename prefix")
args = vars(ap.parse_args())
# load the input image, convert it to a NumPy array, and then reshape it to have an extra dimension
print("[INFO] loading example image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
# construct the image generator for data augmentation then initialize the total number of images generated thus far
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
total = 0
# The rotation_range parameter controls the degree range of the random rotations. Here we’ll allow our input image to
# be randomly rotated +-30 degrees. Both the width_shift_range and height_shift_range are used for horizontal and
# vertical shifts, respectively.The #parameter value is a fraction of the given dimension, in this case,
# 10%.The shear_range controls the angle in counterclockwise direction as radians in which our image will allowed to
# be sheared. We then have the zoom_range, a floating point value that allows the #image to be “zoomed in” or “zoomed
# out” according to the following uniform distribution of values: [1 - zoom_range, 1 + zoom_range]. Finally,
# the horizontal_flip boolean controls whether or not a given input is allowed to be flipped horizontally during the
# #training process. For most computer vision applications a horizontal flip of an image does not change the
# resulting class label – but there are applications where a horizontal (or vertical) flip does change the semantic
# meaning of the image. Take care when #applying this type of data augmentation as our goal is to slightly modify the
# input image, thereby generating a new training #sample, without changing the class label itself. construct the
# actual Python generator
print("[INFO] generating images...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
                    save_prefix=args["prefix"], save_format="jpg")

# loop over examples from our image data augmentation generator
for image in imageGen:
    # increment our counter
    total += 1

    # if we have reached 10 examples, break from the loop
    if total == 10:
        break

# Lines 34 and 35 initialize a Python generator used to construct our augmented images. We’ll pass in our input
# image, a batch_size of 1 (since we are only augmenting one image), along with a few additional parameters to
# specify the output image file paths, the prefix for each file path, and the image file format. Line 38 then starts
# looping over each image in the imageGen generator. Internally, imageGen is automatically generating a new training
# sample each time one is requested via the loop. We then increment the total number of data augmentation examples
# written to disk and stop the script from executing once we’ve reached ten examples. To visualize data augmentation
# in action, we’ll be using Figure 2.2 (left), an image of Jemma, my family beagle. To generate new training example
# images of Jemma, just execute the following command: $ python augmentation_demo.py --image jemma.png --output
# output #after the script executes you should see ten images in the output directory:

# I have constructed a montage of each of these images so you can visualize them in Figure 2.2 (right). Notice how
# each image has been randomly rotated, sheared, zoomed, and horizontally flipped. In each case the image retains the
# original class label: dog; however, each image has been modified slightly, thereby giving our neural network new
# patterns to learn from when #training. Since the input images will constantly be changing (while the class labels
# remain the same), it’s common to see our training accuracy decrease when compared to training without data
# #augmentation.
