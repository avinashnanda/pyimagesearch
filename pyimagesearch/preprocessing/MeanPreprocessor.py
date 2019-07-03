# import the necessary packages
import cv2

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        # store the Red, Green, and Blue channel averages across a
        # training set
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):

        # split the image into its respective Red, Green, and Blue
        # channels
        (B, G, R) = cv2.split(image.astype("float32"))

        # subtract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # merge the channels back together and return the image
        return cv2.merge([B, G, R])

    # Line 15 uses the cv2.split function to split our input image into its respective RGB components. Keep in mind
    # that OpenCV represents images in BGR order rather than RGB ([38], hence why our return tuple has the signature
    # (B, G, R) rather than (R, G, B). We’ll also ensure that these channels are of a floating point data type as
    # OpenCV images are typically represented as unsigned 8-bit integers (in which case we can’t have negative
    # values, and modulo arithmetic would be performed instead). Lines 17-20 perform the mean subtraction itself,
    # subtracting the respective mean RGB values from the RGB channels of the input image. Line 23 then merges the
    # normalized channels back together and returns the resulting image to the calling function.