import json
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import BaseLogger


class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

        """
        figPath: The path to the output plot that we can use to visualize loss and accuracy overtime.
        jsonPath: An optional path used to serialize the loss and accuracy values as a JSON file. This path is useful if you want to use the training history to create custom plots of your own.
        startAt: This is the starting epoch that training is resumed at when using ctrl + c training.We cover ctrl + c training in the Practitioner Bundle so we can ignore this variable for now.
        """

    def on_train_begin(self, logs={}):
        """
        On Line 19 we define H, used to represent the “history” of losses.
        We’ll see how this dictionary is updated in the on_epoch_end function in the next code block.
        Line 22 makes a check to see if a JSON path was supplied. If so, we then check to see if this JSON file exists.
        Provided that the JSON file does exist, we load its contents and update the history
        dictionary H up until the starting epoch (since that is where we will resume training from).
        """
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # The on_epoch_end method is automatically supplied to parameters from Keras. The first is an integer representing the epoch number. The second is a dictionary, logs, which contains the
        # training and validation loss + accuracy for the current epoch. We loop over each of the items in logs and then update our history dictionary (Lines 37-40).
        # After this code executes, the dictionary H now has four keys:
        # 1. train_loss
        # 2. train_acc
        # 3. val_loss
        # 4. val_acc
        # We maintain a list of values for each of these keys. Each list is updated at the end of every epoch, thus enabling us to plot an updated loss and accuracy curve as soon as the epoch completes.
        # In the case that a jsonPath was provided, we serialize the history H to disk:

        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        # check to see if the training history should be serialized to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H["loss"]) > 1:
        # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["acc"], label="train_acc")
            plt.plot(N, self.H["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(
                len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

        # save the figure
        plt.savefig(self.figPath)
        plt.close()
