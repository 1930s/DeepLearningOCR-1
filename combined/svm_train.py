# https://www.tensorflow.org/guide/keras
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/

import codecs
import json
import numpy as np
import os.path as path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
import pickle
import sys
import tensorflow as tf
from tensorflow import keras

import time # only use for demo video

# Don't shorten printing numpy lists (i.e. don't use "...")
# np.set_printoptions(threshold=np.nan)


# Train the Support Vector Machine
# Then output the resulting model to a file to be loaded later
def main():
    if len(sys.argv) < 2:
        print("Specify fontData file name")
        return 1
    elif not path.isfile(path.abspath(path.join(__file__, "../" + sys.argv[1]))):
        print("Given filename for fontData doesn't exist")
        return 1
    else:
        # Pre-trained fontData and expected characters with matching array indices
        dataset = np.genfromtxt(sys.argv[1], dtype='str', delimiter=" ", encoding="utf8")

        X = dataset[:,0:-1].astype(dtype='float')
        Y = dataset[:,-1]

        clf = svm.SVC(kernel='linear', C = 2.0)
        clf.fit(X,Y)

        # save the classifier
        with open('svm_model.pkl', 'wb') as f:
            pickle.dump(clf, f)


if __name__ == "__main__":
    main()
