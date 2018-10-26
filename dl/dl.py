import tensorflow as tf
import numpy as np
import os.path as path

training_epochs = 500
n_neurons_in_h1 = 60
n_neurons_in_h2 = 60
learning_rate = 0.01

n_features = 27
n_classes = 60

# Import pre-trained font data
def load_font(filename):
    trained_font = []
    expected = []
    with open(filename) as f:
        for line in f.readlines():
            splits = line.strip().split(" ")
            character_data = splits[:-1]
            trained_font.append(character_data)
            expected.append(splits[-1])

    return trained_font, expected


# Pre-trained English font data and expected characters with matching array indices
font, expected = load_font(path.abspath(path.join(__file__, "../../fontData/english.data")))

def main():
    # print "Font:", font[0]
    # print "Expected:", expected[0]
    # print "blarg"
    return "a"
