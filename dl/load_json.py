import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

book_name = ""
label_encoder = LabelEncoder()

def load_model(model_json, book_name):
    # load json and create model
    loaded_model = keras.models.model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    # print("Loaded model from disk")

    # Compile the model and test on it
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # test_loss, test_acc = loaded_model.evaluate(test_images, test_labels)

    # print("Test accuracy:", test_acc)

    onehot_json_file = open("onehot.json", 'r')
    onehot_json = onehot_json_file.read()
    onehot_json_file.close()

    global onehot_encoded
    onehot_encoded = json.loads(onehot_json)
    # print(onehot_encoded)

    dataset = np.genfromtxt("../fontData/" + book_name + ".data", dtype='str', delimiter=" ", encoding="utf8")
    expecteds = dataset[:, -1]
    label_encoder.fit_transform(expecteds)

    return loaded_model, onehot_encoded, label_encoder
