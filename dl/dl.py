import tensorflow as tf
import numpy as np
import os.path as path
#from tensorflow.examples.tutorials.mnist import input_data

'''
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print mnist
'''

#begin tensorflow stuff
#ocr = input_data.read_data_sets(tuple, one_hot=True)
n_features = 27
n_classes = 60
batch_size = 10

n_neurons_in_h1 = 100
n_neurons_in_h2 = 100
n_neurons_in_h3 = 100
learning_rate = 0.01

# Length of input
x = tf.placeholder('float', [None, 27])
y = tf.placeholder('float')

def neural_network_model(data):

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([27, n_neurons_in_h1])),
                      'biases': tf.Variable(tf.random_normal([n_neurons_in_h1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2])),
                      'biases': tf.Variable(tf.random_normal([n_neurons_in_h2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_neurons_in_h2, n_neurons_in_h3])),
                      'biases': tf.Variable(tf.random_normal([n_neurons_in_h3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_neurons_in_h3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    training_epochs = 10

    with tf.Session() as sess
        sess.run(tf.initialize_all_variables())

        for epoch in training_epochs:
            epoch_loss = 0
            for _ in range(int(57/batch_size))
                epoch_x, epoch_y = next_batch(epoch, batch_size)
                _, c = ses.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += class
            print('Epoch', epoch, 'completed out of', training_epochs, 'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy',accuracy.eval({x:ocr.test.images, y:ocr.test.labels}))

train_neural_network(x)
#end tensorflow stuff

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

    data = tf.convert_to_tensor(trained_font)
    return data, expected


# Pre-trained English font data and expected characters with matching array indices
font, expected = load_font(path.abspath(path.join(__file__, "../../fontData/english.data")))
# print font

def next_batch(epoch, batch_size):
    font

def ocrValue(tuple_in, max_length):
    # For some reason, the list comes is as length 54 but with only 27 elements
    # Extract the elements from indices 27 through 53
    temp = []
    for dimension in range(max_length, len(tuple_in)):
        temp.append(tuple_in[dimension])
    # print "Length:", len(tuple), "\t data:", tuple
    tuple = tf.convert_to_tensor(temp)
    return "a"
