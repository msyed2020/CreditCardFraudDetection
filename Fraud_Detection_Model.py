import numpy as np
import pandas as pd
import tensorflow as tf
import time

credit_card_data = pd.read_csv('creditcard.csv')

# Data split
# Shuffling data and randomizing it
shuffled_data = credit_card_data.sample(frac=1)
# Changing class data to distinguish fraudulent data from real data
one_hot_data = pd.get_dummies(shuffled_data, columns=['Class'])
# Change all values into numbers between 0 and 1
normalized_data = (one_hot_data - one_hot_data.min()) / (one_hot_data.max() - one_hot_data.min())
# Storing V columns in df_X and Class columns in df_Y
df_X = normalized_data.drop(['Class_0', 'Class_1'], axis=1)
df_Y = normalized_data[['Class_0', 'Class_1']]
# Converts both data frames into np arrays of value float32
ar_X = np.asarray(df_X.values, dtype='float32')
ar_Y = np.asarray(df_Y.values, dtype='float32')
# First 80 percent of data should be in training, while the last 20 percent is in testing
train_size = int(0.8 * len(ar_X))
(raw_X_train, raw_Y_train) = (ar_X[:train_size], ar_Y[:train_size])
(raw_X_test, raw_Y_test) = (ar_X[train_size:], ar_Y[train_size:])

# Procedure for eliminating bias in the data
legitCount, fraudCount = np.unique(credit_card_data['Class'], return_counts=True)[1]
fraudRatio = float(fraudCount / (legitCount + fraudCount))
print('Percent of fraudulent transactions: ', fraudRatio)
weighting = 1 / fraudRatio
raw_Y_train[:, 1] = raw_Y_train[:, 1] * weighting

# Graph creation
input_dimensions = ar_X.shape[1]
output_dimensions = ar_Y.shape[1]
cellsNumLayer1 = 100
cellsNumLayer2 = 150
# Train input model
X_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name='X_train')
Y_train_node = tf.placeholder(tf.float32, [None, output_dimensions], name='Y_train')
# Test input model
X_test_node = tf.constant(raw_X_test, name='X_test')
Y_test_node = tf.constant(raw_Y_test, name='Y_test')
# Weights/biases which pass layers to one another until the third one determines legit/fraud
nodeWeight1 = tf.Variable(tf.zeros([input_dimensions, cellsNumLayer1]), name="weight1")
nodeWeight2 = tf.Variable(tf.zeros([cellsNumLayer1, cellsNumLayer2]), name="weight2")
nodeWeight3 = tf.Variable(tf.zeros([cellsNumLayer2, output_dimensions]), name="weight2")
nodeBiases1 = tf.Variable(tf.zeroes(cellsNumLayer1), name="biases1")
nodeBiases2 = tf.Variable(tf.zeroes(cellsNumLayer2), name="biases2")
nodeBiases3 = tf.Variable(tf.zeroes(output_dimensions), name="biases3")

# Uses input tensor to get a fraud/legit result
def network(input_tensor):
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, nodeWeight1) + nodeBiases1)
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, nodeWeight2) + nodeBiases2), 0.85)
    layer3 = tf.nn.softmax(tf.matmul(layer2, nodeWeight3) + nodeBiases3)
    return layer3

# Predicts what will get train/test data
Y_train_prediction = network(X_train_node)
Y_test_prediction = network(X_test_node)

cross_entropy = tf.losses.softmax_cross_entropy(Y_train_node, Y_train_prediction)
optimizer = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

# Function to calculate accuracy of actual to predicted results
def calculate_accuracy(actual, predicted):
    actual = np.argmax(actual, 1)
    predicted = np.argmax(predicted, 1)
    return (100 * np.sum(np.equal(actual, predicted)) / predicted.shape[0])

# Doing training and testing here
numEpochs = 100
with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(numEpochs):
        startTime = time.time()
        _, cross_entropy_score = session.run([optimizer, cross_entropy], feed_dict={X_train_node: raw_X_train, Y_train_node: raw_Y_train})
        if epoch % 10 == 0:
            timer = time.time() - startTime
            print('Epoch: {}'.format(), 'Current loss: {0:.4f}'.format(cross_entropy_score),
                  'Elapsed time: {0:.2f} seconds'.format(timer))
    final_y_test = Y_test_node.eval()
    final_y_test_prediction = Y_test_prediction.eval()
    final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
    print("Final accuracy: {0:.2f}%".format(final_accuracy))

final_fraud_y_test = final_y_test[final_y_test[:, 1] == 1]
final_fraud_y_test_prediction = final_y_test_prediction[final_y_test[:, 1] == 1]
final_fraud_accuracy = calculate_accuracy(final_fraud_y_test, final_fraud_y_test_prediction)
print('Final fraud specific accuracy: {0:.2f}%'.format(final_fraud_accuracy))

