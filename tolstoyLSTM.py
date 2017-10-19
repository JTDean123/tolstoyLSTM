########################################
#
#
# text generation via a LSTM-rnn trained on the single
# character level on Tolstoy's Anna Karenina
#
# Jason Dean
# jtdean@gmail.com
# 10/16/2017
#
#
########################################


import tensorflow as tf
print('tensorflow version: ', tf.__version__)
import numpy as np
from tensorflow.contrib import rnn
import random
from optparse import OptionParser
import sys


def getbook(path):
    """
    Loads Anna Karenina as a text file and returns a really long string
    :param path: Local path to text file
    """

    try:
        contents = ''
        with open(path) as f:
            letter =f.readlines()
            for char in letter:
                contents += char
        # remove licensing info at the beginning and end
        #contents = contents[1133:-18814]
        # train on ~100k characters
        contents = contents[1133:120000]

        # convert to lowercase and remove \r
        contents = contents.lower().replace('\r', '')

        # only keep ascii characters
        return "".join(i for i in contents if ord(i) < 128)

    except:
        sys.stderr('ERROR:  Unable to load text')
        sys.exit(1)


def textDicts(texts):
    """
    Create text to number and number to text dictionaries
    :param texts:  string create dictionaries from
    """

    # get the unique characters
    chars = list(set(texts))
    num_chars = len(chars)
    print('Number of unique characters:  ', num_chars)

    # create a character to number and number to character dictionary
    char_Num = {}
    num_Char = {}
    for i, j in enumerate(chars):
        char_Num[j] = i
        num_Char[i] = j

    return (char_Num, num_Char)


def corpus(text, charDict, sentLength):
    """
    Create a rnn-LSTM text corpus
    :param texts:  string create corpus from
    :param charDict:  lookup dictionary to covert a character into a one-hot encoded vector
    :param sentLength:  number of time steps
    """

    # number of observations
    sentences = int(len(text) - sentLength)

    # labels represent the next letter after a sentence
    labels = np.zeros([sentences, len(charDict)])
    # features represent every letter (one hot encoded) in every sentence
    features = np.zeros([sentences, sentLength, len(charDict)])

    # populate the labels and features
    for i in range(sentences):
        sentence = text[i:(i + sentLength)]
        for j, letter in enumerate(sentence):
            features[i, j, charDict[letter]] = 1
        labels[i, charDict[text[i + sentLength]]] = 1

    return (features, labels)


def sampler(guess):
    """
    Randomly sample a multinomial probability distribution of predicted characters
    :param guess:  probability distribution
    """
    probs = np.asarray(guess).astype('float64')
    probs = np.log(probs) / 0.4
    exp_probs = np.exp(probs)
    probs = exp_probs / np.sum(exp_probs)
    winner = np.random.multinomial(1, np.squeeze(probs), 1)
    return np.argmax(winner)


def lstm(features, labels, rand_sentence, numDict, epochs, batch_size, nodes):
    """
    train a lstm on text at single character level
    :param features:  text corpus for model training
    :param labels:  truth array of features, or next letter in the sentence (time series)
    :param random_sentence:  random seed for sentence prediction
    :param numDict:  dictionary to convert one-hot encoded characters to characters
    :param epochs:  number of epochs to train
    :param batches:  number of observations in a batch
    :param nodes:  number of nodes in each LSTM cell
    """

    # get sizes of inputs and labels
    sentence_length = features.shape[1]
    num_characters = labels.shape[1]

    #with tf.device("/gpu:0"):

    # define placeholders
    x = tf.placeholder(tf.float32, [None, sentence_length, num_characters])
    _y = tf.placeholder(tf.float32, [None, num_characters])

    # make a stacked RNN
    weights = tf.Variable(tf.random_normal([nodes, num_characters], stddev=0.01))
    biases = tf.Variable(tf.random_normal([num_characters], stddev=0.01))

    _x = tf.unstack(x, sentence_length, 1)

    def lstm_cell():
        cell =tf.contrib.rnn.BasicLSTMCell(nodes)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=0.7)
        return cell

    number_of_layers = 3
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(number_of_layers)])

    # Get lstm cell outputs
    outputs, states = rnn.static_rnn(stacked_lstm, _x, dtype=tf.float32)

    # get last output for prediction
    logits = tf.add(tf.matmul(outputs[-1], weights), biases)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=_y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        init = tf.global_variables_initializer()
        sess.run(init)

        # reshshape the data
        features = features.reshape(features.shape[0], sentence_length, num_characters)

        # train
        for i in range(epochs):
            print('\nepoch:  ', i)
            start = 0
            losses = []
            for j in range(int(features.shape[0] / batch_size)):
                x_batch = features[start:(start + batch_size), :, :]
                y_batch = labels[start:(start + batch_size), :]
                sess.run(train_op, feed_dict={x: x_batch, _y: y_batch})
                loss = sess.run(loss_op, feed_dict={x: x_batch, _y: y_batch})
                losses.append(loss)
                start += batch_size

            print('\nloss:  ', sum(losses)/len(losses))

            # predict some text using a seed from a random location in the text
            if i % 10 == 0:
                random_shiz = ''
                guess = np.copy(rand_sentence)

                for random_letter in range(1000):
                    # generate softmax character prediction
                    letter_prediction = sess.run(prediction, feed_dict={x: guess})

                    # sample a character from the softmax generated probability distribution
                    maximum = np.argmax(letter_prediction)
                    #maximum = sampler(letter_prediction)

                    # add this character to the end of the generated text and trim off the first character
                    pred = np.zeros([1, 1, num_characters])
                    pred[:, :, maximum] = 1
                    guess = np.concatenate((guess[:, 1:, :], pred), axis=1)
                    letter_index = np.argmax(np.reshape(guess[0, -1, :], (1, num_characters)))
                    random_shiz += (numDict[letter_index])

                print('\nprediction:  \n', random_shiz)

        return random_shiz


def output(seed, predicted):
    """
    Write the random seed and the final predicted string a text file
    :param seed:  random seed for prediction
    :param predicted:  predicted character sequence from trained model
    """

    # write the results to text
    try:
        output = open('output.txt', 'w')
        output.write('input:  \n')
        for i in seed:
            output.write(i)

        output.write('\n\n')
        output.write('output:  \n')
        for i in predicted:
            output.write(i)
        output.close()
        return
    except:
        sys.stderr('ERROR writing output')
        sys.exit(1)
        return


def run():
    # read user passed arguments
    parser = OptionParser()
    parser.add_option("-e", "--epochs", dest="epochs", help="number of epochs to train", default="200")
    parser.add_option("-b", "--batchsize", dest="batchsize", help="batchsize", default="128")
    parser.add_option("-n", "--nodes", dest="nodes", help="number of nodes in hidden layer cells", default="500")
    (options, args) = parser.parse_args()

    epochs = int(options.epochs)
    batchsize = int(options.batchsize)
    nodes = int(options.nodes)

    print('Training parameters:  ',
          '\nepochs:  ', epochs,
          '\nbatch size:  ', batchsize,
          '\nnumber of hidden layer nodes:  ', nodes)

    # get contents
    path = 'AnnaKarenina.txt'
    book = getbook(path)
    print('\nLength of text:  ', len(book))

    # get character dictionaries
    charNum, numChar = textDicts(book)

    # now we need to create a corpus, create arrays of the dimensions
    # X = [sentences, sentence length, number of characters]
    # y = [sentences, number of characters]
    sentenceLength = 50
    X, y = corpus(book, charNum, sentenceLength)

    # generate a sentence starting at a random location
    starter = random.randint(0, len(book) - sentenceLength)
    random_sentence = book[starter:starter + sentenceLength]

    # create a corpus from this random sentence
    X_rand = np.zeros([1, sentenceLength, len(charNum)])

    for i, j in enumerate(random_sentence):
        X_rand[0, i, charNum[j]] = 1

    # train a model
    print('\nrandom sentence seed: \n', random_sentence)
    generated = lstm(X, y, X_rand, numChar, epochs, batchsize, nodes)

    # write output to text and goodbye
    output(random_sentence, generated)
    print('Success, exiting....')
    sys.exit(0)


# -------- go time --------
if __name__ == '__main__':
    run()
