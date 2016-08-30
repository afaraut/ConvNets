from keras.layers import Convolution1D, Dense, MaxPooling1D, Reshape, Dropout, Activation, Embedding
from keras.optimizers import SGD, Adadelta
from keras.models import Sequential, model_from_json
import csv
import numpy as np
import json
import time
import os

csv.field_size_limit(int(922337203))

ALPHABET = r"""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'"/\|_@#$%^&*~`+-=<>()[]{} """
#ALPHABET = r"""abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'"/\|_@#$%^&*~`+-=<>()[]{} """
ALPHABET_SIZE = len(ALPHABET)
CLASSIFICATION_MAX = 14
EMBEDDING_SIZE = np.log2(ALPHABET_SIZE) + 1

# Parameters
BATCH_SIZE = 32
NUM_EPOCHS = 20
VAL_SPLIT = 0.1
SENTENCE_LENGTH = 400
LEARNING_RATE = 0.01
VERBOSE = 2
CURRENT_FILE = os.path.splitext(os.path.basename(__file__))[0]
NAME_FILE_MODEL_WEIGHTS = "../{}.h5".format(CURRENT_FILE)
DIRECTORY_INPUT = './.../'
INDEX_CLASS_IN_CORPUS = 0
INDEX_SENTENCE_IN_CORPUS = 2
NUMBER_OF_CONVOLUTION_KERNELS = 96
POOL_LENGTH = 3
STRIDE = 3


def nb_elem_with_length_in_corpus(file, nbelem):
    compteur = 0
    with open(file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if len(row[INDEX_SENTENCE_IN_CORPUS]) >= nbelem:
                compteur = compteur + 1
    return compteur


def create_vector_embeddings_from_class(classification_value):
    vec = np.zeros(CLASSIFICATION_MAX, dtype=np.float32)
    vec[int(classification_value)-1] = 1
    return np.asarray(vec, dtype=np.float32)


def create_vector_embeddings_from_char(char):
    return get_index_of_char(char)


def get_index_of_char(char):
    return ALPHABET.find(char)


def load_all_data(path):

    train_length = nb_elem_with_length_in_corpus(path + 'train.csv', SENTENCE_LENGTH)
    test_length = nb_elem_with_length_in_corpus(path + 'test.csv', SENTENCE_LENGTH)

    train_length = int(train_length)
    test_length = int(test_length)
    
    x_train, y_train = load_data(path + 'train.csv', train_length)  # Training
    x_test, y_test = load_data(path + 'test.csv', test_length)  # Test
    return x_train, y_train, x_test, y_test


def load_data(filename, corpus_length):
    vec_corpus_x = []
    vec_corpus_y = []
    tab = [0] * CLASSIFICATION_MAX
    tmp_compteur = 0
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

        while tmp_compteur < corpus_length:
            row = spamreader.next()
            if len(row[INDEX_SENTENCE_IN_CORPUS]) < SENTENCE_LENGTH:
                continue
            tmp_compteur += 1
            vec_sentence = []
            tab[int(row[INDEX_CLASS_IN_CORPUS])-1] += 1
            vec_corpus_y.append(create_vector_embeddings_from_class(row[INDEX_CLASS_IN_CORPUS]))
            row[INDEX_SENTENCE_IN_CORPUS] = str(row[INDEX_SENTENCE_IN_CORPUS])
            for letter in row[INDEX_SENTENCE_IN_CORPUS][:SENTENCE_LENGTH]:
                vec_sentence.append(create_vector_embeddings_from_char(letter))

            vec_sentence = np.asarray(vec_sentence, dtype=np.float32)
            # --- Add the sentence vector to the corpus vectors
            vec_corpus_x.append(vec_sentence)

    vec_corpus_y = np.asarray(vec_corpus_y, dtype=np.float32)
    vec_corpus_x = np.asarray(vec_corpus_x, dtype=np.float32)

    print tab

    return vec_corpus_x, vec_corpus_y


def create_cnn_model():
    # main sequential model
    model = Sequential()

    emb_dim = np.log2(ALPHABET_SIZE) + 1
    el = Embedding(ALPHABET_SIZE, emb_dim, input_length=SENTENCE_LENGTH)
    model.add(el)

    conv_1 = Convolution1D(NUMBER_OF_CONVOLUTION_KERNELS, 7, input_shape=(SENTENCE_LENGTH, 1))
    model.add(conv_1)

    model.add(Activation('relu'))

    pool_1 = MaxPooling1D(pool_length=POOL_LENGTH, stride=STRIDE)
    model.add(pool_1)
    print("pool_1 " + str(pool_1.output_shape))

    conv_2 = Convolution1D(NUMBER_OF_CONVOLUTION_KERNELS, 7)
    model.add(conv_2)

    model.add(Activation('relu'))

    pool_2 = MaxPooling1D(pool_length=POOL_LENGTH, stride=STRIDE)
    model.add(pool_2)
    print("pool_2 " + str(pool_2.output_shape))

    conv_3 = Convolution1D(NUMBER_OF_CONVOLUTION_KERNELS, 3)
    model.add(conv_3)

    pool_3 = MaxPooling1D(pool_length=POOL_LENGTH, stride=STRIDE)
    model.add(pool_3)
    print("pool_3 " + str(pool_3.output_shape))

    pool_3_new_steps = pool_3.output_shape[1]
    pool_3_nb_filter = pool_3.output_shape[2]
    reshape_size = pool_3_new_steps * pool_3_nb_filter

    model.add(Reshape([reshape_size]))

    print model.output_shape

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(CLASSIFICATION_MAX))
    model.add(Activation('softmax'))

    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adadelta)

    print("OK")
    return model


def save_cnn_model_to_json(json_string, filename):
    json_obj = json.loads(json_string)
    f = open(filename, 'w+')
    json.dump(json_obj, f, indent=4)
    f.close()


def load_cnn_model_from_json(filename):
    return model_from_json(filename)


def train_model(model, x_train, y_train, x_test, y_test):

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
              validation_split=VAL_SPLIT, verbose=VERBOSE)

    print 'Testing...'
    res = model.evaluate(x_test, y_test,
                         batch_size=BATCH_SIZE, verbose=VERBOSE)

    print('Test accuracy loss: ', res)

    loss = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=VERBOSE)
    print('Train accuracy loss: ', loss)


def keras_predict(model, x_test, y_test):
    print '# predict'
    output_array = model.predict(x_test, batch_size=BATCH_SIZE)
    len_output_array = len(output_array)

    numerator = 0
    for i in xrange(len_output_array):
        probable_elem = output_array[i].tolist()
        probable_class = probable_elem.index(max(probable_elem))
        real_elem = y_test[i].tolist()
        real_class = real_elem.index(max(real_elem))
        if int(real_class) == int(probable_class):
            numerator = numerator + 1

    result = str("{} / {}".format(numerator, len_output_array))
    percentage = float(float(numerator)/int(len_output_array)) * 100

    print ("*************************************************************")
    print ("ALPHABET ", ALPHABET)
    print ("ALPHABET_SIZE ", ALPHABET_SIZE)
    print ("CLASSIFICATION_MAX ", CLASSIFICATION_MAX)
    print ("BATCH_SIZE ", BATCH_SIZE)
    print ("NUM_EPOCHS ", NUM_EPOCHS)
    print ("SENTENCE_LENGTH ", SENTENCE_LENGTH)
    print ("LEARNING_RATE ", LEARNING_RATE)
    print ("DIRECTORY_INPUT ", DIRECTORY_INPUT)
    print ("NUMBER_OF_CONVOLUTION_KERNELS ", NUMBER_OF_CONVOLUTION_KERNELS)
    print ("RESULT ", result)
    print ("PERCENTAGE ", percentage)
    print ("*************************************************************")

    print '# end'


def app():
    x_train, y_train, x_test, y_test = load_all_data(DIRECTORY_INPUT)

    print '# create_cnn_model'
    model = create_cnn_model()

    print '# train_model'
    train_model(model, x_train, y_train, x_test, y_test)

    print '# save_weights'
    model.save_weights(NAME_FILE_MODEL_WEIGHTS, overwrite=True)

    # predict
    keras_predict(model, x_test, y_test)


def app_by_loading_weight():
    print 'CNN by loading weight'

    x_train, y_train, x_test, y_test = load_all_data(DIRECTORY_INPUT)

    print '# create_cnn_model'
    model = create_cnn_model()

    print '# load_weights'
    model.load_weights(NAME_FILE_MODEL_WEIGHTS)

    # predict
    keras_predict(model, x_test, y_test)


def main():
    app()
    #  app_by_loading_weight()


if __name__ == '__main__':
    main()