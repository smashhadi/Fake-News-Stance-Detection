"""
Do not run directly as .py file without making custom changes
Python implementation in Google Colab
Reads batch processed training data from saved location
Tests trained model on validation dataset
Test output was not available to competition participants. Choosing a model based on performance on Validation data.
""

import numpy as np
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, BatchNormalization, Activation, Bidirectional, LSTM, TimeDistributed, concatenate
from keras import initializers,regularizers
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.callbacks import LearningRateScheduler
import math
from keras import callbacks

MAX_VOCAB_SIZE = 20000
BATCH_SIZE = 10
N_EPOCHS =20

EMBEDDING_DIM = 300
MAX_SENT_LEN_HEADLINE = 10#15
MAX_SENT_LEN_BODY = 20#30
MAX_BODY_SENTENCES = 30#30
MAX_HEADLINE_SENTENCES = 1



b_train = BatchGenerator(BATCH_SIZE,"train")
b_validation = BatchGenerator(BATCH_SIZE,"validation")
b_test = TestBatchGenerator(1)
# dataset = StanceDetectionDataset('train', classes)
embedding_matrix = np.load('/content/../Data/colab/embeddings_matrix.npy')
# bodies = dataset.tokenized_body_train

embedding_layer = Embedding(embedding_matrix.shape[0],
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LEN_BODY,
                            trainable=True,
                            mask_zero = True)

init_weight = initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
init_bias = initializers.Constant(value=0.01)
l2_weight = regularizers.l2(0.001)
l2_bias =  regularizers.l2(0.0)
class_sample_count = [4944, 1293, 12001, 48439]
class_weight = [sum(class_sample_count)/ cl for cl in class_sample_count]

#headline rnn
headline_input = Input(shape=(MAX_SENT_LEN_HEADLINE,), dtype='int32', name = 'headline_input')
embedded_sequences1 = embedding_layer(headline_input)
#l_lstm1 = Bidirectional(LSTM(128, kernel_initializer = init_weight, bias_initializer = init_bias, kernel_regularizer = l2_weight, bias_regularizer = l2_bias, dropout = 0.2, name = 'headline_encoder'))(embedded_sequences1)
headline_lstm_output = Bidirectional(LSTM(128, return_state=True, kernel_initializer = init_weight, bias_initializer = init_bias, kernel_regularizer = l2_weight, bias_regularizer = l2_bias, dropout = 0.2, name = 'headline_encoder'))(embedded_sequences1)

#Body word rnn
sentence_input = Input(shape=(MAX_SENT_LEN_BODY,), dtype='int32', name='body_word_input')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(256, kernel_initializer = init_weight, bias_initializer = init_bias, kernel_regularizer = l2_weight, bias_regularizer = l2_bias, dropout = 0.2, name = 'word_encoder'))(embedded_sequences)
sentEncoder = Model(sentence_input, l_lstm)

#Body sentence rnn
review_input = Input(shape=(None,MAX_SENT_LEN_BODY), dtype='int32', name = 'body_sent_input')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(128, kernel_initializer = init_weight, bias_initializer = init_bias, kernel_regularizer =l2_weight, bias_regularizer = l2_bias, dropout = 0.2, name = 'sent_encoder'))(review_encoder, initial_state = headline_lstm_output[1:])

x = concatenate([headline_lstm_output[0], l_lstm_sent])
#Dense Output Layer
linear_1 = Dense(128, kernel_initializer = init_weight, bias_initializer = init_bias, kernel_regularizer = l2_weight, bias_regularizer = l2_bias, name = 'linear_layer')(x)
bn_1 = BatchNormalization(name='batchnorm')(linear_1)
relu_1 = Activation(activation='relu', name='relu_activation')(bn_1)
dropout_2 = Dropout(rate=0.5, name='dropout')(relu_1)
preds = Dense(4, activation='softmax', kernel_initializer = init_weight, bias_initializer = init_bias, kernel_regularizer = l2_weight, bias_regularizer = l2_bias, name ='output_layer')(dropout_2)
model = Model(inputs = [review_input, headline_input], outputs = [preds])

filepath="/content/../models/HierarchicalLSTM_{epoch:02d}_{val_loss:.4f}.h5"
checkpoint = callbacks.ModelCheckpoint(filepath, 
                                       monitor='val_acc',
                                       mode = 'max',
                                       verbose=0, 
                                       save_best_only=True)

def step_decay(epoch):
  initial_lrate = 0.01
  drop = 0.5
  epochs_drop = 4
  lrate = initial_lrate*math.pow(drop, math.floor((1+epoch)/epochs_drop))
  print("learning_rate: ", lrate)
  return lrate

def step_decay_warm(epoch):
  l_min = 0.001
  l_max = 0.01
  T = 5
  epoch_since_restart = epoch%T
  lr = l_min + 0.5 * (l_max - l_min) * (1 + np.cos(epoch_since_restart * np.pi/T))
  print("learning_rate: ", lr)
  return lr

opt = SGD(lr=0.01, momentum=0.9, decay=0, nesterov=False)#
# opt = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print("model fitting - Bidirectional LSTM")
model.summary()

shuffle = DataShuffle(b_train)
lrate = LearningRateScheduler(step_decay_warm)
callbacks_list = [lrate, shuffle, checkpoint]



model.fit_generator(b_train, validation_data=(b_validation),
          epochs=N_EPOCHS, class_weight = class_weight, callbacks =callbacks_list, shuffle = True)

a= model.predict_generator(b_test)
predictions = np.argmax(a, axis=-1)

# model.fit_generator(b,b.__len__(), epochs=3, class_weight = class_weight)

np.save('y_test.npy',predictions)
np.save('y_test_ind.npy', b_test.batches)

from keras.backend import eval
a = np.load('y_test_ind.npy')
b = np.load('y_test.npy')

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
ind = []


for l in a:
  ind.append(l)
ind = np.array(ind).squeeze()
ind
print(len(ind))
print(len(b))

final_y = ['none']*len(b)

for i in range(len(final_y)):
  
  final_y[ind[i]] = LABELS[b[i]]

np.save('final_prediction.npy',final_y)

print(final_y.count('none'))

import pandas as pd
test_data = pd.read_csv("colab/test_data.csv", encoding = 'utf-8')
test_data['Stance'] = final_y


test_data.to_csv('answer.csv', index=False, encoding='utf-8')
