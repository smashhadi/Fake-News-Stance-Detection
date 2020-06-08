"""
Does not run directly as .py file
Python implementation in Google Colab
Batch Processing data 
Data already split in train, validation and test sets
"""

#to generate tokenized data. 
from nltk.corpus import stopwords
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords

MAX_SENT_LEN_HEADLINE = 15
MAX_SENT_LEN_BODY = 30
MAX_BODY_SENTENCES = 30
MAX_HEADLINE_SENTENCES = 1

#TODO: Add path to data
train_data = pd.read_csv('/content/../Data/colab/validation_data.csv', encoding = 'utf-8')#load train or validation data according to your need
train_df = train_data.copy()
with open('/content/../Data/colab/tokenizer.pickle', 'rb') as handle:
   tokenizer = pickle.load(handle)
stoplist = stopwords.words('english')

def filter_stopwords(words):
    filtered_words = []
    for word in words:
        if word not in stoplist:
            filtered_words.append(word)
    return filtered_words    

def get_headlines(df):
     return df['Headline']

def get_bodies(list_body_id):
    bodies = pd.read_csv('/content/../Data/colab/bodies.csv', encoding = 'utf-8', index_col = False)
    return [bodies.loc[bodies['Body ID']==body_id]['articleBody'].values.tolist() for body_id in list_body_id]


def tokenize_bodies(bodies):
    return [sent_tokenize(body[0]) for body in bodies] 
  
def tokenize_headlines(headlines):
    return [sent_tokenize(body) for body in headlines]

def trim_headlines(headlines):
    return [headline[0:MAX_HEADLINE_SENTENCES] for headline in headlines] 

def trim_bodies(bodies):
    return [body[0:MAX_BODY_SENTENCES] for body in bodies]
  
def get_headline_words(headlines):
    word_seq = [text_to_word_sequence(sent[0]) for sent in headlines]
    word_seq = filter_stopwords(word_seq)
    X = tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN_HEADLINE]) for seq in word_seq])
    return X
  
def get_body_words(bodies):
    body_words = []
    for body in bodies:
        word_seq = [text_to_word_sequence(sent) for sent in body]
        word_seq = filter_stopwords(word_seq)
        body_words.append(tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN_BODY]) for seq in word_seq]))
    return body_words
  
def create_length_matrix_headlines(headlines):
    len_headline = []
    for headline in headlines:
        len_headline.append(len(headline))
    return len_headline
  
def create_length_matrix_bodies(bodies):
    len_body = []
    for body in bodies:
        len_body.append([[len(body)]] + [[len(line) for line in body]])
    return len_body

def pad_headlines(headlines):
    return pad_sequences(headlines, maxlen=MAX_SENT_LEN_HEADLINE, padding='post', truncating='post')

def pad_bodies(bodies, len_matrix):
    padded_bodies = np.zeros((len(bodies), MAX_BODY_SENTENCES, MAX_SENT_LEN_BODY), dtype = np.uint8)
    for i in range(len(padded_bodies[0])):
      for j in range(len(padded_bodies[1])):
        for k in range(len(padded_bodies[2])):
            if j < len_matrix[i][0][0] and k < len_matrix[i][1][j]:
              padded_bodies[i,j,k] = bodies[i][j][k]
              
    return padded_bodies
    
train_bodies = get_bodies(train_df['Body ID'])
train_headlines = get_headlines(train_df)
train_sentences_body = tokenize_bodies(train_bodies)
train_sentences_headline = tokenize_headlines(train_headlines)
train_sentences_headline_trimmed = trim_headlines(train_sentences_headline)
train_sentences_body_trimmed = trim_bodies(train_sentences_body)
train_headline_words = get_headline_words(train_sentences_headline_trimmed)

train_body_words = get_body_words(train_sentences_body_trimmed)
train_headline_len = create_length_matrix_headlines(train_headline_words)
train_body_len = create_length_matrix_bodies(train_body_words)
train_tokenized_headlines = pad_headlines(train_headline_words)
train_tokenized_bodies = pad_bodies(train_body_words, train_body_len)


#change names 
np.save('colab/tokenized_headline_validation.npy', train_tokenized_headlines)
np.save('colab/tokenized_body_validation.npy', train_tokenized_bodies)
#np.save('colab/validation_headline_len.npy', train_headline_len)
#np.save('colab/validation_body_len.npy', train_body_len)
