# Fake-News-Stance-Detection
Go to www.fakenewschallenge.org for more information on the problem statement and datasets.
Code implementation by Sayema Mashhadi and Kesha Bodawala loosely based on the paper "Hierarchical Attention Networks for Document Classification" by Z. Yang et al.

## Motivation:
Stance detection can be the first step in detecting fake news. Stance detection can also be used to detect out clickbait articles.


## Formal Definition:
(from fakenewschallenge.org)
##### Input
A headline and a body text - either from the same news article or from two different articles.
##### Output
Classify the stance of the body text relative to the claim made in the headline into one of four categories:
Agrees: The body text agrees with the headline.
Disagrees: The body text disagrees with the headline.
Discusses: The body text discuss the same topic as the headline, but does not take a position
Unrelated: The body text discusses a different topic than the headline

## Challenges:
1. Imbalanced classes
2. Word encoding
3. Document encoding
4. Number of words in each article differed vastly

## Experimentation:
1. Using Tensorflow(Keras) vs Pytorch
2. Batch Normalization
3. Word embeddings (GloVe and Word2Vec)
4. Learning rates 
5. Initial state of Bi-LSTMs and DNN
6. Hyperparameter tuning
7. Optimizers
8. Dropout
 
Accuracy after each epoch can be seen in the file training_output.
