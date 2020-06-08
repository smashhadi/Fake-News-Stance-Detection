"""
Does not run directly as .py file without customizing
Python implementation in Google Colab
Batch Processing data 
Data already split in train, validation and test sets prior to this step
"""

import keras
class BatchGenerator(keras.utils.Sequence):
  def __init__(self, BATCH_SIZE, data):
    
    self.data = data
    self.BATCH_SIZE = BATCH_SIZE
    if self.data == "train":
        self.body_len = np.load('/content/../Data/colab/train_body_len.npy')
        self.bodies = np.load('/content/../Data/colab/tokenized_body_train.npy')
        self.headlines = np.load('/content/../Data/colab/tokenized_headline_train.npy')
        self.y = np.load('/content/../Data/colab/y_train.npy')
    if self.data == "validation":
        self.body_len = np.load('/content/../Data/colab/validation_body_len.npy')
        self.bodies = np.load('/content/../Data/colab/tokenized_body_validation.npy')
        self.headlines = np.load('/content/../Data/colab/tokenized_headline_validation.npy')
        self.y = np.load('/content/../Data/colab/y_validation.npy')
   
    self.labels = np.zeros((self.y.shape[0],4))
    for i,y_ in enumerate(self.y):
      self.labels[i, y_] = 1
    
    bodies_sent_len = np.array(self.body_len)[:,0].tolist()
    bodies_sent_len = np.array(bodies_sent_len).T[0]
    
    ind = bodies_sent_len.argsort()
    a = np.unique(bodies_sent_len) #unique lenghts
    b = [bodies_sent_len.tolist().count(i) for i in a] #Counts
    self.batches = []

    
    for i in a:
      idx = np.isin(bodies_sent_len, [i])
      batch_ind = np.where(idx)[0]
      np.random.shuffle(batch_ind)
      j = 0
      while j+self.BATCH_SIZE < len(batch_ind):
        batch = batch_ind[j:j+self.BATCH_SIZE]
        self.batches.append(batch.tolist())
        j+=self.BATCH_SIZE
      batch = batch_ind[j:len(batch_ind)]
      self.batches.append(batch.tolist())  
    
  def __getitem__(self, index):
    ind = self.batches[self.__len__()-index-1]
    first_body = self.bodies[ind[0]]
    
    bodies = np.zeros((len(ind), len(first_body), len(first_body[0])), dtype = np.int32)
    
    for i in range(bodies.shape[0]):
      
      b = self.bodies[ind[i]]
      for j in range(bodies.shape[1]):
        for k in range(bodies.shape[2]):
          bodies[i,j,k] = b[j][k]
        
    headlines = self.headlines[ind]
    y = self.labels[ind]
    
    return([bodies, headlines], y)
    
  def shuffle(self):
    for batch in self.batches:
      np.random.shuffle(batch)
       
  def __len__(self):
    return len(self.batches)
  
class DataShuffle(keras.callbacks.Callback):
  
  def __init__(self, batch):
    super(DataShuffle,self).__init__()
    self.batch = batch

  def on_epoch_end(self, epoch, logs={}):
	  print("shuffled")
	  self.batch.shuffle()
	  return


class TestBatchGenerator(keras.utils.Sequence):
  
  def __init__(self, BATCH_SIZE):
    
    self.BATCH_SIZE = BATCH_SIZE
    self.body_len = np.load('/content/MSCI_project/Data/colab/test_body_len.npy')
    self.bodies = np.load('/content/MSCI_project/Data/colab/tokenized_body_test.npy')
    self.headlines = np.load('/content/MSCI_project/Data/colab/tokenized_headline_test.npy')
    
    
    bodies_sent_len = np.array(self.body_len)[:,0].tolist()
    bodies_sent_len = np.array(bodies_sent_len).T[0]
    
    ind = bodies_sent_len.argsort()
    a = np.unique(bodies_sent_len) #unique lenghts
    b = [bodies_sent_len.tolist().count(i) for i in a] #Counts
    self.batches = []

    for i in a:
      idx = np.isin(bodies_sent_len, [i])
      batch_ind = np.where(idx)[0]
      j = 0
      while j+self.BATCH_SIZE < len(batch_ind):
        batch = batch_ind[j:j+self.BATCH_SIZE]
        self.batches.append(batch.tolist())
        j+=self.BATCH_SIZE
      batch = batch_ind[j:len(batch_ind)]
      self.batches.append(batch.tolist())  

    
  def __getitem__(self, index):
    ind = self.batches[index]
    first_body = self.bodies[ind[0]]
    
    bodies = np.zeros((len(ind), len(first_body), len(first_body[0])), dtype = np.int32)
    
    for i in range(bodies.shape[0]):
      
      b = self.bodies[ind[i]]
      for j in range(bodies.shape[1]):
        for k in range(bodies.shape[2]):
          bodies[i,j,k] = b[j][k]
      headlines = self.headlines[ind]
    return([bodies, headlines])   
       
  def __len__(self):
    return len(self.batches)
