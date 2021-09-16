#!/usr/bin/env python
# coding: utf-8

# In[123]:


import os
import re
import pandas as pd
import numpy as np

import tensorflow as tf

from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, LSTM ,Bidirectional
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras import layers
from keras.utils.vis_utils import plot_model

from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


# In[124]:


file1 = open("X_data.txt", encoding='cp437')
text1 = file1.readlines()
X_data = [x.strip() for x in text1]

file2 = open("y_data.txt", encoding='cp437')
text2 = file2.readlines()
y_data = [y.strip() for y in text2]

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.20, random_state=42)

X_train1 = X_train

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)

vocab_size = len(tokenizer.word_index) + 1

#maxlen = 50
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
                      


# In[125]:


from sklearn.preprocessing import LabelBinarizer
onehot = LabelBinarizer()
Y_train = onehot.fit_transform(y_train)
Y_val   = onehot.transform(y_val)


# In[126]:


model = Sequential()
optimizer = Adam(lr=0.001)

model.add(Embedding(vocab_size, 32, input_length=maxlen))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
#model.add(Bidirectional(LSTM(64)))
model.add(Dense(64 , activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


# In[129]:


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train,Y_train,
                    epochs= 1,
                   validation_data=(X_val, Y_val),
                    callbacks =[callback])


# In[130]:


train_score = model.evaluate(X_train, Y_train)
test_score= model.evaluate(X_val, Y_val)
#loss and accuracy


# In[131]:



print(train_score)
print(test_score)


# In[ ]:



plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[ ]:





# In[ ]:


file1 = open("X_data.txt", encoding='cp437')
text1 = file1.readlines()
X_data = [x.strip() for x in text1]


# In[ ]:


file2 = open("y_data.txt", encoding='cp437')
text2 = file2.readlines()
y_data = [y.strip() for y in text2]

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.20, random_state=42)


# In[ ]:


file3 = open("X_test.txt", encoding='cp437')
text3 = file3.readlines()
X_test = [x.strip() for x in text3]


# In[ ]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# In[ ]:


maxlen = 100

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# In[ ]:


#X_test[0]


# In[ ]:





# In[ ]:


prediction = model.predict_classes(X_test)
#print(prediction)


# In[ ]:



testcase = []
for i in prediction:
    if i == 1:
        testcase.append("1" + "\n")
    else:
        testcase.append("0" + "\n")


# In[ ]:


with open("answer.txt", "w", newline = "\n") as f:
    for item in testcase:
        f.write(item)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




