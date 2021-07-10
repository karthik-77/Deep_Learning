import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers  import Embedding,LSTM,Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split



pd.set_option("Display.max_rows",None,"Display.max_columns",None)
df=pd.read_csv("/home/karthik/Downloads/fake-news/train.csv")

#stats on the dataframe
print(df.shape)
print(df.columns)

#dropping rows with null values and resetting the dataframe
df.dropna(inplace=True)
df=df.reset_index()

#stats after removing null values
print(df.columns)
print(df.shape)

#we will take title as the input and lavel as the output for this particular problem.
#cleaning the respective input.

def clean_data(df):
    wn=WordNetLemmatizer()
    corpus=[]
    for i in range(len(df)):
        review=re.sub('[^a-zA-Z]',' ',df['title'][i])
        review=review.lower()
        review=review.split()
        review=[wn.lemmatize(word) for word in review if word not in stopwords.words('english')]
        review=' '.join(review)
        corpus.append(review)
    return corpus

corpus=clean_data(df)
print(corpus[0])

#converting the cleaned corpus into one_hot_representations
vocab_size=10000
one_hot_repr=[one_hot(word,vocab_size) for word in corpus]
print(one_hot_repr[0])

#padding the one_hot_repr so that everyline in the list has same length
max_length=max(len(x) for x in one_hot_repr)
embedded_docs=pad_sequences(one_hot_repr,padding='pre',maxlen=max_length)
print(embedded_docs[0])

#creating a model with embedding layer and LSTM
feature_dimension=100
model=Sequential()
model.add(Embedding(vocab_size,feature_dimension,input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer=Adam(),loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())

#preparing the training and testing data for model
x=np.array(embedded_docs)
y=np.array(df['label'])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(x_train.shape,y_train.shape)

#training the model and evaluating it
model.fit(x_train,y_train,epochs=10,batch_size=32)
y_pred=model.predict_classes(x_test)

#printing the results confusion matrix and accuracy score
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))







