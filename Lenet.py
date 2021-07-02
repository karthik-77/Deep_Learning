from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,AveragePooling2D,Dense,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical as tc
import numpy as np

def read_dataset():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    rows,cols=28,28
    x_train=x_train.reshape(x_train.shape[0],rows,cols,1)
    x_test=x_test.reshape(x_test.shape[0],rows,cols,1)
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_train=x_train/255.0
    x_test=x_test/255.0
    y_train=tc(y_train,10)
    y_test=tc(y_test,10)
    input_shape=rows,cols,1
    return (x_train,y_train),(x_test,y_test),input_shape

def build_lenet_model(input_shape):
    model=Sequential()
    model.add(Conv2D(filters=6,kernel_size=(5,5),strides=(1,1),padding='valid',activation='tanh',input_shape=input_shape))
    model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',activation='tanh'))
    model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(120,activation='tanh'))
    model.add(Flatten())
    model.add(Dense(84,activation='tanh'))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
    return model


if __name__=="__main__":
    (x_train,y_train),(x_test,y_test),input_shape=read_dataset()
    print("training shape ------->",x_train.shape,y_train.shape)
    print("testing shape----------->",x_test.shape,y_test.shape)
    model=build_lenet_model(input_shape)
    model.fit(x_train,y_train,batch_size=64,epochs=10,verbose=1)
    loss,acc=model.evaluate(x_test,y_test)
    print("accuracy of the model------------>",acc)
    img_index=4444
    pred=model.predict(x_test[img_index].reshape(1,28,28,1))
    print("the predicted number is",pred.argmax())




























# from matplotlib import pyplot as plt
# import numpy as np
# import tensorflow
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D,Dense,AveragePooling2D,Flatten
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical as tc
# from tensorflow.keras.optimizers import Adam
# def read_dataset():
#     (x_train,y_train),(x_test,y_test)=mnist.load_data()
#     rows,cols=28,28
#     x_train=x_train.reshape(x_train.shape[0],rows,cols,1)
#     x_test=x_test.reshape(x_test.shape[0],rows,cols,1)
#     x_train=x_train.astype('float32')
#     x_test=x_test.astype('float32')
#     x_train=x_train/255.0
#     x_test=x_test/255.0
#     y_train= tc(y_train,10)
#     y_test=tc(y_test,10)
#     input_shape=rows,cols,1
#
#     return (x_train,y_train),(x_test,y_test),input_shape
#
# def build_lenet_model(input_shape):
#     model=Sequential()
#     model.add(Conv2D(filters=6,kernel_size=(5,5),strides=(1,1),padding='valid',activation='tanh',input_shape=input_shape))
#     model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
#     model.add(Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),activation='tanh',padding='valid'))
#     model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
#     model.add(Flatten())
#     model.add(Dense(120,activation='tanh'))
#     model.add(Flatten())
#     model.add(Dense(84,activation='tanh'))
#     model.add(Dense(10,activation='softmax'))
#     model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
#     return model
#
#
# if __name__=="__main__":
#     (x_train,y_train),(x_test,y_test),input_shape=read_dataset()
#     model=build_lenet_model(input_shape)
#     model.fit(x_train,y_train,batch_size=64,epochs=10,verbose=1)
#     loss,acc=model.evaluate(x_test,y_test)
#     print("loss----->:",loss,"Accuracy------->:",acc)
#     img_index=1111
#     plt.imshow(x_test[img_index].reshape(28,28),cmap='Greys')
#     pred=model.predict(x_test[img_index].reshape(1,28,28,1))
#     print("prediction is ------>:",pred.argmax())



