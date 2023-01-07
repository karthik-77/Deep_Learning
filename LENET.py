from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten

def build_lenet(input_shape=(32,32,1)):
    model=Sequential()
    model.add(Conv2D(filters=6,kernel_size=(5,5),strides=(1,1),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120,activation='relu'))
    model.add(Dense(80,activation='relu'))
    model.add(Dense(10,activation='sigmoid'))
    return model

if __name__=="__main__":
    model=build_lenet()
    print(model.summary())


