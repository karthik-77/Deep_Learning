from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Dense,Flatten

def build_alexnet(input_shape=(227,227,3)):
    model=Sequential()
    model.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding='valid',input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='sigmoid'))
    return model

if __name__=="__main__":
    model=build_alexnet()
    print(model.summary())

