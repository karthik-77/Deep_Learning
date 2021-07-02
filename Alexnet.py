import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_alexenet(input_shape):
    model=Sequential()
    model.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding='valid',activation='relu',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(6,activation='softmax'))
    model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def read_dataset():
    path='/home/karthik/Desktop/archive/seg_train/seg_train'
    path_test = '/home/karthik/Desktop/archive/seg_test/seg_test'
    train_datagen=ImageDataGenerator(rescale=1./255.0)
    train = train_datagen.flow_from_directory(path, target_size=(227, 227), class_mode='categorical')
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test = test_datagen.flow_from_directory(path_test, target_size=(227, 227), class_mode='categorical')
    return train,test



if __name__=="__main__":
    train,test=read_dataset()
    input_shape=227,227,3
    model=build_alexenet(input_shape)
    print(model.summary())
    model.fit_generator(train,epochs=5)
    loss,acc=model.evaluate_generator(test)
    print("accuracy------>",acc)