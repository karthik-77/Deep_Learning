from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def modify_vgg16(input_shape):
    vgg16=VGG16(input_shape=input_shape+[3],weights='imagenet',include_top=False)
    for layers in vgg16.layers:
        layers.trainable=False
    x=Flatten()(vgg16.output)
    predictions=Dense(6,activation='softmax')(x)
    model=Model(inputs=vgg16.input,outputs=predictions)
    model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def prepare_data(input_shape):
    path_train='/home/karthik/Desktop/archive/seg_train/seg_train'
    path_test='/home/karthik/Desktop/archive/seg_test/seg_test'
    train_datagen=ImageDataGenerator(rescale=1.0/255.0)
    test_datagen=ImageDataGenerator(rescale=1.0/255.0)
    train=train_datagen.flow_from_directory(path_train,target_size=input_shape,class_mode='categorical')
    test = test_datagen.flow_from_directory(path_test, target_size=input_shape, class_mode='categorical')
    return train,test




if __name__=="__main__":
    input_shape=[224,224]
    model=modify_vgg16(input_shape)
    print(model.summary())
    train,test=prepare_data(input_shape)
    model.fit_generator(train,epochs=4)
    loss,acc=model.evaluate(test)
    print("accuracy of the model------>",acc)
