import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score

def trafic_sign():
    data = []
    labels = []
    classes = 43
    cur_path = os.getcwd()

    #Retrieving the images and their labels 
    for i in range(classes):
        path = os.path.join(cur_path,'train',str(i)) #path to the images
        images = os.listdir(path) #list of all images in the path

        for a in images: #iterating over each image in the path
            try:
                image = Image.open(path + '\\'+ a) #reading the image
                image = image.resize((30,30)) #resizing the image
                image = np.array(image) #converting the image to an array
                #sim = Image.fromarray(image)
                data.append(image)
                labels.append(i) #appending the image and its label in the dataset
            except:
                print("Error loading image")

    #Converting lists into numpy arrays
    data = np.array(data)  
    labels = np.array(labels) 

    print(data.shape, labels.shape)
    #Splitting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    #Converting the labels into one hot encoding
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)


    #Building the model
    model = Sequential() #initializing the model
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:])) # adding the first convolutional layer
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu')) # adding the second convolutional layer
    model.add(MaxPool2D(pool_size=(2, 2))) # adding the maxpool layer
    model.add(Dropout(rate=0.25)) # adding the dropout layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) # adding the third convolutional layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) # adding the fourth convolutional layer 
    model.add(MaxPool2D(pool_size=(2, 2))) # adding the maxpool layer 
    model.add(Dropout(rate=0.25)) # adding the dropout layer 
    model.add(Flatten()) # flattening the output from the convolutional layers so that it can be fed to the dense layers 
    model.add(Dense(256, activation='relu')) # adding the first dense layer 
    model.add(Dropout(rate=0.5)) # adding the dropout layer 
    model.add(Dense(43, activation='softmax')) # adding the output layer 

    #Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 15
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
    model.save("my_model.h5")

    #plotting graphs for accuracy 
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    #testing accuracy on test dataset


    y_test = pd.read_csv('Test.csv')

    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values

    data=[]

    for img in imgs:
        image = Image.open(img)
        image = image.resize((30,30))
        data.append(np.array(image))

    X_test=np.array(data)

    pred = model.predict_classes(X_test)

    #Accuracy with the test data

    print(accuracy_score(labels, pred))

trafic_sign()