import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from sklearn.metrics import accuracy_score

class TrafficSignClassifier:
    def __init__(self):
        """Initialize TrafficSignClassifier class
        
        Attributes
        ----------
        data : list
            List of images
        labels : list
            List of labels
        classes : int
            Number of classes
        cur_path : str
            Current path
        model : Sequential
            CNN model
        """
        self.data = []
        self.labels = []
        self.classes = 43
        self.cur_path = os.getcwd()
        self.model = self._build_model()

    def _preprocess_images(self):
        """Preprocess images
        Resize images to 30x30 and append to data list
        Append labels to labels list

        """
        for i in range(self.classes):
            path = os.path.join(self.cur_path, 'train', str(i))
            images = os.listdir(path)

            for image_name in images:
                try:
                    image = Image.open(os.path.join(path, image_name))
                    image = image.resize((30, 30))
                    image = np.array(image)
                    self.data.append(image)
                    self.labels.append(i)
                except Exception as e:
                    print(f"Error loading image: {e}")

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def _split_data(self):
        """Split data into training and testing sets

        Returns
        -------
        X_train : list
            List of training images
        X_test : list
            List of testing images
        y_train : list
            List of training labels
        y_test : list
            List of testing labels
        """
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)
        y_train = to_categorical(y_train, self.classes)
        y_test = to_categorical(y_test, self.classes)
        return X_train, X_test, y_train, y_test

    def _build_model(self):
        """Build CNN model
        
        Returns
        -------
        model : Sequential
            CNN model
        """
        model = Sequential()

        # filters are used to extract features from the input image
        # here we use 32 filters but you can experiment with more or less
        # kernel size is the size of the filter matrix
        # activation is the activation function (ReLU is widely used)

        # other functions are available such as tanh, sigmoid, etc.
        # input shape is the shape of the input image
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(30, 30, 3)))

        # Batch normalization is a technique for improving the speed, performance, and stability of CNNs
        # it works by normalizing the outputs of neurons in a network
        model.add(BatchNormalization())
        
        # relu works by setting all the negative values in the feature map to zero
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2))) # Pooling layer is used to reduce the spatial dimensions of the output volume
        model.add(Dropout(rate=0.25)) # Dropout is used to reduce overfitting
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))

        model.add(Flatten()) # Flatten layer is used to convert the final feature maps into a one single 1D vector
        
        # Dense layers with Dropout and regularization
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(rate=0.5))

        # Output layer
        model.add(Dense(self.classes, activation='softmax'))
        
        # Optimizer with custom learning rate 
        optimizer = Adam(learning_rate=0.001) # Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data
        
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        return model

    def train(self):
        """Train CNN model
        """
        self._preprocess_images()
        X_train, X_test, y_train, y_test = self._split_data() # Get training and testing data
        epochs = 20 # Change epochs as needed (more epochs = higher accuracy but longer training time)
        history = self.model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test)) # Train model with training and testing data
        self._plot_training_history(history) # Plot training and validation accuracy/loss

    def _plot_training_history(self, history):
        """Plot training and validation accuracy/loss

        Parameters
        ----------
        history : History
            Training history
        """

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

    def test(self, test_csv_path='Test.csv'):
        """Test CNN model

        Parameters
        ----------
        test_csv_path : str
            Path to test CSV file
        """
        test_data = pd.read_csv(test_csv_path)
        labels = test_data["ClassId"].values
        imgs = test_data["Path"].values

        test_images = []
        for img in imgs:
            image = Image.open(img)
            image = image.resize((30, 30))
            test_images.append(np.array(image))

        X_test = np.array(test_images)
        pred_probs = self.model.predict(X_test)
        pred_classes = np.argmax(pred_probs, axis=1)

        # Convert labels to integer array if needed
        labels = labels.astype(np.int64)

        accuracy = accuracy_score(labels, pred_classes)
        print(f"Accuracy on test dataset: {accuracy}")
    def save_model(self, filename='my_model.keras'):
        self.model.save(filename)
        print(f"Model saved as {filename}")

# Usage
if __name__ == "__main__":
    classifier = TrafficSignClassifier()
    classifier.train()
    classifier.test()
    classifier.save_model()
