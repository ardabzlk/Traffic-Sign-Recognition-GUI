import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2
from keras.models import load_model

# Load the pre-trained traffic sign recognition model
# Model is trained on German Traffic Sign Dataset 
# https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
model = load_model('traffic_classifier.h5') 

# Define traffic sign classes
# classes are in the order of the labels in the dataset 

# classes from 0 to 42
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }


# Define a function to classify the traffic signs
def classify_traffic_sign(file_path):
    '''
    This function takes the file path of the traffic sign image as input
    and predicts the traffic sign class.
    
    Parameters
    ----------
    file_path : str
        File path of the traffic sign image.
    
    Returns
    -------
    None. 
    '''
    try:
        image = cv2.imread(file_path)
        image = cv2.resize(image, (30, 30))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize pixel values

        # Perform traffic sign classification
        pred = model.predict(image)
        predicted_class = np.argmax(pred)
        sign = classes[predicted_class]

        # Update the label with the predicted traffic sign
        label.configure(text=f"Predicted Traffic Sign: {sign}")
    except Exception as e:
        print(f"Error: {e}")
        label.configure(text="Error: Unable to classify the traffic sign.")

def upload_image():
    '''
    This function opens a file dialog box to upload the traffic sign image.

    Parameters
    ----------
    None.

    Returns
    -------
    None.
    '''
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        classify_traffic_sign(file_path)
    except Exception as e:
        print(f"Error: {e}")
        label.configure(text="Error: Unable to upload image.")

# Create a GUI window



top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Recognition')

# Create GUI components
upload_button = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload_button.pack(side=BOTTOM, pady=50)

sign_image = Label(top)
sign_image.pack(side=BOTTOM, expand=True)

label = Label(top, font=('Arial', 16))
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text="Traffic Sign Recognition", pady=20, font=('Arial', 20))
heading.pack()

top.mainloop()
