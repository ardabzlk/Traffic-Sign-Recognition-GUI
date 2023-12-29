import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained traffic sign recognition model
model = load_model('traffic_classifier.h5')

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


# Function to classify traffic signs in a frame
def classify_traffic_sign(frame):
    try:
        resized_frame = cv2.resize(frame, (30, 30))
        processed_frame = np.expand_dims(resized_frame, axis=0)
        processed_frame = processed_frame / 255.0  # Normalize pixel values

        # Perform traffic sign classification
        pred = model.predict(processed_frame)
        predicted_class = np.argmax(pred)
        sign = classes[predicted_class]

        return sign
    except Exception as e:
        print(f"Error: {e}")
        return "Error: Unable to classify the traffic sign."

# Function to start real-time traffic sign detection
def start_detection():
    def update_frame():
        ret, frame = cap.read()
        if ret:
            # frame = cv2.flip(frame, 1)  # Mirror the frame
            sign = classify_traffic_sign(frame)

            # Display the frame with the predicted sign
            cv2.putText(frame, f"{sign}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Resize frame to fit the window
            frame = cv2.resize(frame, (800, 600))

            # Convert frame to RGB and then to ImageTk format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update the label with the new frame
            video_label.img_tk = img_tk  # Keep reference to avoid garbage collection
            video_label.configure(image=img_tk)
        
        # Repeat after 15 milliseconds (change delay as needed)
        video_label.after(15, update_frame)

    cap = cv2.VideoCapture(0)  # Open the webcam (change the index if using an external camera)
    if not cap.isOpened():
        print("Error: Unable to access the webcam")
        return

    top = tk.Toplevel()
    top.geometry('800x600')
    top.title('Traffic Sign Detection')

    video_label = Label(top)
    video_label.pack()

    # Start updating the frame
    update_frame()

# Create a GUI window
root = tk.Tk()
root.geometry('800x600')
root.title('Real-time Traffic Sign Recognition')

start_button = Button(root, text="Start Detection", command=start_detection, padx=10, pady=5)
start_button.pack(side=BOTTOM, pady=50)

label = Label(root, font=('Arial', 16))
label.pack(side=BOTTOM, expand=True)

heading = Label(root, text="Real-time Traffic Sign Recognition", pady=20, font=('Arial', 20))
heading.pack()

root.mainloop()