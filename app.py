import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
import os
import camera
import model  # Ensure the model module exists
from PIL import Image, ImageTk

class App:
    def __init__(self, window=None, window_title="Camera Classifier"):
        if window is None:
            window = tk.Tk()
        
        self.window = window
        self.window.title(window_title)

        self.counters = [1, 1]
        self.autoprediction = False

        self.camera = camera.Camera()
        self.model = model.Model()  # Ensure model is initialized

        self.init_gui()
        self.delay = 15
        self.update()
        self.window.attributes('-topmost', True)
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = simpledialog.askstring("Classname One", "Enter the name of the first class:", parent=self.window)
        self.classname_two = simpledialog.askstring("Classname Two", "Enter the name of the second class:", parent=self.window)

        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=50, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=50, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=lambda: self.model.train_model(self.counters))
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="CLASS")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def auto_predict_toggle(self):
        self.autoprediction = not self.autoprediction  # Fix variable name

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if ret:
            os.makedirs(str(class_num), exist_ok=True)
            img_path = f'{class_num}/frame{self.counters[class_num - 1]}.jpg'
            
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Fix color conversion
            cv.imwrite(img_path, gray_frame)

            img = Image.open(img_path)
            img.thumbnail((150, 150))  # Removed deprecated Image.ANTIALIAS
            img.save(img_path)
            
            self.counters[class_num - 1] += 1

    def reset(self):
        for directory in ['1', '2']:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
        
        self.counters = [1, 1]  # Fixed reset bug
        self.model = model.Model()
        self.class_label.config(text='CLASS')

    def update(self):
        if self.autoprediction:
            self.predict()  # Now calls predict when autoprediction is enabled
        
        ret, frame = self.camera.get_frame()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.window.after(self.delay, self.update)

    def predict(self):
        """Predicts the class of the current frame and updates the UI."""
        ret, frame = self.camera.get_frame()
        if ret:
            prediction = self.model.predict( frame)  # Ensure correct model input format
            self.class_label.config(text=f"CLASS: {prediction}")
