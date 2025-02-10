from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import cv2 as cv

class Model:

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 1)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, counters):
        img_list = []
        class_list = []

        for i in range(1, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip if image is missing
            img = cv.resize(img, (150, 150))
            img_list.append(img)
            class_list.append(0)  # Book

        for i in range(1, counters[1]):
            img = cv.imread(f'2/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip if image is missing
            img = cv.resize(img, (150, 150))
            img_list.append(img)
            class_list.append(1)  # Phone

        img_list = np.array(img_list).reshape(-1, 150, 150, 1) / 255.0
        class_list = np.array(class_list)

        self.model.fit(img_list, class_list, epochs=10, batch_size=32, verbose=1)
        print("Model successfully trained!")

    def predict(self, frame):
        if frame is None:
            raise ValueError("Error: Frame is None.")

        if len(frame.shape) == 3 and frame.shape[2] == 3:  # Ensure it's a valid RGB image
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        frame = cv.resize(frame, (150, 150))  # Resize
        frame = np.array(frame).reshape(1, 150, 150, 1) / 255.0  # Normalize and reshape

        prediction = self.model.predict(frame)
        class_label = "Book" if round(prediction[0][0]) == 0 else "Phone"

        return class_label  # Return the class name instead of 0/1
