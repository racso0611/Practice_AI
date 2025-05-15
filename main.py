import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import tkinter as tk
from PIL import Image, ImageOps

# Step 1: Train the AI Model
def train_model(save_path="digit_recognizer.h5"):
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Reshape the data for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Build the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    
    # Compile the model
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    
    # Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save the model
    model.save(save_path)
    print(f"Model saved to {save_path}")

# Step 2: Create the Drawing UI
class DigitRecognizerApp:
    def __init__(self, model_path="digit_recognizer.h5"):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Initialize the UI
        self.window = tk.Tk()
        self.window.title("Handwritten Digit Recognizer")
        
        self.canvas = tk.Canvas(self.window, width=280, height=280, bg="white")
        self.canvas.pack()
        
        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack()
        
        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT)
        
        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)
        
        self.label = tk.Label(self.window, text="Draw a digit and click Predict.", font=("Helvetica", 14))
        self.label.pack()
        
        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.draw)
        self.last_x, self.last_y = None, None
    
    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=8)
        self.last_x, self.last_y = x, y
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.last_x, self.last_y = None, None
    
    def predict_digit(self):
        # Get the canvas content as an image
        self.canvas.update()
        ps = self.canvas.postscript(colormode="color")
        img = Image.open(ImageOps.grayscale(Image.frombytes("RGB", (280, 280), self.canvas.winfo_id())))
        
        # Preprocess the image
        img = img.resize((28, 28)).convert("L")
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Predict the digit
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Display the prediction
        self.label.config(text=f"Prediction: {digit} (Confidence: {confidence * 100:.2f}%)")
    
    def run(self):
        self.window.mainloop()

# Step 3: Run the UI
if __name__ == "__main__":
    # Ensure the model is trained or loaded
    # Uncomment the next line to train the model if not already trained
    #train_model()
    
    app = DigitRecognizerApp()
    app.run()
