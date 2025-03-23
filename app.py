# app.py
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import base64
import io
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load and preprocess data from CSV
data = pd.read_csv('https://raw.githubusercontent.com/karanveersinghmassan/AIML/refs/heads/main/data.csv')

def preprocess_image(image_string):
    try:
        image_data = base64.b64decode(image_string)
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=-1)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3: #RGB to Grayscale
            image = image.convert('L')
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=-1)
        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

images = []
labels = []
for _, row in data.iterrows():
    img = preprocess_image(row['image'])
    if img is not None:
        images.append(img)
        labels.append(row['label'])

images = np.array(images)
labels = np.array(labels)

# Define and train the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(images, labels, epochs=10) #Train with the entire dataset

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_uploaded_image(img_path):
    try:
        img = Image.open(img_path).resize((28,28))
        img_array = np.array(img) / 255.0
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img = img.convert('L')
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=-1)

        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_index] * 100
        return predicted_class_index, confidence

    except Exception as e:
        print(f"Error predicting image: {e}")
        return "Error", 0

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            predicted_class, confidence = predict_uploaded_image(filepath)
            return render_template('index.html', filename=filename, prediction=predicted_class, confidence=confidence)
        else:
            return "Invalid file type"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
