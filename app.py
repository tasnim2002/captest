import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

# Load the trained model
model = load_model('car_fault_model3.h5')

@app.route('/')
def home():
    return "Welcome to the Flask App!"

# Define the image dimensions
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Load and preprocess the image
    img = load_img(file, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make a prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)

    # Map the predicted class to the corresponding label
    classes = ['Driver A', 'Driver B', 'Both', 'Neither']
    result = classes[predicted_class[0]]

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
