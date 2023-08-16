from flask import Flask, jsonify, request
import cv2
import numpy as np
from keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model
model = load_model('my_emotion_model.h5')
model2 = load_model('my_breed_model.h5')
# Define the API endpoint
@app.route('/predict/emotion', methods=['POST'])
def predict_emotion():
    # Get the image file from the request
    image_file = request.files['image']
    if 'image' not in request.files or request.files['image'].filename == '':
        return 'Error: No file uploaded or file is empty'
    
    # Read the image data and preprocess it
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (299, 299))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make the prediction using the trained model
    prediction = model.predict(image)[0]
    
    # Map the predicted class index to the corresponding emotion label
    labels = ["angry", "happy", "sad", "scared"]
    predicted_emotion = labels[np.argmax(prediction)]
    
    # Return the prediction as a JSON response
    return jsonify({'emotion': predicted_emotion})

@app.route('/predict/breed', methods=['POST'])
def predict_breed():
    # Get the image file from the request
    image_file = request.files['image']
     if 'image' not in request.files or request.files['image'].filename == '':
        return 'Error: No file uploaded or file is empty'
    # Read the image data and preprocess it
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (299, 299))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make the prediction using the trained model
    prediction = model2.predict(image)[0]
    
    # Map the predicted class index to the corresponding emotion label
    labels = ["Abyssinian","American Bobtail","American Curl","American Shorthair","Applehead Siamese","Balinese",
                  "Bengal","Birman","Bombay","British Shorthair","Burmese","Calico","Cornish Rex","Devon Rex","Dilute Calico","Dilute Tortoiseshell","Domestic Long Hair","Domestic Medium Hair","Domestic Short Hair","Egyptian Mau"
                 ,"Exotic Shorthair","Extra-Toes Cat - Hemingway Polydactyl","Havana","Himalayan","Japanese Bobtail","Maine Coon","Manx","Munchkin","Nebelung","Norwegian Forest Cat","Ocicat","Oriental Short Hair","Oriental Tabby","Persian","Pixiebob","Ragamuffin","Ragdoll","Russian Blue"
,"Scottish Fold","Siamese","Siberian","Snowshoe","Sphynx - Hairless Cat","Tabby","Tiger","Tonkinese","Torbie","Tortoiseshell","Turkish Angora","Turkish Van","Tuxedo"]
    predicted_breed = labels[np.argmax(prediction)]
    
    # Return the prediction as a JSON response
    return jsonify({'Breed': predicted_breed})

@app.route('/predict', methods=['GET'])
def predict_test(): 
    return jsonify({'Hello Hello Hello': 'Hello hello hello'})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)