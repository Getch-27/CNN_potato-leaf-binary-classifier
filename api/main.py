from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Allow CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Get the directory of the current file (main.py)
current_dir = os.path.dirname(__file__)

# Construct the dynamic path to the model
binary_model_path = os.path.join(current_dir, '..', 'saved_models', 'binary_model_20.keras')
disease_classification_model_path = os.path.join(current_dir, '..', 'saved_models', 'model.keras')


# Load the models
OBJECT_CLASSIFICATION_MODEL = tf.keras.models.load_model(binary_model_path, compile=False)
DISEASE_CLASSIFICATION_MODEL = tf.keras.models.load_model(disease_classification_model_path, compile=False)

# Define class names for potato disease classification
OBJECT_CLASS_NAMES = ['non_potato_leaf', 'potato_leaf']
DISEASE_CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# Helper function to read and preprocess the image
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert('RGB')
    image = image.resize((256, 256))  # Resize to the expected input size
    image = np.array(image) / 255.0  # Normalize to [0, 1] range
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Add batch dimension

    # pridiction of object as potato leaf or non-potato leaf then predict disease
    object_leaf_prediction = OBJECT_CLASSIFICATION_MODEL.predict(img_batch)
    pridicted_object_class = OBJECT_CLASS_NAMES[np.argmax(object_leaf_prediction[0])]  # Binary classification: 0 (non-potato), 1 (potato leaf)
    
    
    
    if pridicted_object_class == 'potato_leaf':  # If it's a potato leaf
        # Proceed with disease classification
        disease_predictions = DISEASE_CLASSIFICATION_MODEL.predict(img_batch)
        predicted_class = DISEASE_CLASS_NAMES[np.argmax(disease_predictions[0])]
        confidence = float(np.max(disease_predictions[0]))
        return {
            'class': predicted_class,
            'confidence': confidence
        }
    else:
        # If not a potato leaf, return a message indicating so
        return {
            'message': 'The image is not a potato leaf',
            'confidence': float(np.max(object_leaf_prediction[0]))  # Confidence of the potato leaf detection
        }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
