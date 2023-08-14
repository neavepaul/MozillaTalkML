import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the MobileNetV2 model pretrained on ImageNet
model = MobileNetV2(weights='imagenet')

# Load and preprocess the image
img_path = 'dog.jpeg'  # Replace with your image file path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions using the model
predictions = model.predict(img_array)

# Decode and get the top predicted class label
decoded_predictions = decode_predictions(predictions, top=1)[0]
top_label = decoded_predictions[0][1]

print(top_label)
