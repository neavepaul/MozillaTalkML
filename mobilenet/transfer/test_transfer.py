import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('meme_classifier_model.h5')

# Load and preprocess a new meme image
new_img_path = 'drake1.jpg'
new_img = load_img(new_img_path, target_size=(224, 224))
new_img_array = img_to_array(new_img)
new_img_array = np.expand_dims(new_img_array, axis=0)
new_img_array = new_img_array / 255.0 

# Make predictions using the loaded model
predictions = model.predict(new_img_array)

# Decode and print the predicted class
class_labels = ['drake', 'incredible', 'spidey']
predicted_class_index = np.argmax(predictions)
predicted_class_label = class_labels[predicted_class_index]

print(f"Predicted Class: {predicted_class_label}")
