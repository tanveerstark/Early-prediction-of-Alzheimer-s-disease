import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class AlzheimerInference:
    def __init__(self, model_path):
        
""" Load trained model for inference  """
        
        self.model = tf.keras.models.load_model(model_path)
        self.class_labels = [
            'Non-Demented', 
            'Very Mild Dementia', 
            'Mild Dementia', 
            'Moderate Dementia'
        ]
    
    def preprocess_image(self, image_path, target_size=(224, 224)):

        """  Preprocess single image for prediction   """
         
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    
    def predict(self, image_path):

        """ Make prediction for single image """
         
        preprocessed_img = self.preprocess_image(image_path)
        predictions = self.model.predict(preprocessed_img)[0]
        
        results = {
            label: float(prob) 
            for label, prob in zip(self.class_labels, predictions)
        }
        
        return {
            'predictions': results,
            'most_likely_stage': max(results, key=results.get)
        }
