import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
raw_data_path = './data/raw'
processed_data_path = './data/processed'
metadata_path = './data/metadata/metadata.csv'

# Create directories for processed data
os.makedirs(processed_data_path, exist_ok=True)

# Load metadata
metadata = pd.read_csv(metadata_path)

# Data augmentation and preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    metadata,
    directory=raw_data_path,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    metadata,
    directory=raw_data_path,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
