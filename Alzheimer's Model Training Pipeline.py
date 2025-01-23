import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class AlzheimerTrainer:
    def __init__(self, model, train_dir, val_dir):

        """ Initialize training pipeline """

        self.model = model
        self.train_dir = train_dir
        self.val_dir = val_dir
    
    def create_generators(self, img_size=(224, 224)):

        """  Create data generators with augmentation     """
         
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=img_size,
            batch_size=32,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=img_size,
            batch_size=32,
            class_mode='categorical'
        )
        
        return train_generator, val_generator
    
    def train(self, train_gen, val_gen, epochs=50):

        """ Train model with callbacks """

        # Callbacks
        checkpoint = ModelCheckpoint(
            'best_alzheimer_model.h5', 
            monitor='val_accuracy', 
            save_best_only=True
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Training
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[checkpoint, early_stop]
        )
        
        return history
