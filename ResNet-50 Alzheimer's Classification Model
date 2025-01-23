import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class AlzheimerResNet:
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        """ Create ResNet-50 transfer learning model  """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """ Construct transfer learning model """
        # Base ResNet50 model
        base_model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_shape=self.input_shape
        )
        
        # Freeze base layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        output = Dense(
            self.num_classes, 
            activation='softmax', 
            name='alzheimers_classification'
        )(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def summary(self):
        """
        Display model architecture summary
        """
        return self.model.summary()
