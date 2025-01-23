from preprocessing import OASISPreprocessor
from model_architecture import AlzheimerResNet
from training_pipeline import AlzheimerTrainer
from inference import AlzheimerInference

def main():
    # Preprocessing
    preprocessor = OASISPreprocessor(
        original_dataset_path='/path/to/original/dataset',
        output_path='/path/to/processed/dataset'
    )
    preprocessor.create_classification_directories()
    preprocessor.organize_images('metadata.csv')
    preprocessor.validate_dataset()
    
    # Model Creation
    model = AlzheimerResNet()
    model.summary()
    
    # Training
    trainer = AlzheimerTrainer(
        model.model, 
        train_dir='/path/to/train/dataset',
        val_dir='/path/to/validation/dataset'
    )
    
    train_gen, val_gen = trainer.create_generators()
    history = trainer.train(train_gen, val_gen)
    
    # Inference
    inferencer = AlzheimerInference('best_alzheimer_model.h5')
    prediction = inferencer.predict('/path/to/test/image.jpg')
    print(prediction)

if __name__ == "__main__":
    main()
