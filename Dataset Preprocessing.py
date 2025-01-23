import os
import shutil
import pandas as pd
import numpy as np

class OASISPreprocessor:
    def __init__(self, original_dataset_path, output_path):

        """ Initialize dataset preprocessing """

        self.original_path = original_dataset_path
        self.output_path = output_path
        
    def create_classification_directories(self):
        """ Create directories for Alzheimer's stages """
        stages = ['non_demented', 'very_mild', 'mild', 'moderate']
        for stage in stages:
            os.makedirs(os.path.join(self.output_path, stage), exist_ok=True)
    
    def organize_images(self, metadata_csv):
        """ Organize images into classification directories """
        # Read metadata
        metadata = pd.read_csv(metadata_csv)
        
        # Iterate through metadata
        for _, row in metadata.iterrows():
            image_path = row['image_path']
            alzheimers_stage = row['clinical_dementia_rating']
            
            # Map CDR to stages
            stage_mapping = {
                0: 'non_demented', 
                0.5: 'very_mild', 
                1: 'mild', 
                2: 'moderate'
            }
            
            stage_folder = stage_mapping.get(alzheimers_stage, 'non_demented')
            
            # Copy image to appropriate directory
            destination = os.path.join(self.output_path, stage_folder)
            shutil.copy(image_path, destination)
    
    def validate_dataset(self):
        """
        Validate preprocessed dataset
        """
        stages = ['non_demented', 'very_mild', 'mild', 'moderate']
        for stage in stages:
            stage_path = os.path.join(self.output_path, stage)
            print(f"{stage}: {len(os.listdir(stage_path))} images")
