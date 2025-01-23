import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

class OASISPreprocessor:
    def __init__(self, raw_data_path, processed_data_path, metadata_path):

        """ Initialize OASIS dataset preprocessor """
         
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.metadata_path = metadata_path
        # Create processed data directories
        self._create_directories()

    def _create_directories(self):
        """ Create directory structure for processed dataset"""
        stages = ['non_demented', 'very_mild', 'mild', 'moderate']
        splits = ['train', 'val', 'test']
        for stage in stages:
            for split in splits:
                os.makedirs(os.path.join(self.processed_data_path, split, stage), exist_ok=True)

    def preprocess_dataset(self):
        """Preprocess and split dataset into train/val/test"""
        # Load metadata
        metadata = pd.read_csv(self.metadata_path)
        # Map CDR to stages
        stage_mapping = {0: 'non_demented', 0.5: 'very_mild', 1: 'mild', 2: 'moderate'}
        # Prepare stratified split
        train_val, test = train_test_split(metadata, test_size=0.2, stratify=metadata['clinical_dementia_rating'])
        train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['clinical_dementia_rating'])
        # Process and copy images
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            for _, row in split_data.iterrows():
                stage = stage_mapping.get(row['clinical_dementia_rating'], 'non_demented')
                # Copy image to processed directory
                dest_path = os.path.join(self.processed_data_path, split_name, stage, os.path.basename(row['image_path']))
                shutil.copy(row['image_path'], dest_path)

    def dataset_summary(self):
        """Generate summary of processed dataset"""
        summary = {}
        stages = ['non_demented', 'very_mild', 'mild', 'moderate']
        splits = ['train', 'val', 'test']
        for split in splits:
            summary[split] = {}
            for stage in stages:
                stage_path = os.path.join(self.processed_data_path, split, stage)
                summary[split][stage] = len(os.listdir(stage_path))
        return summary

def main():
    preprocessor = OASISPreprocessor(
        raw_data_path='./data/raw',
        processed_data_path='./data/processed',
        metadata_path='./data/metadata/metadata.csv'
    )
    # Preprocess dataset
    preprocessor.preprocess_dataset()
    # Print dataset summary
    dataset_summary = preprocessor.dataset_summary()
    print("Dataset Distribution:")
    print(dataset_summary)

if __name__ == "__main__":
    main()
