the flow of the Alzheimer's disease prediction project using ResNet-50 with the OASIS dataset into detailed steps:
 
1. Data Preprocessing
                    ->This step involves preparing the dataset for training. The OASIS dataset contains MRI images and metadata.
                    ->Here's how you can preprocess the data:
                                       -Load Metadata: Read the metadata CSV file that contains information about the images and their corresponding labels.
                                       -Create Directories: Organize the data into directories based on the stages of Alzheimer's disease (e.g., non-demented, very mild, mild, moderate).
                                       -Split Data: Split the data into training, validation, and test sets while maintaining the class distribution.
                                       - Copy Images: Copy the images to the appropriate directories based on their labels.

2.Model Architecture
                    Use ResNet-50 as the base model for transfer learning. Add custom layers on top for classification.

3.Training the Model 
                    Train the model using the training and validation generators.
                    
4. Model Evaluation
                      -Evaluate the model's performance on the validation set using various metrics.
                      -Below are some evaluation metrics and visualizations:
                                            -Confusion Matrix
                                            -Classification Report
                                            -ROC Curve and AUC
    

6. Inference
             Use the trained model to make predictions on new images.
                      
