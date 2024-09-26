# Handwritten Character Recognition Using CNN

## Overview
This project implements a **Convolutional Neural Network (CNN)** to perform handwritten character recognition. It takes image inputs of handwritten characters, processes them through several CNN layers, and classifies the images into character classes. The model is trained on a dataset of character images and achieves high accuracy through the use of convolutional layers, pooling, and data augmentation techniques.

## Features
- **Handwritten Character Classification**: The model is capable of identifying and classifying handwritten characters.
- **Convolutional Neural Network (CNN)**: Utilizes CNN architecture for feature extraction and classification.
- **Data Augmentation**: Includes data augmentation techniques to enhance the diversity of training data and improve model performance.
- **Accuracy**: The model achieves high accuracy in recognizing handwritten characters.

## Technologies Used
- **Python**: Programming language used to develop the model.
- **TensorFlow/Keras**: For building and training the CNN model.
- **OpenCV**: For image processing and manipulation.
- **NumPy/Pandas**: For data handling and preprocessing.
  
## Setup and Installation

### Prerequisites
- Python 3.x installed on your system.
- **TensorFlow** and **Keras** libraries installed.
- **OpenCV** for image processing.
  
### Installation Steps
1. Clone the repository:
   ```
   git clone https://github.com/sosophia10/Handwritten-Character-Recognition-using-CNN.git
   ```
   
2. Navigate to the project directory:
   ```
   cd Handwritten-Character-Recognition-using-CNN
   ```
   
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   
4. Download or generate the dataset for training and testing. 

5. Run the training script to train the CNN model:
   ```
   python train_model.py
   ```
   
6. After training, run the prediction script to test the model on new handwritten character images:
   ```
   python predict.py
   ```

## Dataset
You can use any publicly available dataset of handwritten characters, such as the EMNIST or MNIST datasets. Ensure that the dataset is properly preprocessed and split into training and testing sets before feeding it into the model.

## Model Architecture
The CNN model includes:

- Convolutional Layers: For extracting features from the input images.
- Pooling Layers: To reduce the spatial dimensions and prevent overfitting.
- Fully Connected Layers: For classifying the features into respective character classes.

### Data Augmentation
To improve model performance, the project includes data augmentation techniques such as:

- Rotation
- Scaling
- Translation
- Horizontal/Vertical Flipping

### Results
The trained model achieves high accuracy on both the training and test sets. It is capable of recognizing handwritten characters with strong generalization performance across different styles and shapes of writing.
