## Project 1
`deepLearning_tensorFlow_keras.ipynb`

## This project introduces an advanced approach using Convolutional Neural Networks (CNNs) for image classification. By leveraging the dataset, this project will classify images of bikes and cars into distinct categories. The goal is to apply computer vision techniques to classify vehicles based on their images, which could later be used in operational optimization and demand forecasting.

<!-- Develop a CNN-based image classification model that distinguishes between bikes and cars using the given image dataset. The CNN model will be implemented to classify images based on vehicle type, with potential extensions to classify other environmental features such as location and time of day. -->

## Task 1: Data preprocessing:

- Clean and preprocess the images from the Cars and Bikes Prediction Dataset
- Resize and normalize the images to be suitable for the CNN model input
- Split the dataset into training and testing sets

## Task 2: Building the CNN model:

- Design and implement a CNN using Keras or TensorFlow for image classification
- Use layers such as convolution, pooling, and dense layers to process the images
- Apply techniques like dropout, batch normalization, and data augmentation to enhance model robustness

## Task 3: Model training:

- Train the CNN on the images of bikes and cars
- Experiment with hyperparameters like learning rate, batch size, and the number of epochs to optimize model performance

## Task 4: Model evaluation:

Evaluate the CNN model using standard classification metrics like accuracy, precision, recall, and F1-score
Assess the impact of various architecture modifications such as changing layer configurations on classification performance

## Task 5: Visualization and reporting:

Visualize the CNN's predictions on sample images, displaying the confidence level for each class (bike or car)
Provide a final report that includes the CNN model architecture, evaluation metrics, and the final classification results

## Project 2
`home_loan_data.ipynb`

<!-- For a safe and secure lending experience, it's important to analyze the past data. In this project, you have to build a deep learning model to predict the chance of default for future loans using historical data. As you will see, this dataset is highly imbalanced and includes a lot of features that make this problem more challenging. 

Create a model that predicts whether or not an applicant will be able to repay a loan using historical data
-->

- Load the dataset that is given to you
- Check for null values in the dataset
- Print the percentage of default to a payer of the dataset for the TARGET column
- Balance the dataset if the data is imbalanced
- Plot the balanced or imbalanced data
- Encode the columns that are required for the model
- Calculate sensitivity as a metric
- Calculate the area under the receiver operating characteristics curve