welcome to the Course 4, where we will be focusing on Deep learning. Learning about Deep Neural Networks (DNN), Artificial Neural Networks (ANN) and the perceptrons used for binary classification, TensorFlow used for building models, Model Optimization and performace improvements, Transfer learning, object detection and much more!


## Artificial Neural Network Folder

`intro_deep_learning.ipynb`
- intro to deep learning
- deep learning vs machine learning
- Applications of deep learning
- limitations of deep learning
- Keras, TensorFlow, PyTorch
- Deep Learing Lifecycle

`intro_artificial_neural_network.ipynb`
- Artificial Neuron
- Neural Networks and the different types
    - Perceptrons
    - Multilayer perceptron
    - Deep Neural Networks or DNNs 
    - Convolutional neural networks or CNNs
    - Recurrent neural networks or RNNs
- Feedforward Neural Networks
- Activation Function
- types of activation functions
- forward propagation in perceptrons
- backpropagation in perceptrons
- error landscapre
- vanishing gradient
- Exploding Gradient

`perceptron_classification.ipynb`
- Build perceptron-based classification model
- Perform preprocessing and splitting on data
- Initiakize and fit the perceptron

`neural_networks_activation_functions.ipynb`
- configuring Neural Network
- applying activation function
- applying the think function

## Deep Neural Networks Folder

`deep_neural_networks.ipynb`
- deep neural networks architecture
- Loss function in DNN and Types
    - regression loss
        - Mean Absolute error
    - classifon loss
        - Mean squareed error
    - Forward and Backward propagation in DNN

    `forward_propagation.ipynb`
    - predict data output
    - calculate the errors
    - calculate sum squared error
    

## TensorFlow Folder
`intro_tensors.ipynb`
- Tenors
- TensorFlow
- dataflow graph
- TensorFlow APIs
- TensorFlow Playground
- TFLearn
- Keras
     
`tensorFlow_practice.ipynb`
- TensorFlow hands on
- Create sequential model
- applying softmax activation
- Create probability model

`deep_neural_networks_tensorflow.ipynb`

<!-- Building Deep Neural Networks on TensorFlow refers to the process of designing and constructing neural network models using the TensorFlow framework. This involves defining the architecture of the neural network, selecting appropriate layers and activation functions, specifying the optimization algorithm, and training the model using data. -->

`sequential_apis_in_tensorflow.ipynb`
<!-- The sequential API in TensorFlow offers a high-level interface for building and training deep learning models. It allows for the sequential addition of layers, simplifying the process of constructing neural network architectures by specifying the input shape and layer type. -->
- load fashion data set
- build model
- train test split
- compile the model
- evaluate the model
- predict the model

`functional_apis_tensorflow.ipynb`
<!-- The functional APIs in TensorFlow is an alternative way to create and customize complex neural network models. It allows you to build models with more flexibility and handle multiple inputs and outputs. -->
- load a dataset 
- inspect and visualize dataset
- build neural network modle using functional API
- compile model
- evaluate the model

`tensorflow_practice_2`
- load dataset
- define classifiers
- voting classifier
- train and evaluate classifiers
- visualization of decision boundaries

## PyTorch 
`pytorch_intro.ipynb`
- Features of PyTorch
- Modules in PyTorch
    - Basic Layers
    - Activation Function
    - Pooling Layers
    - Normalization Layers
    - Dropout Layers

`torch_dl_model.ipynb`
<!-- build and train a deep learning model using the FashionMNIST dataset, which consists of 28x28 grayscale images of 10 different clothing items. The model, a convolutional neural network (CNN), is trained to classify these images into their respective categories, leveraging PyTorch for the implementation and employing techniques such as normalization and Adam optimization to enhance performance. -->

`torch_classifier_model.ipynb`
<!-- In this example, we develop and train a deep learning model utilizing the MNIST dataset, which comprises 28x28 grayscale images of handwritten digits from 0 to 9. The model, a fully connected neural network, is meticulously designed to classify these images into their respective digit categories. We employ PyTorch, a powerful and flexible deep learning framework, to facilitate our implementation. -->

## Model Optimization & Performance Improvement Folder
`model_opt_perfor_improvement.ipynb`
- Optimization Algorithms
- Importance of Optimization Algorithms
- Optimizers and their Types
    - Gradient Descent
    - Stochastic Gradient Descent
    - Stochastic Gradient Descent mini batch
    - Momentum
    - Nesterov Accelerated Gradient (NAG)
    - RMSProp
    - Adadelta
    - Adam Optimizer
- Batch Normalization
- Regularization
- Modifying the loss function
- loss function strategies
- Data augmentation
- K fold cross-validation
- Vanishing Gradient
- Prevent vanishing gradient
- Exploding Gradient
- Hyperparameter and Parameters

`implementation_of_SGD.ipynb`
<!-- Stochastic gradient descent (SGD) is an optimization algorithm, commonly used in machine learning to train models. It is easier to fit into memory due to a single training sample being processed by the network.
- It is computationally fast as only one sample is processed at a time. For larger datasets, it can converge faster as it causes updates to the parameters more frequently. -->

`implementation_of_Momentum.ipynb`
- calculate the square of **x**, representing the objective function.
- The derivative(x) function computes the derivative of x with respect to the objective function.
- The **gradient_descent(objective, derivative, bounds, n_iter, step_size, momentum)** function implements the gradient descent algorithm. It initializes a solution within the specified bounds and iteratively updates it based on the objective and derivative functions. The function also tracks and stores the solutions and their corresponding scores.
- The random seed is set to 4 using seed(4) to ensure reproducibility.
- The bounds variable defines the lower and upper bounds for the solution space.
- Parameters such as the number of iterations (n_iter), step size (step_size), and momentum (momentum) are specified.
- The **gradient_descent** function is called with the provided arguments, and the resulting solutions and scores are stored.
- An array of input values (inputs) is generated using **arange** within the defined bounds.
- The objective function values (results) are computed for the input values.
- The objective function curve is plotted using **pyplot.plot** with inputs on the x-axis and results on the y-axis.
- The optimization path is visualized by plotting the solutions and scores as red dots connected by lines using pyplot.plot.
- Finally, **pyplot.show()** is called to display the plot.

`implementation_of_AdaGrad.ipynb`
- Define the objective function as the sum of squares of x and y.
- Set the bounds for input variables.
- Generate arrays of x and y values within the specified bounds at 0.1 increments.
- Create a mesh grid from the x and y arrays.
- Compute the objective function values for each combination of x and y in the mesh grid.
- Create a filled contour plot with 50 contour levels and 'jet' color scheme.
- Display the plot.

`implementation_of_RMSProp.ipynb`
- Define the objective function as the sum of squares of x and y.
- Set the bounds for input variables.
- Generate arrays of x and y values within the specified bounds at 0.1 increments.
- Create a mesh grid from the x and y arrays.
- Compute the objective function values for each combination of x and y in the mesh grid.
- Create a filled contour plot with 50 contour levels and 'jet' color scheme.
- Display the plot.

`implementation_of_Adadelta.ipynb`
- Define the objective function as the sum of squares of x and y.
- Set the bounds for input variables.
- Generate arrays of x and y values within the specified bounds at 0.1 increments.
- Create a mesh grid from the x and y arrays.
- Compute the objective function values for each combination of x and y in the mesh grid.
- Create a filled contour plot with 50 contour levels and 'jet' color scheme.
- Display the plot.

`implementation_of_Adam.ipynb`
- Initialization and Setup:

    - Start by initializing the solution within the specified bounds.
    - Prepare variables m and v to store the first and second moments (moving averages of the gradients and squared gradients).

- Gradient Computation and Update:

    - Calculate the gradient of the objective function.
    - Update the moments using exponential decay rates beta1 and beta2.
    - Adjust each parameter based on the biased-corrected first and second moment estimates.

`implementation_of_Dropout.ipynb`
- Randomly selected neurons are ignored during training. They are 'dropped out' randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass, and any weight updates are not applied to the neuron on the backward pass.


- If neurons are randomly dropped out of the network during training, other neurons will have to step in and handle the representation required to make predictions for the missing neurons. This is believed to result in multiple independent internal representations being learned by the network.

- The effect is that the network becomes less sensitive to the specific weights of neurons. This, in turn, results in a network that is capable of better generalization and is less likely to overfit the training data.


`LEP_Implementation_Hyperparameter_Tuning.ipynb`
- Hyperparameter tuning is the process of systematically searching for the best combination of hyperparameter values for a machine learning model.
- It involves selecting a subset of hyperparameters and exploring different values for each hyperparameter to find the configuration that optimizes the model's performance on a given dataset.

## Convolutional Neural Networks folder

`convolutional_neural_networks.ipynb`
- Convolutional Neural Networks
- CNN applications
- image data
- Convolution Operation
- CNN Architecture
    - convolution layer
    - pooling layer
    - fully connected layer
    - output layer
- CNN architecture parameters
    - Strides
    - Padding
- ResNet
- CNN filters
- Pooling in CNN

`image_data.ipynb`
1. Import the necessary libraries
2. Read and display the image
3. Display RGB channels
4. Flip augmentation
5. Perform width shifting augmentation
6. Change the brightness augmentation

`cnn_image_classification.ipynb`
1. Import the necessary libraries and dataset
2. Count and retrieve the images
3. Create a training dataset
4. Create a validation dataset
5. Visualize a subset of images from the training dataset
6. Preprocess and normalize the training dataset
7. Create a convolutional neural network model with data augmentation
8. Summarize and compile the model
9. Train the model
10. Visualize the result
11. Predict the class of a given image

`image_classification.ipynb`
1. Import the necessary libraries
2. Load and normalize the CIFAR10 training and test datasets using TensorFlow
3. Display a batch of training images
4. Define the convolutional neural network
5. Compile the Model
6. Train the network on the training data with validation split
7. Test the network on the test data
8. Predict a batch of test images
9. Perform Classes on Individual Datasets

## Transfer Learning Folder.ipynb

`transfer_learning_intro.ipynb`
- what is transfer learning
- transfer learning in DNN
    - positive transfer
    - negative transfer
- pre-trained model specs
- compare and contrast 
- pre-trained models: image domain
- pre-trained models: text domain
- pre-trained models: audio domain
- pre-traied models: video domain

`tranfer_learning.ipynb`
<!-- learn how to utilize transfer learning with the VGG16 model to adapt pre-trained features for a new classification task, highlighting efficient model adaptation without extensive new training. -->


## Object Detection Folder
`object_detection.ipynb`
- what is object detection
- computer vision
- detection modes
- multiple objects
- bouding boxes
- classes 

`yolo_object_Detection.ipynb`
1. Import the necessary libraries
2. Define the hyperparameter values
3. Define a helper function to download files
4. Pull the data from Roboflow
5. Clone the **YOLOv5** repository
6. Create a directory to store results
7. Run the model
8. Define a function to show validation predictions saved during training
9. Define a helper function for inference on images
10. Visualize inference images

`tensorFlow_lite.ipynb`
1. Import the required libraries
2. Create and save the model
3. Convert the Keras model to a TensorFlow lite model
4. Convert concrete functions

## Recurrent Neural Networks Folder

`recurrent_neural_networks_intro.ipynb`
- Sequential modeling
    - RNNs
    - Autoencoders
    - Seq2Seq
- Recurrent Neural Networks
- Types of RNN
    - one to one
    - one to many
    - many to one
    - many to many
- RNN architecture
    - initialization
    - input processing
    - hidden state update
    - output calculation
    - training
- long short term memory (LSTM)
- Gated Recurrent Network 
- Hybrid Modeling

`classification_rnns.ipynb`
1. Import the libraries
2. Define the hyperparameter
3. Preprocess the data and print the lengths of the labels and article lists
4. Split the data into training and validation sets
5. Initialize a tokenizer and fit it to the training articles
6. Convert the training articles into sequences using the tokenizer
7. Pad the sequence
8. Print the length of validation sequences and the shape of validation padded
9. Train the model
10. Compile the model
11. Plot the graph

`classification_lstm.ipynb`
1. Import the libraries
2. Define the hyperparameter
3. Preprocess the data and print the lengths of the labels and articles lists
4. Split the data into training and validation sets
5. Initialize a tokenizer and fitting it to the training articles
6. Convert the training articles into sequences using the tokenizer
7. Pad the sequence
8. Print the length of validation sequences and the shape of validation padded
9. Train the model
10. Compile the model
11. Plot the graph

`videoClassification_hybridNN.ipynb`
1. Download data and import the required libraries
2. Read the data from datasets and print the ten rows
3. Define the functions for cropping and loading video frames
4. Build a feature extraction model using the InceptionV3 architecture
5. Create a string lookup table for labels and print the vocabulary of the label processor
6. Prepare video data for training and testing by extracting frame features
7. Define and train a sequence model using GRU layers
8. Load a test video, extract frame features, and make predictions using the sequence model

## Transformer Models NLP folder

`bert_v3_intro.ipynb`
- Transformer models for NLP
- Self Attention
- layers of self attention
- Long range dependencies
- Transformer models
- Transformer architecture
- language translation
- BERT Model


1. Import the required libraries
2. Analyze the sentiment using the transformer pipeline
3. Create text generation
4. Create named entity recognition (NER)
5. Generate a masked language model using a model and a tokenizer


`text_bert_classification.ipynb`
1. Import the required libraries
2. Analyze the sentiment using the transformer pipeline
3. Create text generation
4. Create named entity recognition (NER)
5. Generate a masked language model using a model and a tokenizer

## Autoencoders Folder
`autoencoders_intro.ipynb`
- unsupervised deep learning
    - Clustering
    - Association
    - Dimensionality reduction
- Autoencoders and components
- Autoencoder Hyperparameters
- Autoencoder Use cases

## Projects: Course4 Folder

`deepLearning_tensorFlow_keras.ipynb`
-  vist the `README.md` in that folder to get scope of each project! 

`home_loan_data.ipynb`
-  vist the `README.md` in that folder to get scope of each project! 

