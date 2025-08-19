# 06 Deep learning and advanced models

Overview

This cell contains many experiments across deep learning frameworks. It sets seeds for reproducibility, prepares TF-IDF inputs for neural networks, builds and evaluates simple Keras models, trains PyTorch models, compares Keras and PyTorch training times, runs sequential text models, fine tunes ResNet50 on images, and contains guarded YOLO training blocks.

Purpose

- Demonstrate end to end deep learning workflows in both Keras and PyTorch.
- Offer small models for quick demonstration and larger training paths for production experiments.
- Provide code for image layout normalization, data generators, model fine tuning and evaluation.

Line by line explanation and important details

1. Reproducibility and standard library imports
   - Set python random seed and numpy seed to fixed values to reduce randomness in runs.
   - Set `PYTHONHASHSEED` to make hash based operations deterministic where possible.

2. Core ML imports
   - Import scikit learn utilities for splitting data and TF-IDF.
   - Import Keras components including layers for RNNs and image models.
   - Import PyTorch utilities for building and training models on CPU or GPU.

3. Guarded YOLO import
   - Try to import `ultralytics.YOLO`. If it is not available, set a flag so YOLO sections are skipped gracefully during presentation.

4. Prepare TF-IDF features and labels
   - Build a TF-IDF matrix with up to 1000 features and convert it to a dense array for use with neural networks.
   - Prepare `y_keras` as one hot vectors and `y_pytorch` as integer labels for each framework.

5. Train test split for both frameworks
   - Split the same features into train and test sets with stratification so label distribution is similar across splits.

6. Keras perceptron example
   - Create and train a tiny single layer perceptron for quick demonstration of Keras model definition, compile, fit, and predict workflow.
   - Print a classification report comparing predicted labels to ground truth.

7. YOLO caching and conditional training
   - Provide helper to count label files under `dataset`. If labels exist and YOLO is available and there is no existing run, kick off a short training run with the ultralytics API. Use the `timed` decorator to print duration for the training.

8. PyTorch classification DNN
   - Define a small feed forward network using `nn.Sequential`, move data to tensors, create a DataLoader, and train for multiple epochs using a standard training loop with gradient steps.
   - After training run inference under `torch.no_grad()` and print a classification report.

9. Keras vs PyTorch timing comparison
   - Train the best available Keras model with a small number of epochs and time it.
   - Train a PyTorch model on the same data and compare training time and accuracy.
   - This provides a rough apples to apples comparison for demonstration purposes.

10. Sequential text models
   - Tokenize review text, convert to sequences, pad them for fixed length, and train several small sequence models: RNN, LSTM, GRU, and a CRNN that uses Conv1D followed by LSTM.
   - Use a helper `train_eval_keras` to compile, fit, and report classification metrics for each model.

11. Image layout normalization and ResNet50 fine tuning
   - Normalize dataset folder structure by moving images into `dataset/images/{train,val,test}` and grouping loose images into an `unknown` folder.
   - Use `ImageDataGenerator` to create train, val, test generators that feed images to Keras.
   - Build a model from `ResNet50` with a new classification head, unfreeze the last 30 layers for fine tuning, and train with `EarlyStopping`.
   - Evaluate on test set and show a confusion matrix heatmap.

12. YOLOv8 object detection section
   - A second guarded YOLO block that checks for installed ultralytics and the existence of `.txt` label files inside `dataset`. If conditions are met, run a short training session with parameters adjusted for smoke testing.

Inputs and outputs

- Inputs: TF-IDF feature arrays, tokenized sequences, image folders under `dataset/images`, and `SMOKE_TEST` setting.
- Outputs: trained Keras and PyTorch models in memory, evaluation metrics, and saved model exports optionally.

Notes and tips

- Many operations in this cell are heavy. Use `SMOKE_TEST = True` to reduce epochs, batch sizes and dataset sizes for a fast demo.
- For ResNet50 fine tuning, a GPU is strongly recommended. Training on CPU will be slow.
- The YOLO sections require the `ultralytics` package and the proper dataset structure with labels in YOLO format.
- Save heavy artifacts like model weights and vectorizers to disk if you need to preserve experiments across sessions.

---

End of 06 Deep learning and advanced models
