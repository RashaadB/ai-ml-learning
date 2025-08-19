# Capstone Project

This folder contains the files and experiments for a capstone project that uses natural language processing and computer vision methods. The goal of this README is to explain what is in the folder, how the data and models are organized, and how to run or reproduce the main steps. The instructions use plain language and avoid special characters.

## High level overview

- The project mixes two main areas: text processing and object detection. Text processing materials include review data and precomputed text features. Object detection materials include a dataset organized for YOLO, a YOLO model file, and training outputs.
- The main notebook for exploration and reproducible steps is `capstone_project_one.ipynb`.

## Important files and folders

Below is a short description for each important file or folder in this directory.

- `capstone_project_one.ipynb`  
  Main notebook. It contains the project narrative, data exploration steps, and code to create or load artifacts used in the analysis. Open this notebook in Jupyter or VS Code's notebook viewer.

- `amazon_reviews_us_Apparel_v1_00.csv`  
  A CSV file with Amazon apparel review records. This file is used for text analysis and feature extraction.

- `product_data.jsonl`  
  A JSON Lines file that may contain product metadata. It is useful for joining product details with reviews or for additional preprocessing.

- `image_labels.csv`  
  A CSV that contains mapping or labels for images used in the object detection task. Inspect the file to confirm the columns used in your workflow.

- `Images.zip`  
  A zip archive with image files. The dataset folder contains unzipped images organized for model training and validation.

- `yolo_dataset.yaml`  
  A dataset configuration file used by YOLO training code. It points to the image and label directories and lists class names. Use or update it when training a YOLO model.

- `yolov8n.pt`  
  A YOLOv8 model file. This may be a pretrained model or a saved checkpoint. It can be used for inference or as a starting point for further training.

- `artifacts/`  
  Precomputed artifacts. For example:
  - `tfidf.pkl`  contains a fitted TF-IDF vectorizer object.  
  - `tfidf_matrix.pkl`  contains the TF-IDF matrix produced from text features.
  These files are useful for running the text model pipeline without recomputing the vectorizer.

- `dataset/`  
  The dataset folder is organized for YOLO training. It contains `images` and `labels` with `train`, `val`, and `test` subfolders. Each label file follows the YOLO text format where each line is `class x_center y_center width height` normalized to image size.

- `runs/`  
  Training and detection outputs created by the object detection training code. Subfolders such as `detect/capstone_yolo_model` contain inference images and logs from detection runs.

- `flagged/`  
  A folder used by the project to save examples or images that need review. The exact contents may vary across runs.


## Quick setup

1. Python version 3.8 through 3.11 is recommended. Create and activate a virtual environment. On macOS with zsh you can run:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install required packages. There is a `requirements.txt` at the workspace root. From the repository root run:

```bash
pip install -r requirements.txt
```

If your environment does not include packages needed for YOLO, install the official package that provides the training and inference commands. For example, install `ultralytics` if you use YOLOv8. If you install additional packages, pin versions that match your system or GPU drivers.

3. If you need to unzip images, run:

```bash
unzip "Images.zip" -d "Images"
```

Adjust the path if you prefer to extract into the `dataset/images` folder.


## How to open and run the main notebook

1. Start Jupyter Lab or Notebook from the repository root with the environment active.

```bash
jupyter lab
```

2. Open `Capstone Project/capstone_project_one.ipynb`.

3. Follow the notebook cells in order. The notebook contains code to load the CSV files, create or load TF-IDF objects, and run experiments.


## Text processing artifacts

- The `artifacts` folder contains pickled objects that speed up text experiments. If you want to recreate them, look for cells in the notebook that fit a `TfidfVectorizer` or similar transform and save it using `pickle` or `joblib`.
- If you change tokenizer settings or the training data, you should rebuild these artifacts and save them with a new name.


## Object detection workflows

The repository has dataset files and a YOLO model file. The most common workloads are training, evaluating, and doing inference.

1. Training with YOLOv8 using the ultralytics package. A typical command line looks like this:

```bash
yolo detect train data="Capstone Project/yolo_dataset.yaml" model="yolov8n.pt" epochs=50 imgsz=640
```

Change `epochs`, `imgsz`, and other parameters to match your compute limits. Training outputs go to `runs/train` by default.

2. Run detection (inference) on images or a folder. A typical command:

```bash
yolo detect predict model="runs/train/your_run/weights/best.pt" source="dataset/images/test" save=True
```

The predictions will be saved under `runs/detect` unless you set a different `save_dir`.

3. If you only want to run inference with `yolov8n.pt` that is included in this folder, use it as the `model` in the command above.


## Dataset format notes

- Images are under `dataset/images` with `train`, `val`, and `test` splits.
- Labels are under `dataset/labels` with text files named to match image file names. Each label file uses one row per object in the YOLO format.
- If you update the dataset structure, update `yolo_dataset.yaml` so the training code points to the right image and label paths.


## Reproducibility tips and common issues

- Data size. Some files are large. Do not store very large files in version control. Use the provided `Images.zip` only when you need the raw images. If your system runs out of memory during training, reduce the batch size and image size.
- GPU drivers and torch. If you plan to train with a GPU, make sure you have a matching CUDA toolkit and a compatible PyTorch build installed. Use the correct wheel or conda package for your platform.
- Package versions. If you run into errors with a package API, try pinning package versions that were used when the project was developed. Save a new `requirements.txt` after you confirm a working set.


## Where results are stored

- Model weights, logs, and prediction images created by YOLO training or detection go to the `runs` folder. Each run has its own subfolder. Inspect the latest subfolder to find `weights`, `results.csv`, and prediction images.


## Next steps and suggestions

- If you want to continue the project, consider the following steps:
  - Add a script to run a full training pipeline from raw data to final model. That script can call the notebook code or run command line training as shown earlier.
  - Create a smaller example subset for fast iteration. A small, curated subset of images and labels helps test changes quickly.
  - Add unit tests for data loading and label parsing so mistakes are caught early.


## Notes about authorship and license

This repository contains work produced for a capstone project. If you share the project, add a license file to clarify how others can use the code and data.


## Contact and further help

If you need specific help running a notebook, training the YOLO model, or reproducing the TF-IDF artifacts, tell me the operating system, Python version, and whether you have a GPU. I can provide step by step commands to run locally.


## Short checklist to get started

1. Create and activate a virtual environment.  
2. Install dependencies with `pip install -r requirements.txt`.  
3. Unzip `Images.zip` if you need the raw images.  
4. Open `capstone_project_one.ipynb` in Jupyter and run cells in order.  
5. If training detection models, use the `yolo` command shown earlier and monitor outputs in `runs`.


End of README
