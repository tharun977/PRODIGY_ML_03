# Cat and Dog Image Classification Using Support Vector Machines (SVM)

This project implements a Support Vector Machine (SVM) model to classify images of cats and dogs using the Kaggle dataset. The aim is to accurately identify the type of animal in the images by extracting relevant features and training a classifier.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The project consists of several stages:
1. **Data Preparation**: Load and preprocess the dataset by resizing and normalizing the images.
2. **Feature Extraction**: Use Histogram of Oriented Gradients (HOG) to extract relevant features from the images.
3. **Model Development**: Implement the SVM classifier to train on the extracted features.
4. **Performance Evaluation**: Assess the model's accuracy using various metrics.

## Technologies Used

- **Programming Languages**: Python
- **Libraries**:
  - TensorFlow
  - scikit-learn
  - NumPy
  - OpenCV
  - Pandas

## Dataset

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats). It contains images of cats and dogs, which have been labeled accordingly.

## Installation

To run this project, ensure you have the following installed:

1. Python 3.x
2. Required libraries can be installed using pip:

pip install tensorflow scikit-learn numpy opencv-python pandas
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/tharun977/PRODIGY_ML_03.git
cd PRODIGY_ML_03
Place the dataset in the appropriate directory structure as required by the project.

Run the main script:

bash
Copy code
python main.py
Training the Model
The model is trained using the provided dataset. The training process involves feature extraction using HOG and fitting the SVM model with the extracted features.

Evaluation
After training, the model is evaluated using metrics such as accuracy, precision, and recall. Results can be found in the evaluation section of the script.

Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

License
This project is licensed under the MIT License - see the LICENSE file for details.
