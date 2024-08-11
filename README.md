# Bird Species Classification

This project focuses on classifying bird species based on audio recordings. It uses various machine learning classifiers to predict the species of a bird from the extracted audio features. The project involves data preprocessing, feature extraction, model training, evaluation, and visualization.

## Table of Contents

- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

Before you begin, ensure you have the following libraries installed:

- Python 3.6+
- `numpy`
- `librosa`
- `scikit-learn`
- `matplotlib`
- `jupyter`

You can install these dependencies using `pip`:

```bash
pip install numpy librosa scikit-learn matplotlib jupyter

```
### Clone the Repository

```bash
git clone https://github.com/yourusername/bird-species-classification.git
cd bird-species-classification

```

### Dataset Structure

The project assumes that the dataset is organized in the following structure:

```bash 

/path/to/dataset/
    critically_endangered/
        species1/
            file1.mp3
            file2.mp3
            ...
        species2/
            file1.mp3
            ...
    endangered/
        species1/
            file1.mp3
            ...
    vulnerable/
        species1/
            file1.mp3
            ...

```

Additionally, for evaluation purposes, you should have a test set:

```bash 
/path/to/dataset/
    critically_endangered_test/
        species1/
            file1.wav
            ...
    endangered_test/
        species1/
            file1.wav
            ...
    vulnerable_test/
        species1/
            file1.wav
            ...
```


### Feature Extraction
The project extracts the following features from each audio file:

MFCCs (Mel-frequency cepstral coefficients)
* Spectral Centroid
* Zero-Crossing Rate
* Spectral Bandwidth
* Spectral Contrast
* Chroma Features
* Root Mean Square (RMS) Energy
* Spectral Rolloff
* Mel Spectrogram

These features are concatenated into a single feature vector for each audio file.

### Model Training
Several classifiers are used to train on the extracted features:

* Random Forest
* Gradient Boosting
* k-Nearest Neighbors (k-NN)
* Decision Tree
* Support Vector Machine (SVM)
* Multi-layer Perceptron (MLP)
* Naive Bayes
* Logistic Regression

The features are first normalized using StandardScaler, and optionally, dimensionality reduction is applied using PCA (Principal Component Analysis).

### Training Process
1. Data Preprocessing: The data is preprocessed to extract features and labels.
2. Data Shuffling: The dataset is shuffled randomly to avoid any bias.
3. Data Splitting: The dataset is split into training and testing sets.
4. Normalization: Features are normalized using a standard scaler.
5. Dimensionality Reduction: PCA is applied to reduce feature dimensionality.
6. Model Training: Classifiers are trained on the training set.


### Evaluation 

## Initial Split Test
The models are first evaluated on an initial split test dataset (20% of the original data). Accuracy scores for each classifier are calculated and displayed.

## Final Test Evaluation
The models are then evaluated on a separate test set that contains data not seen during the initial training and testing phases. Accuracy scores and a confusion matrix for the best classifier are displayed.

## Results
Accuracy Scores
The accuracy scores for all classifiers on both the initial split test data and the final test data are plotted and printed.

## Confusion Matrix
The confusion matrix for the best-performing classifier is displayed, showing how well the classifier distinguishes between different bird species.


## Usage

To run the project:

1. **Ensure the dataset is structured correctly** as mentioned above.
2. **Place the IPython notebook** (`prediction.ipynb`) in the root directory of the dataset.
3. **Execute the notebook** using Jupyter Notebook or JupyterLab:

    ```bash
    jupyter notebook prediction.ipynb
    ```

    or

    ```bash
    jupyter lab prediction.ipynb
    ```

4. **Run the cells** in the notebook to execute the code. The notebook will output accuracy scores for each classifier and display confusion matrices.

The script will output accuracy scores for each classifier and display confusion matrices.
Contributing
Contributions are welcome! Please fork the repository and submit a pull request if you have any improvements or bug fixes.

### License
This project is licensed under the MIT [LICENSE](LICENSE). See the LICENSE file for more details.