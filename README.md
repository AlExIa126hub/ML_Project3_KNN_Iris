# k-Nearest Neighbours (kNN) on the Iris Dataset

Machine Learning practical assignment focused on implementing the k-Nearest Neighbours (kNN) algorithm from scratch and applying it to Fisher’s Iris dataset.

The project explores how distance-based classification works, evaluates the effect of different k values and distance metrics, and compares the custom implementation with the kNN implementation provided by scikit-learn.

Dataset source:  
https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html


---

# Project Structure

```
knn-iris-from-scratch/
│
├── knn_iris.ipynb
└── README.md
```

---

# Project Overview

The goal of this project is to understand how the k-Nearest Neighbours (kNN) algorithm works by implementing it manually in Python.

The project applies kNN to classify flowers from the Iris dataset based on their physical measurements. The algorithm predicts the class of a data point by analysing the classes of its nearest neighbours in the feature space.

The workflow includes:

1. Implementing the kNN algorithm from scratch  
2. Testing different distance metrics  
3. Evaluating model performance using classification metrics  
4. Comparing the custom implementation with the scikit-learn implementation  

---

# Dataset

The Iris dataset is a classic dataset used for machine learning classification tasks. It contains measurements of iris flowers belonging to three species:

- Setosa  
- Versicolor  
- Virginica  

Each observation includes four numerical features:

- sepal length  
- sepal width  
- petal length  
- petal width  

The goal of the classification task is to correctly predict the **species of the flower** based on these measurements.

---

# k-Nearest Neighbours Algorithm

The kNN algorithm is a **distance-based classification method**.

For a given data point:

1. The distance between the point and all training samples is calculated.
2. The **k closest neighbours** are selected.
3. The predicted class is determined by **majority voting** among those neighbours.

---

# Distance Metrics

Two different distance metrics were implemented and compared.

### Euclidean Distance

```
d(x, y) = sqrt( Σ (xi - yi)^2 )
```

This is the most commonly used distance metric in kNN.

### Manhattan Distance

```
d(x, y) = Σ |xi - yi|
```

This metric measures the distance between points along grid-based paths.

---

# Model Experiments

The algorithm was tested with multiple values of **k**:

- k = 1  
- k = 3  
- k = 5  
- k = 7  

Testing different values of k helps understand how the neighbourhood size affects classification performance.

---

# Evaluation Metrics

Model performance was evaluated using standard classification metrics:

### Accuracy

```
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Precision measures how many predicted positives are actually correct.

### Recall

Recall measures how many actual positives are correctly identified.

These metrics help evaluate how well the algorithm performs in classifying different Iris species.

---

# Visualisation

Visualisations were created to analyse the results of the classifier.

Examples include:

- scatter plots of the dataset  
- visual indication of correctly vs incorrectly classified samples  
- comparisons between predictions from different k values  

These visualisations help illustrate how the algorithm separates the classes.

---

# Comparison with Scikit-learn

After implementing kNN manually, the results were compared with the implementation provided by **scikit-learn**.

The comparison focused on:

- classification accuracy  
- computational speed  

This comparison helps demonstrate the efficiency and optimisation of library implementations compared to basic custom implementations.

---

# Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- Jupyter Notebook  

---

# Running the Project

Install dependencies:

```
pip install numpy pandas matplotlib scikit-learn
```

Launch Jupyter Notebook:

```
jupyter notebook
```

Open and run:

```
knn_iris.ipynb
```

---

# Dataset

Iris Dataset (Scikit-learn)

https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
