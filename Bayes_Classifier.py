from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data          # X represents Data
Y = iris.target        # Y represents labels

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

def train_validate_test_split(data, labels, test_ratio=0.3, val_ratio=0.3, random_seed=None):
    #random_seed is used to ensure that the data is shuffled in a consistent way
    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate the sizes of the training, validation, and test sets
    num_samples = len(data)
    num_test = int(num_samples * test_ratio)
    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_test - num_val

    # Shuffle the data and labels to ensure randomness
    # To ensure that the data is randomly distributed across sets
    # so that the order doesn't affect the outcome.
    indices = np.random.permutation(num_samples)
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]

    # Split the data and labels into training, validation, and test sets

    X_train = shuffled_data[:num_train]
    y_train = shuffled_labels[:num_train]

    X_val = shuffled_data[num_train:num_train + num_val]
    y_val = shuffled_labels[num_train:num_train + num_val]

    X_test = shuffled_data[num_train + num_val:]
    y_test = shuffled_labels[num_train + num_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test



classifier = GaussianNB()
classifier.fit(X_train, Y_train)




# def calculate_accuracy(predicted_y, y):



# Draw decision boundaries



