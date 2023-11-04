from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data          # X represents Data
Y = iris.target        # Y represents labels

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

# def train_test_split(data, labels, testRatio = 0.3, valRatio = 0.3):


classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# def calculate_accuracy(predicted_y, y):

# Draw decision boundaries


