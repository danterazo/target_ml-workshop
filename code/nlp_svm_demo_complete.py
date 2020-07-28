# Title:    NLP SVM Demo for Target BegINNER Con
# Author:   Dante Razo, dante.razo@target.com
# Modified: 2020-07-28
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import sklearn.metrics
import random


# Import data
# Audience challenge: remove "http://t.co/*" links from data
def get_data():
    data_dir = "../data/"  # defined up top for easy refactoring

    a = open(f"{data_dir}X_train", "r", encoding="utf-8")  # open plaintext file
    X_train = a.read().splitlines()  # given file, read line-by-line into a list

    a = open(f"{data_dir}X_test", "r", encoding="utf-8")  # open plaintext file
    X_test = a.read().splitlines()

    a = open(f"{data_dir}y_train", "r", encoding="utf-8")  # open plaintext file
    y_train = a.read().splitlines()
    for i in range(0, len(y_train)):
        y_train[i] = int(y_train[i])

    a = open(f"{data_dir}y_test", "r", encoding="utf-8")  # open plaintext file
    y_test = a.read().splitlines()
    for i in range(0, len(y_test)):
        y_test[i] = int(y_test[i])

    a.close()
    return X_train, X_test, y_train, y_test


def feature_engineering():
    # params
    analyzer = "word"  # CV. "word" OR "char"
    ngram_upper_bound = 3  # CV. Probably best to keep it lower
    kernel = "linear"  # SVM. "linear" OR "poly" OR "rbf" OR "sigmoid"
    gamma = "auto"  # SVM. kernel coefficient for rbf, poly, sigmoid kernels

    # Get data, assign it to variables
    X_train, X_test, y_train, y_test = get_data()

    # Convert plaintext into features
    # ML models need features, not just plaintext. The CountVectorizer converts the latter into the former
    vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, int(ngram_upper_bound)))
    print("Fitting CV...")
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)

    # Shuffle data (keeps indices)
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # Fitting the model
    print("Training SVM...")
    svm = SVC(kernel=kernel, gamma="auto")
    svm.fit(X_train, y_train)
    print("Training complete.")

    # Testing + results
    rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(1, 2) for x in range(0, len(y_test))])
    print(f"Baseline Accuracy: {rand_acc:}\n")  # random baseline
    print(f"Classification Report [{kernel} kernel, {analyzer} analyzer, ngram_range(1,{ngram_upper_bound})]:\n "
          f"{sklearn.metrics.classification_report(y_test, svm.predict(X_test), digits=6)}")  # report


if __name__ == "__main__":
    feature_engineering()
