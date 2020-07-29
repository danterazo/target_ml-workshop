# Title:    Natural Language Processing Support Vector Machine Demo for Target BegINNER Con
# Author:   Dante Razo, dante.razo@target.com
# Modified: 2020-07-29
import random
import time

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import sklearn.metrics


class Demo:
    def __init__(self):
        self.X_train = None  # type None until initialized with `get_data()`
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.svm = None  # blank until model is trained in `train_model()`
        self.analyzer = "word"  # CountVectorizer. "word" OR "char"
        self.ngram_upper_bound = 3  # CountVectorizer. lower is generally better
        self.kernel = "linear"  # SVM. "linear" OR "poly" OR "rbf" OR "sigmoid"
        self.gamma = "auto"  # SVM. kernel coefficient for rbf, poly, sigmoid kernels
        self.start_time = time.perf_counter()

    # Import data
    # Audience challenge: remove "http://t.co/*" links from data
    def get_data(self):
        data_dir = "../data/"  # defined up top for easy refactoring

        a = open(f"{data_dir}X_train", "r", encoding="utf-8")  # open plaintext file
        self.X_train = a.read().splitlines()  # given file, read line-by-line into a list

        a = open(f"{data_dir}X_test", "r", encoding="utf-8")  # open plaintext file
        self.X_test = a.read().splitlines()

        a = open(f"{data_dir}y_train", "r", encoding="utf-8")  # open plaintext file
        y_train = a.read().splitlines()
        self.y_train = [int(y) for y in y_train]  # typecast list members from str -> int using list comprehension

        a = open(f"{data_dir}y_test", "r", encoding="utf-8")  # open plaintext file
        y_test = a.read().splitlines()
        self.y_test = [int(y) for y in y_test]

        a.close()  # close files, free memory

    # Convert plaintext into features
    def feature_engineering(self):
        # ML models need features, not just plaintext. The CountVectorizer converts the latter into the former
        vec = CountVectorizer(analyzer=self.analyzer, ngram_range=(1, int(self.ngram_upper_bound)))
        print("Fitting CV...")
        self.X_train = vec.fit_transform(self.X_train)
        self.X_test = vec.transform(self.X_test)
        print("Fitting complete.")

    # fit SVM model on data
    def train_model(self):
        # Shuffle data (keeps indices)
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test)

        # Fitting the model
        print("Training SVM...")
        svm = SVC(kernel=self.kernel, gamma=self.gamma)
        svm.fit(self.X_train, self.y_train)
        print("Training complete.\n")
        self.svm = svm

    # predictions + print results
    def classification_report(self):
        rand_acc = sklearn.metrics.balanced_accuracy_score(self.y_test, [random.randint(1, 2) for x in range(0, len(self.y_test))])
        print(f"Baseline Accuracy: {rand_acc:}")  # accuracy of random baseline
        print(f"Classification Report [{self.kernel} kernel, {self.analyzer} analyzer, ngram_range(1,{self.ngram_upper_bound})]:\n "
              f"{sklearn.metrics.classification_report(self.y_test, self.svm.predict(self.X_test), digits=6)}")  # report
        print(f"\nTime Elapsed: {time.perf_counter() - self.start_time}s")  # print time elapsed since script was run


if __name__ == "__main__":
    demo = Demo()  # initialize empty Demo object

    demo.get_data()  # populate Demo object's instance attributes with data
    demo.feature_engineering()  # prepare data into something the SVM can utilize
    demo.train_model()  # train SVM on prepared data
    demo.classification_report()  # print scores to console
