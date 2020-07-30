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
        self.start_time = time.perf_counter()  # start timer when object is initialized

    # Import data
    # Audience challenge: remove "http://t.co/*" links from data
    def get_data(self):
        opened_file = self.open_file("X_train")  # get File() object
        self.X_train = opened_file.read().splitlines()  # given File(), read line-by-line into a list

        opened_file = self.open_file("X_test")
        self.X_test = opened_file.read().splitlines()

        opened_file = self.open_file("y_train")
        y_train = opened_file.read().splitlines()
        self.y_train = [int(y) for y in y_train]  # typecast list members from str -> int using list comprehension

        opened_file = self.open_file("y_test")
        y_test = opened_file.read().splitlines()
        self.y_test = [int(y) for y in y_test]

        opened_file.close()  # close files, free memory

    # helper function. given a filename, return a File() object with appropriate params
    @staticmethod  # this method is "static" because it is not affected by the object state
    def open_file(filename):
        data_dir = "../data/"  # params defined up top for easy refactoring
        mode = "r"
        encoding = "utf-8"

        return open(f"{data_dir}{filename}", mode=mode, encoding=encoding)

    # Convert plaintext into features
    def feature_engineering(self):
        # ML models need features, not just plaintext. The CountVectorizer converts the latter into the former
        vec = CountVectorizer(analyzer=self.analyzer, ngram_range=(1, int(self.ngram_upper_bound)))
        print("Fitting CountVectorizer...")
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
        print(f"Time Elapsed: {time.perf_counter() - self.start_time} seconds")  # print time elapsed since script was run


if __name__ == "__main__":
    demo = Demo()  # initialize new Demo object

    demo.get_data()  # populate Demo object's instance attributes with data
    demo.feature_engineering()  # prepare data into something the SVM can utilize
    demo.train_model()  # train SVM on prepared data
    demo.classification_report()  # print scores to console
