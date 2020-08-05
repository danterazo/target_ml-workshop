from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import numpy as np

""" Import data """
file = open("../data/X_train.txt", mode="rt", encoding="utf-8") # go out one folder -> go into `data` folder -> open `X_train.txt`
X_train = file.read().splitlines()  # if printed, would look like: [line1, line2, line3, ...]
file = open("../data/X_test.txt", mode="rt", encoding="utf-8")
X_test = file.read().splitlines()
file = open("../data/y_train.txt", mode="rt", encoding="utf-8")
y_train = file.read().splitlines()
file = open("../data/y_test.txt", mode="rt", encoding="utf-8")
y_test = file.read().splitlines()

file.close()  # close file to save memory

""" Typecast str -> int in `y` sets """
y_train = [int(y) for y in y_train]  # list comprehension. for each member of `y_train`, make it an integer
y_test = [int(y) for y in y_test]

""" Initialize and fit CountVectorizer """
vec = CountVectorizer(analyzer="word", ngram_range=(1, 3))  # ngram(1,3): consider single words or groups of 2-3 words
X_train = vec.fit_transform(X_train)  # fit CountVectorizer to AND transform `X_train` in one step
X_test = vec.transform(X_test)  # transform `X_text`; no need to fit the CountVectorizer again

""" Initialize and fit SVM model """
my_svm = SVC(kernel="linear")  # initalize SVM model. feel free to experiment with kernels!
my_svm.fit(X_train, y_train)  # fit SVM model to training data

""" Get baseline accuracy (random) """
ytest_size = len(y_test)  # size of `y_test`
array_of_random_preds = np.random.randint(1, 3, ytest_size)  # create an random array of 1's and 2's. length is the same as `y_test`
baseline_accuracy = balanced_accuracy_score(y_test, array_of_random_preds)  # compare y_test with random array
print(f"baseline accuracy: {baseline_accuracy}")  # expected: ~0.50, i.e. 50%

""" Get model accuracy """
predictions = my_svm.predict(X_test)  # given the lines we set aside for testing, the model will predict what it thinks its class should be
model_accuracy = balanced_accuracy_score(y_test, predictions)
print(f"model accuracy: {model_accuracy}")  # print report; expected: ~0.81, i.e. 81% accuracy
