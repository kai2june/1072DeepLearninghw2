import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# load file
fisheriris = scipy.io.loadmat('fisheriris.mat')

############# iris
# split into 80%training, 20%testing
iris_train_feature, iris_test_feature, iris_train_label, iris_test_label = \
    train_test_split(fisheriris['meas'], fisheriris['species'],  test_size=0.2, shuffle=True, random_state=113)

for i in range(len(iris_train_label)):
    iris_train_label[i] = iris_train_label[i][0]
for i in range(len(iris_test_label)):
    iris_test_label[i] = iris_test_label[i][0]
iris_train_label = iris_train_label.ravel()
iris_test_label = iris_test_label.ravel()

# logistic regression
iris_clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000)
iris_clf.fit(iris_train_feature, iris_train_label)
iris_clf_predict_train = iris_clf.predict(iris_train_feature)
iris_clf_accuracy_train = iris_clf.score(iris_train_feature, iris_train_label)
print("iris_train actual:\n", iris_train_label)
print("iris_train predict:\n", iris_clf_predict_train)
print("iris_train_LR accuracy: ", iris_clf_accuracy_train)
print("===========================================================================")
iris_clf_predict_test = iris_clf.predict(iris_test_feature)
iris_clf_accuracy_test = iris_clf.score(iris_test_feature, iris_test_label)
print("iris_test actual:\n", iris_test_label)
print("iris_test predict:\n", iris_clf_predict_test)
print("iris_test_LR accuracy: ", iris_clf_accuracy_test)
print(iris_clf.coef_)
