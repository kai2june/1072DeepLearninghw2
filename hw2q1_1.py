import scipy.io
from sklearn.model_selection import train_test_split

fisheriris = scipy.io.loadmat('fisheriris.mat')
ionosphere = scipy.io.loadmat('ionosphere.mat')

# iris
iris_train_feature, iris_test_feature, iris_train_label, iris_test_label = \
    train_test_split(fisheriris['meas'], fisheriris['species'],  test_size=0.2, shuffle=True, random_state=113)
for i in range(len(iris_train_label)):
    iris_train_label[i] = iris_train_label[i][0]
for i in range(len(iris_test_label)):
    iris_test_label[i] = iris_test_label[i][0]
iris_train_label = iris_train_label.ravel()
iris_test_label = iris_test_label.ravel()

# ionosphere
iono_train_feature, iono_test_feature, iono_train_label, iono_test_label = \
    train_test_split(ionosphere['X'], ionosphere['Y'], test_size=0.2, shuffle=True, random_state=113)
for i in range(len(iono_train_label)):
    iono_train_label[i] = iono_train_label[i][0]
for i in range(len(iono_test_label)):
    iono_test_label[i] = iono_test_label[i][0]
iono_train_label = iono_train_label.ravel()
iono_test_label = iono_test_label.ravel()