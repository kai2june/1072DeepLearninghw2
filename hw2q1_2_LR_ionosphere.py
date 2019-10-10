import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# load file
ionosphere = scipy.io.loadmat('ionosphere.mat')

############# ionosphere
# split into 80%training, 20%testing
iono_train_feature, iono_test_feature, iono_train_label, iono_test_label = \
    train_test_split(ionosphere['X'], ionosphere['Y'], test_size=0.2, shuffle=True, random_state=113)

for i in range(len(iono_train_label)):
    iono_train_label[i] = iono_train_label[i][0]
for i in range(len(iono_test_label)):
    iono_test_label[i] = iono_test_label[i][0]
iono_train_label = iono_train_label.ravel()
iono_test_label = iono_test_label.ravel()

# logistic regression
iono_clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr', max_iter=1000)
iono_clf.fit(iono_train_feature, iono_train_label)
iono_clf_predict_train = iono_clf.predict(iono_train_feature)
iono_clf_accuracy_train = iono_clf.score(iono_train_feature, iono_train_label)
print("iono_train actual:\n", iono_train_label)
print("iono_train predict:\n", iono_clf_predict_train)
print("iono_train_LR accuracy: ", iono_clf_accuracy_train)
print("===========================================================================")
iono_clf_predict_test = iono_clf.predict(iono_test_feature)
iono_clf_accuracy_test = iono_clf.score(iono_test_feature, iono_test_label)
print("iono_test actual:\n", iono_test_label)
print("iono_test predict:\n", iono_clf_predict_test)
print("iono_test_LR accuracy: ", iono_clf_accuracy_test)
